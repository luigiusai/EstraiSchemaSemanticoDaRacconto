#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sorgente Python per l'analisi dell'intreccio semantico in un PDF
Secondo il modello teorico dettagliato nei documenti forniti (Autore: Luigi Usai).
Il programma implementa i seguenti passaggi metodologici:
  - Estrazione del testo da ogni pagina del documento PDF.
  - Tokenizzazione del testo di ogni pagina e ottenimento degli embedding vettoriali
    per ciascun token (parola significativa) utilizzando un modello linguistico spaCy.
  - Costruzione di un grafo semantico per ogni pagina:
    - I nodi del grafo rappresentano i token (parole) della pagina.
    - Un arco (connessione) viene creato tra due token se la loro similarità semantica
      (calcolata tramite similarità coseno tra i rispettivi embedding) supera una soglia T predefinita.
  - Calcolo di metriche statistiche sulla rete (grafo semantico) per ogni pagina:
    - Numero di nodi (parole nel grafo).
    - Numero di archi (relazioni semantiche significative).
    - Lista dei gradi di ciascun nodo (utile per analizzare la distribuzione dei gradi P(k) e stimare l'esponente gamma).
    - Coefficiente di clustering medio del grafo.
    - Lunghezza media del cammino minimo (calcolata sul componente connesso più grande, se il grafo non è connesso).
  - Salvataggio dei risultati (metriche per pagina) in un file CSV per successive analisi statistiche
    e per la verifica delle ipotesi di invarianza dell'intreccio semantico, come descritto nel modello teorico.

Autori: Luigi Usai e LLM
Data: Maggio 2025
Pubblicato su (ispirato da): https://zenodo.org/records/15484259 (ID originale)
"""

import argparse
import csv
import fitz  # PyMuPDF, per l'estrazione del testo da PDF
import spacy # Per l'elaborazione del linguaggio naturale (NLP)
import numpy as np
import networkx as nx # Per la creazione e l'analisi dei grafi
from sklearn.metrics.pairwise import cosine_similarity # Per calcolare la similarità coseno
import os

# Parametro globale: soglia di similarità (T) per stabilire un arco nel grafo semantico.
# Questo valore corrisponde alla soglia tau (τ) nel modello teorico.
SOGLIA_SIMILARITA_DEFAULT = 0.7

def estrai_testo_da_pdf(percorso_file_pdf):
    """
    Estrae il testo da ogni pagina di un file PDF.
    Corrisponde al passo di acquisizione del contenuto testuale della pagina p.

    Args:
        percorso_file_pdf (str): Il percorso del file PDF da analizzare.

    Returns:
        list: Una lista di stringhe, dove ogni stringa è il testo di una pagina del PDF.
              Restituisce una lista vuota se il file non può essere aperto o è vuoto.
    """
    try:
        documento_pdf = fitz.open(percorso_file_pdf)
    except Exception as e:
        print(f"Errore nell'apertura del file PDF '{percorso_file_pdf}': {e}")
        return []
        
    testi_pagine = []
    if documento_pdf.page_count == 0:
        print(f"Attenzione: Il file PDF '{percorso_file_pdf}' non contiene pagine.")
        return []

    for numero_pagina in range(len(documento_pdf)):
        pagina = documento_pdf.load_page(numero_pagina)
        testo_pagina = pagina.get_text("text")
        testi_pagine.append(testo_pagina)
    documento_pdf.close()
    return testi_pagine

def elabora_testo_pagina(testo_pagina, modello_linguistico_spacy):
    """
    Elabora il testo di una singola pagina utilizzando un modello linguistico spaCy.
    Esegue la tokenizzazione e l'estrazione degli embedding vettoriali per le parole significative.
    Corrisponde all'estrazione di parole(p) e al calcolo degli embedding f(w) per ogni parola w.

    Args:
        testo_pagina (str): Il testo della pagina da elaborare.
        modello_linguistico_spacy (spacy.lang): Il modello spaCy caricato.

    Returns:
        tuple: Una tupla contenente:
            - lista_parole (list): Lista delle stringhe dei token (parole) significativi.
            - vettori_embedding (np.array): Array NumPy degli embedding corrispondenti ai token.
                                           Array vuoto se non ci sono token significativi.
    """
    documento_spacy = modello_linguistico_spacy(testo_pagina)
    lista_parole = []
    lista_vettori_embedding = []
    
    for token in documento_spacy:
        # Filtro per token significativi:
        # - Non deve essere punteggiatura.
        # - Non deve essere uno spazio.
        # - La lunghezza del testo del token deve essere maggiore di 1 (per escludere lettere singole poco significative).
        # - Il token deve avere un vettore di embedding (token.has_vector).
        if not token.is_punct and not token.is_space and len(token.text) > 1 and token.has_vector:
            lista_parole.append(token.text)
            lista_vettori_embedding.append(token.vector)
            
    if not lista_vettori_embedding: # Se non sono stati trovati token con vettori
        return [], np.array([])

    return lista_parole, np.array(lista_vettori_embedding)

def costruisci_grafo_semantico_pagina(lista_parole, vettori_embedding, soglia_similarita):
    """
    Costruisce il grafo semantico G_p = (V, E) per una pagina, basandosi sulla similarità semantica tra le parole.
    V = parole(p) (i token).
    E = {(w_i, w_j) | S(w_i, w_j) > τ} (archi basati sulla similarità coseno e la soglia).

    Args:
        lista_parole (list): Lista dei token (stringhe) che saranno i nodi del grafo.
        vettori_embedding (np.array): Array NumPy degli embedding per ogni token.
        soglia_similarita (float): La soglia (τ) per la similarità coseno;
                                     se S(w_i, w_j) > τ, viene creato un arco.

    Returns:
        nx.Graph: Il grafo semantico costruito. Può essere un grafo vuoto se non ci sono parole
                  o se nessun embedding è fornito.
    """
    grafo_semantico = nx.Graph()
    numero_parole = len(lista_parole)

    if numero_parole == 0 or vettori_embedding.size == 0:
        return grafo_semantico

    # Aggiunge i nodi al grafo. Ogni nodo corrisponde a una parola (token).
    # L'ID del nodo è l'indice nella lista_parole, e l'attributo 'label' contiene il testo del token.
    for i, parola_testo in enumerate(lista_parole):
        grafo_semantico.add_node(i, label=parola_testo)
    
    # Se c'è una sola parola, non ci possono essere archi.
    if numero_parole < 2:
        return grafo_semantico
        
    # Assicura che vettori_embedding sia un array 2D prima di passarlo a cosine_similarity.
    # Questo è importante se, per qualche motivo, ci fosse un solo vettore passato come 1D.
    vettori_2d = vettori_embedding
    if vettori_embedding.ndim == 1:
        # Questo caso non dovrebbe verificarsi se numero_parole >= 2 e la logica precedente è corretta,
        # ma è una protezione aggiuntiva.
        if numero_parole == 1: # Già gestito sopra, ma per coerenza
             return grafo_semantico 
        else: # Potenziale incoerenza dati
            print(f"Attenzione: Incoerenza rilevata. {numero_parole} parole ma vettori_embedding con dimensione {vettori_embedding.ndim}.")
            return grafo_semantico # Restituisce grafo senza archi

    # Calcola la matrice di similarità coseno tra tutti gli embedding.
    # S(w_i, w_j) = cos_sim(f(w_i), f(w_j))
    matrice_similarita = cosine_similarity(vettori_2d)
    
    # Aggiunge gli archi al grafo.
    # Un arco (i, j) viene aggiunto se la similarità tra la parola i e la parola j supera la soglia.
    # Si considerano solo le coppie (i, j) con i < j per evitare duplicati e auto-loop in un grafo non orientato.
    for i in range(numero_parole):
        for j in range(i + 1, numero_parole):
            if matrice_similarita[i, j] > soglia_similarita:
                # L'attributo 'weight' dell'arco memorizza il valore della similarità.
                grafo_semantico.add_edge(i, j, weight=matrice_similarita[i, j])
                
    return grafo_semantico

def calcola_metriche_grafo(grafo_semantico):
    """
    Calcola le metriche statistiche del grafo semantico G_p, come descritto nel modello teorico.
    Le metriche includono: numero di nodi, numero di archi, lista dei gradi,
    coefficiente di clustering medio (C(G_p)), e lunghezza media del cammino minimo (L(G_p)).

    Args:
        grafo_semantico (nx.Graph): Il grafo semantico da analizzare.

    Returns:
        dict: Un dizionario contenente le metriche calcolate.
    """
    metriche = {}
    
    num_nodi = grafo_semantico.number_of_nodes()
    metriche['numero_nodi_grafo'] = num_nodi
    metriche['numero_archi_grafo'] = grafo_semantico.number_of_edges()
    
    if num_nodi == 0:
        # Se il grafo è vuoto, le altre metriche sono 0 o non definite.
        metriche['gradi_dei_nodi'] = []
        metriche['coefficiente_clustering_medio'] = 0.0
        metriche['lunghezza_media_cammino_minimo'] = 0.0 # Convenzione per grafo vuoto
        return metriche

    # Gradi dei nodi: k_v per ogni nodo v. Utile per P(k).
    gradi = [grado for nodo, grado in grafo_semantico.degree()]
    metriche['gradi_dei_nodi'] = gradi

    # Coefficiente di clustering medio: C(G_p)
    # Gestisce eccezioni per grafi molto piccoli o sparsi dove il clustering potrebbe non essere definito.
    try:
        metriche['coefficiente_clustering_medio'] = nx.average_clustering(grafo_semantico)
    except nx.NetworkXError: 
        # Questo può accadere se il grafo ha meno di 2 nodi o nessun nodo ha almeno 2 vicini.
        metriche['coefficiente_clustering_medio'] = 0.0

    # Lunghezza media del cammino minimo: L(G_p)
    # Calcolata sul componente connesso più grande se il grafo non è connesso.
    # Se il grafo ha nodi ma non archi, o è sconnesso, la gestione è importante.
    if num_nodi > 0:
        if nx.is_connected(grafo_semantico):
            try:
                # Per grafi con un solo nodo, average_shortest_path_length solleva NetworkXError.
                if num_nodi == 1:
                    metriche['lunghezza_media_cammino_minimo'] = 0.0
                else:
                    metriche['lunghezza_media_cammino_minimo'] = nx.average_shortest_path_length(grafo_semantico)
            except nx.NetworkXError: # Gestione per casi imprevisti, anche se num_nodi=1 è il più comune.
                 metriche['lunghezza_media_cammino_minimo'] = 0.0 # Convenzione per un solo nodo
        else:
            # Il grafo non è connesso. Calcola sul componente connesso più grande.
            componenti_connesse = list(nx.connected_components(grafo_semantico))
            if componenti_connesse: # Se ci sono componenti
                componente_piu_grande_nodi = max(componenti_connesse, key=len)
                sottografo_componente_grande = grafo_semantico.subgraph(componente_piu_grande_nodi)
                if sottografo_componente_grande.number_of_nodes() > 1:
                    try:
                        metriche['lunghezza_media_cammino_minimo'] = nx.average_shortest_path_length(sottografo_componente_grande)
                    except Exception: # Gestione generica per errori nel calcolo sul sottografo
                        metriche['lunghezza_media_cammino_minimo'] = float('inf') # Indica che non è calcolabile
                else: # Il componente più grande ha 0 o 1 nodo
                    metriche['lunghezza_media_cammino_minimo'] = 0.0
            else: # Non ci sono componenti (improbabile se num_nodi > 0, ma per sicurezza)
                metriche['lunghezza_media_cammino_minimo'] = 0.0
    else: # num_nodi == 0
         metriche['lunghezza_media_cammino_minimo'] = 0.0

    return metriche

def analizza_documento_pdf(percorso_file_pdf, modello_linguistico_spacy, file_csv_risultati, soglia_similarita_attuale):
    """
    Funzione principale per l'elaborazione dell'intero documento PDF.
    Analizza ogni pagina, costruisce il grafo semantico, calcola le metriche
    e salva i risultati aggregati in un file CSV.
    Implementa la "Procedura Operativa per la Validazione Sperimentale" descritta nel modello teorico,
    generando il database di metriche per pagina.

    Args:
        percorso_file_pdf (str): Percorso del file PDF da analizzare.
        modello_linguistico_spacy (spacy.lang): Modello spaCy caricato.
        file_csv_risultati (str): Nome del file CSV in cui salvare i risultati.
        soglia_similarita_attuale (float): Soglia di similarità da utilizzare.
    """
    print(f"Estrazione del testo dal PDF: {percorso_file_pdf}...")
    testi_pagine_pdf = estrai_testo_da_pdf(percorso_file_pdf)
    
    if not testi_pagine_pdf:
        print("Nessun testo estratto dal PDF. L'analisi non può procedere.")
        return

    print(f"Numero di pagine estratte: {len(testi_pagine_pdf)}.")
    lista_risultati_per_csv = []
    
    for indice_pagina, testo_singola_pagina in enumerate(testi_pagine_pdf):
        numero_pagina_corrente = indice_pagina + 1
        print(f"\nAnalisi della pagina {numero_pagina_corrente} / {len(testi_pagine_pdf)}...")
        
        parole_della_pagina, vettori_embedding_parole = elabora_testo_pagina(testo_singola_pagina, modello_linguistico_spacy)
        
        metriche_pagina_corrente = {'pagina': numero_pagina_corrente, 
                                    'numero_token_originali': len(parole_della_pagina)}

        if len(parole_della_pagina) == 0 or vettori_embedding_parole.size == 0:
            print(f"Pagina {numero_pagina_corrente}: Testo vuoto o nessun token significativo trovato dopo l'elaborazione.")
            # Aggiunge metriche di default per pagine vuote o senza token utili
            metriche_pagina_corrente.update({
                'numero_nodi_grafo': 0, 
                'numero_archi_grafo': 0, 
                'gradi_dei_nodi': "[]", # Lista vuota come stringa
                'coefficiente_clustering_medio': 0.0, 
                'lunghezza_media_cammino_minimo': 0.0
            })
            lista_risultati_per_csv.append(metriche_pagina_corrente)
            continue
        
        print(f"Pagina {numero_pagina_corrente}: Numero di token significativi estratti: {len(parole_della_pagina)}.")
        
        grafo_semantico_costruito = costruisci_grafo_semantico_pagina(parole_della_pagina, 
                                                                     vettori_embedding_parole, 
                                                                     soglia_similarita_attuale)
        
        print(f"Pagina {numero_pagina_corrente}: Grafo semantico costruito con {grafo_semantico_costruito.number_of_nodes()} nodi e {grafo_semantico_costruito.number_of_edges()} archi.")
        
        metriche_del_grafo = calcola_metriche_grafo(grafo_semantico_costruito)
        
        # Unisce le informazioni della pagina con le metriche del grafo
        metriche_pagina_corrente.update(metriche_del_grafo)
        # Converte la lista dei gradi in una stringa per il CSV
        metriche_pagina_corrente['gradi_dei_nodi'] = str(metriche_pagina_corrente['gradi_dei_nodi'])

        lista_risultati_per_csv.append(metriche_pagina_corrente)
        print(f"Pagina {numero_pagina_corrente}: Metriche calcolate: {metriche_pagina_corrente}")

    # Scrittura dei risultati nel file CSV
    # I nomi dei campi (fieldnames) corrispondono alle chiavi nei dizionari dei risultati.
    nomi_colonne_csv = [
        'pagina', 'numero_token_originali', 'numero_nodi_grafo', 'numero_archi_grafo', 
        'gradi_dei_nodi', 'coefficiente_clustering_medio', 'lunghezza_media_cammino_minimo'
    ]
    
    try:
        with open(file_csv_risultati, mode='w', newline='', encoding='utf-8') as file_output:
            writer = csv.DictWriter(file_output, fieldnames=nomi_colonne_csv)
            writer.writeheader()
            for risultato_riga in lista_risultati_per_csv:
                # Assicura che tutte le chiavi definite in nomi_colonne_csv siano presenti
                # nel dizionario della riga, aggiungendo None o un valore di default se mancanti.
                riga_da_scrivere = {campo: risultato_riga.get(campo) for campo in nomi_colonne_csv}
                writer.writerow(riga_da_scrivere)
        print(f"\nAnalisi completata. I risultati sono stati salvati nel file: {file_csv_risultati}")
    except IOError as e:
        print(f"Errore durante la scrittura del file CSV '{file_csv_risultati}': {e}")
    except Exception as e:
        print(f"Errore imprevisto durante il salvataggio dei risultati: {e}")


def principale():
    """
    Funzione principale che gestisce gli argomenti da riga di comando,
    carica il modello linguistico e avvia il processo di analisi del PDF.
    """
    global SOGLIA_SIMILARITA_DEFAULT # Permette la modifica della soglia globale tramite argomenti

    parser = argparse.ArgumentParser(
        description="Script per l'analisi dell'intreccio semantico in un documento PDF, secondo il modello teorico fornito.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Mostra i valori di default nell'help
    )
    parser.add_argument("pdf", 
                        help="Percorso del file PDF da analizzare.")
    parser.add_argument("--output", 
                        default="risultati_analisi_semantica.csv", 
                        help="Nome del file CSV di output per i risultati dell'analisi.")
    parser.add_argument("--soglia", 
                        type=float, 
                        default=SOGLIA_SIMILARITA_DEFAULT, 
                        help="Soglia di similarità coseno (valore tra 0.0 e 1.0) per la creazione degli archi nel grafo semantico.")
    parser.add_argument("--modello_spacy", 
                        default="it_core_news_md", 
                        help="Nome del modello linguistico spaCy da utilizzare (es. 'it_core_news_md' per l'italiano, 'en_core_web_md' per l'inglese).")
    
    argomenti = parser.parse_args()

    # Validazione della soglia di similarità
    if not (0.0 <= argomenti.soglia <= 1.0):
        print(f"Errore: La soglia di similarità ({argomenti.soglia}) deve essere compresa tra 0.0 e 1.0.")
        print(f"Utilizzo del valore di default: {SOGLIA_SIMILARITA_DEFAULT}")
        soglia_da_usare = SOGLIA_SIMILARITA_DEFAULT
    else:
        soglia_da_usare = argomenti.soglia
    
    # Caricamento del modello linguistico spaCy
    try:
        print(f"Caricamento del modello linguistico spaCy: {argomenti.modello_spacy}...")
        modello_spacy_caricato = spacy.load(argomenti.modello_spacy)
        print("Modello caricato con successo.")
    except OSError:
        print(f"Errore: Impossibile caricare il modello spaCy '{argomenti.modello_spacy}'.")
        print("Verifica che il modello sia installato correttamente. Puoi scaricarlo con il comando:")
        print(f"python -m spacy download {argomenti.modello_spacy}")
        return # Interrompe l'esecuzione se il modello non può essere caricato
    except Exception as e:
        print(f"Errore imprevisto durante il caricamento del modello spaCy '{argomenti.modello_spacy}': {e}")
        return

    # Verifica dell'esistenza del file PDF
    if not os.path.isfile(argomenti.pdf):
        print(f"Errore: Il file PDF specificato '{argomenti.pdf}' non è stato trovato o non è un file.")
        return

    print(f"\nAvvio dell'analisi del documento PDF: {argomenti.pdf}")
    print(f"Soglia di similarità impostata a: {soglia_da_usare}")
    print(f"I risultati verranno salvati in: {argomenti.output}")
    
    # Avvio dell'analisi del PDF
    analizza_documento_pdf(argomenti.pdf, modello_spacy_caricato, argomenti.output, soglia_da_usare)

if __name__ == '__main__':
    principale()
