#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sorgente Python per l'analisi statistica e la visualizzazione dei risultati ottenuti
dallo script 'analizzatore_semantico_pdf_refactored_it.py'.
Questo script carica il file CSV generato e:
  - Calcola statistiche descrittive per le metriche di rete.
  - Calcola e visualizza la distribuzione aggregata dei gradi dei nodi P(k).
  - Tenta una stima dell'esponente gamma (γ) della legge di potenza per P(k) e salva la stima.
  - Genera visualizzazioni grafiche per le principali metriche.
  - Utilizza il modulo logging per i messaggi.

Autori: Luigi Usai e LLM
Data: Maggio 2025
https://colab.research.google.com/drive/1mrlTjSmF-I3N1h3-SbNxGdxOwBbWev0Z
https://zenodo.org/records/15484259
"""

import argparse
import pandas as pd
import numpy as np
import ast  # Per convertire stringhe di liste in liste Python (literal_eval)
from collections import Counter # Per calcolare facilmente le frequenze dei gradi
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from scipy.stats import linregress # Per la regressione lineare (stima di gamma)
import os # Aggiunto per gestire percorsi file

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def carica_e_prepara_dati(percorso_file_csv):
    """
    Carica i dati dal file CSV e prepara la colonna dei gradi per l'analisi.
    """
    try:
        df = pd.read_csv(percorso_file_csv)
        logging.info(f"File CSV '{percorso_file_csv}' caricato con successo.")
        
        def converti_stringa_in_lista(stringa_lista):
            try:
                if pd.isna(stringa_lista): # Gestisce valori NaN
                    return []
                return ast.literal_eval(stringa_lista)
            except (ValueError, SyntaxError) as e:
                logging.warning(f"Impossibile convertire la stringa '{stringa_lista}' in una lista: {e}. Sarà trattata come lista vuota.")
                return []

        df['gradi_dei_nodi_lista'] = df['gradi_dei_nodi'].apply(converti_stringa_in_lista)
        logging.info("Colonna 'gradi_dei_nodi' processata e convertita in liste.")
        return df
    except FileNotFoundError:
        logging.error(f"Il file CSV '{percorso_file_csv}' non è stato trovato.")
        return None
    except Exception as e:
        logging.error(f"Errore imprevisto durante il caricamento o la preparazione dei dati: {e}", exc_info=True)
        return None

def calcola_statistiche_descrittive_colonna(dataframe, nome_colonna, nome_metrica_leggibile):
    """
    Calcola e logga statistiche descrittive per una specifica colonna del DataFrame.
    """
    if nome_colonna not in dataframe.columns:
        logging.warning(f"La colonna '{nome_colonna}' non è presente nel DataFrame.")
        return None 

    dati_colonna = dataframe[nome_colonna].dropna() 

    if dati_colonna.empty:
        logging.info(f"Nessun dato valido trovato per la metrica '{nome_metrica_leggibile}'.")
        return None 

    statistiche = {
        'media': np.mean(dati_colonna),
        'mediana': np.median(dati_colonna),
        'deviazione_standard': np.std(dati_colonna),
        'valore_minimo': np.min(dati_colonna),
        'valore_massimo': np.max(dati_colonna),
        'conteggio_pagine': len(dati_colonna)
    }

    logging.info(f"\n--- Statistiche per: {nome_metrica_leggibile} ---")
    logging.info(f"  Media: {statistiche['media']:.4f}")
    logging.info(f"  Mediana: {statistiche['mediana']:.4f}")
    logging.info(f"  Deviazione Standard: {statistiche['deviazione_standard']:.4f}")
    logging.info(f"  Valore Minimo: {statistiche['valore_minimo']:.4f}")
    logging.info(f"  Valore Massimo: {statistiche['valore_massimo']:.4f}")
    logging.info(f"  Numero di pagine analizzate (con dati validi): {statistiche['conteggio_pagine']}")
    return statistiche

def visualizza_distribuzione_metrica(dati_colonna, nome_metrica_leggibile, nome_file_grafico):
    """
    Crea e salva un istogramma per la distribuzione di una metrica.
    """
    if dati_colonna is None or dati_colonna.empty:
        logging.warning(f"Nessun dato da visualizzare per '{nome_metrica_leggibile}'.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(dati_colonna, kde=True, bins=20)
    plt.title(f'Distribuzione di: {nome_metrica_leggibile}')
    plt.xlabel(nome_metrica_leggibile)
    plt.ylabel('Frequenza')
    plt.grid(True, linestyle='--', alpha=0.7)
    try:
        plt.savefig(nome_file_grafico)
        logging.info(f"Grafico della distribuzione di '{nome_metrica_leggibile}' salvato come '{nome_file_grafico}'.")
    except Exception as e:
        logging.error(f"Errore durante il salvataggio del grafico '{nome_file_grafico}': {e}", exc_info=True)
    plt.close()


def analizza_e_visualizza_distribuzione_gradi(dataframe, nome_file_grafico_pk):
    """
    Calcola, logga e visualizza la distribuzione aggregata dei gradi P(k).
    Restituisce il conteggio dei gradi e il numero totale di nodi per la stima di gamma.
    """
    if 'gradi_dei_nodi_lista' not in dataframe.columns:
        logging.warning("Colonna 'gradi_dei_nodi_lista' non trovata per l'analisi dei gradi.")
        return None, 0

    lista_completa_gradi = []
    for i, lista_gradi_pagina in enumerate(dataframe['gradi_dei_nodi_lista']):
        if not isinstance(lista_gradi_pagina, list):
            logging.warning(f"Elemento non valido (non una lista) trovato nella colonna 'gradi_dei_nodi_lista' alla riga {i}. Saltato.")
            continue
        lista_completa_gradi.extend(lista_gradi_pagina)
    
    if not lista_completa_gradi:
        logging.info("Nessun dato sui gradi dei nodi trovato per calcolare la distribuzione P(k).")
        return None, 0

    conteggio_gradi = Counter(lista_completa_gradi)
    numero_totale_nodi_con_grado = sum(conteggio_gradi.values())

    logging.info("\n--- Distribuzione Aggregata dei Gradi P(k) ---")
    logging.info("  Grado (k) | Conteggio | Probabilità P(k)")
    logging.info("  ------------------------------------------")
    
    gradi_k = []
    probabilita_pk = []

    for grado, conteggio in sorted(conteggio_gradi.items()):
        if numero_totale_nodi_con_grado > 0:
            prob = conteggio / numero_totale_nodi_con_grado
            logging.info(f"  {grado:<10} | {conteggio:<9} | {prob:.6f}")
            if grado > 0: 
                gradi_k.append(grado)
                probabilita_pk.append(prob)
        else:
            logging.warning("Numero totale di nodi con grado è zero, impossibile calcolare P(k).")
            return conteggio_gradi, numero_totale_nodi_con_grado


    logging.info(f"  Numero totale di nodi considerati per P(k): {numero_totale_nodi_con_grado}")

    if gradi_k and probabilita_pk:
        plt.figure(figsize=(10, 6))
        plt.loglog(gradi_k, probabilita_pk, marker='o', linestyle='None', label='Dati P(k)')
        plt.title('Distribuzione dei Gradi P(k) (Scala Log-Log)')
        plt.xlabel('Grado k (log)')
        plt.ylabel('Probabilità P(k) (log)')
        plt.grid(True, which="both", ls="--", alpha=0.7)
        
        try:
            plt.savefig(nome_file_grafico_pk)
            logging.info(f"Grafico della distribuzione P(k) salvato come '{nome_file_grafico_pk}'.")
        except Exception as e:
            logging.error(f"Errore durante il salvataggio del grafico P(k) '{nome_file_grafico_pk}': {e}", exc_info=True)
        plt.close()
    else:
        logging.info("Nessun dato valido (k>0) per il plot log-log di P(k).")
        
    return conteggio_gradi, numero_totale_nodi_con_grado

def stima_gamma_legge_potenza(conteggio_gradi, numero_totale_nodi_con_grado):
    """
    Stima l'esponente gamma (γ) della legge di potenza P(k) ~ k^(-γ)
    utilizzando una regressione lineare sui dati log-trasformati.
    Restituisce gamma stimato e R^2.
    """
    if not conteggio_gradi or numero_totale_nodi_con_grado == 0:
        logging.warning("Dati sui gradi insufficienti per la stima di gamma.")
        return None, None

    log_k = []
    log_pk = []

    for grado, conteggio in conteggio_gradi.items():
        if grado > 0 and conteggio > 0: 
            log_k.append(np.log(grado))
            log_pk.append(np.log(conteggio / numero_totale_nodi_con_grado))
    
    if len(log_k) < 2: 
        logging.info("Non ci sono abbastanza punti dati (k>0) per eseguire la regressione lineare per gamma.")
        return None, None

    try:
        slope, intercept, r_value, p_value, std_err = linregress(log_k, log_pk)
        gamma_stimato = -slope 
        r_squared = r_value**2
        logging.info(f"\n--- Stima dell'Esponente Gamma (γ) della Legge di Potenza ---")
        logging.info(f"  Gamma (γ) stimato: {gamma_stimato:.4f}")
        logging.info(f"  Coefficiente di determinazione (R^2): {r_squared:.4f}")
        return gamma_stimato, r_squared
    except Exception as e:
        logging.error(f"Errore durante la stima di gamma: {e}", exc_info=True)
        return None, None

def salva_stima_gamma_su_file(gamma_stimato, r_squared, nome_file_output):
    """
    Salva la stima di gamma e R^2 in un file di testo.
    """
    if gamma_stimato is None or r_squared is None:
        logging.warning("Nessuna stima di gamma da salvare.")
        return

    try:
        with open(nome_file_output, 'w', encoding='utf-8') as f:
            f.write("--- Stima dell'Esponente Gamma (γ) della Legge di Potenza ---\n")
            f.write(f"  Gamma (γ) stimato: {gamma_stimato:.4f}\n")
            f.write(f"  Coefficiente di determinazione (R^2): {r_squared:.4f}\n")
        logging.info(f"Stima di Gamma salvata nel file: '{nome_file_output}'")
    except IOError as e:
        logging.error(f"Errore durante il salvataggio della stima di gamma nel file '{nome_file_output}': {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Errore imprevisto durante il salvataggio della stima di gamma: {e}", exc_info=True)


def principale():
    parser = argparse.ArgumentParser(
        description="Script per l'analisi statistica dei risultati dell'intreccio semantico da un file CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("file_csv", 
                        help="Percorso del file CSV generato dallo script di analisi PDF (es. 'risultati_analisi_semantica.csv').")
    parser.add_argument("--output_prefisso",
                        default="analisi_semantica_",
                        help="Prefisso per i nomi dei file di output generati (grafici, stima gamma).")
    
    argomenti = parser.parse_args()

    logging.info(f"Avvio analisi del file: {argomenti.file_csv}")
    # Crea la directory di output se non esiste, basata sul prefisso
    output_dir = os.path.dirname(argomenti.output_prefisso)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Directory di output '{output_dir}' creata.")
        except OSError as e:
            logging.error(f"Impossibile creare la directory di output '{output_dir}': {e}. I file verranno salvati nella directory corrente.")
            # Se la creazione della directory fallisce, i file verranno salvati dove si trova lo script.
            # Potremmo voler gestire questo in modo diverso, ad es. uscendo o usando un prefisso senza percorso.
            # Per ora, lasciamo che i file vengano creati nella directory corrente se il percorso specificato nel prefisso non è valido/creabile.


    dataframe_risultati = carica_e_prepara_dati(argomenti.file_csv)

    if dataframe_risultati is not None and not dataframe_risultati.empty:
        # Analisi e visualizzazione del Coefficiente di Clustering Medio
        stat_clustering = calcola_statistiche_descrittive_colonna(dataframe_risultati, 
                                                'coefficiente_clustering_medio', 
                                                'Coefficiente di Clustering Medio (C)')
        if stat_clustering is not None: 
            visualizza_distribuzione_metrica(dataframe_risultati['coefficiente_clustering_medio'].dropna(),
                                             'Coefficiente di Clustering Medio',
                                             f"{argomenti.output_prefisso}distribuzione_clustering.png")

        # Analisi e visualizzazione della Lunghezza Media del Cammino Minimo
        stat_cammino = calcola_statistiche_descrittive_colonna(dataframe_risultati, 
                                             'lunghezza_media_cammino_minimo', 
                                             'Lunghezza Media del Cammino Minimo (L)')
        if stat_cammino is not None:
             visualizza_distribuzione_metrica(dataframe_risultati['lunghezza_media_cammino_minimo'].dropna(),
                                             'Lunghezza Media del Cammino Minimo',
                                             f"{argomenti.output_prefisso}distribuzione_cammino_medio.png")
        
        # Analisi, visualizzazione della Distribuzione dei Gradi P(k) e stima di Gamma
        conteggio_gradi_calc, num_nodi_tot_pk = analizza_e_visualizza_distribuzione_gradi(
            dataframe_risultati,
            f"{argomenti.output_prefisso}distribuzione_gradi_pk_loglog.png"
        )
        
        if conteggio_gradi_calc and num_nodi_tot_pk > 0:
            gamma_stimato, r_quadro_stimato = stima_gamma_legge_potenza(conteggio_gradi_calc, num_nodi_tot_pk)
            if gamma_stimato is not None and r_quadro_stimato is not None:
                salva_stima_gamma_su_file(gamma_stimato, r_quadro_stimato, f"{argomenti.output_prefisso}stima_gamma.txt")
        
        logging.info("\nAnalisi completata.")
        logging.info(f"I grafici e la stima di gamma sono stati salvati con il prefisso: '{argomenti.output_prefisso}'")
        logging.info("Ulteriori passi potrebbero includere l'uso di metodi più sofisticati per il fitting della legge di potenza o analisi comparative.")
    else:
        logging.error("Impossibile procedere con l'analisi a causa di errori nel caricamento dei dati o file vuoto/invalido.")

if __name__ == '__main__':
    principale()
