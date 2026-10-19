import json
import sys
import warnings
from pathlib import Path
import pandas as pd

from modules.cleaning import esegui_pulizia_globale
from modules.imputation import applica_missforest
from modules.normalization import aggregate_madrs_subscales, normalizza_e_standardizza
from modules.diagnostics import esegui_diagnostica
from modules.analysis import esegui_estrazione_efa
from modules.testing import esegui_test_ipotesi_1
from reporting.latex import genera_report_completo

warnings.filterwarnings('ignore')

INPUT_PATH = Path("data/matrice_fenotipica_completa.csv")
OUTPUT_DIR = Path("output_pipeline")


def _chiedi_fase_partenza(paths: dict, cached_metrics: dict) -> int:
    scelta = None
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        val = int(sys.argv[1])
        if 1 <= val <= 6:
            scelta = val

    if scelta is None:
        print("\n" + "=" * 60)
        print(" SELEZIONE FASE DI PARTENZA PIPELINE")
        print("=" * 60)
        print(" [1] Esecuzione completa (da zero)")
        print(" [2] Da Imputazione MissForest (usa matrice_pulita.csv)")
        print(" [3] Da Normalizzazione/MADRS (usa matrice_imputata.csv)")
        print(" [4] Da Diagnostica Algebrica (usa matrice_standardizzata.csv)")
        print(" [5] Da Estrazione Fattoriale EFA (usa matrice_standardizzata.csv)")
        print(" [6] Solo Compilazione Report LaTeX (usa checkpoint_metrics.json)")
        print("=" * 60)

        while True:
            risp = input("Da quale fase vuoi partire? [1-6] (default=5): ").strip()
            if risp == "":
                scelta = 5
                break
            if risp.isdigit() and 1 <= int(risp) <= 6:
                scelta = int(risp)
                break
            print("Inserisci un numero valido compreso tra 1 e 6.")

    requisiti = {
        2: (paths['clean'].exists() and 'cleaning' in cached_metrics, "Fase 1 (Cleaning)"),
        3: (paths['imputed'].exists() and 'imputation' in cached_metrics, "Fase 2 (Imputazione)"),
        4: (paths['final'].exists() and 'normalization' in cached_metrics, "Fase 3 (Normalizzazione)"),
        5: (paths['final'].exists() and 'diagnostics' in cached_metrics, "Fase 4 (Diagnostica)"),
        6: ('analysis' in cached_metrics, "Fase 5 (Analisi Fattoriale)"),
    }

    for step in range(2, scelta + 1):
        presente, nome_step = requisiti[step]
        if not presente:
            print(f"\n[AVVISO] Dati o metriche per '{nome_step}' assenti su disco.")
            print(f"Ritorno forzato alla Fase {step - 1}.\n")
            return step - 1

    return scelta


def _aggiorna_cache(path_cache: Path, metrics: dict):
    with open(path_cache, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    paths = {
        'clean': OUTPUT_DIR / "matrice_pulita.csv",
        'imputed': OUTPUT_DIR / "matrice_imputata.csv",
        'final': OUTPUT_DIR / "matrice_standardizzata.csv",
        'scores': OUTPUT_DIR / "matrice_punteggi_fattoriali.csv",
        'loadings': OUTPUT_DIR / "matrice_pattern_loadings.csv",
        'cache': OUTPUT_DIR / "checkpoint_metrics.json",
        'latex': OUTPUT_DIR / "report_metodologico.tex",
    }

    report_metrics = {}
    if paths['cache'].exists():
        try:
            with open(paths['cache'], 'r', encoding='utf-8') as f:
                report_metrics = json.load(f)
        except Exception:
            report_metrics = {}

    start_step = _chiedi_fase_partenza(paths, report_metrics)
    print(f"\n>>> Avvio pipeline dalla FASE {start_step} <<<\n")

    # FASE 1: Pulizia Dati
    if start_step <= 1:
        print("[1/6] Esecuzione Pulizia Globale (Soggetti e Variabili)...")
        df_raw = pd.read_csv(INPUT_PATH, low_memory=False)
        df_clean, metrics_clean = esegui_pulizia_globale(df_raw)
        report_metrics['cleaning'] = metrics_clean
        df_clean.to_csv(paths['clean'], index=False)
        _aggiorna_cache(paths['cache'], report_metrics)
    elif start_step == 2:
        print("[1/6] Salto Cleaning -> Carico matrice_pulita.csv...")
        df_clean = pd.read_csv(paths['clean'], low_memory=False)

    # FASE 2: Imputazione
    if start_step <= 2:
        print("\n[2/6] Imputazione Non-Lineare (MissForest)...")
        df_imputed, metrics_imp = applica_missforest(df_clean, OUTPUT_DIR)
        report_metrics['imputation'] = metrics_imp
        df_imputed.to_csv(paths['imputed'], index=False)
        _aggiorna_cache(paths['cache'], report_metrics)
    elif start_step == 3:
        print("[2/6] Salto MissForest -> Carico matrice_imputata.csv...")
        df_imputed = pd.read_csv(paths['imputed'], low_memory=False)

    # FASE 3: Normalizzazione e Standardizzazione
    if start_step <= 3:
        print("\n[3/6] Aggregazione MADRS, Normalizzazione e Standardizzazione...")
        df_imputed, report_metrics = aggregate_madrs_subscales(df_imputed, report_metrics)
        df_final, metrics_norm = normalizza_e_standardizza(df_imputed)
        report_metrics['normalization'] = metrics_norm
        df_final.to_csv(paths['final'], index=False)
        _aggiorna_cache(paths['cache'], report_metrics)
    elif start_step in (4, 5):
        print("[3/6] Salto Normalizzazione -> Carico matrice_standardizzata.csv...")
        df_final = pd.read_csv(paths['final'], low_memory=False)

    # FASE 4: Diagnostica Preliminare
    if start_step <= 4:
        print("\n[4/6] Esecuzione Diagnostica di Fattorizzabilita' (VIF, KMO, Bartlett)...")
        metrics_diag = esegui_diagnostica(df_final)
        report_metrics['diagnostics'] = metrics_diag
        _aggiorna_cache(paths['cache'], report_metrics)

    # FASE 5: Dimensionalità (Horn/MAP) ed Estrazione Fattoriale (EFA)
    if start_step <= 5:
        print("\n[5/6] Triangolazione Dimensionalita' (Horn/MAP) ed Estrazione EFA...")
        df_scores, df_loadings, metrics_efa = esegui_estrazione_efa(
            df_final, output_dir=OUTPUT_DIR
        )
        df_scores.to_csv(paths['scores'], index=False)
        df_loadings.to_csv(paths['loadings'])
        report_metrics['analysis'] = metrics_efa
        _aggiorna_cache(paths['cache'], report_metrics)

    # FASE 6: Test dell'Ipotesi 1 (Continuità vs Cluster nello Spazio Latente Z)
    if start_step <= 6:
        print("\n[6/6] Esecuzione Test dell'Ipotesi 1 (GMM-BIC & Silhouette)...")
        metrics_test = esegui_test_ipotesi_1(df_scores, output_dir=OUTPUT_DIR)
        report_metrics['testing'] = metrics_test
        _aggiorna_cache(paths['cache'], report_metrics)
        
    # REPORTING LATEX
    print("\n[REPORTING] Generazione report LaTeX centralizzato...")
    genera_report_completo(report_metrics, paths['latex'])
    print(f"\nPipeline conclusa. Tutti i file operativi e il LaTeX sono in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()