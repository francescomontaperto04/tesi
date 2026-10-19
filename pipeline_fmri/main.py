import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from modules.network_mapping import costruisci_connettoma_network
from modules.spls_engine import esegui_sPLS_cv_permutazionale
from modules.edge_decoder import decodifica_connessioni_attive
from modules.reporting_fmri import genera_report_spls
from modules.plotting import genera_plot_spls

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

PATH_Z_INPUT = DATA_DIR / "matrice_punteggi_fattoriali.csv"
PATH_LABELS = DATA_DIR / "whole_brain_region_labels_ordered_Cerebellum_Seperate.npy"
PATH_FMRI_TS = DATA_DIR / "fMRI_timeseries_clean_denoised_GSR_parcellated"

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("\n" + "=" * 65)
    print(" PIPELINE MULTIVARIATA fMRI-COMPORTAMENTO (sPLS)")
    print("=" * 65)

    # FASE 1: Allineamento
    print("\n[1/4] Estrazione serie fMRI e allineamento matriciale...")
    df_z_allineata, Y_raw, unique_networks = costruisci_connettoma_network(
        percorso_z=PATH_Z_INPUT,
        labels_path=PATH_LABELS,
        data_dir=PATH_FMRI_TS
    )
    
    df_z_allineata.to_csv(OUTPUT_DIR / "matrice_Z_allineata.csv", index=False)
    np.save(OUTPUT_DIR / "matrice_Y_connettoma.npy", Y_raw)
    
    fattori_estratti = [c for c in df_z_allineata.columns if c.startswith('F') and c[1:].isdigit()]
    Z_raw = df_z_allineata[fattori_estratti].values.astype(np.float32)
    Y_raw = Y_raw.astype(np.float32)

    # FASE 2: Calcolo sPLS
    print("\n[2/4] Avvio Motore sPLS con Cross-Validation e Permutation Test...")
    spls_results = esegui_sPLS_cv_permutazionale(Z_raw, Y_raw, n_perms=1000)

    # FASE 3: Decodifica Connessioni
    print("\n[3/4] Decodifica e ordinamento della firma neurale attiva...")
    decoder_metrics = decodifica_connessioni_attive(
        v_weights=np.array(spls_results['v_weights']),
        unique_networks=unique_networks,
        output_dir=OUTPUT_DIR
    )

    # FASE 4: Esportazione Risultati e Reporting
    print("\n[4/4] Salvataggio vettori e generazione Report LaTeX...")
    
    # Escludiamo le distribuzioni nulle massicce dal JSON per mantenerlo leggero
    metrics_to_save = {k: v for k, v in spls_results.items() if k not in ['u_weights', 'v_weights', 'null_dist_oos_r', 'null_dist_oos_cov']}
    with open(OUTPUT_DIR / "spls_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=4)
        
    np.save(OUTPUT_DIR / "spls_u_weights_Z.npy", np.array(spls_results['u_weights']))
    np.save(OUTPUT_DIR / "spls_v_weights_Y.npy", np.array(spls_results['v_weights']))
    np.save(OUTPUT_DIR / "spls_null_oos_r.npy", np.array(spls_results['null_dist_oos_r']))
    np.save(OUTPUT_DIR / "spls_null_oos_cov.npy", np.array(spls_results['null_dist_oos_cov']))

    # Generazione dei grafici vettoriali
    print("  -> Generazione grafici vettoriali...")
    genera_plot_spls(
        output_dir=OUTPUT_DIR,
        u_weights=np.array(spls_results['u_weights']),
        v_weights=np.array(spls_results['v_weights']),
        null_oos_cov=np.array(spls_results['null_dist_oos_cov']),
        oos_cov_obs=spls_results['oos_cov'],
        null_oos_r=np.array(spls_results['null_dist_oos_r']),
        oos_r_obs=spls_results['oos_r'],
        unique_networks=unique_networks,
        factor_names=fattori_estratti
    )
    
    genera_report_spls(
        metrics=spls_results, 
        filepath=OUTPUT_DIR / "report_spls_fmri.tex",
        n_subj=Z_raw.shape[0], 
        p_fmri=Y_raw.shape[1], 
        q_behav=Z_raw.shape[1],
        decoder_metrics=decoder_metrics
    )

    print(f"\nPipeline conclusa. Tutti i file operativi e il report LaTeX sono in: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()