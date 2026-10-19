from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


def esegui_test_ipotesi_1(
    df_scores: pd.DataFrame, output_dir: Path
) -> dict:
    colonne_testuali = df_scores.select_dtypes(
        include=['object', 'string']
    ).columns.tolist()

    # Isolamento deterministico dei metadati (preserva ordine delle colonne)
    metadati_noti = [
        c
        for c in [
            'participant_id',
            'Group',
            'Site',
            'sex',
            'age',
            'rel_to_proband',
        ]
        if c in df_scores.columns
    ]
    meta_cols = list(dict.fromkeys(metadati_noti + colonne_testuali))

    factor_cols = sorted(
        [c for c in df_scores.columns if c not in meta_cols],
        key=lambda x: int(x[1:]) if x.startswith('F') and x[1:].isdigit() else x,
    )
    Z = df_scores[factor_cols].astype(float).to_numpy()

    max_clusters = 8
    c_range = np.arange(1, max_clusters + 1)
    bic_full, bic_diag = [], []

    for c in c_range:
        gmm_f = GaussianMixture(
            n_components=c,
            covariance_type='full',
            n_init=10,
            reg_covar=1e-5,
            random_state=42,
        ).fit(Z)
        gmm_d = GaussianMixture(
            n_components=c,
            covariance_type='diag',
            n_init=10,
            reg_covar=1e-5,
            random_state=42,
        ).fit(Z)
        bic_full.append(float(gmm_f.bic(Z)))
        bic_diag.append(float(gmm_d.bic(Z)))

    best_c_bic_full = int(c_range[np.argmin(bic_full)])
    best_c_bic_diag = int(c_range[np.argmin(bic_diag)])
    delta_bic_1_vs_2 = float(bic_full[1] - bic_full[0])

    sil_kmeans, sil_gmm = [], []
    for c in range(2, max_clusters + 1):
        labels_km = KMeans(
            n_clusters=c, n_init=20, random_state=42
        ).fit_predict(Z)
        labels_gm = GaussianMixture(
            n_components=c,
            covariance_type='full',
            n_init=10,
            reg_covar=1e-5,
            random_state=42,
        ).fit_predict(Z)
        sil_kmeans.append(float(silhouette_score(Z, labels_km)))
        sil_gmm.append(float(silhouette_score(Z, labels_gm)))

    max_sil_km = float(np.max(sil_kmeans))
    best_c_sil = int(np.argmax(sil_kmeans) + 2)

    # Generazione Plot di Controllo (BIC e Silhouette)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(
        c_range,
        bic_full,
        'o-',
        color='#1f77b4',
        linewidth=2,
        label=f'GMM Full Cov (Min a $C={best_c_bic_full}$)',
    )
    ax1.plot(
        c_range,
        bic_diag,
        's--',
        color='#ff7f0e',
        linewidth=2,
        label=f'GMM Diag Cov (Min a $C={best_c_bic_diag}$)',
    )
    ax1.axvline(best_c_bic_full, color='#1f77b4', linestyle=':', alpha=0.7)
    ax1.set_title('A. Selezione Modello Informativo (BIC - GMM)')
    ax1.set_xlabel('Numero di Componenti / Cluster ($C$)')
    ax1.set_ylabel('Bayesian Information Criterion (BIC)')
    ax1.set_xticks(c_range)
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    c_sil_range = np.arange(2, max_clusters + 1)
    ax2.plot(
        c_sil_range,
        sil_kmeans,
        'o-',
        color='#2ca02c',
        linewidth=2,
        label='K-Means Silhouette',
    )
    ax2.plot(
        c_sil_range,
        sil_gmm,
        'd--',
        color='#d62728',
        linewidth=1.8,
        label='GMM Silhouette',
    )
    ax2.axhline(
        0.25,
        color='black',
        linestyle='--',
        linewidth=1.2,
        label='Soglia Minima Struttura Discreta ($0.25$)',
    )
    ax2.set_ylim(0, max(0.55, max_sil_km + 0.05))
    ax2.set_title(f'B. Coesione Topologica (Max Sil = {max_sil_km:.3f})')
    ax2.set_xlabel('Numero di Cluster Forzati ($C$)')
    ax2.set_ylabel('Coefficiente di Silhouette Medio')
    ax2.set_xticks(c_sil_range)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / 'topology_continuity_vs_clusters.png', dpi=300)
    plt.close()

    metrics = {
        'best_c_bic_full': best_c_bic_full,
        'best_c_bic_diag': best_c_bic_diag,
        'delta_bic_c2_c1': round(delta_bic_1_vs_2, 2),
        'max_silhouette_km': round(max_sil_km, 4),
        'best_c_silhouette': best_c_sil,
        'bic_full_curve': [round(x, 2) for x in bic_full],
        'bic_diag_curve': [round(x, 2) for x in bic_diag],
        'sil_km_curve': [round(x, 4) for x in sil_kmeans],
        'sil_gmm_curve': [round(x, 4) for x in sil_gmm],
        'topology_plot_path': 'topology_continuity_vs_clusters.png',
    }
    return metrics