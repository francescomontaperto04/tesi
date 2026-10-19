import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def genera_plot_spls(
    output_dir: Path,
    u_weights: np.ndarray,
    v_weights: np.ndarray,
    null_oos_cov: np.ndarray,
    oos_cov_obs: float,
    null_oos_r: np.ndarray,
    oos_r_obs: float,
    unique_networks: np.ndarray,
    factor_names: list[str]
):
    """
    Traduce i tensori sPLS in grafici PDF vettoriali di alta qualità per la tesi.
    """
    sns.set_theme(style="white", font_scale=1.1)
    
    # ---------------------------------------------------------
    # 1. GRAFICO DEL PROFILO COMPORTAMENTALE (Vettore u)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#d62728' if x > 0 else '#1f77b4' for x in u_weights]
    
    y_pos = np.arange(len(factor_names))
    ax.barh(y_pos, u_weights, color=colors, edgecolor='black', height=0.6)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(factor_names, fontweight='bold')
    ax.invert_yaxis()  # Il Fattore 1 in alto
    ax.axvline(0, color='black', linewidth=1.2)
    ax.set_xlabel('Peso Associativo sPLS ($u$)', fontweight='bold')
    ax.set_title('Profilo Latente Comportamentale', fontsize=14, fontweight='bold', pad=15)
    
    sns.despine(left=True, bottom=False)
    plt.tight_layout()
    plt.savefig(output_dir / "spls_comportamento_u.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    # ---------------------------------------------------------
    # 2. HEATMAP DEL CONNETTOMA (Vettore v -> Matrice (19x19)
    # ---------------------------------------------------------
    K = len(unique_networks)
    mat_v = np.zeros((K, K))
    
    # Ricostruzione della matrice simmetrica dal vettore del triangolo superiore
    triu_idx = np.triu_indices(K, k=1)
    mat_v[triu_idx] = v_weights
    mat_v = mat_v + mat_v.T  # Rende la matrice speculare
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Mappa divergente centrata sullo zero (coolwarm o vlag)
    vmax = np.max(np.abs(v_weights)) if np.max(np.abs(v_weights)) > 0 else 1.0
    sns.heatmap(
        mat_v, 
        cmap='coolwarm', 
        center=0, 
        vmin=-vmax, 
        vmax=vmax,
        xticklabels=unique_networks, 
        yticklabels=unique_networks,
        square=True, 
        linewidths=0.5, 
        linecolor='lightgray',
        cbar_kws={"shrink": .8, "label": "Peso sPLS ($v$)"},
        ax=ax
    )
    
    ax.set_title('Topologia Neurale: Connettività Inter-Network', fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "spls_connettoma_v.pdf", format='pdf', bbox_inches='tight')
    plt.close()

   # ---------------------------------------------------------
    # 3. ISTOGRAMMI DELLA PERMUTAZIONE 
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
    
    # Pannello Superiore: Covarianza
    sns.histplot(null_oos_cov, bins=30, color='gray', alpha=0.6, kde=True, ax=ax1)
    ax1.axvline(oos_cov_obs, color='#d62728', linestyle='--', linewidth=2.5, label=f'Reale OOS Cov ({oos_cov_obs:+.4f})')
    ax1.set_xlabel('Covarianza Out-Of-Sample', fontweight='bold')
    ax1.set_ylabel('Frequenza', fontweight='bold')
    ax1.set_title('A. Significatività Covarianza (Obj. Function sPLS)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Pannello Inferiore: Correlazione di Pearson r
    sns.histplot(null_oos_r, bins=30, color='lightgray', alpha=0.6, kde=True, ax=ax2)
    ax2.axvline(oos_r_obs, color='#1f77b4', linestyle='--', linewidth=2.5, label=f'Reale OOS $r$ ({oos_r_obs:+.3f})')
    ax2.set_xlabel('Correlazione Out-Of-Sample (Pearson $r$)', fontweight='bold')
    ax2.set_ylabel('Frequenza', fontweight='bold')
    ax2.set_title('B. Significatività Correlazione Lineare', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_dir / "spls_permutazioni_null.pdf", format='pdf', bbox_inches='tight')
    plt.close()