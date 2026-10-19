from pathlib import Path
import factor_analyzer.factor_analyzer as _fa_mod
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.utils.validation


def _patched_check_array(*args, **kwargs):
    if 'force_all_finite' in kwargs:
        kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
    return sklearn.utils.validation.check_array(*args, **kwargs)


_fa_mod.check_array = _patched_check_array


def _compute_smc_matrix(R: np.ndarray) -> np.ndarray:
    R_inv_diag = np.diag(np.linalg.pinv(R))
    smc = 1.0 - (1.0 / np.clip(R_inv_diag, 1.0, None))
    R_reduced = R.copy()
    np.fill_diagonal(R_reduced, smc)
    return R_reduced


def esegui_parallel_analysis_e_map(
    df_active: pd.DataFrame, output_dir: Path
) -> dict:
    rng = np.random.default_rng(42)
    X = df_active.to_numpy(dtype=float)
    N, P = X.shape
    n_iter = 500
    max_k_plot = 25

    R_emp = np.corrcoef(X, rowvar=False)
    R_emp = np.nan_to_num(R_emp, nan=0.0)
    np.fill_diagonal(R_emp, 1.0)

    eig_pca_real = np.linalg.eigvalsh(R_emp)[::-1]
    R_smc_emp = _compute_smc_matrix(R_emp)
    eig_efa_real = np.linalg.eigvalsh(R_smc_emp)[::-1]

    sim_eigs_pca = np.zeros((n_iter, P))
    sim_eigs_efa = np.zeros((n_iter, P))

    for i in range(n_iter):
        X_perm = rng.permuted(X, axis=0)
        R_perm = np.corrcoef(X_perm, rowvar=False)
        R_perm = np.nan_to_num(R_perm, nan=0.0)
        np.fill_diagonal(R_perm, 1.0)

        sim_eigs_pca[i, :] = np.linalg.eigvalsh(R_perm)[::-1]
        R_smc_perm = _compute_smc_matrix(R_perm)
        sim_eigs_efa[i, :] = np.linalg.eigvalsh(R_smc_perm)[::-1]

    p95_pca = np.percentile(sim_eigs_pca, 95, axis=0)
    p95_efa = np.percentile(sim_eigs_efa, 95, axis=0)

    k_horn_pca = int(np.argmin(eig_pca_real > p95_pca))
    k_horn_efa = int(np.argmin(eig_efa_real > p95_efa))
    k_kaiser = int(np.sum(eig_pca_real > 1.0))

    eigvals, eigvecs = np.linalg.eigh(R_emp)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    max_map_steps = min(max_k_plot, P - 2)
    map_sq = np.zeros(max_map_steps + 1)
    map_4th = np.zeros(max_map_steps + 1)

    off_diag_mask = ~np.eye(P, dtype=bool)
    map_sq[0] = np.mean(R_emp[off_diag_mask] ** 2)
    map_4th[0] = np.mean(R_emp[off_diag_mask] ** 4)

    for m in range(1, max_map_steps + 1):
        A = eigvecs[:, :m] @ np.diag(np.sqrt(np.maximum(eigvals[:m], 0)))
        R_cov_res = R_emp - (A @ A.T)
        d = np.diag(R_cov_res)
        inv_sqrt_d = 1.0 / np.sqrt(np.maximum(d, 1e-10))
        D_inv_sqrt = np.diag(inv_sqrt_d)
        R_part = D_inv_sqrt @ R_cov_res @ D_inv_sqrt

        map_sq[m] = np.mean(R_part[off_diag_mask] ** 2)
        map_4th[m] = np.mean(R_part[off_diag_mask] ** 4)

    k_map_sq = int(np.argmin(map_sq[1:]) + 1)
    k_map_4th = int(np.argmin(map_4th[1:]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x_vals = np.arange(1, max_k_plot + 1)

    ax1.plot(
        x_vals,
        eig_pca_real[:max_k_plot],
        'o-',
        color='#1f77b4',
        linewidth=2,
        label='Autovalori Reali (PCA)',
    )
    ax1.plot(
        x_vals,
        p95_pca[:max_k_plot],
        '--',
        color='#1f77b4',
        alpha=0.7,
        label='Horn 95° Perc. (PCA)',
    )
    ax1.plot(
        x_vals,
        eig_efa_real[:max_k_plot],
        's-',
        color='#d62728',
        linewidth=2,
        label='Autovalori Ridotti (EFA/SMC)',
    )
    ax1.plot(
        x_vals,
        p95_efa[:max_k_plot],
        '--',
        color='#d62728',
        alpha=0.7,
        label='Horn 95° Perc. (EFA/SMC)',
    )
    ax1.axhline(
        1.0,
        color='gray',
        linestyle=':',
        linewidth=1.2,
        label=r'Soglia di Kaiser ($\lambda=1$)',
    )
    ax1.axvline(
        k_horn_pca,
        color='#1f77b4',
        linestyle='-.',
        alpha=0.6,
        label=f'Cut-off Horn PCA ($k={k_horn_pca}$)',
    )
    ax1.axvline(
        k_horn_efa,
        color='#d62728',
        linestyle='-.',
        alpha=0.6,
        label=f'Cut-off Horn EFA ($k={k_horn_efa}$)',
    )
    ax1.set_title(f'A. Parallel Analysis di Horn ({n_iter} Permutazioni)')
    ax1.set_xlabel('Numero di Dimensioni ($k$)')
    ax1.set_ylabel(r'Autovalore ($\lambda$)')
    ax1.set_xticks(range(1, max_k_plot + 1, 2))
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.25)

    m_vals = np.arange(1, max_map_steps + 1)
    ax2.plot(
        m_vals,
        map_sq[1:],
        'o-',
        color='#2ca02c',
        linewidth=2,
        label=f'Velicer MAP Classico ($r^2$, min a $k={k_map_sq}$)',
    )
    ax2.plot(
        m_vals,
        map_4th[1:] * 10,
        's--',
        color='#9467bd',
        linewidth=1.8,
        label=rf'Revised MAP ($r^4 \times 10$, min a $k={k_map_4th}$)',
    )
    ax2.axvline(k_map_sq, color='#2ca02c', linestyle='-.', alpha=0.7)
    ax2.set_title('B. Test Minimum Average Partial (MAP) di Velicer')
    ax2.set_xlabel('Numero di Componenti Estratte ($m$)')
    ax2.set_ylabel('Correlazione Parziale Residua Media')
    ax2.set_xticks(range(1, max_map_steps + 1, 2))
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(output_dir / 'scree_parallel_map.png', dpi=300)
    plt.close()

    return {
        'k_kaiser': k_kaiser,
        'k_horn_pca': k_horn_pca,
        'k_horn_efa': k_horn_efa,
        'k_map_sq': k_map_sq,
        'k_map_4th': k_map_4th,
        'eig_pca_real': eig_pca_real[:15].round(4).tolist(),
        'p95_pca': p95_pca[:15].round(4).tolist(),
        'eig_efa_real': eig_efa_real[:15].round(4).tolist(),
        'p95_efa': p95_efa[:15].round(4).tolist(),
        'scree_plot_path': 'scree_parallel_map.png',
    }


def esegui_estrazione_efa(
    df_final: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    colonne_testuali = df_final.select_dtypes(
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
        if c in df_final.columns
    ]
    meta_cols = list(dict.fromkeys(metadati_noti + colonne_testuali))

    active_cols = [
        c
        for c in df_final.columns
        if c not in meta_cols and df_final[c].nunique(dropna=True) > 2
    ]

    df_meta = df_final[meta_cols].copy()
    df_active = df_final[active_cols].astype(float)

    dim_metrics = esegui_parallel_analysis_e_map(df_active, output_dir)
    n_factors = dim_metrics['k_horn_pca']
    loading_threshold = 0.32

    fa = FactorAnalyzer(
        n_factors=n_factors, rotation='oblimin', method='minres', use_smc=True
    )
    fa.fit(df_active)

    factor_names = [f'F{i+1}' for i in range(n_factors)]

    # Sign-Flipping canonico per orientare positivamente il polo dominante di ogni fattore
    loadings_arr = fa.loadings_.copy()
    scores_arr = fa.transform(df_active).copy()
    phi_arr = fa.phi_.copy() if fa.phi_ is not None else np.eye(n_factors)

    signs = np.where(np.sum(loadings_arr**3, axis=0) < 0, -1.0, 1.0)
    loadings_arr *= signs
    scores_arr *= signs
    phi_arr = (phi_arr * signs[:, None]) * signs[None, :]

    pattern_matrix = pd.DataFrame(
        loadings_arr, index=active_cols, columns=factor_names
    )
    communalities = pd.Series(
        fa.get_communalities(), index=active_cols, name='h2'
    )
    uniquenesses = pd.Series(
        fa.get_uniquenesses(), index=active_cols, name='u2'
    )

    heywood_vars = communalities[communalities >= 0.999].index.tolist()

    df_loadings = pattern_matrix.copy()
    df_loadings['Fattore_Dominante'] = pattern_matrix.abs().idxmax(axis=1)
    df_loadings['_f_num'] = (
        df_loadings['Fattore_Dominante'].str.replace('F', '').astype(int)
    )
    df_loadings['Max_Loading_Abs'] = pattern_matrix.abs().max(axis=1)
    df_loadings['h2'] = communalities
    df_loadings['u2'] = uniquenesses

    df_loadings = df_loadings.sort_values(
        by=['_f_num', 'Max_Loading_Abs'], ascending=[True, False]
    ).drop(columns=['_f_num'])

    ss_loadings, prop_var, cum_var = fa.get_factor_variance()
    phi_matrix = pd.DataFrame(phi_arr, index=factor_names, columns=factor_names)

    factor_scores = pd.DataFrame(
        scores_arr, columns=factor_names, index=df_final.index
    )
    df_scores = pd.concat([df_meta, factor_scores], axis=1)

    factors_summary = []
    for i, f_name in enumerate(factor_names):
        sub_f = df_loadings[df_loadings['Fattore_Dominante'] == f_name]
        sub_salient = sub_f[sub_f['Max_Loading_Abs'] >= loading_threshold]
        if sub_salient.empty:
            sub_salient = sub_f.head(3)
        top_formatted = [
            f'{var} ({row[f_name]:+.2f})'
            for var, row in sub_salient.head(4).iterrows()
        ]
        factors_summary.append({
            'fattore': f_name,
            'ss_loading': round(float(ss_loadings[i]), 2),
            'prop_var_pct': round(float(prop_var[i] * 100), 2),
            'cum_var_pct': round(float(cum_var[i] * 100), 2),
            'top_markers': ', '.join(top_formatted),
        })

    df_loadings_clean = df_loadings.drop(columns=['Max_Loading_Abs'])
    metrics = {
        **dim_metrics,
        'n_factors_extracted': int(n_factors),
        'extraction_method': 'MINRES',
        'rotation_method': 'Oblimin',
        'loading_threshold': loading_threshold,
        'total_variance_explained_pct': round(float(cum_var[-1] * 100), 2),
        'ss_loadings': ss_loadings.round(3).tolist(),
        'prop_var_pct': (prop_var * 100).round(2).tolist(),
        'cum_var_pct': (cum_var * 100).round(2).tolist(),
        'phi_matrix': phi_matrix.round(3).to_dict(),
        'heywood_cases': heywood_vars,
        'mean_communality': round(float(communalities.mean()), 4),
        'low_communality_vars': (
            communalities[communalities < 0.20].round(3).to_dict()
        ),
        'factors_summary': factors_summary,
    }

    return df_scores, df_loadings_clean, metrics