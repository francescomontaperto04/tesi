from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def applica_missforest(
    df: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, dict]:
    colonne_testuali = df.select_dtypes(
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
        if c in df.columns
    ]
    meta_cols = list(dict.fromkeys(metadati_noti + colonne_testuali))

    df_meta = df[meta_cols].copy()
    df_numeric = df.drop(columns=meta_cols).astype(float)
    df_pre_imputation = df_numeric.copy()

    rf_core = RandomForestRegressor(
        n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1
    )
    missForest_engine = IterativeImputer(
        estimator=rf_core, max_iter=10, random_state=42, tol=1e-3
    )

    df_imputed_array = missForest_engine.fit_transform(df_numeric)
    df_imputed = pd.DataFrame(
        df_imputed_array, columns=df_numeric.columns, index=df.index
    )

    # Estrazione Metriche sulle 4 variabili chiave ad alta missingness
    variabili_chiave = ['tmb_word_score', 'ctqscore_sa', 'asi', 'bapq_aloof_ave']
    variabili_presenti = [
        v for v in variabili_chiave if v in df_imputed.columns
    ]

    diagnostics = {}
    for var in variabili_presenti:
        std_pre = df_pre_imputation[var].std(skipna=True)
        std_post = df_imputed[var].std()
        diagnostics[var] = {
            'std_pre': float(std_pre),
            'std_post': float(std_post),
            'var_pct': float(((std_post - std_pre) / std_pre) * 100),
        }

    corr_pre = df_pre_imputation.corr(method='pearson')
    corr_post = df_imputed.corr(method='pearson')
    delta_matrice = np.abs(corr_post - corr_pre)
    np.fill_diagonal(delta_matrice.values, np.nan)

    # Generazione Plot di Controllo (allineato esattamente alle variabili della Tabella 1)
    cols_to_plot = variabili_presenti

    if cols_to_plot:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for i, col in enumerate(cols_to_plot):
            # cut=0 impedisce al KDE di estrapolare code inesistenti su valori negativi
            sns.kdeplot(
                df_pre_imputation[col].dropna(),
                ax=axes[i],
                label='Pre (Osservati)',
                color='black',
                linewidth=2,
                cut=0,
            )
            sns.kdeplot(
                df_imputed[col],
                ax=axes[i],
                label='Post (Imputati)',
                color='red',
                linestyle='--',
                linewidth=2,
                cut=0,
            )
            axes[i].set_title(f'Distribuzione: {col}')
            axes[i].set_xlabel('Punteggio')
            axes[i].set_ylabel('Densità')
            axes[i].legend()

        # Rimuove eventuali assi vuoti se le variabili presenti sono meno di 4
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(output_dir / 'controllo_imputazione_densita.png', dpi=300)
        plt.close()

    df_final = pd.concat([df_meta, df_imputed], axis=1)
    metrics = {
        'imputation_diagnostics': diagnostics,
        'delta_medio_covarianza': float(np.nanmean(delta_matrice.values)),
    }
    return df_final, metrics