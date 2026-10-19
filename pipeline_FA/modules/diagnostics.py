import pandas as pd
import numpy as np
from scipy.stats import chi2
import warnings

"""
MODULO DI DIAGNOSTICA DI FATTORIZZABILITA'
------------------------------------------
Scopo: Analizzare la matrice standardizzata per determinare se è matematicamente 
idonea all'estrazione di dimensioni latenti (Factor Analysis / PCA), senza alterare i dati.
"""

def esegui_diagnostica(df: pd.DataFrame) -> dict:
    metrics = {}
    
    # 1. Isolamento variabili psicometriche (bypass metadati)
    colonne_testuali = df.select_dtypes(include=['object', 'string']).columns.tolist()
    metadati_strutturali = ['participant_id', 'Site', 'Group', 'sex', 'age']
    tutti_i_metadati = list(set(colonne_testuali + [c for c in metadati_strutturali if c in df.columns]))
    
    active_cols = [col for col in df.columns if col not in tutti_i_metadati]
    X = df[active_cols].astype(float)
    # ISOLAMENTO DEL COLPEVOLE
    stds = X.std(skipna=True)
    colonne_piatte = stds[stds <= 1e-8].index.tolist()
    if colonne_piatte:
        print(f"\n[ATTENZIONE] Trovate {len(colonne_piatte)} variabili a varianza nulla (piatte): {colonne_piatte}")
        
    # Controllo di eventuali NaN residui post-standardizzazione
    colonne_con_nan = X.columns[X.isna().any()].tolist()
    if colonne_con_nan:
        print(f"\n[ATTENZIONE] Trovate {len(colonne_con_nan)} variabili con NaN residui: {colonne_con_nan}")
        
    # PULIZIA PREVENTIVA: Rimuove colonne con varianza nulla o NaN per evitare crash del solver
    X = X.loc[:, X.std(skipna=True) > 1e-8]
    X = X.dropna(axis=1, how='all')
    
    n, p = X.shape
    metrics['n_soggetti'] = n
    metrics['p_variabili'] = p
    
    # Matrice di correlazione sicura (ripulita da eventuali NaN residui)
    R = X.corr().values
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    
    # 2. Autovalori e Condition Index di Belsley (Kappa)
    try:
      evals = np.linalg.eigvalsh(R)
      evals = np.clip(evals, a_min=1e-10, a_max=None)
      # Radice quadrata per allinearsi alla soglia canonica < 30 (Belsley et al.)
      kappa = float(np.sqrt(np.max(evals) / np.min(evals)))
    except Exception:
      kappa = np.nan
    metrics['condition_number_kappa'] = kappa

    # 3. VIF (Variance Inflation Factor)
    try:
      vif = np.diag(np.linalg.pinv(R))
      vif_series = pd.Series(vif, index=X.columns).sort_values(ascending=False)
      metrics['top_10_vif'] = vif_series.head(10).to_dict()
      metrics['max_vif'] = float(vif_series.iloc[0])
    except Exception:
      metrics['top_10_vif'] = {}
      metrics['max_vif'] = np.nan

    # 4. Determinante (tramite slogdet per prevenire underflow numerico)
    sign, logdet = np.linalg.slogdet(R)
    det_R = float(np.exp(logdet)) if logdet > -700 else 0.0
    metrics['determinante'] = det_R
    metrics['log10_determinante'] = float(logdet / np.log(10))

    # 5. Test di Sfericità di Bartlett (usando logdet esatto)
    chi2_stat = -(n - 1 - (2 * p + 5) / 6) * logdet
    df_bartlett = p * (p - 1) // 2
    p_value_bartlett = chi2.sf(chi2_stat, max(df_bartlett, 1))
    metrics['bartlett'] = {
        'chi2': float(chi2_stat),
        'gdl': int(df_bartlett),
        'p_value': float(p_value_bartlett),
    }
    # 6. KMO (Kaiser-Meyer-Olkin) Globale
    try:
        R_inv = np.linalg.inv(R)
        D = np.diag(1 / np.sqrt(np.diag(R_inv)))
        Partial_R = - np.dot(D, np.dot(R_inv, D))
        np.fill_diagonal(Partial_R, 0)
        
        R_no_diag = R.copy()
        np.fill_diagonal(R_no_diag, 0)
        
        sum_sq_R = np.sum(R_no_diag ** 2)
        sum_sq_PR = np.sum(Partial_R ** 2)
        kmo_global = sum_sq_R / (sum_sq_R + sum_sq_PR)
        metrics['kmo_global'] = kmo_global
    except np.linalg.LinAlgError:
        metrics['kmo_global'] = np.nan
        
    return metrics