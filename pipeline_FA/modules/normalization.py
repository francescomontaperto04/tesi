import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from typing import Dict, Tuple
import pandas as pd


def aggregate_madrs_subscales(
    df: pd.DataFrame, report_metrics: Dict
) -> Tuple[pd.DataFrame, Dict]:
    """Aggrega i 10 item atomici della MADRS nelle 3 sottoscale cliniche

    post-imputazione ed elimina i singoli indicatori atomici.
    """
    # Definizione univoca dei mapping fenomenologici
    madrs_clusters = {
        "madrs_dysphoria": [
            "madrssad",
            "madrsaps",
            "madrsten",
            "madrspes",
            "madrssui",
        ],
        "madrs_vegetative": ["madrsslp", "madrsapp"],
        "madrs_retardation": ["madrslas", "madrscon", "madrsfee"],
    }

    all_raw_items = [
        item for sublist in madrs_clusters.values() for item in sublist
    ]

    # Verifica di sicurezza: presenza di tutti gli item attesi
    missing_cols = [col for col in all_raw_items if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Item MADRS mancanti nel DataFrame post-imputazione: {missing_cols}"
        )

    # Verifica assenza di NaN residui prima dell'aggregazione
    if df[all_raw_items].isna().any().any():
        raise ValueError(
            "Rilevati NaN residui negli item MADRS. L'aggregazione post-imputazione "
            "esige una matrice densa."
        )

    # Calcolo deterministico della media per ciascuna sottoscala
    df_out = df.copy()
    for subscale, items in madrs_clusters.items():
        df_out[subscale] = df_out[items].mean(axis=1)

    # Rimozione atomica degli item grezzi
    df_out = df_out.drop(columns=all_raw_items)

    # Tracciamento nel dizionario di reporting
    report_metrics["madrs_aggregation"] = {
        "items_removed": all_raw_items,
        "subscales_created": list(madrs_clusters.keys()),
        "p_delta": -len(all_raw_items) + len(madrs_clusters),  # -10 + 3 = -7
    }

    return df_out, report_metrics
def normalizza_e_standardizza(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df_norm = df.copy()
    results = []

    colonne_testuali = df.select_dtypes(include=['object', 'string']).columns.tolist()
    metadati_strutturali = ['participant_id', 'Site', 'Group', 'sex', 'age']
    metadati_da_escludere = [c for c in metadati_strutturali if c in df.columns]
    tutti_i_metadati = list(set(colonne_testuali + metadati_da_escludere))

    for col in df.columns:
        if col in tutti_i_metadati:
            results.append({
                'Variabile': col, 
                'Trasformazione': 'Nessuna (Preservata Grezza)', 
                'Shapiro_W_Iniziale': np.nan, 
                'Shapiro_W_Finale': np.nan
            })
            continue
            
        x = df[col].values
        if len(np.unique(x)) <= 2:
            results.append({
                'Variabile': col, 
                'Trasformazione': 'Nessuna (Binaria/Costante)', 
                'Shapiro_W_Iniziale': np.nan, 
                'Shapiro_W_Finale': np.nan
            })
            continue
            
        candidates = {'Originale': x}
        shift = np.abs(np.min(x)) + 1e-5 if np.min(x) <= 0 else 0
            
        candidates['Logaritmo'] = np.log(x + shift)
        candidates['Radice Quadrata'] = np.sqrt(x + shift)
        candidates['Arcsinh'] = np.arcsinh(x)
        
        try:
            pt = PowerTransformer(method='yeo-johnson', standardize=False)
            candidates['Yeo-Johnson'] = pt.fit_transform(x.reshape(-1, 1)).flatten()
        except: pass
            
        best_name, best_stat, best_transformed = 'Originale', -1, x
        
        try: stat_init, _ = stats.shapiro(x)
        except: stat_init = np.nan
        
        for name, tx in candidates.items():
            tx_clean = tx[~np.isnan(tx) & ~np.isinf(tx)]
            if len(np.unique(tx_clean)) > 2:
                try:
                    stat, _ = stats.shapiro(tx_clean)
                    if stat > best_stat:
                        best_stat, best_name, best_transformed = stat, name, tx
                except: pass
                    
        df_norm[col] = best_transformed
        results.append({
            'Variabile': col, 'Trasformazione': best_name, 
            'Shapiro_W_Iniziale': stat_init, 'Shapiro_W_Finale': best_stat if best_stat != -1 else stat_init
        })

    # Fase Standardizzazione
    metadata = df[tutti_i_metadati].copy()
    active_vars = [col for col in df_norm.columns if col not in tutti_i_metadati]
    df_active = df_norm[active_vars].copy()
    df_active_z = (df_active - df_active.mean()) / df_active.std()
    
    df_final = pd.concat([metadata.reset_index(drop=True), df_active_z.reset_index(drop=True)], axis=1)

    res_df = pd.DataFrame(results)
    res_df['Miglioramento_W'] = res_df['Shapiro_W_Finale'] - res_df['Shapiro_W_Iniziale']
    
    # Filtro corretto escludendo le diciture "Nessuna"
    valid_improvements = res_df[~res_df['Trasformazione'].str.contains('Nessuna', na=False)]
    top_10 = valid_improvements.sort_values('Miglioramento_W', ascending=False).head(10)

    metrics = {
        'top_shapiro': top_10.to_dict('records'),
        'full_results': res_df.to_dict('records') # Utile per eventuali log globali nel main
    }
    return df_final, metrics