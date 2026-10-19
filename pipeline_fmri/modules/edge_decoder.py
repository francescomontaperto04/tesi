import numpy as np
import pandas as pd
from pathlib import Path

def decodifica_connessioni_attive(
    v_weights: np.ndarray, 
    unique_networks: np.ndarray, 
    output_dir: Path
) -> dict:
    """
    Mappa i pesi del vettore fMRI v sulle rispettive coppie di macro-network,
    estraendo la firma neurale non nulla.
    """
    # Ricostruzione algoritmica coerente con np.triu_indices
    pairs = []
    for i in range(len(unique_networks)):
        for j in range(i + 1, len(unique_networks)):
            pairs.append(f"{unique_networks[i]} $\\leftrightarrow$ {unique_networks[j]}")

    if len(v_weights) != len(pairs):
        raise ValueError(f"Disallineamento: {len(v_weights)} pesi vs {len(pairs)} edge teorici.")

    df_edges = pd.DataFrame({
        "Connessione_Network": pairs,
        "Peso_sPLS": v_weights,
        "Importanza_Assoluta": np.abs(v_weights)
    })

    # Vincolo di sparsità (Witten et al., 2009)
    df_active = df_edges[df_edges["Importanza_Assoluta"] > 0].copy()
    df_active = df_active.sort_values(by="Importanza_Assoluta", ascending=False).reset_index(drop=True)

    # Salvataggio dinamico
    num_attive = len(df_active)
    file_name = f"connessioni_attive_{num_attive}_sPLS.csv"
    df_active.to_csv(output_dir / file_name, index=False)

    return {
        'totale_edge_analizzati': len(df_edges),
        'edge_attivi': num_attive,
        'nome_file_csv': file_name,
        'top_edges_list': df_active.head(10).to_dict('records')
    }