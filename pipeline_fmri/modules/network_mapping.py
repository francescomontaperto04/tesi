import os
import glob
import h5py
import numpy as np
import pandas as pd
from pathlib import Path


def costruisci_connettoma_network(
    percorso_z: Path, labels_path: Path, data_dir: Path
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Esegue il parsing delle etichette corticali/subcorticali, costruisce 
    la matrice di proiezione spaziale W e calcola il connettoma ridotto (Y)
    solo per i soggetti presenti nella matrice comportamentale Z.
    """
    # 1. Parsing Etichette e Matrice W
    region_labels = np.load(labels_path, allow_pickle=True)
    network_assignment = []
    
    for lab in region_labels:
        parts = str(lab).split('_')
        if parts[0] == '17networks' and len(parts) > 2:
            net = parts[2]
        elif 'cereb' in str(lab).lower():
            net = 'Cerebellum'
        elif 'subc' in str(lab).lower() or 'thal' in str(lab).lower() or 'striat' in str(lab).lower():
            net = 'Subcortical'
        else:
            net = 'Other'
        network_assignment.append(net)

    network_assignment = np.array(network_assignment)
    unique_networks = np.unique(network_assignment)
    K = len(unique_networks)

    print(f"[MAPPING] Trovati {K} macro-network: {unique_networks}")

    W = np.zeros((len(region_labels), K))
    for i, net in enumerate(network_assignment):
        idx = np.where(unique_networks == net)[0][0]
        W[i, idx] = 1.0

    # 2. Intersezione rigorosa Soggetti Z vs fMRI H5
    df_z = pd.read_csv(percorso_z)

    def normalize_id(raw_id):
        clean = str(raw_id).strip()
        if clean.startswith("sub-"):
            clean = clean[4:]
        if not "_" in clean and clean.startswith("NDAR"):
            clean = "NDAR_" + clean[4:]
        return clean

    id_col = 'participant_id' if 'participant_id' in df_z.columns else df_z.columns[0]
    df_z['id_norm'] = df_z[id_col].apply(normalize_id)

    all_folders = sorted(
        glob.glob(os.path.join(data_dir, "NDAR_*") + os.path.sep) +
        glob.glob(os.path.join(data_dir, "sub-*") + os.path.sep)
    )

    local_h5_dict = {}
    for folder in all_folders:
        base_name = os.path.basename(os.path.normpath(folder))
        local_h5_dict[normalize_id(base_name)] = folder

    df_z_valid = df_z[df_z['id_norm'].isin(local_h5_dict.keys())].copy()
    df_z_valid = df_z_valid.sort_values('id_norm').reset_index(drop=True)
    valid_ids = df_z_valid['id_norm'].tolist()

    print(f"[MAPPING] Soggetti allineati post-intersezione: {len(valid_ids)}")

    # 3. Estrazione TS, Proiezione Spaziale e Calcolo Covarianza
    y_list = []
    failed_subjects = []

    for sid in valid_ids:
        subj_path = local_h5_dict[sid]
        rest_files = glob.glob(os.path.join(subj_path, "*rest*_parcellated.h5"))
        
        if not rest_files:
            failed_subjects.append(sid)
            continue
            
        try:
            with h5py.File(rest_files[0], 'r') as f:
                key = list(f.keys())[0]
                ts = np.array(f[key])
                
            if ts.shape[0] == 434:
                ts = ts.T
                
            if ts.shape[1] != 434:
                raise ValueError(f"Shape anomala: {ts.shape}. Attese 434 regioni.")
                
            # Proiezione W e calcolo connettività FC
            ts_net = ts @ W
            ts_net = ts_net / W.sum(axis=0)
            corr_matrix = np.corrcoef(ts_net, rowvar=False)
            
            triu_idx = np.triu_indices(K, k=1)
            fc_vector = corr_matrix[triu_idx]
            fc_vector = np.nan_to_num(fc_vector, nan=0.0)
            
            y_list.append(fc_vector)
            
        except Exception as e:
            print(f"  [ERRORE] Soggetto {sid}: {e}")
            failed_subjects.append(sid)

    if failed_subjects:
        df_z_valid = df_z_valid[~df_z_valid['id_norm'].isin(failed_subjects)].reset_index(drop=True)

    df_z_valid = df_z_valid.drop(columns=['id_norm'])
    Y = np.array(y_list)

    return df_z_valid, Y, unique_networks