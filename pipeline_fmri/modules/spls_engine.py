import os
# BLOCCO OVERSUBSCRIPTION
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
from sklearn.model_selection import KFold
from joblib import Parallel, delayed

def project_l1_l2(w, c, tol=1e-6, max_iter=60):
    if c < 1.0:
        raise ValueError(f"Vincolo L1 impossibile: c={c} < 1.0")
    norm2 = np.linalg.norm(w, 2)
    if norm2 <= 1e-12:
        return np.zeros_like(w)
        
    abs_w = np.abs(w)
    if (abs_w.sum() / norm2) <= c:
        return w / norm2

    max_idx = int(np.argmax(abs_w))
    low, high = 0.0, abs_w[max_idx]
    abs_proj = np.zeros_like(abs_w)
    abs_proj[max_idx] = 1.0
    
    for _ in range(max_iter):
        delta = (low + high) * 0.5
        v_soft = np.maximum(0.0, abs_w - delta)
        norm2_soft = np.linalg.norm(v_soft, 2)
        
        if norm2_soft <= 1e-12:
            high = delta
            continue
            
        l1_val = v_soft.sum() / norm2_soft

        if np.abs(l1_val - c) < tol:
            return np.sign(w) * (v_soft / norm2_soft)
        elif l1_val > c:
            low = delta
        else:
            high = delta
            abs_proj = v_soft / norm2_soft
            
    return np.sign(w) * abs_proj

def safe_standardize(train, test=None):
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std_safe = np.where(std <= 1e-12, 1.0, std)
    train_std = (train - mean) / std_safe
    if test is not None:
        test_std = (test - mean) / std_safe
        return train_std, test_std
    return train_std

def fast_first_left_singular(C_zy):
    M = C_zy @ C_zy.T
    _, eigvecs = np.linalg.eigh(M)
    return eigvecs[:, -1]

def fit_spls_from_cov(C_zy, c_Y, u_init=None, max_iter=100, tol=1e-6):
    u = fast_first_left_singular(C_zy) if u_init is None else u_init.copy()
    v = np.zeros(C_zy.shape[1], dtype=C_zy.dtype)

    for _ in range(max_iter):
        u_old = u.copy()
        w_v = C_zy.T @ u
        v = project_l1_l2(w_v, c=c_Y)

        w_u = C_zy @ v
        norm_u = np.linalg.norm(w_u, 2)
        if norm_u <= 1e-12:
            return np.zeros_like(u), np.zeros_like(v)
        u = w_u / norm_u

        if np.dot(u, u_old) < 0:
            u = -u
            v = -v

        if np.linalg.norm(u - u_old) < tol:
            break
    return u, v

def fast_cov_and_corr(x, y):
    x_c = x - x.mean()
    y_c = y - y.mean()
    ss_x = float(np.dot(x_c, x_c))
    ss_y = float(np.dot(y_c, y_c))
    if ss_x <= 1e-24 or ss_y <= 1e-24:
        return 0.0, 0.0
    xy_dot = float(np.dot(x_c, y_c))
    return xy_dot / len(x), xy_dot / np.sqrt(ss_x * ss_y)

def precompute_Y_folds(Y_data, kf_splits):
    y_folds = []
    for train_idx, test_idx in kf_splits:
        Y_tr, Y_te = safe_standardize(Y_data[train_idx], Y_data[test_idx])
        y_folds.append((train_idx, test_idx, Y_tr, Y_te))
    return y_folds

def evaluate_pipeline_cv_fast(Z_data, y_folds, c_grid):
    n_c, n_folds = len(c_grid), len(y_folds)
    q, p = Z_data.shape[1], y_folds[0][2].shape[1]
    fold_covs, fold_corrs = np.zeros((n_folds, n_c)), np.zeros((n_folds, n_c))
    U_tr_all = np.empty((q, n_c), dtype=Z_data.dtype)
    V_tr_all = np.empty((p, n_c), dtype=Z_data.dtype)

    for f_idx, (train_idx, test_idx, Y_tr, Y_te) in enumerate(y_folds):
        Z_tr, Z_te = safe_standardize(Z_data[train_idx], Z_data[test_idx])
        C_zy_tr = Z_tr.T @ Y_tr
        u_svd = fast_first_left_singular(C_zy_tr)

        for c_idx, c_val in enumerate(c_grid):
            u_tr, v_tr = fit_spls_from_cov(C_zy_tr, c_Y=c_val, u_init=u_svd, max_iter=100)
            U_tr_all[:, c_idx] = u_tr
            V_tr_all[:, c_idx] = v_tr

        Xi_te = Z_te @ U_tr_all
        Omega_te = Y_te @ V_tr_all

        for c_idx in range(n_c):
            cov_oos, r_oos = fast_cov_and_corr(Xi_te[:, c_idx], Omega_te[:, c_idx])
            fold_covs[f_idx, c_idx] = cov_oos
            fold_corrs[f_idx, c_idx] = r_oos

    cv_covs, cv_corrs = fold_covs.mean(axis=0), fold_corrs.mean(axis=0)
    best_idx = int(np.nanargmax(cv_corrs))
    return c_grid[best_idx], float(cv_covs[best_idx]), float(cv_corrs[best_idx]), cv_covs, cv_corrs

def _perm_worker(seed, Z_full, Y_full, y_folds, c_grid):
    rng_local = np.random.default_rng(seed)
    Z_perm = Z_full[rng_local.permutation(Z_full.shape[0])]
    best_c_p, oos_cov_p, oos_r_p, _, _ = evaluate_pipeline_cv_fast(Z_perm, y_folds, c_grid)

    C_zy_p = Z_perm.T @ Y_full
    u_p, v_p = fit_spls_from_cov(C_zy_p, c_Y=best_c_p, max_iter=100)
    active_idx = np.flatnonzero(v_p)
    if len(active_idx) == 0:
        return oos_cov_p, oos_r_p, 0.0, 0.0
    
    cov_in_p, r_in_p = fast_cov_and_corr(Z_perm @ u_p, Y_full[:, active_idx] @ v_p[active_idx])
    return oos_cov_p, oos_r_p, cov_in_p, r_in_p

def esegui_sPLS_cv_permutazionale(Z_raw: np.ndarray, Y_raw: np.ndarray, n_perms: int = 1000) -> dict:
    Y = np.arctanh(np.clip(Y_raw, -0.9999, 0.9999))
    N, q = Z_raw.shape
    _, p = Y.shape

    c_grid = [round(c, 2) for c in np.linspace(1.1, np.sqrt(p) * 0.8, num=15)]
    
    print(f"[sPLS] Dimensioni -> Z: {Z_raw.shape}, Y: {Y.shape}")
    
    t_pre = time.time()
    y_folds = precompute_Y_folds(Y, list(KFold(n_splits=5, shuffle=True, random_state=42).split(Z_raw)))
    print(f"[sPLS] Fold pre-computati in {time.time() - t_pre:.2f}s")

    t0 = time.time()
    best_c, best_oos_cov, best_oos_r, _, _ = evaluate_pipeline_cv_fast(Z_raw, y_folds, c_grid)
    print(f"[sPLS] Ottimo trovato (c_Y = {best_c}) in {time.time() - t0:.2f}s (OOS r = {best_oos_r:.4f})")

    Z_full, Y_full = safe_standardize(Z_raw), safe_standardize(Y)
    u_opt, v_opt = fit_spls_from_cov(Z_full.T @ Y_full, c_Y=best_c, max_iter=100)
    
    xi_full, omega_full = Z_full @ u_opt, Y_full @ v_opt
    if u_opt[np.argmax(np.abs(u_opt))] < 0:
        u_opt, v_opt, xi_full, omega_full = -u_opt, -v_opt, -xi_full, -omega_full
        
    cov_obs_in, r_obs_in = fast_cov_and_corr(xi_full, omega_full)
    active_edges = int(np.sum(np.abs(v_opt) > 0))

    print(f"[sPLS] Inizio Permutation Test ({n_perms} iterazioni)...")
    t_perm = time.time()
    results = np.array(Parallel(n_jobs=-1)(
        delayed(_perm_worker)(s, Z_full, Y_full, y_folds, c_grid) 
        for s in np.random.SeedSequence(42).spawn(n_perms)
    ))
    null_oos_cov, null_oos_r, null_in_cov, null_in_r = results.T
    
    p_oos_cov = float((np.sum(null_oos_cov >= best_oos_cov) + 1.0) / (n_perms + 1.0))
    p_oos_r   = float((np.sum(null_oos_r >= best_oos_r) + 1.0) / (n_perms + 1.0))

    print(f"[sPLS] Permutazioni completate in {time.time() - t_perm:.1f}s. p-value r_OOS = {p_oos_r:.5f}")

    return {
        'best_c': float(best_c),
        'active_edges': active_edges,
        'pct_active': float((active_edges / p) * 100),
        'oos_cov': float(best_oos_cov),
        'oos_r': float(best_oos_r),
        'in_cov': float(cov_obs_in),
        'in_r': float(r_obs_in),
        'p_oos_cov': p_oos_cov,
        'p_oos_r': p_oos_r,
        'u_weights': u_opt.tolist(),
        'v_weights': v_opt.tolist(),
        'null_dist_oos_cov': null_oos_cov.tolist(),
        'null_dist_oos_r': null_oos_r.tolist()
    }