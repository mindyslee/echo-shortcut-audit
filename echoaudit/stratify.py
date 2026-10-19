"""Metadata-only baselines and error stratification.

Three questions:

1. How much EF can be predicted from acquisition metadata *alone*? (If FPS or clip length
   already explains EF variance, a video model can learn it as a shortcut.)
2. How much does the embedding add beyond metadata? (partial R²)
3. Is the embedding's EF error uneven across acquisition strata? (Kruskal-Wallis across bins)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from .probes import probe
from .removal import nuisance_design


def metadata_only_baseline(
    meta: pd.DataFrame, target: str, nuisance_cols: list[str], n_splits: int = 5, seed: int = 0
) -> dict:
    """Predict ``target`` from acquisition metadata alone (linear and boosted)."""
    y = meta[target].to_numpy(dtype=float)
    N = nuisance_design(meta, nuisance_cols)[:, 1:]  # drop intercept; models add their own
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    lin = probe(N.astype(np.float32), y, target, n_splits=n_splits, n_permutations=20, seed=seed)
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed
    )
    pred = cross_val_predict(gbr, N, y, cv=kf)
    return {
        "target": target,
        "nuisance_columns": nuisance_cols,
        "linear_r2": lin.score,
        "linear_p_value": lin.p_value,
        "boosted_r2": float(r2_score(y, pred)),
    }


def partial_r2(
    X: np.ndarray,
    meta: pd.DataFrame,
    target: str,
    nuisance_cols: list[str],
    n_splits: int = 5,
    seed: int = 0,
) -> dict:
    """R² from embedding+metadata minus R² from metadata; the embedding's unique contribution."""
    y = meta[target].to_numpy(dtype=float)
    N = nuisance_design(meta, nuisance_cols)[:, 1:]
    both = np.hstack([X, N]).astype(np.float32)
    r_both = probe(both, y, target, n_splits=n_splits, n_permutations=0, seed=seed).score
    r_meta = probe(
        N.astype(np.float32), y, target, n_splits=n_splits, n_permutations=0, seed=seed
    ).score
    r_emb = probe(X, y, target, n_splits=n_splits, n_permutations=0, seed=seed).score
    return {
        "target": target,
        "r2_embedding": r_emb,
        "r2_metadata": r_meta,
        "r2_both": r_both,
        "embedding_beyond_metadata": float(r_both - r_meta),
        "metadata_beyond_embedding": float(r_both - r_emb),
    }


def _bin(v: np.ndarray, max_bins: int = 4) -> np.ndarray:
    if len(np.unique(v)) <= max_bins:
        return v.astype(str)
    q = pd.qcut(v, q=max_bins, duplicates="drop")
    return q.astype(str).to_numpy()


def stratified_error(
    X: np.ndarray,
    meta: pd.DataFrame,
    target: str,
    strata_cols: list[str],
    n_splits: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Out-of-fold |error| of the embedding probe, broken down by each acquisition column."""
    y = meta[target].to_numpy(dtype=float)
    _, pred = probe(
        X, y, target, n_splits=n_splits, n_permutations=0, seed=seed, return_predictions=True
    )
    err = np.abs(pred - y)
    out = []
    for c in strata_cols:
        b = _bin(meta[c].to_numpy())
        groups = [err[b == g] for g in np.unique(b)]
        if len(groups) < 2:
            continue
        kw = stats.kruskal(*groups)
        rows = {str(g): {"n": len(e), "mae": float(e.mean())} for g, e in zip(np.unique(b), groups)}
        out.append(
            {
                "target": target,
                "stratum": c,
                "bins": rows,
                "kruskal_H": float(kw.statistic),
                "kruskal_p": float(kw.pvalue),
                "mae_range": float(
                    max(r["mae"] for r in rows.values()) - min(r["mae"] for r in rows.values())
                ),
            }
        )
    return out
