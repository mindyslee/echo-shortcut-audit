"""Erase nuisance directions from an embedding and price what it costs the clinical signal.

Two erasers:

* ``regress_out``   one-shot linear residualisation of X on the nuisance design matrix.
* ``inlp``          Iterative Null-space Projection (Ravfogel et al., 2020): fit a linear
                    probe for the nuisance, project its weight directions out of X, repeat
                    until the nuisance is no longer linearly decodable. Works for categorical
                    (logistic) and continuous (ridge) nuisances.

``cost_curve`` reports, per INLP iteration, how decodable the nuisance still is and how
decodable each clinical target remains. The gap between iteration 0 and the last iteration
is the price of the shortcut: if EF survives untouched while FPS drops to chance, the model
did not need FPS; if EF falls with it, the embedding was leaning on acquisition geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from .probes import ProbeResult, is_categorical, probe


def nuisance_design(meta: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """One-hot categorical nuisances, standardise continuous ones, add an intercept."""
    parts = [np.ones((len(meta), 1))]
    for c in columns:
        v = meta[c].to_numpy()
        if is_categorical(v):
            parts.append(pd.get_dummies(v.astype(str), drop_first=True).to_numpy(dtype=float))
        else:
            v = v.astype(float)
            parts.append(((v - v.mean()) / (v.std() + 1e-9))[:, None])
    return np.hstack(parts)


def regress_out(X: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Residualise every embedding dimension on the nuisance design matrix N."""
    beta, *_ = np.linalg.lstsq(N, X, rcond=None)
    return (X - N @ beta).astype(np.float32)


def _probe_directions(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Fit one linear nuisance probe on (already standardised) X; return its weight rows (k, d)."""
    # X is expected to be already standardised ONCE (see ``standardise``); re-standardising
    # after a projection would blow up the collapsed directions and send the probe after noise.
    if is_categorical(y):
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=seed)
        clf.fit(X, y.astype(str))
        W = clf.coef_
    else:
        reg = Ridge(alpha=1.0).fit(X, y.astype(float))
        W = reg.coef_[None, :]
    return W


def standardise(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """PCA-whiten once; all erasure happens in this space.

    Whitening matters: with an anisotropic covariance a ridge/logistic probe's weight vector is
    (Σ+αI)⁻¹Σw, not w, so projecting it out removes the probe's *output* but leaves most of the
    planted direction, and INLP crawls. In a whitened space the probe direction is the signal
    direction and a one-dimensional leak dies in one step. Linear decodability is invariant to
    this invertible map, so clinical probes measured here equal those on the raw embedding.
    Near-zero-variance components are dropped.
    """
    Xc = X - X.mean(0)
    U, s, _ = np.linalg.svd(Xc, full_matrices=False)
    keep = s > eps * s[0]
    return (U[:, keep] * np.sqrt(X.shape[0])).astype(np.float32)


def inlp_projection(
    X: np.ndarray, y: np.ndarray, n_iter: int = 8, seed: int = 0
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return the final projection matrix P (d, d) and the directions removed at each step."""
    X = standardise(X)
    d = X.shape[1]
    P = np.eye(d)
    removed: list[np.ndarray] = []
    Xp = X.copy()
    for _ in range(n_iter):
        W = _probe_directions(Xp, y, seed)
        # orthonormal basis of the rowspace of W
        Q, _ = np.linalg.qr(W.T)
        P_step = np.eye(d) - Q @ Q.T
        P = P_step @ P
        Xp = (X @ P.T).astype(np.float32)
        removed.append(Q.T)
    return P, removed


@dataclass
class CostCurve:
    nuisance: str
    iterations: list[int] = field(default_factory=list)
    nuisance_score: list[float] = field(default_factory=list)
    nuisance_chance: float = 0.0
    clinical: dict[str, list[float]] = field(default_factory=dict)
    rank_removed: list[int] = field(default_factory=list)
    # clinical decodability after removing the SAME rank of random directions (mean, sd)
    random_control: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "nuisance": self.nuisance,
            "iterations": self.iterations,
            "nuisance_score": self.nuisance_score,
            "nuisance_chance": self.nuisance_chance,
            "clinical": self.clinical,
            "rank_removed": self.rank_removed,
            "random_control": {k: list(v) for k, v in self.random_control.items()},
            "price": {k: self.price(k) for k in self.clinical},
            "price_vs_random": {k: self.price_vs_random(k) for k in self.clinical},
        }

    def price(self, target: str) -> float:
        """Drop in clinical decodability from untouched to fully-erased embedding."""
        s = self.clinical[target]
        return float(s[0] - s[-1])

    def price_vs_random(self, target: str) -> float:
        """Price beyond what removing the same number of random directions costs.

        This is the number to read: any projection loses a little, and the random control
        prices that. Only the excess says the clinical signal shared directions with the nuisance.
        """
        if target not in self.random_control:
            return float("nan")
        return float(self.random_control[target][0] - self.clinical[target][-1])


def cost_curve(
    X: np.ndarray,
    nuisance: np.ndarray,
    nuisance_name: str,
    clinical: dict[str, np.ndarray],
    n_iter: int = 8,
    n_splits: int = 5,
    seed: int = 0,
    n_random_controls: int = 3,
) -> CostCurve:
    """Iteratively erase ``nuisance`` and re-probe every clinical target after each step.

    Afterwards, remove the same number of *random* directions ``n_random_controls`` times and
    re-probe, so the nuisance's price can be read against a baseline rather than against zero.
    """
    curve = CostCurve(nuisance=nuisance_name, clinical={k: [] for k in clinical})
    X = standardise(X)
    d = X.shape[1]
    P = np.eye(d)
    Xp = X.copy()
    rank = 0

    def record(i: int) -> None:
        r: ProbeResult = probe(
            Xp, nuisance, nuisance_name, n_splits=n_splits, n_permutations=0, seed=seed
        )
        curve.iterations.append(i)
        curve.nuisance_score.append(r.score)
        curve.nuisance_chance = r.chance
        curve.rank_removed.append(rank)
        for k, y in clinical.items():
            rc = probe(Xp, y, k, n_splits=n_splits, n_permutations=0, seed=seed)
            curve.clinical[k].append(rc.score)

    record(0)
    for i in range(1, n_iter + 1):
        W = _probe_directions(Xp, nuisance, seed)
        Q, _ = np.linalg.qr(W.T)
        P = (np.eye(d) - Q @ Q.T) @ P
        rank += Q.shape[1]
        Xp = (X @ P.T).astype(np.float32)
        record(i)
        # stop early once the nuisance is at chance (within a small margin)
        if curve.nuisance_score[-1] <= curve.nuisance_chance + 0.01 and i >= 2:
            break

    if rank and n_random_controls:
        rng = np.random.default_rng(seed + 7)
        scores: dict[str, list[float]] = {k: [] for k in clinical}
        for _ in range(n_random_controls):
            Qr, _ = np.linalg.qr(rng.normal(size=(d, rank)))
            Xr = (X @ (np.eye(d) - Qr @ Qr.T).T).astype(np.float32)
            for k, y in clinical.items():
                scores[k].append(
                    probe(Xr, y, k, n_splits=n_splits, n_permutations=0, seed=seed).score
                )
        curve.random_control = {k: (float(np.mean(v)), float(np.std(v))) for k, v in scores.items()}
    return curve
