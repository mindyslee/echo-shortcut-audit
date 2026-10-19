"""Linear probes: how decodable is a target from a frozen embedding?

Everything is cross-validated, and every reported score comes with a permutation null so
that "decodable" means "decodable beyond chance at this sample size", not just "R² > 0".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = np.logspace(-2, 4, 13)
MAX_CATEGORIES = 12  # a numeric target with at most this many distinct values is categorical

# "linear" (default) bounds linear decodability; "mlp" asks whether a nuisance survives
# non-linearly after linear erasure. Set once via ``set_probe_family`` (the CLI's --probe).
PROBE_FAMILY = "linear"
MLP_HIDDEN = (256,)


def set_probe_family(family: str) -> None:
    global PROBE_FAMILY
    if family not in {"linear", "mlp"}:
        raise ValueError("probe family must be 'linear' or 'mlp'")
    PROBE_FAMILY = family


def _regressor(seed: int):
    if PROBE_FAMILY == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN,
            alpha=1e-3,
            early_stopping=True,
            max_iter=300,
            random_state=seed,
        )
    return RidgeCV(alphas=RIDGE_ALPHAS)


def _classifier(seed: int):
    if PROBE_FAMILY == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN,
            alpha=1e-3,
            early_stopping=True,
            max_iter=300,
            random_state=seed,
        )
    return LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")


@dataclass
class ProbeResult:
    target: str
    kind: str  # "regression" | "classification"
    n: int
    score: float  # R² (regression) or balanced accuracy (classification)
    score_std: float  # across folds
    secondary: float  # MAE (regression) or macro one-vs-rest AUC (classification)
    chance: float  # 0 for R²; 1/k for balanced accuracy
    null_mean: float
    null_std: float
    p_value: float  # fraction of permutations scoring >= observed (with +1 smoothing)
    n_permutations: int
    classes: list[str] | None = None

    @property
    def above_chance(self) -> float:
        """Score minus chance; comparable across regression and classification."""
        return float(self.score - self.chance)

    @property
    def z_vs_null(self) -> float:
        """How many null standard deviations the observed score sits above the null mean."""
        if self.n_permutations == 0 or not np.isfinite(self.null_std) or self.null_std == 0:
            return float("nan")
        return float((self.score - self.null_mean) / self.null_std)

    @property
    def decodable(self) -> bool:
        """Above chance by a margin, and clearly outside the permutation null when one exists.

        The permutation p-value has a floor of 1/(k+1), so with few permutations it can never
        reach 0.05; the z-score against the null is the robust criterion.
        """
        margin = self.above_chance > 0.02
        if self.n_permutations == 0:
            return margin
        return margin and (self.z_vs_null > 3.0 or self.p_value < 0.05)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["above_chance"] = self.above_chance
        d["z_vs_null"] = self.z_vs_null
        d["decodable"] = self.decodable
        return d


def is_categorical(y: np.ndarray) -> bool:
    if y.dtype.kind in {"O", "U", "S", "b"}:
        return True
    return len(np.unique(y)) <= MAX_CATEGORIES


def _regression_cv(
    X: np.ndarray, y: np.ndarray, n_splits: int, seed: int
) -> tuple[np.ndarray, list[float]]:
    """Return out-of-fold predictions and per-fold R²."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros_like(y, dtype=float)
    folds: list[float] = []
    for tr, te in kf.split(X):
        model = make_pipeline(StandardScaler(), _regressor(seed))
        model.fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
        folds.append(float(r2_score(y[te], pred[te])))
    return pred, folds


def _classification_cv(
    X: np.ndarray, y: np.ndarray, n_splits: int, seed: int
) -> tuple[np.ndarray, np.ndarray, list[float], np.ndarray]:
    """Return out-of-fold labels, probabilities, per-fold balanced accuracy, and class list."""
    classes, y_enc = np.unique(y, return_inverse=True)
    min_count = np.bincount(y_enc).min()
    n_splits = max(2, min(n_splits, int(min_count)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros_like(y_enc)
    proba = np.zeros((len(y_enc), len(classes)))
    folds: list[float] = []
    for tr, te in skf.split(X, y_enc):
        model = make_pipeline(StandardScaler(), _classifier(seed))
        model.fit(X[tr], y_enc[tr])
        p = model.predict_proba(X[te])
        # align columns in case a fold is missing a class
        proba[np.ix_(te, model.classes_)] = p
        pred[te] = model.classes_[p.argmax(1)]
        folds.append(float(balanced_accuracy_score(y_enc[te], pred[te])))
    return pred, proba, folds, classes


def probe(
    X: np.ndarray,
    y: np.ndarray,
    target: str,
    n_splits: int = 5,
    n_permutations: int = 20,
    seed: int = 0,
    return_predictions: bool = False,
) -> ProbeResult | tuple[ProbeResult, np.ndarray]:
    """Cross-validated probe (linear by default, see PROBE_FAMILY) with a permutation null."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    if is_categorical(y):
        pred, proba, folds, classes = _classification_cv(X, y, n_splits, seed)
        _, y_enc = np.unique(y, return_inverse=True)
        score = float(balanced_accuracy_score(y_enc, pred))
        if len(classes) == 2:
            auc = float(roc_auc_score(y_enc, proba[:, 1]))
        else:
            auc = float(roc_auc_score(y_enc, proba, multi_class="ovr", average="macro"))
        null = []
        for _ in range(n_permutations):
            yp = rng.permutation(y)
            p_pred, _, _, _ = _classification_cv(X, yp, n_splits, seed)
            _, yp_enc = np.unique(yp, return_inverse=True)
            null.append(balanced_accuracy_score(yp_enc, p_pred))
        res = ProbeResult(
            target=target,
            kind="classification",
            n=len(y),
            score=score,
            score_std=float(np.std(folds)),
            secondary=auc,
            chance=float(1.0 / len(classes)),
            null_mean=float(np.mean(null)) if null else float("nan"),
            null_std=float(np.std(null)) if null else float("nan"),
            p_value=float((np.sum(np.asarray(null) >= score) + 1) / (len(null) + 1)),
            n_permutations=n_permutations,
            classes=[str(c) for c in classes],
        )
        out_pred = classes[pred]
    else:
        y = y.astype(float)
        pred, folds = _regression_cv(X, y, n_splits, seed)
        score = float(r2_score(y, pred))
        null = []
        for _ in range(n_permutations):
            yp = rng.permutation(y)
            p_pred, _ = _regression_cv(X, yp, n_splits, seed)
            null.append(r2_score(yp, p_pred))
        res = ProbeResult(
            target=target,
            kind="regression",
            n=len(y),
            score=score,
            score_std=float(np.std(folds)),
            secondary=float(mean_absolute_error(y, pred)),
            chance=0.0,
            null_mean=float(np.mean(null)) if null else float("nan"),
            null_std=float(np.std(null)) if null else float("nan"),
            p_value=float((np.sum(np.asarray(null) >= score) + 1) / (len(null) + 1)),
            n_permutations=n_permutations,
        )
        out_pred = pred
    if return_predictions:
        return res, out_pred
    return res


def random_embedding_control(
    n: int, d: int, y: np.ndarray, target: str, seed: int = 0, **kw
) -> ProbeResult:
    """Probe a Gaussian random matrix of the same shape: the floor any score must beat."""
    rng = np.random.default_rng(seed + 1000)
    return probe(rng.normal(size=(n, d)).astype(np.float32), y, f"{target}~random", seed=seed, **kw)


def probe_many(
    X: np.ndarray,
    targets: dict[str, np.ndarray],
    n_splits: int = 5,
    n_permutations: int = 20,
    seed: int = 0,
) -> dict[str, ProbeResult]:
    return {
        name: probe(X, y, name, n_splits=n_splits, n_permutations=n_permutations, seed=seed)
        for name, y in targets.items()
    }
