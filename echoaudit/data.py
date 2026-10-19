"""Loading embeddings and EchoNet-style metadata, and joining them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Columns of EchoNet-Dynamic's FileList.csv. Other datasets can map onto this.
CLINICAL_TARGETS = ["EF", "ESV", "EDV"]
NUISANCE_TARGETS = ["FPS", "FPS_modal", "FrameHeight", "FrameWidth", "NumberOfFrames"]
MIN_CLASS_COUNT = 20  # a categorical target whose rarest class is smaller than this is skipped
EF_BINS = [-np.inf, 40.0, 50.0, np.inf]
EF_BIN_LABELS = ["EF<40", "40<=EF<50", "EF>=50"]


@dataclass
class Dataset:
    """An embedding matrix aligned with a metadata table."""

    ids: np.ndarray  # (n,) video ids, in row order of X
    X: np.ndarray  # (n, d) float32 embeddings
    meta: pd.DataFrame  # (n, ...) metadata, same row order as X
    source: str = "unknown"
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])

    def target(self, name: str) -> np.ndarray:
        return self.meta[name].to_numpy()

    def available(self, names: list[str]) -> list[str]:
        return [c for c in names if c in self.meta.columns and self.meta[c].notna().all()]


def load_embeddings_txt(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the whitespace-delimited ``<id> v1 v2 ... vd`` format used by r3d-v0-embeddings.

    Returns (ids, X). Rows with inconsistent dimensionality raise ValueError.
    """
    ids: list[str] = []
    rows: list[list[float]] = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            ids.append(parts[0])
            rows.append([float(v) for v in parts[1:]])
    if not rows:
        raise ValueError(f"no embeddings found in {path}")
    d = len(rows[0])
    if any(len(r) != d for r in rows):
        raise ValueError("inconsistent embedding dimensionality across rows")
    return np.asarray(ids), np.asarray(rows, dtype=np.float32)


def load_embeddings(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch on extension: .txt (r3d format), .npz (ids,X), .csv (first col id)."""
    p = Path(path)
    if p.suffix == ".txt":
        return load_embeddings_txt(p)
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        return np.asarray(z["ids"]).astype(str), np.asarray(z["X"], dtype=np.float32)
    if p.suffix in {".csv", ".parquet"}:
        df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
        return df.iloc[:, 0].astype(str).to_numpy(), df.iloc[:, 1:].to_numpy(dtype=np.float32)
    raise ValueError(f"unsupported embedding file type: {p.suffix}")


def _strip_ext(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.avi$", "", regex=True)


def load_filelist(path: str | Path) -> pd.DataFrame:
    """Load EchoNet-Dynamic-style FileList.csv (FileName, EF, ESV, EDV, FPS, ...)."""
    df = pd.read_csv(path)
    if "FileName" not in df.columns:
        raise ValueError("FileList must have a FileName column")
    df = df.copy()
    df["FileName"] = _strip_ext(df["FileName"])
    return df


def join(ids: np.ndarray, X: np.ndarray, meta: pd.DataFrame, source: str = "embeddings") -> Dataset:
    """Inner-join embeddings and metadata on video id, preserving embedding order."""
    meta = meta.copy()
    meta["FileName"] = _strip_ext(meta["FileName"])
    m = meta.set_index("FileName")
    id_series = pd.Series(ids).str.replace(r"\.avi$", "", regex=True)
    mask = id_series.isin(m.index).to_numpy()
    kept = id_series[mask]
    out_meta = m.loc[kept].reset_index()
    ds = Dataset(ids=kept.to_numpy(), X=X[mask], meta=out_meta, source=source)
    dropped = int((~mask).sum())
    if dropped:
        ds.notes.append(f"{dropped} embeddings had no metadata row and were dropped")
    if "EF" in ds.meta.columns:
        ds.meta["EF_bin"] = pd.cut(ds.meta["EF"], EF_BINS, labels=EF_BIN_LABELS, right=False)
    if "FPS" in ds.meta.columns and "FPS_modal" not in ds.meta.columns:
        # "was this clip acquired at the scanner's default frame rate" — a clean binary that
        # survives when FPS itself is near-continuous (EchoNet-Dynamic: 79% at 50 fps)
        mode = ds.meta["FPS"].mode().iloc[0]
        ds.meta["FPS_modal"] = np.where(ds.meta["FPS"] == mode, f"fps={mode}", "other")
    return ds


def load_dataset(embeddings: str | Path, filelist: str | Path) -> Dataset:
    ids, X = load_embeddings(embeddings)
    meta = load_filelist(filelist)
    return join(ids, X, meta, source=str(embeddings))


def make_synthetic_filelist(
    ids: np.ndarray,
    X: np.ndarray | None = None,
    seed: int = 0,
    leak_strength: float = 0.6,
) -> pd.DataFrame:
    """Generate a *synthetic* FileList for demos and tests.

    EF is tied to a random projection of X (so probes have something to find) and the
    acquisition fields are given a *known* planted correlation with another projection
    of X, controlled by ``leak_strength``. Nothing here is real patient data: it exists so
    the pipeline can be exercised end-to-end without the Stanford AIMI data-use agreement,
    and so that the audit can be validated against a ground-truth leak.
    """
    rng = np.random.default_rng(seed)
    n = len(ids)
    if X is None:
        X = rng.normal(size=(n, 16)).astype(np.float32)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    w_ef = rng.normal(size=Z.shape[1])
    w_nz = rng.normal(size=Z.shape[1])
    s_ef = Z @ w_ef / np.sqrt(Z.shape[1])
    s_nz = Z @ w_nz / np.sqrt(Z.shape[1])
    ef = np.clip(58 + 10 * s_ef + rng.normal(scale=6, size=n), 10, 85)
    edv = np.clip(110 + 25 * rng.normal(size=n) - 0.8 * (ef - 58), 40, 300)
    esv = np.clip(edv * (1 - ef / 100) + rng.normal(scale=4, size=n), 10, 250)
    # acquisition fields: a planted leak + noise
    mix = leak_strength * s_nz + np.sqrt(max(1 - leak_strength**2, 0)) * rng.normal(size=n)
    fps = np.where(mix > 0.4, 50, np.where(mix < -0.4, 30, 60)).astype(int)
    nframes = np.clip((120 + 60 * mix + rng.normal(scale=25, size=n)).round(), 24, 600).astype(int)
    size = np.where(mix > 0.0, 112, 112)  # EchoNet-Dynamic is uniformly 112x112
    split = rng.choice(["TRAIN", "VAL", "TEST"], p=[0.75, 0.125, 0.125], size=n)
    return pd.DataFrame(
        {
            "FileName": ids,
            "EF": ef.round(2),
            "ESV": esv.round(2),
            "EDV": edv.round(2),
            "FrameHeight": size,
            "FrameWidth": size,
            "FPS": fps,
            "NumberOfFrames": nframes,
            "Split": split,
        }
    )
