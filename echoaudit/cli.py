"""Command-line entry point.

echoaudit run  --embeddings echonet_embeddings.txt --filelist FileList.csv --out results/
echoaudit demo --embeddings echonet_embeddings.txt --out results_demo/   # synthetic labels
echoaudit demo --out results_demo/                                        # fully synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .data import join, load_dataset, load_embeddings, make_synthetic_filelist
from .probes import set_probe_family
from .report import AuditConfig, run_audit, write_report


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--permutations", type=int, default=20)
    p.add_argument("--inlp-iterations", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=None, help="subsample for a quick pass")
    p.add_argument("--pca", type=int, default=None, help="probe a PCA-reduced embedding instead")
    p.add_argument(
        "--probe",
        choices=["linear", "mlp"],
        default="linear",
        help="measurement probe family; erasure directions are always linear",
    )
    p.add_argument("--clinical", nargs="*", default=None, help="override clinical target columns")
    p.add_argument("--nuisance", nargs="*", default=None, help="override nuisance target columns")


def _config(a: argparse.Namespace) -> AuditConfig:
    cfg = AuditConfig(
        n_splits=a.splits,
        n_permutations=a.permutations,
        inlp_iterations=a.inlp_iterations,
        seed=a.seed,
        max_rows=a.max_rows,
        pca_dim=a.pca,
    )
    if a.clinical:
        cfg.clinical = a.clinical
    if a.nuisance:
        cfg.nuisance = a.nuisance
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="echoaudit", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="audit real embeddings against a FileList.csv")
    run.add_argument("--embeddings", required=True)
    run.add_argument("--filelist", required=True)
    _add_common(run)

    demo = sub.add_parser("demo", help="audit with a synthetic FileList (no DUA needed)")
    demo.add_argument(
        "--embeddings", default=None, help="real embeddings to pair with synthetic labels"
    )
    demo.add_argument("--n", type=int, default=2000, help="rows when no embeddings are given")
    demo.add_argument(
        "--leak", type=float, default=0.6, help="planted nuisance leak strength in [0,1]"
    )
    _add_common(demo)

    a = ap.parse_args(argv)
    cfg = _config(a)
    set_probe_family(a.probe)

    if a.cmd == "run":
        ds = load_dataset(a.embeddings, a.filelist)
    else:
        if a.embeddings:
            ids, X = load_embeddings(a.embeddings)
            source = f"{a.embeddings} + synthetic labels"
        else:
            rng = np.random.default_rng(a.seed)
            ids = np.array([f"SYN{i:06d}" for i in range(a.n)])
            X = rng.normal(size=(a.n, 64)).astype(np.float32)
            source = "fully synthetic"
        meta = make_synthetic_filelist(ids, X, seed=a.seed, leak_strength=a.leak)
        ds = join(ids, X, meta, source=source)
        ds.notes.append("labels are SYNTHETIC; numbers here validate the pipeline, not the model")

    print(f"{ds.n} videos x {ds.d} dims from {ds.source}", file=sys.stderr)
    res = run_audit(ds, cfg, log=lambda s: print(s, file=sys.stderr))
    paths = write_report(res, Path(a.out))
    print(paths["markdown"].read_text())
    print(f"wrote {', '.join(str(p) for p in paths.values())}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
