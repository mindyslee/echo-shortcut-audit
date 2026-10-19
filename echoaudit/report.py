"""Run the full audit and write JSON, Markdown and figures."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .data import CLINICAL_TARGETS, NUISANCE_TARGETS, Dataset
from .probes import ProbeResult, probe, random_embedding_control
from .removal import cost_curve
from .stratify import metadata_only_baseline, partial_r2, stratified_error


@dataclass
class AuditConfig:
    clinical: list[str] = field(default_factory=lambda: list(CLINICAL_TARGETS))
    nuisance: list[str] = field(default_factory=lambda: list(NUISANCE_TARGETS))
    n_splits: int = 5
    n_permutations: int = 20
    inlp_iterations: int = 8
    seed: int = 0
    max_rows: int | None = None  # subsample for quick runs
    pca_dim: int | None = None  # optionally probe a PCA-reduced embedding


def _constant(v: np.ndarray) -> bool:
    return len(np.unique(v)) < 2


def run_audit(ds: Dataset, cfg: AuditConfig | None = None, log=print) -> dict:
    cfg = cfg or AuditConfig()
    t0 = time.time()
    X, meta = ds.X, ds.meta
    if cfg.max_rows and ds.n > cfg.max_rows:
        rng = np.random.default_rng(cfg.seed)
        idx = np.sort(rng.choice(ds.n, cfg.max_rows, replace=False))
        X, meta = X[idx], meta.iloc[idx].reset_index(drop=True)
    if cfg.pca_dim:
        from sklearn.decomposition import PCA

        X = PCA(n_components=cfg.pca_dim, random_state=cfg.seed).fit_transform(X).astype(np.float32)

    clinical = [c for c in ds.available(cfg.clinical) if not _constant(meta[c].to_numpy())]
    nuisance = [c for c in ds.available(cfg.nuisance) if not _constant(meta[c].to_numpy())]
    skipped = [c for c in cfg.clinical + cfg.nuisance if c not in clinical + nuisance]
    out: dict = {
        "version": __version__,
        "source": ds.source,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "config": cfg.__dict__,
        "skipped_targets": skipped,
        "notes": list(ds.notes),
    }
    if skipped:
        log(f"skipping constant/missing targets: {skipped}")

    # 1. decodability of every target, plus a random-matrix control
    log("probing clinical and nuisance targets ...")
    probes: dict[str, ProbeResult] = {}
    controls: dict[str, ProbeResult] = {}
    for c in clinical + nuisance:
        y = meta[c].to_numpy()
        probes[c] = probe(X, y, c, cfg.n_splits, cfg.n_permutations, cfg.seed)
        controls[c] = random_embedding_control(
            X.shape[0], X.shape[1], y, c, cfg.seed, n_splits=cfg.n_splits, n_permutations=0
        )
        log(
            f"  {c:>15s}: {probes[c].kind:14s} score={probes[c].score:.3f} chance={probes[c].chance:.3f} p={probes[c].p_value:.3f}"
        )
    out["probes"] = {k: v.to_dict() for k, v in probes.items()}
    out["random_controls"] = {k: v.to_dict() for k, v in controls.items()}
    out["clinical_targets"] = clinical
    out["nuisance_targets"] = nuisance

    # 2. metadata-only baselines and partial R² for the primary clinical target
    primary = clinical[0] if clinical else None
    out["primary_target"] = primary
    if primary and nuisance:
        log(f"metadata-only baseline for {primary} ...")
        out["metadata_only"] = metadata_only_baseline(
            meta, primary, nuisance, cfg.n_splits, cfg.seed
        )
        out["partial_r2"] = partial_r2(X, meta, primary, nuisance, cfg.n_splits, cfg.seed)
        log("stratified error ...")
        out["stratified_error"] = stratified_error(
            X, meta, primary, nuisance, cfg.n_splits, cfg.seed
        )

    # 3. erase each nuisance and price it against every clinical target
    out["cost_curves"] = {}
    for nz in nuisance:
        if not probes[nz].decodable:
            log(f"{nz} not decodable above chance; skipping erasure")
            continue
        log(f"erasing {nz} with INLP ...")
        cc = cost_curve(
            X,
            meta[nz].to_numpy(),
            nz,
            {c: meta[c].to_numpy() for c in clinical},
            n_iter=cfg.inlp_iterations,
            n_splits=cfg.n_splits,
            seed=cfg.seed,
        )
        out["cost_curves"][nz] = cc.to_dict()
        log(
            f"  {nz}: {cc.nuisance_score[0]:.3f} -> {cc.nuisance_score[-1]:.3f} after rank {cc.rank_removed[-1]}; "
            + ", ".join(f"{c} {cc.clinical[c][0]:.3f}->{cc.clinical[c][-1]:.3f}" for c in clinical)
        )

    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------- rendering


def _fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def to_markdown(res: dict) -> str:
    lines = [f"# Shortcut audit: `{Path(res['source']).name}`", ""]
    lines.append(f"n = {res['n']} videos, d = {res['d']} dims. {res['seconds']} s.")
    for n in res.get("notes", []):
        lines.append(f"- note: {n}")
    lines += ["", "## Decodability (cross-validated linear probe)", ""]
    lines.append(
        "| target | role | kind | score | chance | random-matrix control | permutation null | z | p | decodable |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, p in res["probes"].items():
        role = "clinical" if k in res["clinical_targets"] else "nuisance"
        ctrl = res["random_controls"][k]["score"]
        lines.append(
            f"| {k} | {role} | {p['kind']} | **{_fmt(p['score'])}** ± {_fmt(p['score_std'], 2)} | {_fmt(p['chance'], 2)} | "
            f"{_fmt(ctrl)} | {_fmt(p['null_mean'])} ± {_fmt(p['null_std'])} | {_fmt(p['z_vs_null'], 1)} | {_fmt(p['p_value'])} | "
            f"{'yes' if p['decodable'] else 'no'} |"
        )
    lines.append("")
    n_perm = next(iter(res["probes"].values()))["n_permutations"] if res["probes"] else 0
    lines.append(
        "Regression score is R²; classification score is balanced accuracy (chance = 1/k). "
        f"The permutation p-value is floored at 1/({n_perm}+1); z is (score − null mean)/null sd."
    )
    if "metadata_only" in res:
        m, pr = res["metadata_only"], res["partial_r2"]
        lines += ["", f"## Metadata-only baseline for {m['target']}", ""]
        lines.append(
            f"- linear R² from acquisition metadata alone: **{_fmt(m['linear_r2'])}** (p = {_fmt(m['linear_p_value'])})"
        )
        lines.append(f"- boosted-trees R² from metadata alone: **{_fmt(m['boosted_r2'])}**")
        lines.append(
            f"- embedding R² = {_fmt(pr['r2_embedding'])}; embedding + metadata R² = {_fmt(pr['r2_both'])}"
        )
        lines.append(
            f"- embedding beyond metadata: **{_fmt(pr['embedding_beyond_metadata'])}**; metadata beyond embedding: **{_fmt(pr['metadata_beyond_embedding'])}**"
        )
    if res.get("stratified_error"):
        lines += ["", f"## {res['primary_target']} error by acquisition stratum", ""]
        lines.append("| stratum | bins (n, MAE) | MAE range | Kruskal-Wallis p |")
        lines.append("|---|---|---|---|")
        for s in res["stratified_error"]:
            bins = "; ".join(f"{b}: {v['n']}, {v['mae']:.2f}" for b, v in s["bins"].items())
            lines.append(
                f"| {s['stratum']} | {bins} | {s['mae_range']:.2f} | {_fmt(s['kruskal_p'])} |"
            )
    if res.get("cost_curves"):
        lines += ["", "## Price of erasing each nuisance (INLP)", ""]
        lines.append(
            "| nuisance | decodability before → after | rank removed | "
            + " | ".join(
                f"{c}: before → after (random ctrl) → excess price" for c in res["clinical_targets"]
            )
            + " |"
        )
        lines.append("|---|---|---|" + "---|" * len(res["clinical_targets"]))
        for nz, cc in res["cost_curves"].items():
            cells = []
            for c in res["clinical_targets"]:
                before, after = cc["clinical"][c][0], cc["clinical"][c][-1]
                rc = cc.get("random_control", {}).get(c)
                rc_txt = f"{rc[0]:.3f} ± {rc[1]:.3f}" if rc else "n/a"
                cells.append(
                    f"{before:.3f} → {after:.3f} ({rc_txt}) → **{cc['price_vs_random'][c]:+.3f}**"
                )
            lines.append(
                f"| {nz} | {cc['nuisance_score'][0]:.3f} → {cc['nuisance_score'][-1]:.3f} (chance {cc['nuisance_chance']:.2f}) | {cc['rank_removed'][-1]} | "
                + " | ".join(cells)
                + " |"
            )
        lines.append("")
        lines.append(
            "Erasure runs in a PCA-whitened copy of the embedding until the nuisance is no longer linearly decodable. "
            "Every projection costs a little, so the same number of random directions is removed as a control; "
            "the **excess price** (random-control score minus post-erasure score) is the amount of clinical signal that "
            "shared directions with the nuisance. Near zero: encoded but not load-bearing. Clearly positive: the clinical "
            "signal was riding on acquisition geometry. A nuisance score that overshoots below chance is an INLP artifact "
            "(directions are fit on all rows), not evidence of anything."
        )
    return "\n".join(lines) + "\n"


def save_figures(res: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    # decodability bar chart
    names = list(res["probes"])
    above = [res["probes"][k]["score"] - res["probes"][k]["chance"] for k in names]
    ctrl = [res["random_controls"][k]["score"] - res["random_controls"][k]["chance"] for k in names]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(names))
    ax.bar(
        x - 0.2,
        above,
        0.4,
        label="embedding",
        color=["#2b6cb0" if k in res["clinical_targets"] else "#c05621" for k in names],
    )
    ax.bar(x + 0.2, ctrl, 0.4, label="random-matrix control", color="#a0aec0")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("score above chance")
    ax.set_title("What the embedding knows (blue = clinical, orange = acquisition)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    p = out_dir / "decodability.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    # cost curves
    if res.get("cost_curves"):
        n = len(res["cost_curves"])
        fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4), squeeze=False)
        for ax, (nz, cc) in zip(axes[0], res["cost_curves"].items()):
            it = cc["iterations"]
            ax.plot(it, cc["nuisance_score"], "o-", color="#c05621", label=f"{nz} (nuisance)")
            ax.axhline(cc["nuisance_chance"], color="#c05621", ls=":", lw=0.8)
            for c, s in cc["clinical"].items():
                (line,) = ax.plot(it, s, "s-", label=c)
                rc = cc.get("random_control", {}).get(c)
                if rc:
                    ax.plot(
                        it[-1],
                        rc[0],
                        "x",
                        color=line.get_color(),
                        ms=9,
                        mew=2,
                        label=f"{c}: random-rank control",
                    )
            ax.set_xlabel("INLP iteration")
            ax.set_ylabel("decodability")
            ax.set_title(f"Erasing {nz}")
            ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        p = out_dir / "cost_curves.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def write_report(res: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j = out / "audit.json"
    j.write_text(json.dumps(res, indent=2, default=float))
    m = out / "audit.md"
    m.write_text(to_markdown(res))
    figs = save_figures(res, out)
    return {"json": j, "markdown": m, **{f.stem: f for f in figs}}
