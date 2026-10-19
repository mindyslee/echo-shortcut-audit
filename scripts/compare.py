"""Side-by-side table from several audit.json files (one per embedding).

python scripts/compare.py results_r3d/audit.json results_echoclip/audit.json ...
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(paths: list[str]) -> None:
    runs = [(Path(p).parent.name, json.loads(Path(p).read_text())) for p in paths]
    targets: list[str] = []
    for _, r in runs:
        for t in r["probes"]:
            if t not in targets:
                targets.append(t)
    print(
        "| embedding | d | "
        + " | ".join(targets)
        + " | EF beyond metadata | "
        + " | ".join(
            f"price of {t}" for t in targets if any(t in r.get("cost_curves", {}) for _, r in runs)
        )
        + " |"
    )
    print(
        "|---|---|"
        + "---|" * len(targets)
        + "---|"
        + "---|" * len([t for t in targets if any(t in r.get("cost_curves", {}) for _, r in runs)])
    )
    for name, r in runs:
        cells = []
        for t in targets:
            p = r["probes"].get(t)
            cells.append(f"{p['score']:.3f} (chance {p['chance']:.2f})" if p else "–")
        beyond = r.get("partial_r2", {}).get("embedding_beyond_metadata")
        prices = []
        for t in targets:
            cc = r.get("cost_curves", {}).get(t)
            if any(t in rr.get("cost_curves", {}) for _, rr in runs):
                prices.append(
                    ", ".join(f"{c} {v:+.3f}" for c, v in cc["price"].items()) if cc else "–"
                )
        beyond_cell = f"{beyond:.3f}" if beyond is not None else "–"
        row = [name, str(r["d"]), *cells, beyond_cell, *prices]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
