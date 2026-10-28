"""Regenerate every figure used by presentation/slides.md.

Run from the repository root:

    .venv/bin/python presentation/figs/make_all.py

The script only reads committed results (plus the local COCO images needed by
the qualitative panels) and writes into presentation/figs/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPTS = [
    "make_pipeline.py",
    "make_arch.py",
    "make_dfl.py",
    "make_quant_affine.py",
    "make_latency_vs_map.py",
    "make_checks.py",
    "make_split.py",
    "make_split_graph.py",
    "make_qualitative.py",
]


def main() -> None:
    histogram = ROOT / "results" / "activation_histogram.png"
    if not histogram.exists():
        raise SystemExit(f"missing reused figure: {histogram}")
    for name in SCRIPTS:
        script = HERE / name
        print(f"\n== {name} ==", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    print("\nAll figures regenerated. Reused:", histogram.relative_to(ROOT))


if __name__ == "__main__":
    main()
