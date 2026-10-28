"""Shared style and data helpers for the LPRA presentation figures.

Run the make_*.py scripts from the repository root, for example:

    .venv/bin/python presentation/figs/make_pipeline.py

Every script resolves the repository from its own location, so the working
directory does not change which files are read.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS_CSV = ROOT / "results.csv"

# Colour-blind friendly palette (Okabe-Ito + neutrals).
C = {
    "fp32": "#4D4D4D",
    "pc_headfp32": "#0072B2",
    "pt_headfp32": "#56B4E9",
    "pc_all": "#D55E00",
    "pt_all": "#E69F00",
    "split": "#009E73",
    "boxq": "#0072B2",
    "scoreq": "#CC79A7",
    "rr": "#CC79A7",
    "s8": "#7A3E9D",
    "phase1": "#4C78A8",
    "phase2": "#E08A1E",
    "phase3": "#2A9D6F",
    "ink": "#222222",
    "muted": "#6B7280",
    "grid": "#D7DDE5",
    "paper": "#FFFFFF",
    "light_blue": "#E8F0F8",
    "light_orange": "#FDF0DF",
    "light_green": "#E4F3EC",
    "red": "#C0392B",
}

LABELS = {
    "yolov8n_prep.onnx": "FP32",
    "yolov8n_prep": "FP32",
    "yolov8n_int8_pc_headfp32.onnx": "INT8 pc / head FP32",
    "yolov8n_int8_pc_headfp32": "INT8 pc / head FP32",
    "yolov8n_int8_pt_headfp32.onnx": "INT8 pt / head FP32",
    "yolov8n_int8_pt_headfp32": "INT8 pt / head FP32",
    "yolov8n_int8_pc_all.onnx": "INT8 pc / todo",
    "yolov8n_int8_pc_all": "INT8 pc / todo",
    "yolov8n_int8_pt_all.onnx": "INT8 pt / todo",
    "yolov8n_int8_pt_all": "INT8 pt / todo",
    "yolov8n_int8_split_pc_all.onnx": "split / todo INT8",
    "yolov8n_int8_split_pc_all": "split / todo INT8",
    "yolov8n_int8_split_pc_boxq.onnx": "split / boxq",
    "yolov8n_int8_split_pc_scoreq.onnx": "split / scoreq",
    "yolov8n_int8_pc_headfp32_rr.onnx": "reduce_range=True",
    "yolov8n_int8_pc_headfp32_s8.onnx": "S8 (activaciones)",
}

COLORS = {
    "yolov8n_prep.onnx": C["fp32"],
    "yolov8n_int8_pc_headfp32.onnx": C["pc_headfp32"],
    "yolov8n_int8_pt_headfp32.onnx": C["pt_headfp32"],
    "yolov8n_int8_pc_all.onnx": C["pc_all"],
    "yolov8n_int8_pt_all.onnx": C["pt_all"],
    "yolov8n_int8_split_pc_all.onnx": C["split"],
    "yolov8n_int8_pc_headfp32_rr.onnx": C["rr"],
    "yolov8n_int8_pc_headfp32_s8.onnx": C["s8"],
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": C["ink"],
        "axes.labelcolor": C["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": ":",
        "grid.alpha": 0.8,
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        "figure.facecolor": C["paper"],
        "axes.facecolor": C["paper"],
        "savefig.facecolor": C["paper"],
        "svg.fonttype": "path",
        "legend.frameon": False,
    }
)


def read_results_csv() -> pd.DataFrame:
    return pd.read_csv(RESULTS_CSV, dtype={"phase": str})


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def read_json(name: str):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def value(df: pd.DataFrame, phase: str, model: str, metric: str) -> float:
    rows = df[(df["phase"] == phase) & (df["model"] == model) & (df["metric"] == metric)]
    if rows.empty:
        return float("nan")
    raw = rows.iloc[0]["value"]
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float("nan")


def label(model: str) -> str:
    return LABELS.get(model, model.replace(".onnx", "").replace("yolov8n_", ""))


def color(model: str) -> str:
    return COLORS.get(model, C["muted"])


def save(fig: plt.Figure, name: str, dpi: int = 180) -> None:
    path = FIGDIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, dpi=dpi)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def rounded_box(ax, x, y, w, h, fc, ec=None, lw=1.2, radius=0.08, alpha=1.0, z=2):
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec or fc,
        linewidth=lw,
        alpha=alpha,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, xy_from, xy_to, color=None, lw=1.5, style="-|>", mutation=10, z=3, **kw):
    from matplotlib.patches import FancyArrowPatch

    a = FancyArrowPatch(
        xy_from,
        xy_to,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=lw,
        color=color or C["ink"],
        shrinkA=2,
        shrinkB=2,
        zorder=z,
        **kw,
    )
    ax.add_patch(a)
    return a


def source_footer(fig, text: str, y: float = 0.015) -> None:
    fig.text(0.01, y, text, ha="left", va="bottom", fontsize=8, color=C["muted"])
