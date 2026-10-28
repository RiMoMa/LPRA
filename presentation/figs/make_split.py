"""Phase 3: latency and mAP of the output-split experiment, one run."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import C, read_results_csv, save, source_footer, value

MODELS = [
    ("yolov8n_prep.onnx", "FP32", C["fp32"]),
    ("yolov8n_int8_pc_headfp32.onnx", "pc_headfp32", C["pc_headfp32"]),
    ("yolov8n_int8_split_pc_all.onnx", "split_pc_all", C["split"]),
    ("yolov8n_int8_pc_all.onnx", "pc_all\n(sin mAP)", C["pc_all"]),
]
MAP_MODELS = [
    ("yolov8n_prep.onnx", "FP32", C["fp32"]),
    ("yolov8n_int8_pc_headfp32.onnx", "pc_headfp32", C["pc_headfp32"]),
    ("yolov8n_int8_split_pc_all.onnx", "split_pc_all", C["split"]),
]


def main() -> None:
    df = read_results_csv()
    lat = [value(df, "3-split", m, "mean_ms") for m, _, _ in MODELS]
    fp32 = lat[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 6.2), gridspec_kw={"width_ratios": [1.15, 1.0]})

    bars = ax1.bar(range(len(MODELS)), lat, color=[c for _, _, c in MODELS], width=0.62, edgecolor="white", linewidth=1.2)
    for i, (bar, v) in enumerate(zip(bars, lat)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.45,
            f"{v:.2f} ms\n{fp32 / v:.2f}×",
            ha="center",
            va="bottom",
            fontsize=11.6,
            fontweight="bold" if i == 2 else "normal",
            color=C["split"] if i == 2 else C["ink"],
        )
    ax1.set_xticks(range(len(MODELS)))
    ax1.set_xticklabels([lab for _, lab, _ in MODELS], fontsize=10.8)
    ax1.set_ylabel("latencia media (ms)")
    ax1.set_ylim(0, 23)
    ax1.set_title("(a) Latencia", loc="left")
    ax1.grid(axis="x", visible=False)
    ax1.axhline(fp32, color=C["muted"], lw=0.9, ls=":", alpha=0.8)

    x = np.arange(len(MAP_MODELS))
    width = 0.34
    ap50 = [value(df, "3-split", m, "mAP50") for m, _, _ in MAP_MODELS]
    ap5095 = [value(df, "3-split", m, "mAP50_95") for m, _, _ in MAP_MODELS]
    b1 = ax2.bar(x - width / 2, ap50, width, color=[c for _, _, c in MAP_MODELS], edgecolor="white", linewidth=1.0, label="mAP@0.5")
    b2 = ax2.bar(
        x + width / 2,
        ap5095,
        width,
        color=[c for _, _, c in MAP_MODELS],
        alpha=0.45,
        edgecolor="white",
        linewidth=1.0,
        hatch="//",
        label="mAP@.5:.95",
    )
    for bars_ in (b1, b2):
        for bar in bars_:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.010,
                f"{h:.4f}",
                ha="center",
                va="bottom",
                fontsize=9.6,
                rotation=0,
                color=C["ink"],
            )
    ax2.set_xticks(x)
    ax2.set_xticklabels([lab for _, lab, _ in MAP_MODELS], fontsize=10.8)
    ax2.set_ylim(0, 0.72)
    ax2.set_ylabel("métrica COCO")
    ax2.set_title("(b) Precisión", loc="left")
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="upper right", fontsize=9.6, ncol=1)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.94, bottom=0.14, wspace=0.30)
    source_footer(
        fig,
        "Fuente: results.csv, fase 3-split; latencia de results/latency_split.csv y mAP de results/map_split.json. "
        "La fila pc_all es solo latencia; no se mezcla con la fase 2.",
    )
    save(fig, "split.png")


if __name__ == "__main__":
    main()
