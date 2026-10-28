"""Single-setting checks: latency bars from the 2026-10-25 run."""

from __future__ import annotations

import matplotlib.pyplot as plt

from _common import C, read_results_csv, save, source_footer, value

MODELS = [
    ("yolov8n_prep.onnx", "FP32", C["fp32"]),
    ("yolov8n_int8_pc_headfp32.onnx", "pc_headfp32\n(U8)", C["pc_headfp32"]),
    ("yolov8n_int8_pc_headfp32_rr.onnx", "reduce_range\n=True", C["rr"]),
    ("yolov8n_int8_pc_headfp32_s8.onnx", "S8\n(activaciones)", C["s8"]),
]


def main() -> None:
    df = read_results_csv()
    labels, ms, maps = [], [], []
    for model, lab, _ in MODELS:
        labels.append(lab)
        ms.append(value(df, "2-check", model, "mean_ms"))
        maps.append(value(df, "2-check", model, "mAP50"))

    fp32 = ms[0]
    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    bars = ax.bar(range(len(MODELS)), ms, color=[c for _, _, c in MODELS], width=0.62, edgecolor="white", linewidth=1.2)
    for i, (bar, v, ap) in enumerate(zip(bars, ms, maps)):
        speed = fp32 / v
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.55,
            f"{v:.2f} ms\n{speed:.2f}×",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold" if i == 3 else "normal",
            color=C["red"] if i == 3 else C["ink"],
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            max(0.5, v / 2),
            f"mAP@0.5\n{ap:.4f}",
            ha="center",
            va="center",
            fontsize=9.2,
            color="white" if i != 0 else "white",
            fontweight="bold",
        )

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("latencia media (ms)")
    ax.set_ylim(0, 39)
    ax.set_title("Chequeos de un solo ajuste · misma corrida (2026-10-25)", loc="left")
    ax.text(
        0.99,
        0.96,
        "S8: 32,45 ms = 0,50× FP32\nNo es una regresión de precisión: es el grafo que ORT ejecuta.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.4,
        color=C["red"],
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF5F2", ec=C["red"], lw=0.8),
    )
    ax.grid(axis="x", visible=False)
    source_footer(
        fig,
        "Fuente: results.csv, fase 2-check; latencia de results/latency_checks.csv y mAP de results/map_checks.json. "
        "Solo comparaciones dentro de esta corrida.",
    )
    save(fig, "checks.png")


if __name__ == "__main__":
    main()
