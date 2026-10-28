"""Pipeline 00 -> 13, drawn from the Pipeline table in README.md."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from _common import C, FIGDIR, ROOT, arrow, rounded_box, save, source_footer

# (number, title, output, phase)
STEPS = [
    ("00", "Entorno", "results/env.json", 1),
    ("01", "Export ONNX", "models/yolov8n*.onnx", 1),
    ("02", "Inspección", "stdout: firmas y\nhistograma de ops", 1),
    ("03", "Paridad", "results/parity_*.json", 1),
    ("04", "Split COCO", "data/split.json", 2),
    ("05", "Cuantización", "models/yolov8n_int8_*.onnx", 2),
    ("06", "Grafo QDQ", "stdout: grafo en ejecución", 2),
    ("07", "Histograma", "results/activation_histogram.png", 2),
    ("08", "Benchmark", "results/latency_raw.csv", 2),
    ("08b", "Perfil", "results/profile_*.json", 2),
    ("09", "mAP", "results/map_overall.json", 2),
    ("10", "Recolectar", "results.csv", 2),
    ("11", "Chequeos", "models/*_rr.onnx\nmodels/*_s8.onnx", 2),
    ("12", "Cirugía", "models/yolov8n_split.onnx\nmodels/*_split_pc_all.onnx", 3),
    ("13", "Atribución", "results/attribution_split.json", 3),
]

PHASE_COLOR = {1: C["phase1"], 2: C["phase2"], 3: C["phase3"]}
PHASE_LIGHT = {1: C["light_blue"], 2: C["light_orange"], 3: C["light_green"]}
PHASE_NAME = {1: "Fase 1 · exportación", 2: "Fase 2 · INT8", 3: "Fase 3 · cirugía"}


def main() -> None:
    # Snake layout: top row 00..04, middle row 05..08b (right to left),
    # bottom row 09..13 (left to right). This keeps every card legible.
    grid = [
        [0, 1, 2, 3, 4],
        [9, 8, 7, 6, 5],  # 05 at the right, then 06, 07, 08, 08b at the left
        [10, 11, 12, 13, 14],
    ]
    x = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
    y = {0: 2.56, 1: 1.42, 2: 0.28}
    w, h = 0.86, 0.74

    fig, ax = plt.subplots(figsize=(16.2, 9.0))
    ax.set_xlim(-0.55, 4.55)
    ax.set_ylim(-0.22, 3.38)
    ax.axis("off")

    ax.text(0.0, 3.24, "Pipeline reproducible · 00 → 13", fontsize=20, fontweight="bold", color=C["ink"])
    ax.text(
        0.0,
        3.10,
        "Cada paso lee los archivos del anterior; las flechas dobles indican el cambio de fila.",
        fontsize=10.5,
        color=C["muted"],
    )

    handles = [Patch(facecolor=PHASE_LIGHT[k], edgecolor=PHASE_COLOR[k], label=PHASE_NAME[k]) for k in (1, 2, 3)]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.04), ncol=3, fontsize=9.5, handlelength=1.3)

    # Draw cards.
    for row, indices in enumerate(grid):
        for col, idx in enumerate(indices):
            num, title, output, phase = STEPS[idx]
            cx = x[col]
            cy = y[row]
            rounded_box(
                ax,
                cx - w / 2,
                cy - h / 2,
                w,
                h,
                PHASE_LIGHT[phase],
                PHASE_COLOR[phase],
                lw=1.3,
                radius=0.07,
            )
            # Number badge.
            ax.scatter([cx - w / 2 + 0.10], [cy + h / 2 - 0.10], s=250, color=PHASE_COLOR[phase], zorder=5)
            ax.text(
                cx - w / 2 + 0.10,
                cy + h / 2 - 0.10,
                num,
                ha="center",
                va="center",
                fontsize=8.4 if len(num) == 2 else 7.4,
                fontweight="bold",
                color="white",
                zorder=6,
            )
            ax.text(
                cx,
                cy + 0.08,
                title,
                ha="center",
                va="center",
                fontsize=10.0,
                fontweight="bold",
                color=C["ink"],
                zorder=6,
            )
            ax.text(
                cx,
                cy - 0.19,
                output,
                ha="center",
                va="center",
                fontsize=6.9,
                family="DejaVu Sans Mono",
                color=C["muted"],
                linespacing=1.08,
                zorder=6,
            )

    # Horizontal arrows within each row, following the snake direction.
    for row, indices in enumerate(grid):
        direction = 1 if row != 1 else -1
        seq = indices if direction > 0 else list(reversed(indices))
        pairs = list(zip(seq, seq[1:]))
        for a, b in pairs:
            xa = x[grid[row].index(a)]
            xb = x[grid[row].index(b)]
            if direction > 0:
                start = (xa + w / 2 + 0.015, y[row])
                end = (xb - w / 2 - 0.015, y[row])
            else:
                start = (xa - w / 2 - 0.015, y[row])
                end = (xb + w / 2 + 0.015, y[row])
            arrow(ax, start, end, color=C["muted"], lw=1.4, mutation=9)

    # Vertical arrows at the two row changes: 04 -> 05 (right) and 08b -> 09 (left).
    arrow(ax, (4.0, 2.56 - h / 2 - 0.02), (4.0, 1.42 + h / 2 + 0.02), color=C["muted"], lw=1.4, mutation=9)
    arrow(ax, (0.0, 1.42 - h / 2 - 0.02), (0.0, 0.28 + h / 2 + 0.02), color=C["muted"], lw=1.4, mutation=9)

    ax.text(4.12, 1.99, "sigue\nabajo", ha="left", va="center", fontsize=8, color=C["muted"], style="italic")
    ax.text(-0.12, 0.85, "sigue\nabajo", ha="right", va="center", fontsize=8, color=C["muted"], style="italic")

    source_footer(fig, "Fuente: README.md, tabla «Pipeline»; los nombres de salida son los del repositorio.")
    save(fig, "pipeline.svg")


if __name__ == "__main__":
    main()
