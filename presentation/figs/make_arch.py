"""YOLOv8n architecture at a glance: backbone, neck and decoupled head."""

from __future__ import annotations

import matplotlib.pyplot as plt

from _common import C, arrow, rounded_box, save, source_footer


def add_box(ax, x, y, w, h, title, fc=C["light_blue"], ec=C["phase1"], fs=9.0, bold=False, color=None):
    rounded_box(ax, x, y, w, h, fc, ec, lw=1.1, radius=0.06)
    ax.text(
        x + w / 2,
        y + h / 2,
        title,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        color=color or C["ink"],
        linespacing=1.15,
        zorder=5,
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(16.4, 8.2))
    ax.set_xlim(0, 16.4)
    ax.set_ylim(1.55, 8.8)
    ax.axis("off")

    ax.text(0.25, 8.48, "YOLOv8n por dentro", fontsize=21, fontweight="bold", color=C["ink"])
    ax.text(
        0.25,
        8.17,
        "Entrada 640×640 · backbone CSP · neck FPN/PAN · cabeza desacoplada sin anclas",
        fontsize=11,
        color=C["muted"],
    )
    # Parameters badge (from YOLO('weights/yolov8n.pt').info()).
    rounded_box(ax, 12.55, 7.88, 3.55, 0.62, "#F3F4F6", C["grid"], lw=0.8, radius=0.05)
    ax.text(
        14.32,
        8.19,
        "3.157.200 parámetros  ·  8,9 GFLOPs",
        ha="center",
        va="center",
        fontsize=9.6,
        fontweight="bold",
        color=C["ink"],
    )

    # Input.
    add_box(ax, 0.25, 5.85, 1.25, 1.0, "Entrada\n[B,3,640,640]", "#EEF1F5", C["muted"], fs=7.8, bold=True)
    arrow(ax, (1.52, 6.35), (1.75, 6.35), color=C["muted"], lw=1.4)

    # Backbone panel.
    rounded_box(ax, 1.75, 2.72, 4.65, 4.95, C["light_blue"], C["phase1"], lw=1.1, radius=0.08)
    ax.text(4.08, 7.43, "Backbone", ha="center", fontsize=12, fontweight="bold", color=C["phase1"])
    backbone = [
        (6.85, "Stem Conv s2\n[B, 16, 320, 320]"),
        (5.95, "Conv s2 + C2f\n[B, 32, 160, 160]"),
        (5.05, "Conv s2 + C2f\n[B, 64, 80, 80]   P3"),
        (4.15, "Conv s2 + C2f\n[B, 128, 40, 40]  P4"),
        (3.25, "Conv s2 + C2f + SPPF\n[B, 256, 20, 20]  P5"),
    ]
    for i, (yy, txt) in enumerate(backbone):
        add_box(ax, 1.95, yy - 0.34, 4.25, 0.68, txt, "#FFFFFF", C["phase1"], fs=8.7)
        if i < len(backbone) - 1:
            arrow(ax, (4.07, yy - 0.35), (4.07, yy - 0.46), color=C["phase1"], lw=1.2, mutation=7)
    ax.text(
        4.08,
        2.90,
        "64 Conv · 57 SiLU (Sigmoid→Mul) · 3 MaxPool",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color=C["muted"],
    )

    # Neck panel.
    rounded_box(ax, 6.65, 2.72, 4.15, 4.95, C["light_orange"], C["phase2"], lw=1.1, radius=0.08)
    ax.text(8.72, 7.43, "Neck: FPN + PAN", ha="center", fontsize=12, fontweight="bold", color=C["phase2"])
    add_box(ax, 6.88, 6.55, 3.70, 0.72, "P5 → Conv 1×1\n[B, 128, 20, 20]", "#FFFFFF", C["phase2"], fs=8.7)
    add_box(ax, 6.88, 5.52, 3.70, 0.78, "Resize ×2 + Concat P4\n[B, 192, 40, 40]", "#FFFFFF", C["phase2"], fs=8.7)
    add_box(ax, 6.88, 4.55, 3.70, 0.72, "C2f → N4\n[B, 128, 40, 40]", "#FFFFFF", C["phase2"], fs=8.7)
    add_box(ax, 6.88, 3.55, 3.70, 0.72, "Resize ×2 + Concat P3\n[B, 128, 80, 80]", "#FFFFFF", C["phase2"], fs=8.7)
    add_box(ax, 6.88, 2.90, 3.70, 0.48, "C2f → N3  ·  PAN → N4′, N5", "#FFFFFF", C["phase2"], fs=8.2)
    for y0, y1 in [(6.53, 6.32), (6.30, 6.05), (5.50, 5.29), (4.53, 4.29), (3.53, 3.40)]:
        arrow(ax, (8.73, y0), (8.73, y1), color=C["phase2"], lw=1.2, mutation=7)

    # Arrows backbone -> neck (P3, P4, P5).
    for yy, lab in [(5.05, "P3"), (4.15, "P4"), (3.25, "P5")]:
        arrow(ax, (6.23, yy), (6.64, yy), color=C["muted"], lw=1.1, mutation=7)
        ax.text(6.43, yy + 0.09, lab, ha="center", va="bottom", fontsize=7.4, color=C["muted"])

    # Head panel.
    rounded_box(ax, 11.15, 2.72, 5.0, 4.95, C["light_green"], C["phase3"], lw=1.1, radius=0.08)
    ax.text(13.65, 7.43, "Head: 3 escalas, desacoplada", ha="center", fontsize=12, fontweight="bold", color=C["phase3"])
    rows = [
        (6.55, "N3 · 80×80", "cv2: 4×16 DFL\ncv3: 80 clases", "6400"),
        (5.55, "N4 · 40×40", "cv2: 4×16 DFL\ncv3: 80 clases", "1600"),
        (4.55, "N5 · 20×20", "cv2: 4×16 DFL\ncv3: 80 clases", "400"),
    ]
    for yy, scale, branches, npoints in rows:
        add_box(ax, 11.32, yy - 0.30, 1.12, 0.60, scale, "#FFFFFF", C["phase3"], fs=8.0, bold=True)
        add_box(ax, 12.60, yy - 0.36, 1.65, 0.72, branches, "#FFFFFF", C["phase3"], fs=7.8)
        add_box(ax, 14.42, yy - 0.30, 1.35, 0.60, f"{npoints} puntos\n84 canales", "#FFFFFF", C["phase3"], fs=7.8)
        arrow(ax, (12.46, yy), (12.58, yy), color=C["muted"], lw=1.0, mutation=6)
        arrow(ax, (14.27, yy), (14.40, yy), color=C["muted"], lw=1.0, mutation=6)
    add_box(ax, 11.32, 3.38, 4.45, 0.58, "Concat de las 3 escalas → [B, 84, 8400]", "#D9F0E5", C["phase3"], fs=8.8, bold=True)
    arrow(ax, (13.65, 4.18), (13.65, 3.98), color=C["phase3"], lw=1.2, mutation=7)
    ax.text(
        13.65,
        3.08,
        "8400 = 80² + 40² + 20²  ·  sin objectness",
        ha="center",
        va="top",
        fontsize=8.2,
        color=C["muted"],
    )

    # Arrows neck -> head rows.
    for yy, lab in [(6.55, "N3"), (5.55, "N4"), (4.55, "N5")]:
        arrow(ax, (10.82, yy), (11.13, yy), color=C["muted"], lw=1.1, mutation=7)

    # Bottom notes.
    ax.text(
        0.25,
        2.25,
        "El cuello proyecta tres mapas de características; la cabeza comparte pesos entre escalas y "
        "predice cajas y clases en ramas separadas.",
        fontsize=9.2,
        color=C["muted"],
    )
    ax.text(
        0.25,
        1.85,
        "La salida final mezcla coordenadas en píxeles y probabilidades [0,1] en un solo tensor: "
        "el peligro de cuantización de la fase 2.",
        fontsize=9.2,
        color=C["red"],
        fontweight="bold",
    )
    source_footer(
        fig,
        "Fuente: LEARNING_NOTES.md, «Graph structure»; parámetros y GFLOPs: salida de "
        "YOLO('weights/yolov8n.pt').info().",
    )
    save(fig, "yolov8n_arch.svg")


if __name__ == "__main__":
    main()
