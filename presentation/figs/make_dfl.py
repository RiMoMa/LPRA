"""DFL: each box side is a 16-bin distribution, reduced to an expectation."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import C, arrow, rounded_box, save, source_footer


def main() -> None:
    fig = plt.figure(figsize=(14.6, 7.2))
    fig.suptitle("DFL: una caja como cuatro distribuciones de 16 posiciones", fontsize=19, fontweight="bold", x=0.02, ha="left", y=0.965)
    fig.text(
        0.02,
        0.90,
        "La cabeza no regresa un número por lado: regresa una distribución y toma su valor esperado.",
        fontsize=10.5,
        color=C["muted"],
    )

    # Left: illustrative distribution for one side.
    ax = fig.add_axes([0.06, 0.16, 0.38, 0.62])
    i = np.arange(16)
    p = np.exp(-0.5 * ((i - 6.2) / 2.0) ** 2)
    p = p / p.sum()
    bars = ax.bar(i, p, color=C["phase1"], width=0.72, alpha=0.9, edgecolor="white", linewidth=0.6)
    mean = float((i * p).sum())
    ax.axvline(mean, color=C["red"], lw=2.0, ls="--")
    ax.annotate(
        f"E[i] = Σ i·pᵢ = {mean:.1f}",
        xy=(mean, max(p) * 0.94),
        xytext=(mean + 1.6, max(p) * 1.02),
        fontsize=10,
        color=C["red"],
        arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.2),
    )
    ax.set_xlabel("posición DFL  i = 0 … 15")
    ax.set_ylabel("probabilidad  pᵢ")
    ax.set_xticks(i)
    ax.set_ylim(0, max(p) * 1.16)
    ax.set_title("Un lado = Softmax de 16 bins (ejemplo ilustrativo)", fontsize=12, loc="left")
    ax.grid(axis="x", visible=False)

    # Right: pipeline.
    ax2 = fig.add_axes([0.48, 0.08, 0.50, 0.82])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis("off")

    boxes = [
        (8.55, "cv2: 4 lados × 16 bins\n[B, 64, H, W]", C["light_blue"], C["phase1"]),
        (7.25, "Reshape → [B, 4, 16, H, W]\nSoftmax sobre 16 posiciones", C["light_blue"], C["phase1"]),
        (5.95, "Conv 1×1 con pesos fijos 0…15\n→ valor esperado por lado", C["light_orange"], C["phase2"]),
        (4.65, "4 distancias en celdas\n[B, 4, H, W]", C["light_orange"], C["phase2"]),
        (3.35, "× stride (8, 16, 32)\n→ coordenadas en píxeles", C["light_green"], C["phase3"]),
        (2.05, "Caja decodificada\n[x, y, w, h]", C["light_green"], C["phase3"]),
    ]
    for yc, txt, fc, ec in boxes:
        rounded_box(ax2, 0.5, yc - 0.42, 8.6, 0.84, fc, ec, lw=1.2, radius=0.06)
        ax2.text(4.8, yc, txt, ha="center", va="center", fontsize=9.8, color=C["ink"], linespacing=1.15)
        if yc > boxes[-1][0]:
            arrow(ax2, (4.8, yc - 0.44), (4.8, yc - 0.80), color=C["muted"], lw=1.4, mutation=8)


    source_footer(
        fig,
        "Fuente: LEARNING_NOTES.md, «Graph structure» y «Concepts»; la distribución de la izquierda es ilustrativa, no una medida.",
    )
    save(fig, "dfl.svg")


if __name__ == "__main__":
    main()
