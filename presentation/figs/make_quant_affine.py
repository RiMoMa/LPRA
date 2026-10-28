"""Affine quantization: rounding inside the range, clipping outside it."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _common import C, save, source_footer


def main() -> None:
    # Didactic example with 7 levels, so the staircase is visible. The project
    # uses 256 levels; the formula and the two error mechanisms are identical.
    s = 0.4
    z = 0.0
    qmax = 6
    x = np.linspace(-0.45, 2.85, 1400)
    q = np.clip(np.rint(x / s) + z, 0, qmax)
    xhat = s * (q - z)
    lo, hi = 0.0, qmax * s

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # (a) staircase.
    ax1.axvspan(x.min(), lo, color=C["red"], alpha=0.08)
    ax1.axvspan(hi, x.max(), color=C["red"], alpha=0.08)
    ax1.plot(x, x, color=C["muted"], lw=1.2, ls="--", label="ideal  x̂ = x")
    ax1.step(x, xhat, where="mid", color=C["phase1"], lw=2.2, label="x̂ cuantizado")
    ax1.axvline(lo, color=C["red"], lw=1.0, ls=":")
    ax1.axvline(hi, color=C["red"], lw=1.0, ls=":")
    ax1.annotate("recorte\n(q = 0)", xy=(lo - 0.02, 0.05), xytext=(-0.34, 0.55), fontsize=9, color=C["red"])
    ax1.annotate("recorte\n(q = 6)", xy=(hi + 0.02, hi - 0.05), xytext=(hi + 0.13, hi - 0.55), fontsize=9, color=C["red"])
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(-0.08, 2.72)
    ax1.set_xlabel("x (valor FP32)")
    ax1.set_ylabel("x̂ (valor reconstruido)")
    ax1.set_title("(a) La escalera de cuantización", loc="left")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.6)

    # (b) error.
    err = xhat - x
    ax2.axvspan(x.min(), lo, color=C["red"], alpha=0.08)
    ax2.axvspan(hi, x.max(), color=C["red"], alpha=0.08)
    ax2.axhspan(-s / 2, s / 2, color=C["phase1"], alpha=0.10, label="±s/2")
    ax2.plot(x, err, color=C["phase2"], lw=2.0)
    ax2.axhline(0, color=C["muted"], lw=0.9)
    ax2.axvline(lo, color=C["red"], lw=1.0, ls=":")
    ax2.axvline(hi, color=C["red"], lw=1.0, ls=":")
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(-1.15, 0.72)
    ax2.set_xlabel("x (valor FP32)")
    ax2.set_ylabel("error  x̂ − x")
    ax2.set_title("(b) Redondeo frente a recorte", loc="left")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.text(
        0.02,
        0.97,
        "dentro: diente de sierra de ±s/2\nfuera: el error crece sin límite",
        transform=ax2.transAxes,
        fontsize=8.4,
        color=C["muted"],
        va="top",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
    )
    ax2.grid(alpha=0.6)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.94, bottom=0.20, wspace=0.22)
    source_footer(
        fig,
        "Fuente: LEARNING_NOTES.md, «Concepts»; fórmula de la sección 4 del brief. Ejemplo ilustrativo, sin cifras del modelo.",
    )
    save(fig, "quant_affine.png")


if __name__ == "__main__":
    main()
