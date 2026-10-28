"""Before/after graph surgery around the final Concat_9.

The scales and shapes are read from the ONNX files, not copied from a table.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import onnx
from onnx import numpy_helper

from _common import C, ROOT, arrow, rounded_box, save, source_footer


def output_info(path: Path):
    model = onnx.load(str(path))
    graph = model.graph
    prod = {out: node for node in graph.node for out in node.output}
    inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    result = {}
    for out in graph.output:
        node = prod.get(out.name)
        cur = node
        guard = 0
        while cur is not None and cur.op_type != "QuantizeLinear" and guard < 40:
            cur = prod.get(cur.input[0])
            guard += 1
        scale = zp = None
        if cur is not None and cur.op_type == "QuantizeLinear":
            if len(cur.input) > 1 and cur.input[1] in inits:
                scale = float(inits[cur.input[1]])
            if len(cur.input) > 2 and cur.input[2] in inits:
                zp = int(inits[cur.input[2]])
        shape = [d.dim_value or d.dim_param for d in out.type.tensor_type.shape.dim]
        result[out.name] = {"shape": shape, "scale": scale, "zp": zp}
    return result


def main() -> None:
    before = output_info(ROOT / "models" / "yolov8n_int8_pc_all.onnx")["output0"]
    after = output_info(ROOT / "models" / "yolov8n_int8_split_pc_all.onnx")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 6.6))
    for ax in (ax1, ax2):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    # ---------- before ----------
    ax1.text(0.0, 9.45, "Antes: un solo tensor", fontsize=16, fontweight="bold", color=C["ink"], ha="left")
    ax1.text(0.0, 9.05, "output0 mezcla cajas en píxeles y scores en [0,1]", fontsize=10.6, color=C["muted"], ha="left")

    rounded_box(ax1, 0.2, 5.7, 3.0, 1.05, C["light_blue"], C["phase1"], lw=1.1)
    ax1.text(1.7, 6.22, "Mul_2\ncajas px  [B,4,8400]", ha="center", va="center", fontsize=11.0, linespacing=1.15)
    rounded_box(ax1, 0.2, 3.9, 3.0, 1.05, C["light_orange"], C["phase2"], lw=1.1)
    ax1.text(1.7, 4.42, "Sigmoid\nscores  [B,80,8400]", ha="center", va="center", fontsize=11.0, linespacing=1.15)

    rounded_box(ax1, 3.9, 4.45, 2.1, 1.4, "#F4F4F4", C["muted"], lw=1.2)
    ax1.text(4.95, 5.15, "Concat_9", ha="center", va="center", fontsize=12.0, fontweight="bold", family="DejaVu Sans Mono")
    ax1.text(4.95, 4.78, "unión", ha="center", va="center", fontsize=9.6, color=C["muted"])
    arrow(ax1, (3.25, 6.20), (3.85, 5.45), color=C["muted"], lw=1.3)
    arrow(ax1, (3.25, 4.42), (3.85, 4.85), color=C["muted"], lw=1.3)

    rounded_box(ax1, 6.65, 4.45, 2.8, 1.4, "#FFFFFF", C["red"], lw=1.3)
    ax1.text(8.05, 5.15, "output0", ha="center", va="center", fontsize=12.0, fontweight="bold", family="DejaVu Sans Mono")
    shape = " × ".join(str(v) for v in before["shape"])
    ax1.text(8.05, 4.78, f"[{shape}]", ha="center", va="center", fontsize=10.6, color=C["muted"])
    arrow(ax1, (6.02, 5.15), (6.62, 5.15), color=C["muted"], lw=1.4)

    ax1.text(
        4.95,
        3.25,
        f"Q/DQ: escala {before['scale']:.3f} · zero point {before['zp']}\n"
        "un solo paso para dos magnitudes",
        ha="center",
        va="center",
        fontsize=10.2,
        color=C["red"],
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF5F2", ec=C["red"], lw=0.8),
    )
    ax1.text(
        4.95,
        2.05,
        "Consecuencia: scores < 0,0019 → 0; 0 detecciones, mAP 0.",
        ha="center",
        va="center",
        fontsize=10.6,
        color=C["red"],
        fontweight="bold",
    )

    # ---------- after ----------
    ax2.text(0.0, 9.45, "Después: dos tensores", fontsize=16, fontweight="bold", color=C["ink"], ha="left")
    ax2.text(0.0, 9.05, "se elimina Concat_9; cada salida tiene su propia escala", fontsize=10.6, color=C["muted"], ha="left")

    rounded_box(ax2, 0.2, 6.0, 3.0, 1.05, C["light_blue"], C["phase1"], lw=1.1)
    ax2.text(1.7, 6.52, "Mul_2\ncajas px  [B,4,8400]", ha="center", va="center", fontsize=11.0, linespacing=1.15)
    rounded_box(ax2, 0.2, 3.6, 3.0, 1.05, C["light_orange"], C["phase2"], lw=1.1)
    ax2.text(1.7, 4.12, "Sigmoid\nscores  [B,80,8400]", ha="center", va="center", fontsize=11.0, linespacing=1.15)

    arrow(ax2, (3.25, 6.52), (4.05, 6.52), color=C["muted"], lw=1.4)
    arrow(ax2, (3.25, 4.12), (4.05, 4.12), color=C["muted"], lw=1.4)

    rounded_box(ax2, 4.1, 5.85, 2.5, 1.35, "#FFFFFF", C["phase3"], lw=1.3)
    ax2.text(5.35, 6.68, "boxes", ha="center", va="center", fontsize=12.0, fontweight="bold", family="DejaVu Sans Mono")
    shp_b = " × ".join(str(v) for v in after["boxes"]["shape"])
    ax2.text(5.35, 6.30, f"[{shp_b}]", ha="center", va="center", fontsize=9.3, color=C["muted"])
    ax2.text(5.35, 5.98, f"escala {after['boxes']['scale']:.3f}  ·  zp {after['boxes']['zp']}", ha="center", va="center", fontsize=9.3, color=C["phase3"])

    rounded_box(ax2, 4.1, 3.45, 2.5, 1.35, "#FFFFFF", C["phase3"], lw=1.3)
    ax2.text(5.35, 4.28, "scores", ha="center", va="center", fontsize=12.0, fontweight="bold", family="DejaVu Sans Mono")
    shp_s = " × ".join(str(v) for v in after["scores"]["shape"])
    ax2.text(5.35, 3.90, f"[{shp_s}]", ha="center", va="center", fontsize=9.3, color=C["muted"])
    ax2.text(5.35, 3.58, f"escala {after['scores']['scale']:.5f}  ·  zp {after['scores']['zp']}", ha="center", va="center", fontsize=9.3, color=C["phase3"])

    # Removed node.
    ax2.text(8.15, 6.45, "Concat_9\neliminado", ha="center", va="center", fontsize=10.6, color=C["red"], family="DejaVu Sans Mono")
    ax2.plot([7.35, 8.95], [6.56, 6.56], color=C["red"], lw=1.4)

    ax2.text(
        5.35,
        2.25,
        f"boxes: ±{after['boxes']['scale']/2:.2f} px   ·   scores: ±{after['scores']['scale']/2:.5f}",
        ha="center",
        va="center",
        fontsize=11.0,
        color=C["phase3"],
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#E4F3EC", ec=C["phase3"], lw=0.8),
    )
    ax2.text(
        5.35,
        1.25,
        "Cada magnitud conserva su paso; el problema se resuelve en la salida,\n"
        "pero no en las escalas intermedias (fase 3).",
        ha="center",
        va="center",
        fontsize=10.3,
        color=C["muted"],
    )

    fig.subplots_adjust(left=0.02, right=0.99, top=0.96, bottom=0.09, wspace=0.12)
    source_footer(
        fig,
        "Fuente: models/yolov8n_int8_pc_all.onnx y models/yolov8n_int8_split_pc_all.onnx; "
        "nombres, formas y escalas de los QuantizeLinear leídos con onnx.",
    )
    save(fig, "split_graph.svg")


if __name__ == "__main__":
    main()
