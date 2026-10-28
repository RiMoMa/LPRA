"""Qualitative comparison: FP32, pc_headfp32 and split_pc_all on two images.

Boxes come from results/dets/*.json (COCO xywh, already in original-image
coordinates). Vehicle classes only: car=3, motorcycle=4, bus=6, truck=8.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from _common import C, RESULTS, ROOT, label, save, source_footer

MODELS = [
    ("yolov8n_prep", "FP32"),
    ("yolov8n_int8_pc_headfp32", "INT8 pc / head FP32"),
    ("yolov8n_int8_split_pc_all", "split / todo INT8"),
]
VEHICLES = {3: ("coche", "#0072B2"), 4: ("motocicleta", "#E69F00"), 6: ("autobús", "#009E73"), 8: ("camión", "#D55E00")}
THRESHOLD = 0.25

# Chosen from the evaluation split: one close-up with large vehicles, one
# street scene with many small vehicles. The ids are COCO val2017 image ids.
CASES = [
    (523782, "Vehículos grandes · imagen 523782", "qualitative_large.png", 1.9, None),
    (307074, "Vehículos pequeños · imagen 307074", "qualitative_small.png", 1.25, (85, 235, 380, 350)),
]


def load_dets() -> dict[str, dict[int, list[dict]]]:
    out = {}
    for model, _ in MODELS:
        by = defaultdict(list)
        with open(RESULTS / "dets" / f"{model}.json", encoding="utf-8") as f:
            for d in json.load(f):
                if d["category_id"] in VEHICLES and d["score"] >= THRESHOLD:
                    by[d["image_id"]].append(d)
        out[model] = by
    return out


def main() -> None:
    split = json.loads((ROOT / "data" / "split.json").read_text(encoding="utf-8"))
    dets = load_dets()

    for image_id, title, filename, lw, crop in CASES:
        image_path = ROOT / "data" / "coco" / "val2017" / split["files"][str(image_id)]
        image = plt.imread(image_path)
        h, w = image.shape[:2]

        fig_h = 7.0 if crop is None else 5.0
        fig, axes = plt.subplots(1, 3, figsize=(18.0, fig_h), gridspec_kw={"wspace": 0.025})
        for ax, (model, nice) in zip(axes, MODELS):
            ax.imshow(image)
            if crop is None:
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
            else:
                x0, y0, x1, y1 = crop
                ax.set_xlim(x0, x1)
                ax.set_ylim(y1, y0)
            ax.axis("off")
            boxes = dets[model].get(image_id, [])
            for d in sorted(boxes, key=lambda x: x["score"], reverse=True):
                x, y, bw, bh = d["bbox"]
                color = VEHICLES[d["category_id"]][1]
                ax.add_patch(Rectangle((x, y), bw, bh, fill=False, edgecolor=color, linewidth=lw, alpha=0.95))
            ax.set_title(f"{nice} · {len(boxes)} cajas", fontsize=11.2, pad=6)

        handles = [Patch(facecolor=color, edgecolor="white", label=name) for name, color in VEHICLES.values()]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10.2, bbox_to_anchor=(0.5, 0.065))
        fig.suptitle(title, fontsize=19, fontweight="bold", x=0.02, y=0.985, ha="left")
        fig.text(
            0.02,
            0.905,
            f"Detecciones con score ≥ {THRESHOLD:.2f} · clases COCO car, motorcycle, bus, truck · "
            f"misma imagen, tres modelos{' · recorte de la zona de vehículos' if crop else ''}",
            fontsize=10.2,
            color=C["muted"],
        )
        fig.subplots_adjust(left=0.015, right=0.985, top=0.845, bottom=0.19)
        source_footer(
            fig,
            "Fuente: results/dets/*.json y data/split.json; imagen original en data/coco/val2017 (no incluida en git). "
            "Cajas dibujadas por presentation/figs/make_qualitative.py.",
        )
        save(fig, filename, dpi=170)

    # Combined two-row figure for slide 26.
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.0), gridspec_kw={"hspace": 0.18, "wspace": 0.02})
    row_names = ["Grandes", "Pequeños"]
    for row, (image_id, title, filename, lw, crop) in enumerate(CASES):
        image_path = ROOT / "data" / "coco" / "val2017" / split["files"][str(image_id)]
        image = plt.imread(image_path)
        for col, (model, nice) in enumerate(MODELS):
            ax = axes[row, col]
            ax.imshow(image)
            if crop is None:
                ax.set_xlim(0, image.shape[1])
                ax.set_ylim(image.shape[0], 0)
            else:
                x0, y0, x1, y1 = crop
                ax.set_xlim(x0, x1)
                ax.set_ylim(y1, y0)
            ax.axis("off")
            boxes = dets[model].get(image_id, [])
            for d in sorted(boxes, key=lambda x: x["score"], reverse=True):
                x, y, bw, bh = d["bbox"]
                ax.add_patch(
                    Rectangle((x, y), bw, bh, fill=False, edgecolor=VEHICLES[d["category_id"]][1], linewidth=lw, alpha=0.95)
                )
            ax.set_title(f"{row_names[row]} · {nice} · {len(boxes)} cajas", fontsize=10.2, pad=4)

    handles = [Patch(facecolor=color, edgecolor="white", label=name) for name, color in VEHICLES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9.8, bbox_to_anchor=(0.5, 0.055))
    fig.suptitle("Cualitativo: vehículos grandes (arriba) y pequeños (abajo)", fontsize=18, fontweight="bold", x=0.02, y=0.985, ha="left")
    fig.text(
        0.02,
        0.945,
        "Score ≥ 0,25 · clases COCO car, motorcycle, bus, truck · tres modelos sobre la misma imagen · la fila inferior es un recorte",
        fontsize=9.2,
        color=C["muted"],
    )
    fig.subplots_adjust(left=0.015, right=0.985, top=0.91, bottom=0.14)
    source_footer(
        fig,
        "Fuente: results/dets/*.json y data/split.json; imágenes originales en data/coco/val2017 (no incluidas en git).",
    )
    save(fig, "qualitative_both.png", dpi=170)


if __name__ == "__main__":
    main()
