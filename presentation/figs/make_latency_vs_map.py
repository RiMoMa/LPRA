"""Phase 2: latency vs mAP@0.5, one benchmark run only."""

from __future__ import annotations

import matplotlib.pyplot as plt

from _common import C, color, label, read_results_csv, save, source_footer, value

MODELS = [
    "yolov8n_prep.onnx",
    "yolov8n_int8_pc_headfp32.onnx",
    "yolov8n_int8_pt_headfp32.onnx",
    "yolov8n_int8_pc_all.onnx",
    "yolov8n_int8_pt_all.onnx",
]


def make_zoom(points: list[tuple[str, float, float]]) -> None:
    """Readable detail for the slide: only models with non-zero mAP.

    The two all-INT8 points stay in the table and in the caption; putting them
    on a full 0–0.65 y-axis would compress the three useful points into a few
    pixels. The y-axis is labelled as zoomed.
    """
    fig, ax = plt.subplots(figsize=(12.0, 4.6))
    keep = [p for p in points if not p[0].endswith("_all.onnx")]
    fp32_ms, fp32_ap = keep[0][1], keep[0][2]
    ax.axhline(fp32_ap, color=C["muted"], lw=1.0, ls="--", alpha=0.7)
    ax.axvline(fp32_ms, color=C["muted"], lw=1.0, ls="--", alpha=0.7)

    offsets = {
        "yolov8n_prep.onnx": (14, 12),
        "yolov8n_int8_pc_headfp32.onnx": (14, -34),
        "yolov8n_int8_pt_headfp32.onnx": (14, 10),
    }
    for m, ms, ap in keep:
        ax.scatter(ms, ap, s=190, color=color(m), edgecolor="white", linewidth=1.6, zorder=4)
        ax.annotate(
            f"{label(m)}\n{ms:.2f} ms · mAP {ap:.4f}",
            xy=(ms, ap),
            xytext=offsets.get(m, (12, 12)),
            textcoords="offset points",
            fontsize=11.0,
            color=C["ink"],
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color(m), lw=0.9, alpha=0.95),
        )

    ax.set_xlim(11.25, 17.35)
    ax.set_ylim(0.589, 0.6125)
    ax.set_xlabel("latencia media (ms), 6 hilos fijados a CPUs 0–5")
    ax.set_ylabel("mAP@0.5 (eje ampliado)")
    ax.set_title("Fase 2 · modelos con detecciones — misma corrida (2026-10-23)", loc="left", pad=10)
    ax.text(
        0.985,
        0.06,
        "Los dos modelos all-INT8 (≈9,0 ms) tienen 0 detecciones y mAP 0;\nno caben en este eje. Están en la tabla.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.6,
        color=C["red"],
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF5F2", ec=C["red"], lw=0.9, alpha=0.95),
    )
    ax.grid(alpha=0.7)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.23)
    source_footer(
        fig,
        "Fuente: results.csv, fase 2 (2026-10-23). Eje y ampliado; los modelos con mAP 0 se muestran en la tabla. "
        "No mezclar con chequeos ni fase 3.",
    )
    save(fig, "latency_vs_map_phase2_zoom.png")


def main() -> None:
    df = read_results_csv()
    points = []
    for m in MODELS:
        ms = value(df, "2", m, "mean_ms")
        ap = value(df, "2", m, "mAP50")
        points.append((m, ms, ap))

    make_zoom(points)

    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    fp32_ms, fp32_ap = points[0][1], points[0][2]
    ax.axhline(fp32_ap, color=C["muted"], lw=1.0, ls="--", alpha=0.7)
    ax.axvline(fp32_ms, color=C["muted"], lw=1.0, ls="--", alpha=0.7)

    for m, ms, ap in points:
        if m.endswith("_all.onnx"):
            continue
        offsets = {
            "yolov8n_prep.onnx": (12, 10),
            "yolov8n_int8_pc_headfp32.onnx": (12, 18),
            "yolov8n_int8_pt_headfp32.onnx": (12, -30),
        }
        ax.scatter(ms, ap, s=150, color=color(m), edgecolor="white", linewidth=1.4, zorder=4)
        ax.annotate(
            f"{label(m)}\n{ms:.2f} ms · mAP {ap:.4f}",
            xy=(ms, ap),
            xytext=offsets.get(m, (8, 12)),
            textcoords="offset points",
            fontsize=9.3,
            color=C["ink"],
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color(m), lw=0.8, alpha=0.9),
        )

    # Both *_all variants are at mAP = 0; show one shared callout.
    all_x = [ms for m, ms, ap in points if m.endswith("_all.onnx")]
    ax.scatter(all_x, [0.0, 0.0], s=130, color=[C["pc_all"], C["pt_all"]], edgecolor="white", linewidth=1.4, zorder=4)
    ax.annotate(
        "INT8 pc / todo e INT8 pt / todo\n≈9,0 ms · 0 detecciones · mAP 0",
        xy=(sum(all_x) / len(all_x), 0.0),
        xytext=(0, 34),
        textcoords="offset points",
        ha="center",
        fontsize=9.2,
        color=C["red"],
        arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.0),
        bbox=dict(boxstyle="round,pad=0.28", fc="#FFF5F2", ec=C["red"], lw=0.8),
    )

    ax.set_xlim(8.2, 17.6)
    ax.set_ylim(-0.045, 0.72)
    ax.set_xlabel("latencia media (ms), 6 hilos fijados a CPUs 0–5")
    ax.set_ylabel("mAP@0.5 (646 imágenes de evaluación)")
    ax.set_title("Fase 2 · latencia vs mAP@0.5 — misma corrida (2026-10-23)", loc="left", pad=12)
    ax.text(
        0.99,
        0.02,
        "más rápido  ←",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=C["muted"],
    )
    ax.text(
        0.99,
        0.06,
        "mejor mAP  ↑",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=C["muted"],
    )
    ax.grid(alpha=0.7)
    source_footer(
        fig,
        "Fuente: results.csv, fase 2; latencia de results/latency_raw.csv y mAP de results/map_overall.json. "
        "No mezclar con las corridas de chequeos ni de fase 3.",
    )
    save(fig, "latency_vs_map_phase2.png")


if __name__ == "__main__":
    main()
