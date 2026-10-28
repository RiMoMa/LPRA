"""Rebuild results.csv (long format) from the raw files every script writes, and print the summary tables.

results.csv is never edited by hand: every row traces back to a script's output file.
"""
import csv, json, glob, pathlib, pandas as pd

rows = []
def add(phase, model, metric, value, unit, source, **kw):
    rows.append(dict(phase=phase, model=model, metric=metric, value=value, unit=unit, source=source,
                     provider=kw.get("provider", "CPUExecutionProvider"), threads=kw.get("threads", ""),
                     pinning=kw.get("pinning", ""), note=kw.get("note", "")))

for f in sorted(glob.glob("results/parity_*.json")):
    tag = pathlib.Path(f).stem.split("_", 1)[1]
    for k, v in json.load(open(f)).items():
        add(1, "yolov8n.onnx", f"parity[{tag}] {k}", v, "", "03_parity.py")

lat = pd.read_csv("results/latency_raw.csv")
agg = lat.groupby("model").agg(mean_ms=("mean_ms", "median"), p95_ms=("p95_ms", "median"),
                               mean_spread=("mean_ms", lambda s: s.max() - s.min()), size_mb=("size_mb", "first"),
                               threads=("threads", "first"), pinning=("pinning", "first"), reps=("repeat", "nunique"))
fp32 = agg.loc["yolov8n_prep.onnx"]
agg["speedup"] = fp32.mean_ms / agg.mean_ms
for m, r in agg.iterrows():
    kw = dict(threads=r.threads, pinning=r.pinning, note=f"median of {r.reps} repeats x 200 runs")
    for k, u in (("mean_ms", "ms"), ("p95_ms", "ms"), ("mean_spread", "ms"), ("speedup", "x"), ("size_mb", "MB")):
        add(2, m, k, round(float(r[k]), 4), u, "08_benchmark.py", **kw)

mp = {r["model"] + ".onnx": r for r in json.load(open("results/map_overall.json"))}
for m, r in mp.items():
    for k in ("mAP50", "AP50_car", "AP50_motorcycle", "AP50_bus", "AP50_truck", "mAP50_95", "n_dets"):
        add(2, m, k, r[k], "", "09_eval_map.py", note=f"{r['n_images']} eval images")

# Single-setting checks (11_quant_checks.py): own run, own baseline rows -> compare only within this table
chk = pd.read_csv("results/latency_checks.csv").groupby("model").agg(
    mean_ms=("mean_ms", "median"), p95_ms=("p95_ms", "median"), size_mb=("size_mb", "first"))
chk_map = {r["model"] + ".onnx": r for r in json.load(open("results/map_checks.json"))}
chk["speedup"] = chk.loc["yolov8n_prep.onnx", "mean_ms"] / chk.mean_ms
for m, r in chk.iterrows():
    for k, u in (("mean_ms", "ms"), ("p95_ms", "ms"), ("speedup", "x")):
        add("2-check", m, k, round(float(r[k]), 4), u, "08_benchmark.py --out results/latency_checks.csv",
            threads=6, pinning="cpus 0-5 (CCD0)", note="separate run; compare within 2-check rows only")
    for k in ("mAP50", "mAP50_95"):
        add("2-check", m, k, chk_map[m][k], "", "09_eval_map.py --out results/map_checks.json")

# Phase 3 output split (12_split_outputs.py): own run with its own FP32 row, same rule as the checks
if pathlib.Path("results/latency_split.csv").exists() and pathlib.Path("results/map_split.json").exists():
    spl = pd.read_csv("results/latency_split.csv").groupby("model").agg(
        mean_ms=("mean_ms", "median"), p95_ms=("p95_ms", "median"))
    spl_map = {r["model"] + ".onnx": r for r in json.load(open("results/map_split.json"))}
    spl["speedup"] = spl.loc["yolov8n_prep.onnx", "mean_ms"] / spl.mean_ms
    for m, r in spl.iterrows():
        for k, u in (("mean_ms", "ms"), ("p95_ms", "ms"), ("speedup", "x")):
            add("3-split", m, k, round(float(r[k]), 4), u, "08_benchmark.py --out results/latency_split.csv",
                threads=6, pinning="cpus 0-5 (CCD0)", note="separate run; compare within 3-split rows only")
        if m in spl_map:  # a latency row may exist without an mAP row (e.g. a known 0-detection control)
            for k in ("mAP50", "mAP50_95", "n_dets"):
                add("3-split", m, k, spl_map[m][k], "", "09_eval_map.py --out results/map_split.json")

# Phase 3 attribution (13_attribute_split.py): accuracy only, AP by object size
if pathlib.Path("results/attribution_split.json").exists():
    for r in json.load(open("results/attribution_split.json")):
        for k in ("mAP50", "mAP50_95", "n_dets", "AP50_small", "AP50_medium", "AP50_large",
                  "AP_small", "AP_medium", "AP_large"):
            add("3-attr", r["model"] + ".onnx", k, r[k], "", "13_attribute_split.py",
                note=f"{r['n_images']} eval images")

with open("results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

t = agg.join(pd.DataFrame({m: {"mAP50": r["mAP50"], "mAP50_95": r["mAP50_95"]} for m, r in mp.items()}).T)
t["dmAP50_pts"] = 100 * (t.mAP50 - t.loc["yolov8n_prep.onnx", "mAP50"])
print(t[["size_mb", "mean_ms", "p95_ms", "mean_spread", "speedup", "mAP50", "dmAP50_pts", "mAP50_95"]]
      .round(3).to_markdown())
