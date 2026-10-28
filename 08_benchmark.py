"""Phase 2: latency + size, CPU EP, pinned to CCD0 physical cores, identical settings for every model.

Usage: python 08_benchmark.py [--out results/x.csv] [model.onnx ...]   (default: FP32 baseline + the 4 INT8 variants)
"""
import os, sys, csv, json, time, pathlib, cv2, numpy as np, onnxruntime as ort
from common import preprocess

CORES = {0, 1, 2, 3, 4, 5}         # CCD0 (V-Cache) physical cores; SMT siblings 12-17 left idle
THREADS, WARMUP, RUNS, REPEATS = 6, 20, 200, 3
os.sched_setaffinity(0, CORES)      # = taskset -c 0-5; ORT's thread pool inherits the process mask

args = sys.argv[1:]
OUT = args.pop(args.index("--out") + 1) if "--out" in args else "results/latency_raw.csv"
args = [a for a in args if a != "--out"]
models = args or ["models/yolov8n_prep.onnx"] + [f"models/yolov8n_int8_{v}.onnx" for v in
                                                         ("pt_all", "pt_headfp32", "pc_all", "pc_headfp32")]
x, _ = preprocess(cv2.imread("data/coco/val2017/000000000139.jpg"))   # a real image, same for every model

def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = THREADS          # explicit: default = #physical cores system-wide (12) > pinned 6
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])

sessions = {m: session(m) for m in models}
rows = []
for rep in range(REPEATS):              # interleave models across repeats: drift/thermal hits all of them alike
    for m, s in sessions.items():
        for _ in range(WARMUP):
            s.run(None, {"images": x})
        t = np.empty(RUNS)
        for i in range(RUNS):
            t0 = time.perf_counter_ns(); s.run(None, {"images": x}); t[i] = (time.perf_counter_ns() - t0) / 1e6
        rows.append(dict(model=pathlib.Path(m).name, repeat=rep, provider=s.get_providers()[0], threads=THREADS,
                         pinning="cpus 0-5 (CCD0)", mean_ms=t.mean(), p50_ms=np.median(t), p95_ms=np.percentile(t, 95),
                         std_ms=t.std(), size_mb=os.path.getsize(m) / 1e6))
        print(f"rep{rep} {rows[-1]['model']:32s} mean={t.mean():6.2f} p95={rows[-1]['p95_ms']:6.2f} ms")

print("affinity:", sorted(os.sched_getaffinity(0)), "| governor:",
      pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip())
out = pathlib.Path(OUT)
with out.open("a" if out.exists() else "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    if f.tell() == 0: w.writeheader()
    w.writerows(rows)
