"""Phase 2: where does the time go? ORT's built-in profiler, per node, same pinning/threads as 08_benchmark.

Usage: python 08b_profile.py model_a.onnx model_b.onnx ...
"""
import os, sys, json, collections, cv2, onnxruntime as ort
from common import preprocess

WARMUP, RUNS = 20, 100
os.sched_setaffinity(0, {0, 1, 2, 3, 4, 5})
x, _ = preprocess(cv2.imread("data/coco/val2017/000000000139.jpg"))
for path in sys.argv[1:]:
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = 6, 1
    so.enable_profiling = True
    so.profile_file_prefix = f"results/profile_{os.path.basename(path)[:-5]}"
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    for _ in range(WARMUP + RUNS):
        s.run(None, {"images": x})
    events = json.load(open(s.end_profiling()))
    runs = sorted(e["ts"] for e in events if e.get("name") == "model_run")
    t_start = runs[WARMUP]                                    # drop warmup runs by timestamp
    by_op, by_node = collections.Counter(), collections.Counter()
    for e in events:
        if e.get("cat") == "Node" and e["name"].endswith("_kernel_time") and e["ts"] >= t_start:
            by_op[e["args"]["op_name"]] += e["dur"]; by_node[e["name"][:-len("_kernel_time")]] += e["dur"]
    tot = sum(by_op.values())
    print(f"\n== {os.path.basename(path)}: summed kernel time/run = {tot/RUNS/1000:.2f} ms")
    for op, d in by_op.most_common(8):
        print(f"   {op:22s} {d/RUNS/1000:6.3f} ms  {100*d/tot:5.1f}%")
    print("   top nodes:", [(n, round(d / RUNS / 1000, 3)) for n, d in by_node.most_common(4)])
