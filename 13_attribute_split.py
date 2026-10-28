"""Phase 3: WHY does split_pc_all lose 5.3 mAP@0.5? Separate the two candidate mechanisms, one at a time.

Reading the scales inside yolov8n_int8_split_pc_all.onnx showed two error sources the prediction ignored:
  boxes : decoded in GRID units before the stride multiply. Concat_8 (xy 0..80 mixed with wh 0..28) gets one scale,
          0.314 cells, shared by all three strides -> 2.5 px at stride 8, 10 px at stride 32. The output-concat hazard
          again, one level up the graph.
  scores: step 0.0038 after the Sigmoid -> anything below 0.0019 rounds to 0, under the eval's conf 0.001, so the
          low-confidence tail of the PR curve disappears.
Each variant keeps ONE path FP32 and quantizes everything else, so its loss belongs to the other path alone:
  _boxq  : score tail (Concat_7, Sigmoid) FP32  -> measures the box path error
  _scoreq: box decode after the DFL conv FP32   -> measures the score path error (DFL Softmax stays INT8)
Plus AP by object size (COCO small/medium/large) for every model: box-step errors should hurt small objects most.

Usage: python 13_attribute_split.py     (needs data/coco; reuses results/dets/<model>.json when present)
"""
import importlib, json, pathlib, contextlib, io, numpy as np, onnx, onnxruntime as ort
from pycocotools.cocoeval import COCOeval

q = importlib.import_module("05_quantize")
ev = importlib.import_module("09_eval_map")
SPLIT = "models/yolov8n_split.onnx"
OUT = "results/attribution_split.json"

def descendants(g, node_name):
    """Every node reachable downstream of node_name's outputs: graph traversal by tensor name, the only edge ONNX has."""
    consumers = {}
    for n in g.node:
        for i in n.input:
            consumers.setdefault(i, []).append(n)
    start = next(n for n in g.node if n.name == node_name)
    seen, stack = set(), list(start.output)
    while stack:
        for n in consumers.get(stack.pop(), []):
            if n.name not in seen:
                seen.add(n.name); stack.extend(n.output)
    return sorted(seen)

def variants():
    g = onnx.load(SPLIT).graph
    score_tail = ["/model.22/Concat_7", "/model.22/Sigmoid"]
    box_decode = descendants(g, "/model.22/dfl/conv/Conv")      # Reshape_1 ... Concat_8, Mul_2 (anchor math)
    assert not set(score_tail) & set(box_decode)
    return {"models/yolov8n_int8_split_pc_boxq.onnx": score_tail,
            "models/yolov8n_int8_split_pc_scoreq.onnx": box_decode}

def score_by_size(dets, img_ids):
    """ev.score() plus COCO's area split. precision[T,R,K,A,M]: A = all, small(<32²), medium, large(>96²) px."""
    r = ev.score(dets, img_ids)
    if not dets:
        return {**r, **{f"{k}_{a}": 0.0 for k in ("AP50", "AP") for a in ("small", "medium", "large")}}
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(ev.gt, ev.gt.loadRes(dets), "bbox")
        E.params.imgIds, E.params.catIds = list(img_ids), list(ev.VEHICLE)
        E.evaluate(); E.accumulate(); E.summarize()
    for a, name in ((1, "small"), (2, "medium"), (3, "large")):
        p = E.eval["precision"][0, :, :, a, 2]                  # IoU=0.5, all classes pooled per class then averaged
        ap = [p[:, k][p[:, k] > -1].mean() for k in range(p.shape[1]) if (p[:, k] > -1).any()]
        r[f"AP50_{name}"] = float(np.mean(ap)) if ap else float("nan")
    r.update(AP_small=float(E.stats[3]), AP_medium=float(E.stats[4]), AP_large=float(E.stats[5]))
    return r

if __name__ == "__main__":
    reader = q.calib_reader()
    for out, exclude in variants().items():
        print(f"{pathlib.Path(out).stem}: {len(exclude)} nodes kept FP32")
        q.build(out, True, exclude, reader, src=SPLIT)

    ids = ev.split["eval"]
    models = ["yolov8n_prep", "yolov8n_int8_pc_headfp32", "yolov8n_int8_split_pc_all",
              "yolov8n_int8_split_pc_boxq", "yolov8n_int8_split_pc_scoreq"]
    pathlib.Path("results/dets").mkdir(parents=True, exist_ok=True)
    rows = []
    for name in models:
        cache = pathlib.Path(f"results/dets/{name}.json")
        if cache.exists() and name in models[:3]:     # the two new variants were just rebuilt: never reuse their dets
            dets = json.load(open(cache))
        else:
            sess = ort.InferenceSession(f"models/{name}.onnx", providers=["CPUExecutionProvider"])
            dets = [d for i in ids for d in ev.detect(sess, i)]
            json.dump(dets, open(cache, "w"))
        r = {"model": name, "n_images": len(ids), "n_dets": len(dets), **score_by_size(dets, ids)}
        rows.append(r); print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))
    json.dump(rows, open(OUT, "w"), indent=2)
