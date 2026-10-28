"""Phase 2: mAP@0.5 on the 646-image eval split, 4 vehicle classes, COCO protocol (pycocotools).

Saves every detection per model to results/dets/<model>.json so Phase 3 can re-score any image subset
without re-running inference. Post-processing = ultralytics val defaults (conf 0.001, IoU 0.7, multi-label).
"""
import sys, json, pathlib, contextlib, io, cv2, numpy as np, torch, onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics.utils.nms import non_max_suppression
from ultralytics.data.converter import coco80_to_coco91_class
from common import preprocess

VEHICLE = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
TO_COCO = coco80_to_coco91_class()                       # model class index 0..79 -> COCO category id
split = json.load(open("data/split.json"))
gt = COCO("data/coco/annotations/instances_val2017.json")

def detect(sess, img_id):
    bgr = cv2.imread(f"data/coco/val2017/{split['files'][str(img_id)]}")
    x, (g, left, top) = preprocess(bgr)
    y = torch.from_numpy(sess.run(None, {"images": x})[0])
    d = non_max_suppression(y, conf_thres=0.001, iou_thres=0.7, multi_label=True, max_det=300)[0].numpy()
    h, w = bgr.shape[:2]
    d[:, [0, 2]] = ((d[:, [0, 2]] - left) / g).clip(0, w)  # undo letterbox -> original pixel coords
    d[:, [1, 3]] = ((d[:, [1, 3]] - top) / g).clip(0, h)
    return [{"image_id": img_id, "category_id": TO_COCO[int(c)], "score": float(s),
             "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]} for x1, y1, x2, y2, s, c in d]

def score(dets, img_ids):
    """AP@0.5 per vehicle class + their mean, on the given images only."""
    if not dets:
        return {"mAP50": 0.0, **{f"AP50_{n}": 0.0 for n in VEHICLE.values()}, "mAP50_95": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(dets), "bbox")
        E.params.imgIds, E.params.catIds = list(img_ids), list(VEHICLE)
        E.evaluate(); E.accumulate(); E.summarize()
    p = E.eval["precision"][0, :, :, 0, 2]                # IoU=0.5, all recall pts, classes, area=all, maxDet=100
    ap = [float(np.mean(p[:, k][p[:, k] > -1])) if (p[:, k] > -1).any() else float("nan") for k in range(p.shape[1])]
    return {"mAP50": float(np.nanmean(ap)), **{f"AP50_{n}": a for n, a in zip(VEHICLE.values(), ap)},
            "mAP50_95": float(E.stats[0])}

if __name__ == "__main__":
    args = sys.argv[1:]
    OUT = args.pop(args.index("--out") + 1) if "--out" in args else "results/map_overall.json"
    args = [a for a in args if a != "--out"]
    models = args or ["models/yolov8n_prep.onnx"] + [f"models/yolov8n_int8_{v}.onnx" for v in
                                                             ("pt_all", "pt_headfp32", "pc_all", "pc_headfp32")]
    pathlib.Path("results/dets").mkdir(parents=True, exist_ok=True)
    rows = []
    for m in models:
        sess = ort.InferenceSession(m, providers=["CPUExecutionProvider"])
        dets = [d for i in split["eval"] for d in detect(sess, i)]
        name = pathlib.Path(m).stem
        json.dump(dets, open(f"results/dets/{name}.json", "w"))
        r = {"model": name, "provider": sess.get_providers()[0], "n_images": len(split["eval"]), "n_dets": len(dets),
             **score(dets, split["eval"])}
        rows.append(r); print(json.dumps(r))
    json.dump(rows, open(OUT, "w"), indent=2)
