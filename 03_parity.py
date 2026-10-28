"""Phase 1: numerical parity PyTorch FP32 vs ONNX Runtime FP32, both on CPU.

Boxes (channels 0-3, pixel units ~0..640) and class scores (4-83, in [0,1]) are reported
separately: one max-abs number over the concat would be dominated by the box channels.
"""
import glob, sys, json, pathlib, numpy as np, cv2, torch, onnxruntime as ort
from ultralytics import YOLO
from common import preprocess

paths = sys.argv[1:] or sorted(glob.glob(".venv/lib/python3.11/site-packages/ultralytics/assets/*.jpg"))
x = np.concatenate([preprocess(cv2.imread(p))[0] for p in paths])   # [N,3,640,640]

model = YOLO("weights/yolov8n.pt").model.fuse().eval().float()
with torch.no_grad():
    ref = model(torch.from_numpy(x))[0].numpy()                        # eval mode returns (y, feats)

so = ort.SessionOptions()   # default graph_optimization_level = ORT_ENABLE_ALL, as deployed
sess = ort.InferenceSession("models/yolov8n.onnx", so, providers=["CPUExecutionProvider"])
print("provider:", sess.get_providers()[0], "| images:", len(paths))

outs = {"batch=N": sess.run(None, {"images": x})[0],
        "batch=1 loop": np.concatenate([sess.run(None, {"images": x[i:i+1]})[0] for i in range(len(x))])}
rec = {'n_images': len(paths)}
for name, out in outs.items():
    for grp, sl in (("boxes px", slice(0, 4)), ("scores", slice(4, 84))):
        d = np.abs(out[:, sl] - ref[:, sl])
        rel = d / (np.abs(ref[:, sl]) + 1e-6)
        rec[f"{name}|{grp}|max_abs"] = float(d.max())
        print(f"{name:13s} {grp:9s} max|d|={d.max():.3e}  mean|d|={d.mean():.3e}  "
              f"max rel (|ref|>1e-2)={rel[np.abs(ref[:, sl]) > 1e-2].max():.3e}")
    # Decision-level check: does the top-scoring anchor/class agree?
    agree = (out[:, 4:].reshape(len(x), -1).argmax(1) == ref[:, 4:].reshape(len(x), -1).argmax(1)).all()
    rec[f"{name}|top1_identical"] = bool(agree)
    print(f"{name:13s} top-1 (anchor,class) identical across images: {agree}")
tag = "coco50" if sys.argv[1:] else "assets2"
pathlib.Path("results").mkdir(exist_ok=True)
json.dump(rec, open(f"results/parity_{tag}.json", "w"), indent=2)
