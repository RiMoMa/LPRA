"""Phase 1: export yolov8n to ONNX three ways.

  models/yolov8n_static.onnx   legacy tracer, batch fixed to 1   (reference for the diff)
  models/yolov8n.onnx          legacy tracer, dynamic batch      (MAIN artifact, used downstream)
  models/yolov8n_dynamo.onnx   dynamo exporter, dynamic batch    (graph comparison only)
"""
import pathlib, torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

IMG = 640
OPSET = 17  # >=13 needed for per-channel QDQ (axis attr); 17 = mature in ORT, supported by legacy tracer
out = pathlib.Path("models"); out.mkdir(exist_ok=True)

model = YOLO("weights/yolov8n.pt").model
model = model.fuse()          # fold BatchNorm into Conv weights: the graph should hold the INFERENCE math
model.eval().float()          # eval: BN/dropout frozen; float: weights are stored as fp16 in the .pt
for p in model.parameters():
    p.requires_grad_(False)
for m in model.modules():
    if isinstance(m, Detect):
        m.export = True       # return only the decoded [B, 84, 8400] tensor, not (y, raw_feats)
        m.format = "onnx"

dummy = torch.zeros(1, 3, IMG, IMG)
model(dummy)                  # run once so Detect caches anchors for 640x640 -> they become constants
common = dict(input_names=["images"], output_names=["output0"], opset_version=OPSET,
              do_constant_folding=True)

torch.onnx.export(model, (dummy,), out / "yolov8n_static.onnx", dynamo=False, **common)
torch.onnx.export(model, (dummy,), out / "yolov8n.onnx", dynamo=False,
                  dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}}, **common)
try:
    # batch=2 dummy: torch.export specializes any dim traced at size 0 or 1 into a constant,
    # silently ignoring Dim("batch"). Tracing at 1 gave a static [1,3,640,640] graph (mistake #1).
    torch.onnx.export(model, (torch.zeros(2, 3, IMG, IMG),), out / "yolov8n_dynamo.onnx", dynamo=True,
                      dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=64)}},
                      **{**common, "opset_version": 18}, external_data=False)
    print("dynamo export OK")
except Exception as e:  # dynamo is the comparison, not the deliverable: report, don't block
    print("dynamo export FAILED:", type(e).__name__, str(e)[:500])

for f in sorted(out.glob("*.onnx")):
    print(f"{f.name:28s} {f.stat().st_size/1e6:6.2f} MB")
