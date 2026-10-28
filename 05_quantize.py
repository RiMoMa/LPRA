"""Phase 2: INT8 PTQ, 2x2 design = {per-tensor, per-channel weights} x {all ops, head decode kept FP32}."""
import json, numpy as np, onnx, onnxruntime as ort
from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationMethod
from onnxruntime.quantization.shape_inference import quant_pre_process
from calib_reader import CocoCalibReader

SRC, PREP = "models/yolov8n.onnx", "models/yolov8n_prep.onnx"

def head_nodes(prep=PREP):
    # Head decode = everything in /model.22/ after the last cv2 (box) / cv3 (class) convs: DFL, anchor math,
    # the final Concat. Its tensors mix pixel and probability scales (the hazard), and it is ~0% of the FLOPs.
    g = onnx.load(prep).graph
    return [n.name for n in g.node if n.name.startswith("/model.22/") and "/cv2." not in n.name and "/cv3." not in n.name]

def calib_reader():
    split = json.load(open("data/split.json"))
    return CocoCalibReader([f"data/coco/val2017/{split['files'][str(i)]}" for i in split["calib"]])

def build(out, per_channel, exclude, reader, **overrides):
    """One INT8 variant. overrides lets 11_quant_checks change exactly one setting against this baseline."""
    kw = dict(
        quant_format=QuantFormat.QDQ,          # annotate FP32 graph; ORT fuses DQ->Op->Q into int kernels at load
        activation_type=QuantType.QUInt8,      # u8 x s8 = native VPDPBUSD operands; asymmetric z fits post-SiLU ranges
        weight_type=QuantType.QInt8,           # symmetric weights: z=0 so no cross term in the int32 accumulator
        per_channel=per_channel,               # weights only: a per-output-channel scale factors out of sum_k w[o,k]x[k]
        reduce_range=False,                    # VNNI accumulates into int32; the 7-bit workaround is for AVX2's int16 sat
        calibrate_method=CalibrationMethod.MinMax,
        nodes_to_exclude=exclude,
    )
    kw.update(overrides)
    reader.rewind()
    quantize_static(PREP, out, reader, **kw)
    print("wrote", out)

if __name__ == "__main__":
    # Symbolic shape inference: the quantizer needs every tensor's shape/type to place Q/DQ, and the legacy
    # exporter left output dims unresolved. auto_merge: the legacy graph re-derives batch via Shape->Gather,
    # which symbolic inference names as a NEW symbol; it can't prove it equals 'batch' and aborts. Merging is
    # safe here (it is the same batch) but in general it disables a consistency check (mistake/gotcha #3).
    quant_pre_process(SRC, PREP, auto_merge=True)

    # Parity of the pre-processed FP32 graph vs the export: it becomes the FP32 baseline, so prove it's identical.
    x = np.random.default_rng(0).random((2, 3, 640, 640), dtype=np.float32)
    a, b = (ort.InferenceSession(p, providers=["CPUExecutionProvider"]).run(None, {"images": x})[0] for p in (SRC, PREP))
    print(f"prep vs export max|d| = {np.abs(a - b).max():.3e}")

    head = head_nodes()
    print(f"head-decode nodes excluded in *_headfp32: {len(head)}")
    reader = calib_reader()
    for per_channel in (False, True):
        for tag, exclude in (("all", []), ("headfp32", head)):
            build(f"models/yolov8n_int8_{'pc' if per_channel else 'pt'}_{tag}.onnx", per_channel, exclude, reader)
