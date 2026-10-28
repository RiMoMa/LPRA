"""Phase 3: fix the output-concat hazard in the GRAPH instead of in the quantizer's exclusion list.

output0 = Concat_9(boxes [B,4,8400] in pixels 0..640, scores [B,80,8400] in [0,1]). One tensor -> one per-tensor
scale -> ~2.5 per step -> every score rounds to 0 (the *_all models). Phase 2 dodged it by keeping the head decode
FP32. Here we delete Concat_9 and expose its two inputs as two graph outputs, so each gets its own scale, then
quantize the WHOLE network with no exclusions.

Predictions, written before running (see LEARNING_NOTES.md, Phase 3):
  - non-zero detections: scores get a ~1/255 step of their own;
  - boxes pay a ~640/255 = 2.5 px step (+-1.25 px rounding), so mAP@.5:.95 should drop more than mAP@0.5;
  - latency near *_all (~9 ms), since the DFL Softmax runs as QLinearSoftmax again.

Usage: python 12_split_outputs.py            (surgery + FP32 parity + quantize + inspect; needs data/coco for calibration)
"""
import importlib, sys, collections, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh

q = importlib.import_module("05_quantize")       # same baseline settings, not a copy of them
SRC, SPLIT = q.PREP, "models/yolov8n_split.onnx"
INT8 = "models/yolov8n_int8_split_pc_all.onnx"
NAMES = ("boxes", "scores")                      # new output names, in Concat_9's input order

def split_outputs(src=SRC, dst=SPLIT):
    """Graph surgery with the plain onnx API: a graph output is nothing but a tensor NAME plus its type/shape."""
    m = onnx.load(src)
    g = m.graph
    cat = next(n for n in g.node if g.output[0].name in n.output)        # find by producer, not by node name
    assert cat.op_type == "Concat" and len(cat.input) == 2, cat
    old = list(cat.input)
    for t in old:   # nothing else may read these tensors, or renaming them below would cut a live edge
        assert [n.name for n in g.node if t in n.input] == [cat.name], t
    g.node.remove(cat)

    # Rename the two tensors so the public interface reads boxes/scores, not '/model.22/Mul_2_output_0'
    for n in g.node:
        for i, o in enumerate(n.output):
            if o in old:
                n.output[i] = NAMES[old.index(o)]
    # Their type/shape lives in value_info (written by quant_pre_process's symbolic inference): move it to outputs
    vi = {v.name: v for v in g.value_info}
    del g.output[:]
    for o, new in zip(old, NAMES):
        v = onnx.ValueInfoProto(); v.CopyFrom(vi[o]); v.name = new
        g.output.append(v)
        g.value_info.remove(vi[o])

    onnx.checker.check_model(m, full_check=True)   # full_check also runs strict shape inference
    onnx.save(m, dst)
    print(f"wrote {dst}: removed {cat.name}, outputs " +
          ", ".join(f"{o.name}{[d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim]}" for o in g.output))

def parity(a=SRC, b=SPLIT):
    """concat(boxes, scores) must reproduce output0: same nodes, same kernels, one Concat fewer."""
    x = np.random.default_rng(0).random((2, 3, 640, 640), dtype=np.float32)
    ref = ort.InferenceSession(a, providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
    out = np.concatenate(ort.InferenceSession(b, providers=["CPUExecutionProvider"]).run(None, {"images": x}), 1)
    print(f"split vs prep: shape {out.shape} == {ref.shape}, max|d| = {np.abs(out - ref).max():.3e}")

def inspect(path=INT8):
    """Which scale did each output get, and what does ORT actually run?"""
    g = onnx.load(path).graph
    prod = {o: n for n in g.node for o in n.output}
    inits = {i.name: nh.to_array(i) for i in g.initializer}
    for out in g.output:
        n = prod[out.name]                     # quantizer renames outputs; walk back DQ -> Q to the scale
        while n is not None and n.op_type != "QuantizeLinear":
            n = prod.get(n.input[0])
        if n is None:
            print(f"  {out.name}: no Q/DQ before it (stays FP32)"); continue
        s, z = float(inits[n.input[1]]), int(inits[n.input[2]])
        print(f"  {out.name}: scale={s:.5f} zp={z} -> range [{s*(0-z):.2f}, {s*(255-z):.2f}], "
              f"max rounding error {s/2:.4f}")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.optimized_model_filepath = "models/_runtime_split_pc_all.onnx"
    ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    rt = collections.Counter(n.op_type for n in onnx.load(so.optimized_model_filepath).graph.node)
    print("  runtime graph:", dict(rt.most_common(12)))

if __name__ == "__main__":
    split_outputs()
    parity()
    q.build(INT8, True, [], q.calib_reader(), src=SPLIT)    # per-channel, NO exclusions: the point of the surgery
    inspect()
