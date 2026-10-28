"""Phase 2: where did Q/DQ land, what scale did the output concat get, and what does ORT actually run?"""
import collections, numpy as np, onnx, onnxruntime as ort
from onnx import numpy_helper as nh

V = ["pt_all", "pt_headfp32", "pc_all", "pc_headfp32"]

def init_map(g):
    return {i.name: nh.to_array(i) for i in g.initializer}

for v in V:
    path = f"models/yolov8n_int8_{v}.onnx"
    g = onnx.load(path).graph
    prod = {o: n for n in g.node for o in n.output}
    inits = init_map(g)
    ops = collections.Counter(n.op_type for n in g.node)
    # An op "runs quantized" if at least one activation input comes straight from a DequantizeLinear
    q_in = collections.Counter(n.op_type for n in g.node if n.op_type not in ("QuantizeLinear", "DequantizeLinear")
                               and any(prod.get(i) is not None and prod[i].op_type == "DequantizeLinear" for i in n.input))
    print(f"\n=== {v}: Q={ops['QuantizeLinear']} DQ={ops['DequantizeLinear']} | ops fed by DQ: {dict(q_in)}")

    # Walk back from output0: is there a Q->DQ right before the graph output?
    n = prod[g.output[0].name]
    chain = []
    while n is not None and len(chain) < 4:
        chain.append(n.op_type)
        n = prod.get(n.input[0])
    print("  output0 <- " + " <- ".join(chain))
    last_q = [m for m in g.node if m.op_type == "QuantizeLinear" and prod.get(m.input[0]) is not None
              and prod[m.input[0]].name == "/model.22/Concat_9"]
    if last_q:
        s, z = inits[last_q[0].input[1]], inits[last_q[0].input[2]]
        print(f"  Concat_9 quantized: scale={float(s):.4f} zp={int(z)} -> representable [{float(s)*(0-int(z)):.1f}, "
              f"{float(s)*(255-int(z)):.1f}]; a score of 0.9 becomes {float(s)*(np.clip(np.round(0.9/float(s))+int(z),0,255)-int(z)):.3f}")
    else:
        print("  Concat_9 NOT quantized (stays FP32)")

    # What ORT actually executes after load-time fusion (EXTENDED: serializable, before NCHWc layout passes)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    so.optimized_model_filepath = f"models/_runtime_{v}.onnx"
    ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    rt = collections.Counter(n.op_type for n in onnx.load(so.optimized_model_filepath).graph.node)
    print("  runtime graph:", dict(rt.most_common(12)))

# Why per-channel: spread of per-output-channel weight ranges in one BN-folded conv
g = onnx.load("models/yolov8n_prep.onnx").graph
W = init_map(g)
for conv in [n for n in g.node if n.op_type == "Conv"][:6]:
    w = W[conv.input[1]]
    r = np.abs(w.reshape(w.shape[0], -1)).max(1)
    print(f"{conv.name:38s} out_ch={w.shape[0]:3d} per-channel max|w|: min={r.min():.4f} max={r.max():.4f} "
          f"ratio={r.max()/max(r.min(),1e-12):8.1f}  -> per-tensor leaves weakest channel {int(127*r.min()/r.max())} levels")
