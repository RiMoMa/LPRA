"""Phase 2: what did the MinMax range choice cost, on real calibration activations?

(a) a typical backbone activation: post-SiLU output of /model.2/cv1 (first C2f block)
(b) the output0 concat: boxes and scores under ONE per-tensor scale (the hazard)
Histograms are accumulated over ALL elements of all 150 calibration images (exact, no subsampling).
"""
import json, cv2, numpy as np, onnx, onnxruntime as ort, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from onnx import numpy_helper as nh
from common import preprocess

TENSOR = "/model.2/cv1/act/Mul_output_0"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

def qparams(model_path, tensor):
    """The (scale, zp) the quantizer actually froze for this tensor = its Q node's initializers."""
    g = onnx.load(model_path).graph
    init = {i.name: nh.to_array(i) for i in g.initializer}
    prod = {o: n.name for n in g.node for o in n.output}   # match by producer too: graph outputs get renamed
    q = next(n for n in g.node if n.op_type == "QuantizeLinear" and tensor in (n.input[0], prod.get(n.input[0])))
    return float(init[q.input[1]]), int(init[q.input[2]])

s_act, z_act = qparams("models/yolov8n_int8_pt_all.onnx", TENSOR)
s_out, z_out = qparams("models/yolov8n_int8_pt_all.onnx", "/model.22/Concat_9")
lo, hi = s_act * (0 - z_act), s_act * (255 - z_act)          # MinMax range = clipping thresholds

m = onnx.load("models/yolov8n_prep.onnx")
m.graph.output.append(onnx.helper.make_tensor_value_info(TENSOR, onnx.TensorProto.FLOAT, None))
sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])

edges = np.linspace(lo, hi, 2049)                             # 8 bins per INT8 step
h_act = np.zeros(2048); out_edges = np.linspace(-12, 16, 561); h_box = np.zeros(560); h_sc = np.zeros(560)  # zoom: 0.05 bins
split = json.load(open("data/split.json"))
for i in split["calib"]:
    y, a = sess.run(None, {"images": preprocess(cv2.imread(f"data/coco/val2017/{split['files'][str(i)]}"))[0]})
    h_act += np.histogram(a, edges)[0]
    h_box += np.histogram(y[:, :4], out_edges)[0]; h_sc += np.histogram(y[:, 4:], out_edges)[0]

cdf = np.cumsum(h_act) / h_act.sum()
p_lo, p_hi = edges[np.searchsorted(cdf, 0.0005)], edges[np.searchsorted(cdf, 0.9995)]
levels_used = (p_hi - p_lo) / s_act
near_zero = h_act[np.abs(edges[:-1] - 0) < s_act / 2].sum() / h_act.sum()
stats = {"tensor": TENSOR, "scale": s_act, "zero_point": z_act, "clip_lo": lo, "clip_hi": hi,
         "p0.05": float(p_lo), "p99.95": float(p_hi), "int8_levels_spanned_by_central_99.9pct": float(levels_used),
         "frac_values_in_zero_bin": float(near_zero), "output_concat_scale": s_out, "output_concat_zp": z_out}
print(json.dumps(stats, indent=2)); json.dump(stats, open("results/activation_hist_stats.json", "w"), indent=2)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
c = (edges[:-1] + edges[1:]) / 2
ax[0].bar(c, h_act, width=edges[1] - edges[0], color=BLUE, linewidth=0)
ax[0].set_yscale("log")
for x, lab in ((lo, f"clip min {lo:.2f}"), (hi, f"clip max {hi:.2f}")):
    ax[0].axvline(x, color=INK, lw=1.5); ax[0].text(x, h_act.max(), " " + lab, color=INK, va="top", fontsize=9)
ax[0].axvspan(p_lo, p_hi, color=MUTED, alpha=0.12)
ax[0].text(p_hi, h_act.max() / 30, f" central 99.9% uses\n {levels_used:.0f} of 256 levels", color=MUTED, fontsize=9)
ax[0].set_title(f"(a) {TENSOR.split('_output')[0]} post-SiLU, MinMax: step = {s_act:.4f}", fontsize=10, color=INK)
ax[0].set_xlabel("activation value"); ax[0].set_ylabel("count (log)")

oc = (out_edges[:-1] + out_edges[1:]) / 2
w = out_edges[1] - out_edges[0]
ax[1].bar(oc, h_box, width=w, color=BLUE, linewidth=0, label="box coords px (range continues to ~650)")
ax[1].bar(oc, h_sc, width=w, color=ORANGE, linewidth=0, label="class scores [0,1]")
ax[1].set_yscale("log")
grid = s_out * (np.arange(0, 12) - z_out)                    # the only values INT8 can represent here
for gv in grid[(grid > -12) & (grid < 16)]:
    ax[1].axvline(gv, color=MUTED, lw=0.8, ls=":")
ax[1].axvline(s_out / 2, color=INK, lw=1.5, ls="--")
ax[1].text(s_out / 2, h_sc.max() / 300, f" rounding boundary {s_out/2:.2f}: every score\n is below it, so every score -> 0.000",
           color=INK, va="top", fontsize=9)
ax[1].set_title(f"(b) output0 concat, one per-tensor scale = {s_out:.3f} (zp {z_out}); dotted = INT8 grid",
                fontsize=10, color=INK)
ax[1].set_xlabel("value"); ax[1].set_xlim(-12, 16); ax[1].legend(frameon=False, loc="upper left", fontsize=8)
for a_ in ax: a_.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("results/activation_histogram.png", dpi=130)
