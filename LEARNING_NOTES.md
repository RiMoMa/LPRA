# Learning notes — ONNX export & INT8 PTQ for an ALPR-style detector

Legend: **[ONNX]** = genuinely new, ONNX/ORT-specific knowledge. **[ML]** = general concept, new syntax only.

## Environment facts that shaped the design
- Target: **CPU**, Ryzen 9 7900X3D (Zen 4). `avx512_vnni` read from /proc/cpuinfo (`00_env_check.py`).
  VNNI = `VPDPBUSD`: u8 x s8 products accumulated straight into int32 lanes. This is *why* INT8 can be fast here.
- GPU (GTX 1070, Pascal sm_61): no Tensor Cores; ORT's CUDA EP has no fused INT8 conv kernels, so QDQ there
  mostly means "dequantize then FP32". Kept only as a final negative control.
- Two CCDs: CCD0 = cpus 0-5 (+SMT 12-17), 96 MB L3 (V-Cache). CCD1 = cpus 6-11, 32 MB. Benchmarks pin to CCD0
  physical cores, `intra_op_num_threads=6`. Governor already `performance` (amd-pstate-epp, EPP=performance).

## Phase 1 — export

### What export is
- **[ML]** Legacy `torch.onnx.export(dynamo=False)` = tracing: one forward pass with a real tensor, every ATen op
  executed is recorded and mapped to an ONNX operator at a fixed **opset**. Weights become **initializers**.
  Python control flow is frozen to the path the dummy took.
- **[ONNX]** The opset decides what the file can even express (per-channel QDQ needs `axis`, opset >= 13). We use 17.
- **[ONNX]** Since torch 2.9 the **default is `dynamo=True`** (torch.export -> onnxscript). Old tutorials describe
  the legacy path. The legacy exporter is deprecated but still what most deployment code uses.

### Decisions in `01_export.py`
- `fuse()` first: BN at inference is a per-channel affine map, so it folds into the conv weights/bias. Quantization must
  see the *folded* weights, because those are what run.
- Run `model(dummy)` once: Detect caches anchors, so they enter the trace as constants.
- `Detect.export=True`: the head returns only the decoded `[B, 84, 8400]` tensor.

### What `dynamic_axes` changed (from `02_inspect_graph.py`)
- I/O: `images [1,3,640,640]` -> `['batch',3,640,640]`. Mechanically, `dim_value=1` becomes `dim_param="batch"`.
- Inside: +44 nodes (363 -> 407). Every Reshape in the head that used to read a literal Constant
  (e.g. DFL reshape `[1, 4, 16, 8400]`) now reads a runtime `Shape -> Gather -> Unsqueeze -> Concat`.
- Legacy tracer also made the 8400 runtime-derived (the code reads `x.shape[2]`), and its shape inference lost the
  output dims: `['batch','Concatoutput0_dim_1','Concatoutput0_dim_2']`. Dynamo kept `['batch', 84, 8400]`
  because it tracks shapes symbolically end to end.

### Graph structure (yolov8n, legacy, 407 nodes)
- 64 Conv (BN folded), 57 SiLU expanded as `Sigmoid -> Mul` (+1 Sigmoid for class scores = 58), 3 MaxPool (SPPF),
  2 Resize (FPN upsampling), Concat for C2f/FPN skip joins. Head (`/model.22/`): DFL = Reshape -> Transpose
  -> Softmax(16 bins) -> 1x1 Conv (fixed weights 0..15 = expectation), then anchor decode (Sub/Add/Div/Mul by stride).
- **The last node is `Concat_9(Mul_2 [boxes in px], Sigmoid [class probs])` -> `output0`.** Pixel-scale boxes and
  [0,1] probabilities share one tensor (this is the quantization hazard, see Phase 2).
- NMS is **not** in the graph. It runs in numpy afterwards.

### Parity (`03_parity.py`, CPU EP vs PyTorch CPU, default ORT_ENABLE_ALL optimizations)
- Boxes max|d| = 1.8e-3 px on values up to ~640 (rel 2.3e-5). Scores max|d| = 3.6e-6. Top-1 decision identical.
- Rerun on first 50 COCO val2017 images: boxes max|d| = 8.9e-3 px (rel 3.8e-5), scores max|d| = 5.3e-5, top-1
  identical on all 50. Larger max than on 2 images (more samples reach the tail), relative error unchanged.
- batch=N and batch-1 loop give identical results, so dynamic batch is numerically harmless.
- Why it's not 0: ORT fuses ops and reorders reductions (different conv algorithms, sum order). FP32 addition isn't
  associative, so ~1e-7 relative error per op accumulating over ~60 layers is expected.
- **Why parity matters:** it's the baseline that lets us attribute any later degradation to quantization alone.

### Mistakes made
1. **Dynamo export traced with batch=1 came out static** despite `Dim("batch")`. torch.export *specializes* dims of
   size 0/1 (size-1 broadcasts differently, so 1 is treated as a special value). Fix: trace with batch=2.
   I had claimed dynamo "fails loudly". In this case it silently specialized instead.
2. Diffing graphs by node name is fragile: exporters renumber `Constant_N`. Diff by *structure* (who produces the
   Reshape's shape input), as in section 3b of `02_inspect_graph.py`.

### Commands that mattered
```
python 00_env_check.py        # hardware facts -> results/env.json
python 01_export.py           # 3 exports
python 02_inspect_graph.py    # histograms, dynamic_axes diff
python 03_parity.py           # FP32 parity
netron --port 18765 models/yolov8n.onnx   # visual graph, http://localhost:18765 (8081/8090 taken by other local services)
```

## Phase 2 — INT8 PTQ (QDQ, MinMax), 2x2 design

### Concepts
- **[ML]** Affine quant: `q = clamp(round(x/s)+z, 0, 255)`, `x̂ = s(q−z)`. Error = clipping (outside range) +
  rounding (±s/2 inside). A wider range clips less but makes every step coarser.
- **[ONNX]** Static PTQ calibration: ORT copies the model, exposes EVERY intermediate tensor as an output, runs the
  CalibrationDataReader's images through it, and records min/max per tensor. So the reader must use the exact
  deployment preprocessing (`common.preprocess`: letterbox, 114 pad, /255).
- **[ONNX]** QDQ format = annotation. FP32 ops stay, `QuantizeLinear→DequantizeLinear` pairs mark where tensors are
  INT8. At session load ORT pattern-matches `DQ→Op→Q` into integer kernels (`QLinearConv`, ...). An op with no
  integer kernel just runs FP32 after a DQ. Inspect what ACTUALLY runs via `SessionOptions.optimized_model_filepath`.
- **[ML]** Per-channel applies to weights only. In `y[o] = Σ_k w[o,k]x[k]` a scale per output channel `o` factors out
  of the sum (free rescale of the int32 result). A scale per input channel `k` would not, so activations stay
  per-tensor. BN folding makes channel ranges very uneven: `/model.2/cv1` max|w| per channel spans 0.075–1.23
  (16.5x), so per-tensor leaves the weakest channel 7 of 127 levels.
- **[ML/HW]** `reduce_range=False`: AVX2 `vpmaddubsw` adds pairs of u8×s8 products into a *saturating int16*
  (2·255·127 = 64,770 > 32,767). reduce_range = 7-bit weights (2·255·63 = 32,130 fits). VNNI `vpdpbusd` accumulates
  4 products into int32, so there's no overflow and the bit isn't needed. U8 activations × S8 weights = exactly vpdpbusd's
  operand types.
- **[ONNX]** INT8 elementwise unary ops are exact lookup tables: a u8 input has only 256 values, so `QLinearSigmoid` = 256-entry LUT.

### Gotchas found
- **ORT 1.31 default `activation_type` is QInt8 (S8S8), not U8.** On x86 the kernel then has to shift activations to u8
  and correct the result. We chose QUInt8 deliberately.
- **Default `op_types_to_quantize=None` = union of QLinearOps + QDQ registries** (`quantize.py:722`), which includes
  Concat, Mul, Sigmoid, Softmax, Add. So the final output Concat gets quantized.
- **`quant_pre_process` failed: "Incomplete symbolic shape inference".** The legacy graph re-derives batch via
  Shape→Gather, inference gives it a new symbol, and it can't prove it equals `batch`. `auto_merge=True` fixes it (safe
  here, but it disables a consistency check). A cost of the legacy exporter.
- Graph outputs get renamed when Q/DQ are inserted. Find nodes by producer, not by tensor name.

### Where Q/DQ landed (`06_inspect_qdq.py`)
- On the activation inputs of every quantizable op: 64 Conv, 58 Sigmoid + 58 Mul (SiLU halves), Concat, Add,
  MaxPool, Resize. Shape plumbing sits behind a DQ but never runs in INT8.
- Runtime graph: `QLinearConv` ×64, `QLinearSigmoid` ×58, `QLinearMul` ×58, `QLinearConcat` ×17, 15 leftover DQ (the
  exits from the integer path, mostly in the head). headfp32: 63 QLinearConv (the DFL conv, weights 0..15, now FP32).

### The output-concat hazard (`07_activation_histogram.py`, `results/activation_histogram.png`)
- `output0` = Concat(boxes −10..657 px, scores [0,1]). MinMax per-tensor scale = **2.618, zp = 4**. Rounding
  boundary 1.31 > max score, so **every score dequantizes to exactly 0.000**.
- per_channel can't help: it's an activation. Measured: `*_all` produced **0 detections, mAP = 0**, as predicted
  before measuring.
- Fix: keep the head decode (34 nodes in `/model.22/` outside cv2/cv3) in FP32 via `nodes_to_exclude`.
- Typical backbone activation (`/model.2/cv1` post-SiLU): MinMax range [−0.34, 42.99], step 0.170. The central 99.9% of
  values lie in [−0.30, 12.6], so they use **76 of 256 levels** (≈6.2 bits). 0.05% of the values in the tail cost about 1.75
  bits of resolution for everything else. This is the motivation for entropy/percentile calibration (Phase 4a).

### Measurements (`08_benchmark.py`, CPU EP, cpus 0-5 pinned, 6 intra-op threads, 20 warmup + 200 runs, ×3 repeats)
See `10_collect_results.py` for the table. FP32 16.67 ms. `*_all` 9.0 ms (1.85×, but mAP 0). `*_headfp32` 11.9 ms
(1.40×). Repeat-to-repeat spread of the mean ≤ 0.17 ms.

### Profiling explained a surprise (`08b_profile.py`, ORT `enable_profiling`)
- headfp32 was 2.9 ms slower than all, from excluding ~0% of the FLOPs. Cause: ONE FP32 node,
  `/model.22/dfl/Softmax` (axis=1 of `[B,16,4,8400]`, a strided non-last axis) = **3.25 ms**. As QLinearSoftmax it's
  0.25 ms. The FP32 baseline pays it too (19% of FP32 runtime). An FP32 kernel/layout issue, not a quantization effect.
  The exact kernel mechanism (transpose to last axis?) is unverified.
- Kernel-level: FP32 Conv 11.4 ms vs QLinearConv 3.8 ms = **3.0×**. End to end is diluted by INT8 overheads: SiLU as
  QLinearSigmoid+QLinearMul 2.5 ms vs FP32's fused `QuickGelu` 0.93 ms (ORT fuses x·σ(x) into QuickGelu, α=1),
  and QLinearConcat 1.4 ms (requantizes every input to a common scale).
- FP32 also shows ReorderInput/Output: NCHWc blocked layout for AVX-512 convs.

### Accuracy (`09_eval_map.py`, 646 eval images, 4 vehicle classes, COCO AP@0.5)
- FP32 0.6076. pt_headfp32 0.5957 (−1.19 pts). pc_headfp32 0.6043 (−0.33 pts). Both `*_all`: 0 dets.
- Per-channel recovers ~70% of the per-tensor loss. No confidence intervals yet, so Phase 3 should add a paired bootstrap.

### VNNI evidence so far
- Flag present (CPUID). `libonnxruntime.so.1.31.0` contains 2,543 `vpdpbusd` instructions (VNNI kernels compiled in).
  Runtime selection unproven: `perf` is blocked (`kernel.perf_event_paranoid=4`).

### Mistakes made
3. Wrong claim that ORT's default activation type is U8. It's S8 in 1.31. Check `inspect.signature`, not memory.
4. First histogram panel (b) hid the scores under a 1 px bar. A plot that doesn't show its point has to be redrawn.

### Single-setting checks (`11_quant_checks.py`, baseline pc_headfp32, run 2026-10-25)
Separate benchmark run with its own FP32 row. Day-to-day drift: FP32 16.36 ms vs 16.67 ms two days earlier (~2%),
so compare only within one run.

| variant | mean ms | p95 ms | vs FP32 | mAP@0.5 | mAP@.5:.95 |
|---|---|---|---|---|---|
| FP32 | 16.36 | 16.54 | 1.00× | 0.6076 | 0.4223 |
| pc_headfp32 (U8 act, reduce_range=False) | 11.60 | 11.82 | 1.41× | 0.6043 | 0.4119 |
| + reduce_range=True | 11.59 | 11.77 | 1.41× | 0.6002 | 0.4123 |
| + S8 activations (ORT default) | **32.45** | 32.97 | **0.50×** | 0.6030 | 0.4129 |

- **reduce_range**: latency identical (Δ 0.01 ms), so on VNNI it buys nothing, as predicted. mAP@0.5 −0.41 pts but
  mAP@.5:.95 +0.04 pts. The two metrics disagree in sign, and there's no CI, so the accuracy cost is **not established**.
  Conclusion supported by data: no speed benefit, so there's no reason to pay even a possible bit of precision.
- **S8 activations: my prediction ("equal or slightly slower") was wrong by 2.8×.** The runtime graph shows why: only 6
  of 64 convs fuse into QLinearConv, exactly the head's final 1×1 convs, the only convs not followed by SiLU. No
  QLinearSigmoid/QLinearMul exist in the S8 graph. The other 58 run as FP32 Conv wrapped in live Q/DQ ("fake quant"):
  202 Q + 418 DQ, DequantizeLinear alone 12.1 ms/run. Result: INT8 accuracy *and* slower than FP32.
  Hypothesis (unverified in ORT source): the CPU EP's QDQ fusion lacks S8 kernels for Sigmoid/Mul, and that blocks the
  conv fusion upstream. **Lesson: the default activation type of ORT 1.31 is the wrong one for x86 CPU here,
  and the only way to see it is the runtime graph or the profiler. The file looks identical in structure.**

### Mistakes made
5. Predicted S8 ≈ U8 latency from the instruction-level argument alone. The bigger effect was a missing graph fusion,
   which sits one level above instructions. Check the runtime graph before reasoning about kernels.

## Phase 3 — fix the output-concat hazard in the graph (`12_split_outputs.py`)

### Concepts
- **[ONNX]** A graph output is only a tensor *name* plus a `ValueInfoProto` (type, shape). Graph surgery = delete
  `Concat_9`, rename its two input tensors to `boxes`/`scores`, move their `value_info` entries into `graph.output`.
  `onnx.checker.check_model(full_check=True)` then runs strict shape inference over the edited graph.
- **[ONNX]** Rename tensors only after checking who reads them. Here `Concat_9` was the only consumer of both; if
  anything else read them, renaming the producer's output would silently cut that edge.
- **[ONNX]** Phase 2 changed the *quantizer's* view (`nodes_to_exclude`). Phase 3 changes the *graph* so it is
  quantizable as is. Same hazard, two levels.

### Predictions (written 2026-10-28, before any COCO run)
- `split_pc_all` has non-zero detections: `scores` gets its own scale (~1/255) instead of ~2.5.
- `boxes` keeps a ~640/255 ≈ 2.5 px step (±1.25 px rounding, in letterboxed 640 space). Expect mAP@.5:.95 to drop
  more than mAP@0.5 vs `pc_headfp32`, since tight IoU thresholds feel 1 px errors and IoU 0.5 barely does.
- Latency close to `*_all` (~9 ms in Phase 2's run), because the DFL Softmax runs as QLinearSoftmax again.

### Checks done without COCO (Claude web, ORT 1.29, not results)
- Surgery is exact: concat(boxes, scores) vs `output0` max|d| = 0 on random input, batch 2.
- Smoke quantization with 4 random-noise calibration images: `boxes` scale 2.503 (range 0–638), as predicted. But
  `scores` got range 0–0.08: noise images make no confident detections, so MinMax fits the scores to 0.08 and any
  real score above that would clip. **Calibration data defines the ranges; this is why the reader must be real,
  deployment-preprocessed images.** The COCO run decides the actual scores range.
