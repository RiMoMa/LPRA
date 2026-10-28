# Interview notes — what this project shows, in talking points

Every number below is in `results.csv` or `LEARNING_NOTES.md`. Latency is from one machine (Ryzen 9 7900X3D,
6 pinned cores, ORT 1.31 CPU EP); compare only within a run.

## 30-second version
I exported YOLOv8n to ONNX, proved numerical parity with PyTorch, then did static INT8 post-training quantization
(QDQ, MinMax) with ONNX Runtime on CPU. The recommended model is 1.41× faster than FP32 for −0.33 mAP@0.5 on a
fixed 646-image COCO vehicle split. Along the way I found that the naive quantized model returns zero detections,
that ORT's default activation type makes the model 2× *slower* than FP32 on this CPU, and that the only way to see
either was to inspect the graph ONNX Runtime actually executes, not the file.

## Five findings worth telling

1. **Mixed scales in one tensor kill quantization.** `output0` concatenates boxes (pixels, −10…657) and scores
   ([0,1]). One per-tensor scale (2.62) rounds every score to 0: mAP 0, while being the fastest model.
   Fix A: keep the head decode FP32 (~0% of FLOPs). Fix B: graph surgery, split the output into `boxes` and
   `scores` so each gets its own scale.
2. **The file is not what runs.** QDQ is an annotation; ORT fuses `DQ→Op→Q` into integer kernels at load time.
   With S8 activations (ORT 1.31's default) only 6 of 64 convs fused; the rest ran FP32 wrapped in live Q/DQ:
   32 ms vs 16 ms FP32. Same file structure, opposite result. I check `optimized_model_filepath` and the profiler.
3. **Per-channel weights matter because of BN folding.** Folded conv channels span 16× in range; per-tensor leaves
   the weakest channel ~7 of 127 levels. Per-channel recovers ~70% of the per-tensor accuracy loss.
4. **Hardware decides the settings.** VNNI (`vpdpbusd`) takes u8 × s8 into int32, so U8 activations + S8 weights
   are its native operands and `reduce_range` (an AVX2 int16-saturation workaround) buys nothing: measured equal.
5. **Profiling beats guessing.** The head-FP32 model was 2.9 ms slower than the all-INT8 one because of ONE node,
   the DFL Softmax on a strided axis (3.25 ms FP32 vs 0.25 ms INT8). Kernel-level INT8 conv speedup is 3.0×;
   end to end it is 1.4× because SiLU and Concat requantization add overhead.

## The split-output experiment (graph surgery)
- Removing `Concat_9` and exposing two outputs is exact in FP32 (max|d| = 0) and restores detections.
- Result: 8.88 ms (1.88×) but −5.34 mAP@0.5 vs the head-FP32 model. My prediction got the direction right and the
  magnitude wrong: the output step is only ±1.3 px. Reading the scales inside the graph showed the same mixed-scale
  problem one level up (`Concat_8`, grid units shared across strides: up to 10 px per step) plus a score floor.
- `13_attribute_split.py` separates the two by keeping one path FP32 at a time. `_boxq` (score tail FP32)
  reproduces the split almost exactly (0.5509/0.3396 vs 0.5509/0.3403), while `_scoreq` (box decode FP32) stays
  within 0.14 mAP@0.5 and 0.57 mAP@.5:.95 of `pc_headfp32` and cuts detections from 153,420 to 127,782. The box
  path owns the loss; the score path mainly removes the low-confidence tail. The mixed-scale culprit is
  `Concat_8`, not the ±1.3 px output step. These are point estimates, without confidence intervals.

## Likely questions, short answers
- *QDQ vs QOperator?* QDQ keeps FP32 ops and marks INT8 tensors with Q/DQ pairs; the runtime picks fusions per EP.
  It is portable and is what ORT recommends. QOperator bakes `QLinearConv` etc. into the file.
- *Static vs dynamic quantization?* Dynamic computes activation scales per inference: fine for MatMul-heavy NLP,
  poor for convs. Static calibrates ranges once on representative data, which must use the deployment preprocessing.
- *How did you avoid fooling yourself?* Fixed calibration/eval split (no ranges fit on eval images), one preprocessing
  function for everything, FP32 baseline re-measured in every benchmark run, one setting changed per check,
  predictions written before running, and a "Mistakes made" log.
- *What would you do next?* Confidence intervals (paired bootstrap on per-image detections), calibration methods
  (percentile/entropy), and a deployment test on real ALPR images, since COCO vehicles are a proxy.
