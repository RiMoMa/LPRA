# LPRA — YOLOv8n ONNX export & INT8 PTQ on CPU

Reproducible study of exporting YOLOv8n to ONNX and applying post-training static
quantization (QDQ, MinMax) for CPU inference. The workload is an ALPR-style vehicle
detector evaluated on COCO val2017 (car, motorcycle, bus, truck).

The emphasis is on measurement discipline rather than a single accuracy number:
FP32 parity, a calibration/evaluation split that never leaks, latency measured on
pinned cores, inspection of the graph that ONNX Runtime actually runs, and COCO mAP.

## Headline results

Hardware: AMD Ryzen 9 7900X3D (Zen 4, AVX-512 VNNI), CPU execution provider,
ONNX Runtime 1.31, 6 intra-op threads pinned to CCD0 (CPUs 0–5), 150 calibration
images and 646 evaluation images. Values below are from the 2026-10-25 check run;
compare only within the same run.

| Model | Mean latency | Speedup vs FP32 | mAP@0.5 | mAP@.5:.95 |
|---|---:|---:|---:|---:|
| `yolov8n_prep.onnx` (FP32) | 16.36 ms | 1.00× | 0.6076 | 0.4223 |
| `yolov8n_int8_pc_headfp32.onnx` (recommended) | **11.60 ms** | **1.41×** | 0.6043 | 0.4119 |
| `yolov8n_int8_pc_headfp32_rr.onnx` (`reduce_range=True`) | 11.59 ms | 1.41× | 0.6002 | 0.4123 |
| `yolov8n_int8_pc_headfp32_s8.onnx` (S8 activations) | 32.45 ms | 0.50× | 0.6030 | 0.4129 |
| `yolov8n_int8_pc_all.onnx` / `pt_all` (head quantized too) | ≈9.0 ms | ≈1.85× | **0.0000** | **0.0000** |

Why this shape:

- **The output head must stay FP32.** `output0` concatenates boxes in pixel units
  (roughly −10…657) with class scores in [0, 1]. A single per-tensor MinMax scale is
  forced to 2.618 (zero point 4), so every score rounds to zero. The `*_headfp32`
  variants exclude the 34 head-decode nodes (`/model.22/` outside `cv2`/`cv3`) from
  quantization. Those nodes are ~0% of the FLOPs, so this is almost free.
- **Per-channel weights recover most of the loss.** `pc_headfp32` is −0.33 mAP points
  vs FP32; per-tensor `pt_headfp32` is −1.19 points.
- **U8 activations, not ORT's default S8.** S8 looks structurally identical in the
  ONNX file but only 6 of 64 convolutions fuse into `QLinearConv`; the rest run FP32
  wrapped in live Q/DQ, making it 2.8× slower than FP32 on this CPU.
- **`reduce_range=True` buys nothing on VNNI**, where INT8 accumulates into int32
  (`vpdpbusd`). It is not used in the recommended model.

`LEARNING_NOTES.md` contains the full reasoning, graph details, mistakes and
hypotheses. `results.csv` is generated from the raw result files, never edited by hand.

## Pipeline

Run the scripts from the repository root, in order.

| Script | Purpose | Main output |
|---|---|---|
| `00_env_check.py` | Record CPU flags, cache topology, versions | `results/env.json` |
| `01_export.py` | Export three ONNX variants (static, dynamic, dynamo) | `models/yolov8n*.onnx` |
| `02_inspect_graph.py` | I/O signatures, op histograms, dynamic-axes diff | stdout |
| `03_parity.py` | PyTorch FP32 vs ORT FP32 parity | `results/parity_*.json` |
| `04_split.py` | Fixed calibration/eval split of COCO vehicle images | `data/split.json` |
| `05_quantize.py` | 2×2 INT8 PTQ design: per-tensor/per-channel × all-ops/head-FP32 | `models/yolov8n_int8_*.onnx` |
| `06_inspect_qdq.py` | Where Q/DQ landed and what the runtime graph really executes | stdout |
| `07_activation_histogram.py` | MinMax clipping cost and the output-concat hazard | `results/activation_histogram.png`, `*_stats.json` |
| `08_benchmark.py` | Latency + size, pinned cores, 20 warmup + 200 runs × 3 repeats | `results/latency_raw.csv` |
| `08b_profile.py` | Per-node ORT profiler | `results/profile_*.json` |
| `09_eval_map.py` | COCO AP@0.5 on 646 images, 4 vehicle classes | `results/map_overall.json`, `results/dets/*.json` |
| `10_collect_results.py` | Rebuild the long-format table from raw files | `results.csv` |
| `11_quant_checks.py` | One-setting checks: `reduce_range`, S8 activations | `models/*_rr.onnx`, `*_s8.onnx` |

## Repository layout

```
.
├── 00_env_check.py … 11_quant_checks.py   # numbered pipeline
├── common.py                              # preprocessing contract
├── calib_reader.py                        # ORT calibration data reader
├── data/split.json                        # committed; COCO images are not
├── models/*.onnx                          # committed artifacts (~87 MB)
├── results/                               # small metrics/plots committed
├── weights/yolov8n.pt                     # committed (6.5 MB)
├── requirements.txt
├── setup_data.py
├── LEARNING_NOTES.md
└── CLAUDE.md
```

## Setup

Requires Python 3.11 and a Linux x86-64 machine for the current benchmarks. The
pipeline also runs elsewhere, but the absolute latency numbers assume the hardware
and pinning described above.

```bash
git clone git@github.com:RiMoMa/LPRA.git
cd LPRA
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The model weights and ONNX artifacts are already in the repository. The COCO images
are not; fetch them once (~1.1 GB download, ~1.8 GB extracted):

```bash
python setup_data.py                 # keeps the archives
# or: python setup_data.py --remove-zips
```

`data/split.json` is committed and deterministic (`SEED = 0`, 150 calibration +
646 evaluation images). Regenerate it only if you deliberately change the split:

```bash
python 04_split.py
```

## Run

```bash
python 00_env_check.py
python 01_export.py
python 02_inspect_graph.py
python 03_parity.py
python 05_quantize.py
python 06_inspect_qdq.py
python 07_activation_histogram.py
python 08_benchmark.py
python 09_eval_map.py
python 10_collect_results.py
```

Optional one-setting checks, which add the `_rr` and `_s8` variants:

```bash
python 11_quant_checks.py
python 08_benchmark.py --out results/latency_checks.csv \
  models/yolov8n_prep.onnx \
  models/yolov8n_int8_pc_headfp32.onnx \
  models/yolov8n_int8_pc_headfp32_rr.onnx \
  models/yolov8n_int8_pc_headfp32_s8.onnx
python 09_eval_map.py --out results/map_checks.json \
  models/yolov8n_prep.onnx \
  models/yolov8n_int8_pc_headfp32.onnx \
  models/yolov8n_int8_pc_headfp32_rr.onnx \
  models/yolov8n_int8_pc_headfp32_s8.onnx
python 10_collect_results.py
```

`09_eval_map.py` writes ~23 MB of detections per model and `08b_profile.py` writes
45–128 MB profiles. Both are gitignored but useful locally: detections let you
re-score an image subset without re-running inference.

## Working with Claude web

The repository is intentionally self-contained so a Claude project can read and
modify the code, while the heavy measurements stay local:

1. Connect GitHub to Claude and grant access to the private `RiMoMa/LPRA` repository.
2. Ask Claude to work on a branch and open a pull request instead of pushing to `main`.
3. The cloud environment does not have the COCO dataset or this Ryzen CPU. Use it for
   code changes, graph reasoning and review; run the latency/mAP commands locally.
4. `CLAUDE.md` records the repository-specific constraints (preprocessing contract,
   fixed split, head-FP32 decision, activation type, no hand-edited results). Keep it
   updated when a design decision changes.

After a pull request is merged:

```bash
git pull --ff-only
source .venv/bin/activate
python 03_parity.py
python 08_benchmark.py
python 09_eval_map.py
python 10_collect_results.py
```

## Reproducibility notes

- **Preprocessing is a contract.** `common.preprocess` (letterbox, 114 gray padding,
  RGB, /255) is used by parity, calibration, latency and mAP. Changing it invalidates
  all calibration ranges and every measurement.
- **Never calibrate on evaluation images.** The split is fixed in `data/split.json`;
  calibration and evaluation image IDs are disjoint.
- **Latency is hardware- and run-specific.** The scripts pin CPUs 0–5, set 6 intra-op
  threads and expect the `performance` governor. Do not compare absolute milliseconds
  across machines; rerun FP32 alongside any variant in the same session.
- **`results.csv` is derived.** Change a script, rerun it, then run
  `10_collect_results.py`; never edit the CSV by hand.
- **Large files stay out of git.** The checked-out repository is ~95 MB (~120 MB
  including git history). The COCO dataset, virtual environments, profiles and
  detections are ignored and regenerable.

## Not in the repository

| Artifact | Size | How to regenerate |
|---|---:|---|
| `data/coco/` | ~1.8 GB | `python setup_data.py` |
| `results/profile_*.json` | ~219 MB | `python 08b_profile.py <models>` |
| `results/dets/*.json` | ~113 MB | `python 09_eval_map.py` |
| `.venv/`, `.py311/` | ~2 GB | `python -m venv .venv && pip install -r requirements.txt` |
