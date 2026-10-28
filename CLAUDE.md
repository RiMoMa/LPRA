# CLAUDE.md

Repository-specific instructions for Claude when working on this project.

## What this repository is

A reproducible study of ONNX export and static INT8 post-training quantization
(QDQ, MinMax) for YOLOv8n on CPU. The detector is evaluated as an ALPR-style vehicle
detector on COCO val2017 (car, motorcycle, bus, truck).

The value of the repository is not just the final model: it is the chain of evidence
(parity, runtime graph inspection, latency, COCO mAP). Preserve that chain when
changing anything.

Start with `README.md` for the overview and `LEARNING_NOTES.md` for the detailed
reasoning, gotchas and negative results.

## Environment

- Python 3.11, Linux x86-64, CPU execution provider (`onnxruntime` 1.31).
- Exact versions: `requirements.txt`. Do not upgrade ONNX Runtime casually; the
  reported behavior (default S8 activations, QDQ fusion patterns) is version-specific.
- The COCO dataset is not in the repository. `python setup_data.py` fetches it.
- Weights and ONNX artifacts are committed, so `01_export.py` and `05_quantize.py`
  are only needed when changing those stages.
- Benchmarks assume CPUs 0–5 pinned, 6 threads, `performance` governor, Ryzen 9
  7900X3D. On a machine with fewer than 6 CPUs, `08_benchmark.py` and
  `08b_profile.py` will fail at `sched_setaffinity`; adapt the pinning rather than
  reporting numbers measured under different conditions.

## Commands

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python setup_data.py            # only if data/coco is missing

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
python 12_split_outputs.py     # Phase 3, optional; see README for its benchmark/mAP commands
```

All scripts use relative paths and must be run from the repository root. The pipeline
is numbered; keep that order in documentation and examples.

## Hard constraints

1. **`common.preprocess` is the single source of truth.** Letterbox, 114 gray padding,
   BGR→RGB, `/255`. Calibration, parity, latency and mAP all depend on it. Changing
   preprocessing invalidates every quantization range and every reported number.
2. **The calibration/evaluation split is fixed.** `data/split.json` uses `SEED = 0`,
   150 calibration images and 646 evaluation images. Never fit ranges or thresholds on
   evaluation images. If the split changes intentionally, regenerate every downstream
   result.
3. **Keep the head decode FP32 in quantized models.** `output0` concatenates pixel-scale
   boxes with [0,1] scores; a single per-tensor scale collapses all scores to zero.
   The `nodes_to_exclude` list from `05_quantize.head_nodes()` is not optional for the
   recommended model. The `*_all` variants exist only as a documented negative control.
   Phase 3 measured alternative: `12_split_outputs.py` removes the output concat by graph
   surgery, so its `*_split_pc_all` model quantizes everything. It restores non-zero
   detections and is the fastest usable model, but loses 5.34 mAP@0.5 and 7.16
   mAP@.5:.95 vs `pc_headfp32`; it is not the default. Reproduce with `12_split_outputs.py`
   followed by the `latency_split.csv` / `map_split.json` runs.
4. **Baseline quantization settings:** `QuantFormat.QDQ`, `QUInt8` activations,
   `QInt8` weights, `per_channel=True`, `reduce_range=False`, `MinMax`, head decode
   excluded. S8 activations and `reduce_range=True` are deliberate one-setting checks,
   not defaults.
5. **Do not hand-edit `results.csv`.** It is rebuilt by `10_collect_results.py` from
   `results/parity_*.json`, `latency_raw.csv`, `map_overall.json`, `latency_checks.csv`, `latency_split.csv`, `map_split.json`
   and `map_checks.json`.
6. **Do not commit generated heavy files:** `data/coco/`, `results/profile_*.json`,
   `results/dets/`, `.venv/`, `.py311/`, `*.data`, or editor scratch files. See
   `.gitignore`.
7. **Do not invent measurements.** Claude web does not have the COCO images or the
   target CPU. Code changes can be reviewed there, but latency and mAP must be rerun
   locally before being reported as results.

## Validation after a change

Minimum, depending on what changed:

- Preprocessing or export: rerun `01_export.py`, `03_parity.py`, then everything
  downstream.
- Quantization settings: rerun `05_quantize.py`, `06_inspect_qdq.py`,
  `08_benchmark.py`, `09_eval_map.py`, `10_collect_results.py`.
- Output split (`12_split_outputs.py`): rerun `12_split_outputs.py`, then the
  `latency_split.csv` / `map_split.json` commands, then `10_collect_results.py`.
- Only analysis/reporting: rerun the affected script and `10_collect_results.py`.
- Always check that `03_parity.py` shows FP32 agreement (boxes ~1e-3 px max, scores
  ~1e-5) and that the recommended INT8 model still produces non-zero detections.

## Documentation

- `README.md` — setup, pipeline, headline results, Claude web workflow.
- `LEARNING_NOTES.md` — conceptual notes, graph details, mistakes. Update it when a
  finding changes; keep the "Mistakes made" sections honest.
- `CLAUDE.md` — this file; update it if a constraint or command changes.
