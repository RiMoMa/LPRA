"""Phase 2 checks: test two documented claims on THIS machine instead of trusting them.

Baseline = pc_headfp32 (per-channel, head decode FP32, U8 activations, reduce_range=False). Each variant changes ONE
setting, so any difference in latency or mAP is attributable to that setting alone.
  _rr : reduce_range=True  -> 7-bit weights. Claim: on VNNI it buys nothing (no int16 saturation to avoid) and
        costs ~1 bit of weight precision, so expect ~equal latency and a small mAP loss.
  _s8 : activation_type=QInt8 (ORT 1.31's default). Claim: VPDPBUSD wants u8 activations, so the kernel must shift
        s8 -> u8 and correct; expect equal-or-slower latency. Symmetric s8 also wastes half the levels on
        post-SiLU tensors (min ~ -0.28), so expect a small mAP loss.
"""
import importlib
from onnxruntime.quantization import QuantType

q = importlib.import_module("05_quantize")       # reuse the exact baseline settings, not a copy of them
head, reader = q.head_nodes(), q.calib_reader()
q.build("models/yolov8n_int8_pc_headfp32_rr.onnx", True, head, reader, reduce_range=True)
q.build("models/yolov8n_int8_pc_headfp32_s8.onnx", True, head, reader, activation_type=QuantType.QInt8)
