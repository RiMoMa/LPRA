"""Single source of truth for preprocessing. Parity, calibration, latency and mAP all import
this, because quantization ranges are only valid for inputs distributed exactly like these."""
import cv2, numpy as np

IMG = 640
PAD = 114  # ultralytics letterbox fill value; calibration must see the same gray borders

def letterbox(bgr, size=IMG):
    """Resize keeping aspect ratio, pad to size x size. Returns image + (gain, padw, padh) to undo it."""
    h, w = bgr.shape[:2]
    g = min(size / h, size / w)
    nh, nw = round(h * g), round(w * g)
    img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pw, ph = (size - nw) / 2, (size - nh) / 2
    top, left = round(ph - 0.1), round(pw - 0.1)
    img = cv2.copyMakeBorder(img, top, size - nh - top, left, size - nw - left,
                             cv2.BORDER_CONSTANT, value=(PAD, PAD, PAD))
    return img, (g, left, top)

def preprocess(bgr):
    """BGR uint8 HxWx3 -> float32 1x3x640x640 RGB in [0,1]."""
    img, meta = letterbox(bgr)
    x = img[:, :, ::-1].transpose(2, 0, 1)            # BGR->RGB, HWC->CHW
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None], meta
