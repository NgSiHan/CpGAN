"""
iris_norm.py — shared strip-normalization module.

Imported by BOTH prepare_strips.py (preprocessing) and the FastAPI server (inference).
This guarantees train == deploy for the normalization pipeline.

Pin open-iris==1.11.1 on every machine that runs this module.
"""

from pathlib import Path

import cv2
import numpy as np

STRIP_H, STRIP_W = 64, 512   # radial (rows) x angular (cols). Must match training + inference.


def to_mono(path: Path, modality: str) -> np.ndarray:
    """Load an image and return a uint8 single-channel array ready for IRISPipeline.

    VIS: red channel (BGR index 2). Red penetrates melanin best and is the closest
         VIS analog to NIR for pigmented irides — better than luminance grayscale.
    NIR: standard grayscale.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"could not read {path}")
    if img.ndim == 3 and img.shape[2] >= 3:
        if modality == "VIS":
            mono = img[:, :, 2]           # red channel in OpenCV BGR ordering
        else:
            mono = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        mono = img
    if mono.dtype != np.uint8:            # some TIFFs are 16-bit
        mono = cv2.normalize(mono, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return np.ascontiguousarray(mono)


def enhance_strip(strip: np.ndarray, block: int = 8) -> np.ndarray:
    """Background subtraction (block mean) followed by CLAHE, applied on the strip.

    Nigam order: subtract background first, then adaptive histogram equalization.
    Both steps are applied on the 64×512 strip — NOT on the full raw image.
    """
    h, w = strip.shape
    small = cv2.resize(strip, (max(w // block, 1), max(h // block, 1)),
                       interpolation=cv2.INTER_AREA)
    background = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    sub = strip.astype(np.int16) - background.astype(np.int16) + 128
    sub = np.clip(sub, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(sub)


def normalize_strip(mono: np.ndarray, pipeline, eye_side: str):
    """Run IRISPipeline on a single-channel uint8 image and return a 64×512 strip.

    Args:
        mono:      uint8 H×W grayscale image (output of to_mono).
        pipeline:  an iris.IRISPipeline instance (caller owns lifecycle).
        eye_side:  "left" or "right" (required by IRISPipeline).

    Returns:
        strip   (np.ndarray uint8, shape 64×512) — normalized, enhanced, soft-filled.
        mask    (np.ndarray bool,  shape 64×512) — True = valid iris pixel.
        success (bool) — False if segmentation/normalization failed.
    """
    out = pipeline(img_data=mono, eye_side=eye_side)
    if out.get("error") is not None:
        return None, None, False

    norm = pipeline.call_trace.get("normalization")
    if norm is None:
        return None, None, False

    strip = np.asarray(norm.normalized_image, dtype=np.uint8)
    mask = np.asarray(norm.normalized_mask).astype(np.uint8)

    # Resize to fixed geometry — open-iris LinearNormalization does NOT emit 64×512 directly.
    # This resize must be identical at train time and at inference time.
    strip = cv2.resize(strip, (STRIP_W, STRIP_H), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (STRIP_W, STRIP_H), interpolation=cv2.INTER_NEAREST).astype(bool)

    strip = enhance_strip(strip)

    if not mask.any():
        return None, None, False

    # Soft-fill occluded pixels (eyelid / eyelash / specular) with the iris-region mean
    # so the encoder does not learn occlusion patterns as identity features.
    strip[~mask] = int(strip[mask].mean())

    return strip, mask, True
