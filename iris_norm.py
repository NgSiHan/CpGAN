"""
iris_norm.py — shared strip-normalization module (train == deploy).

Imported by BOTH prepare_strips.py (preprocessing) and the FastAPI server (inference),
so the normalization pipeline is identical at train and deploy.

Two segmentation/normalization backends are supported behind one interface:

  - "cvrl"     : CVRL/Notre Dame NestedSharedAtrousResUNet + ResNet18 circles + rubber-sheet
                 (cvrl_seg.CVRLSegmenter). VIS-capable (training corpus includes UBIRIS v2),
                 emits the 64x512 polar strip natively. THIS IS THE DEFAULT.
  - "openiris" : Worldcoin open-iris IRISPipeline (NIR-only; ~33% VIS failures). Kept so we
                 can A/B the two on the same eyes with tools/diagnose_alignment.py.

Post-processing (CLAHE/background-subtract via enhance_strip, and mask soft-fill) is applied
in COMMON for both backends and is independently toggleable, because both steps are applied
per-strip and can inject modality-asymmetric differences that hurt cross-spectral matching.
They were never ablated — treat enhance/soft_fill as hyperparameters, not gospel.

NOTE: torch lives only in cvrl_seg (imported lazily by build_segmenter). Importing this
module on the Windows dev box (no torch) is safe.
"""

from pathlib import Path

import cv2
import numpy as np

STRIP_H, STRIP_W = 64, 512   # radial (rows) x angular (cols). Must match training + inference.


def to_mono(path: Path, modality: str) -> np.ndarray:
    """Load an image and return a uint8 single-channel array.

    VIS: red channel (best melanin penetration, closest VIS analog to NIR).
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
    """
    h, w = strip.shape
    small = cv2.resize(strip, (max(w // block, 1), max(h // block, 1)),
                       interpolation=cv2.INTER_AREA)
    background = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    sub = strip.astype(np.int16) - background.astype(np.int16) + 128
    sub = np.clip(sub, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(sub)


def build_segmenter(backend: str = "cvrl", device: str = "cuda",
                    mask_model_path: str = None, circle_model_path: str = None):
    """Construct a segmentation backend. torch is imported here, not at module top."""
    if backend == "cvrl":
        from cvrl_seg import CVRLSegmenter
        if not mask_model_path or not circle_model_path:
            raise ValueError("cvrl backend needs --mask_model and --circle_model paths")
        return CVRLSegmenter(mask_model_path, circle_model_path, device=device,
                             polar_h=STRIP_H, polar_w=STRIP_W)
    if backend == "openiris":
        import iris
        return iris.IRISPipeline()
    raise ValueError(f"unknown backend: {backend}")


def _raw_strip(mono: np.ndarray, segmenter, eye_side: str):
    """Backend dispatch -> (strip uint8 64x512, mask bool 64x512) or (None, None)."""
    if getattr(segmenter, "is_cvrl", False):
        strip, mask, ok = segmenter.run(mono)
        if not ok:
            return None, None
        return strip, mask

    # --- open-iris path ---
    try:
        import iris as _iris
        ir_image = _iris.IRImage(img_data=mono, eye_side=eye_side)
        segmenter(ir_image)
    except Exception:
        return None, None
    try:
        norm = segmenter.call_trace["normalization"]
    except (KeyError, AttributeError, TypeError):
        return None, None
    if norm is None:
        return None, None
    strip = np.asarray(norm.normalized_image, dtype=np.uint8)
    mask = np.asarray(norm.normalized_mask).astype(np.uint8)
    # open-iris does NOT emit 64x512 directly — resize (identical at train + deploy).
    strip = cv2.resize(strip, (STRIP_W, STRIP_H), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (STRIP_W, STRIP_H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return strip, mask


def normalize_strip(mono: np.ndarray, segmenter, eye_side: str = "left",
                    enhance: bool = True, soft_fill: bool = True):
    """Segment + normalize a single-channel uint8 image to a 64x512 strip.

    Args:
        mono:       uint8 H×W grayscale (output of to_mono).
        segmenter:  a CVRLSegmenter (preferred) or an open-iris IRISPipeline.
        eye_side:   "left"/"right" — used by open-iris only; CVRL ignores it.
        enhance:    apply background-subtract + CLAHE (enhance_strip). Ablatable.
        soft_fill:  overwrite non-iris pixels with the iris-region mean. Ablatable —
                    can hurt cross-spectral matching when masks disagree between modalities.

    Returns:
        (strip uint8 64x512, mask bool 64x512, success bool)
    """
    strip, mask = _raw_strip(mono, segmenter, eye_side)
    if strip is None or mask is None or not mask.any():
        return None, None, False

    if enhance:
        strip = enhance_strip(strip)
    if soft_fill:
        strip = strip.copy()
        strip[~mask] = int(strip[mask].mean())

    return strip, mask, True
