"""
diagnose_alignment.py — Phase-1 diagnostic for the PolyU EER gap.

Question we are answering: is our ~17% EER ceiling caused by bad / inconsistent
VIS segmentation (open-iris is NIR-only) that destroys the geometric alignment
co-registered PolyU otherwise gives us for free?

For N sampled eyes that have BOTH a VIS and an NIR instance, this script produces:

  1. Segmentation-quality overlay — fitted pupil/iris/eyeball contours (from
     open-iris `call_trace["geometry_estimation"]`) drawn on the exact mono image
     open-iris saw. VIS | NIR side by side. *Are the VIS contours on the real iris?*

  2. Strip-alignment montage — the finished 64x512 VIS strip stacked over the NIR
     strip over their abs-difference, plus a horizontal-shift NCC scan. A genuine
     co-registered pair, if well aligned, peaks near shift=0. A large best-shift or
     low NCC@0 = misaligned genuine pairs = the suspected ceiling.

  3. Per-modality segmentation success/fail counts on the sample.

  4. (--oracle, diagnostic ONLY — never a shipping path) a VIS strip unwrapped with
     the NIR image's geometry via iris.LinearNormalization().run(...). If oracle pairs
     align and self-segmented VIS pairs do not, segmentation is proven to be the cause.

This runs on the Linux training box (needs `iris` + the cached HF seg model + raw data).
It imports the SAME normalization code the trainer uses (iris_norm.normalize_strip), so
the strips here are byte-identical to training strips.

Usage:
    python tools/diagnose_alignment.py --src ~/PolyU_raw --out eval_results/diag --n 12
    python tools/diagnose_alignment.py --src ~/PolyU_raw --out eval_results/diag --n 12 --oracle
"""

import argparse
import os
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Make repo root importable when run as `python tools/diagnose_alignment.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iris_norm import STRIP_H, STRIP_W, enhance_strip, normalize_strip, to_mono  # noqa: E402
from prepare_strips import IMG_EXTS, parse_meta  # noqa: E402


# --------------------------------------------------------------------------- #
# PolyU eye indexing: group files by (subject, eye) and instance index
# --------------------------------------------------------------------------- #
def instance_index(stem: str):
    """'001_L_NIR_1' -> 1.  Returns None if no trailing integer token."""
    tok = stem.split("_")[-1]
    return int(tok) if tok.isdigit() else None


def build_eye_index(src_root: Path, dataset: str):
    """{(subject, eye): {modality: {instance: path}}} for eyes seen in both spectra."""
    idx = defaultdict(lambda: defaultdict(dict))
    files = [f for f in src_root.rglob("*") if f.suffix.lower() in IMG_EXTS]
    for path in files:
        meta = parse_meta(path, dataset)
        if meta is None:
            continue
        subject, eye, modality, side = meta
        k = instance_index(path.stem)
        if k is None:
            continue
        idx[(subject, eye, side)][modality][k] = path
    return idx


# --------------------------------------------------------------------------- #
# open-iris helpers (defensive — API objects vary slightly across builds)
# --------------------------------------------------------------------------- #
def safe_trace(pipeline, key):
    """Return call_trace[key] or None, never raising."""
    try:
        val = pipeline.call_trace[key]
    except Exception:
        return None
    return val


def run_pipeline(pipeline, iris_mod, mono, side):
    """Run the full pipeline on a mono image. Returns True on success."""
    try:
        ir = iris_mod.IRImage(img_data=mono, eye_side=side)
        pipeline(ir)
        return True
    except Exception as e:
        print(f"      pipeline error: {e}")
        return False


def geometry_arrays(gp):
    """Pull (pupil, iris, eyeball) Nx2 point arrays from a GeometryPolygons-like obj."""
    out = {}
    for name in ("pupil_array", "iris_array", "eyeball_array"):
        arr = getattr(gp, name, None)
        if arr is not None:
            out[name] = np.asarray(arr, dtype=np.float32)
    return out


def draw_overlay(mono, gp, title):
    """Draw fitted contours on a BGR copy of the mono image open-iris saw."""
    bgr = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    colors = {"pupil_array": (0, 255, 0), "iris_array": (0, 0, 255), "eyeball_array": (255, 128, 0)}
    if gp is not None:
        for name, arr in geometry_arrays(gp).items():
            pts = np.round(arr).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(bgr, [pts], isClosed=True, color=colors.get(name, (255, 255, 255)), thickness=1)
    cv2.putText(bgr, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return bgr


# --------------------------------------------------------------------------- #
# Alignment metric: zero-mean normalized cross-correlation over horizontal rolls
# --------------------------------------------------------------------------- #
def ncc(a, b):
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float((a * b).sum() / den)


def shift_scan(vis_strip, nir_strip, max_shift=64):
    """Roll NIR over [-max_shift, max_shift] columns; return (best_shift, ncc@0, ncc@best)."""
    ncc0 = ncc(vis_strip, nir_strip)
    best_s, best_v = 0, -2.0
    for s in range(-max_shift, max_shift + 1):
        v = ncc(vis_strip, np.roll(nir_strip, s, axis=1))
        if v > best_v:
            best_v, best_s = v, s
    return best_s, ncc0, best_v


def strip_montage(vis_strip, nir_strip, label, scale=2):
    """Stack VIS / NIR / abs-diff vertically, upscale, annotate."""
    diff = cv2.absdiff(vis_strip, nir_strip)
    stack = np.vstack([vis_strip, nir_strip, diff])              # (3*64, 512)
    stack = cv2.cvtColor(stack, cv2.COLOR_GRAY2BGR)
    stack = cv2.resize(stack, (STRIP_W * scale, STRIP_H * 3 * scale), interpolation=cv2.INTER_NEAREST)
    for i, txt in enumerate(("VIS", "NIR", "|VIS-NIR|")):
        cv2.putText(stack, txt, (6, 18 + i * STRIP_H * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(stack, label, (6, STRIP_H * 3 * scale - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return stack


# --------------------------------------------------------------------------- #
# Optional oracle: unwrap VIS using NIR geometry (diagnostic only)
# --------------------------------------------------------------------------- #
def oracle_vis_strip(iris_mod, pipeline, vis_mono, nir_mono, side):
    """Unwrap VIS with NIR-derived geometry. Returns 64x512 strip or None.

    NOTE: requires a paired NIR image -> NOT usable at deployment. Measurement only,
    to prove whether perfect geometric alignment collapses the genuine/impostor gap.
    """
    try:
        if not run_pipeline(pipeline, iris_mod, nir_mono, side):
            return None
        nir_geom = safe_trace(pipeline, "geometry_estimation")
        nir_orient = safe_trace(pipeline, "eye_orientation")
        nir_noise = safe_trace(pipeline, "noise_masks_aggregation")
        if nir_geom is None or nir_orient is None or nir_noise is None:
            print("      oracle: missing NIR trace (geometry/orientation/noise)")
            return None
        norm_node = iris_mod.LinearNormalization()
        vis_ir = iris_mod.IRImage(img_data=vis_mono, eye_side=side)
        res = norm_node.run(
            image=vis_ir,
            noise_mask=nir_noise,
            extrapolated_contours=nir_geom,
            eye_orientation=nir_orient,
        )
        strip = np.asarray(res.normalized_image, dtype=np.uint8)
        strip = cv2.resize(strip, (STRIP_W, STRIP_H), interpolation=cv2.INTER_LINEAR)
        return enhance_strip(strip)
    except Exception as e:
        print(f"      oracle error: {e}")
        return None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Diagnose VIS/NIR segmentation alignment")
    ap.add_argument("--src", required=True, help="PolyU raw root (recursively searched)")
    ap.add_argument("--out", default="eval_results/diag", help="output dir for PNGs")
    ap.add_argument("--dataset", default="polyu", choices=["polyu", "cuviris"])
    ap.add_argument("--n", type=int, default=12, help="number of eyes to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oracle", action="store_true", help="also emit NIR-geometry oracle VIS strip")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    import iris  # noqa: E402  (lazy; needs cached HF seg model)
    pipeline = iris.IRISPipeline()

    idx = build_eye_index(Path(args.src), args.dataset)
    eyes = [k for k, v in idx.items() if "VIS" in v and "NIR" in v]
    if not eyes:
        sys.exit("No eyes with BOTH a VIS and NIR instance found — check --src / parse_meta.")
    random.shuffle(eyes)
    eyes = eyes[: args.n]
    print(f"Sampled {len(eyes)} eyes (of {len(eyes)} candidates with both spectra).")

    seg_ok = defaultdict(int)
    seg_fail = defaultdict(int)
    rows = []   # (eye_label, best_shift, ncc0, ncc_best)

    for (subject, eye, side) in eyes:
        rec = idx[(subject, eye, side)]
        # pick a shared instance index (co-registered pair if PolyU is per-index aligned)
        shared = sorted(set(rec["VIS"]) & set(rec["NIR"]))
        k = shared[0] if shared else None
        vis_path = rec["VIS"][k] if k is not None else rec["VIS"][sorted(rec["VIS"])[0]]
        nir_path = rec["NIR"][k] if k is not None else rec["NIR"][sorted(rec["NIR"])[0]]
        label = f"{subject}_{eye}_{side}_inst{k}"
        print(f"  {label}")

        vis_mono = to_mono(vis_path, "VIS")
        nir_mono = to_mono(nir_path, "NIR")

        # finished, training-identical strips (each independently segmented)
        vis_strip, _, vis_ok = normalize_strip(vis_mono, pipeline, side)
        nir_strip, _, nir_ok = normalize_strip(nir_mono, pipeline, side)
        seg_ok["VIS"] += int(vis_ok); seg_fail["VIS"] += int(not vis_ok)
        seg_ok["NIR"] += int(nir_ok); seg_fail["NIR"] += int(not nir_ok)

        # geometry overlays (separate runs to capture call_trace geometry)
        vis_gp = safe_trace(pipeline, "geometry_estimation") if run_pipeline(pipeline, iris, vis_mono, side) else None
        vis_overlay = draw_overlay(vis_mono, vis_gp, f"VIS {label}")
        nir_gp = safe_trace(pipeline, "geometry_estimation") if run_pipeline(pipeline, iris, nir_mono, side) else None
        nir_overlay = draw_overlay(nir_mono, nir_gp, f"NIR {label}")
        h = max(vis_overlay.shape[0], nir_overlay.shape[0])
        pad = lambda im: cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        cv2.imwrite(os.path.join(args.out, f"seg_{label}.png"), np.hstack([pad(vis_overlay), pad(nir_overlay)]))

        # strip alignment montage + NCC scan
        if vis_ok and nir_ok:
            best_s, ncc0, ncc_best = shift_scan(vis_strip, nir_strip)
            rows.append((label, best_s, ncc0, ncc_best))
            mlabel = f"shift*={best_s}px  ncc@0={ncc0:.3f}  ncc*={ncc_best:.3f}"
            cv2.imwrite(os.path.join(args.out, f"strip_{label}.png"),
                        strip_montage(vis_strip, nir_strip, mlabel))

            if args.oracle:
                ov = oracle_vis_strip(iris, pipeline, vis_mono, nir_mono, side)
                if ov is not None:
                    b2, n0, nb = shift_scan(ov, nir_strip)
                    olabel = f"ORACLE(NIR-geom) shift*={b2}px ncc@0={n0:.3f} ncc*={nb:.3f}"
                    cv2.imwrite(os.path.join(args.out, f"oracle_{label}.png"),
                                strip_montage(ov, nir_strip, olabel))

    # ---- summary ----
    print("\n--- segmentation success on sample ---")
    for mod in ("VIS", "NIR"):
        print(f"  {mod}: ok={seg_ok[mod]}  fail={seg_fail[mod]}")

    if rows:
        shifts = np.array([r[1] for r in rows])
        ncc0s = np.array([r[2] for r in rows])
        print("\n--- genuine-pair alignment (self-segmented strips) ---")
        print(f"  |best-shift|  : mean={np.abs(shifts).mean():.1f}px  max={np.abs(shifts).max()}px")
        print(f"  ncc@shift=0   : mean={ncc0s.mean():.3f}  min={ncc0s.min():.3f}")
        print("  (well-aligned co-registered pairs -> |best-shift|~0 and higher ncc@0)")
        print("\n  per-eye:")
        for label, s, n0, nb in rows:
            print(f"    {label:28s} shift*={s:+4d}px  ncc@0={n0:+.3f}  ncc*={nb:+.3f}")

    print(f"\nPNGs written to {args.out}/  (seg_*, strip_*" + (", oracle_*" if args.oracle else "") + ")")
    print("Read the gate: VIS contours off the iris / large |best-shift| / low ncc@0 => Phase 2A.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
