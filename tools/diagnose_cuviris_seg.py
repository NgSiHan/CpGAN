"""
diagnose_cuviris_seg.py — visualize what the CVRL segmenter does on raw images.

For N sampled images per modality it runs the SAME pipeline prepare_strips uses
(fix_image -> segment mask -> circApprox circles -> rubber-sheet strip) and saves a
composite PNG:  [ raw + fitted pupil(green)/iris(red) circles + mask(green tint) ]
                [ resulting 64x512 strip ]

Use it to decide WHY CUVIRIS strips look bad:
  - circles OFF the real iris (too big / off-centre / on skin-lash) -> localization bug (fixable)
  - circles OK but strip still lash/dark -> inherent capture occlusion/contrast (data limit)

Works for any dataset parsed by prepare_strips.parse_meta (cuviris / polyu), so you can
run it on PolyU too and compare "good" vs "bad" overlays side by side.

Usage:
  python tools/diagnose_cuviris_seg.py --src ~/CUVIRIS --dataset cuviris --n 8 \
      --out eval_results/seg_cuviris
  python tools/diagnose_cuviris_seg.py --src ~/PolyU_raw --dataset polyu --n 8 \
      --out eval_results/seg_polyu        # for comparison
"""

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iris_norm import build_segmenter, to_mono                 # noqa: E402
from prepare_strips import IMG_EXTS, parse_meta                # noqa: E402


def overlay(seg, mono):
    """Return (composite_bgr, info_dict) or (None, None) on failure."""
    import torch
    from PIL import Image
    pil = seg._fix_image(Image.fromarray(np.ascontiguousarray(mono), "L"))
    try:
        with torch.no_grad():
            mask = seg._segment(pil)                            # uint8 [480,640] 0/255
            pupil_xyr, iris_xyr = seg._circ(pil)
            strip, _ = seg._cart_to_pol(pil, mask, pupil_xyr, iris_xyr)
    except Exception as e:
        print(f"      pipeline error: {e}")
        return None, None

    base = np.array(pil)                                        # [480,640] grayscale
    bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    # mask tint (green) where iris pixels detected
    if mask is not None and mask.shape == base.shape:
        green = bgr[:, :, 1].astype(np.int16)
        green[mask > 0] = np.minimum(255, green[mask > 0] + 70)
        bgr[:, :, 1] = green.astype(np.uint8)
    px, py, pr = [int(round(v)) for v in pupil_xyr]
    ix, iy, ir = [int(round(v)) for v in iris_xyr]
    cv2.circle(bgr, (px, py), max(pr, 1), (0, 255, 0), 2)      # pupil = green
    cv2.circle(bgr, (ix, iy), max(ir, 1), (0, 0, 255), 2)      # iris  = red

    ok = seg._quality_ok(pupil_xyr, iris_xyr, mask)
    mask_cov = float((mask > 0).mean())

    # composite: overlay (resized to 512 wide) over the strip (512 wide)
    ov = cv2.resize(bgr, (512, 384), interpolation=cv2.INTER_AREA)
    strip_bgr = cv2.cvtColor(cv2.resize(strip, (512, 128), interpolation=cv2.INTER_NEAREST),
                             cv2.COLOR_GRAY2BGR)
    gap = np.full((6, 512, 3), 40, np.uint8)
    comp = np.vstack([ov, gap, strip_bgr])
    info = dict(pr=pr, ir=ir, ratio=(pr / ir if ir else 0), ok=ok, mask_cov=mask_cov,
                concentric=float(np.hypot(px - ix, py - iy) / ir if ir else 9))
    txt = f"p_r={pr} i_r={ir} ratio={info['ratio']:.2f} maskcov={mask_cov:.2f} qok={ok}"
    cv2.putText(comp, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return comp, info


def main():
    ap = argparse.ArgumentParser(description="Visualize CVRL segmentation on raw images")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dataset", default="cuviris", choices=["cuviris", "polyu"])
    ap.add_argument("--n", type=int, default=8, help="images per modality")
    ap.add_argument("--out", default="eval_results/seg_diag")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mask_model", default="nestedsharedatrousresunet-006-0.028214-maskIoU-0.938446.pth")
    ap.add_argument("--circle_model", default="resnet18-027-0.008222-maskIoU-0.967159.pth")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    seg = build_segmenter("cvrl", device=args.device,
                          mask_model_path=args.mask_model, circle_model_path=args.circle_model)

    by_mod = defaultdict(list)
    for f in Path(args.src).rglob("*"):
        if f.suffix.lower() in IMG_EXTS:
            meta = parse_meta(f, args.dataset)
            if meta:
                by_mod[meta[2]].append((f, meta[3]))           # modality -> [(path, side)]

    stats = defaultdict(list)
    for mod in ("VIS", "NIR"):
        sample = random.sample(by_mod[mod], min(args.n, len(by_mod[mod])))
        print(f"\n=== {mod} ({len(sample)} samples of {len(by_mod[mod])}) ===")
        for path, side in sample:
            try:
                mono = to_mono(path, mod)
            except IOError:
                continue
            comp, info = overlay(seg, mono)
            if comp is None:
                continue
            name = f"seg_{mod}_{path.stem}.png"
            cv2.imwrite(os.path.join(args.out, name), comp)
            stats[mod].append(info)
            print(f"  {path.name:32s} p_r={info['pr']:3d} i_r={info['ir']:3d} "
                  f"ratio={info['ratio']:.2f} maskcov={info['mask_cov']:.2f} "
                  f"concentric={info['concentric']:.2f} qok={info['ok']}")

    print("\n--- summary (means) ---")
    for mod in ("VIS", "NIR"):
        s = stats[mod]
        if not s:
            continue
        print(f"  {mod}: iris_r={np.mean([x['ir'] for x in s]):.0f}  "
              f"pupil/iris={np.mean([x['ratio'] for x in s]):.2f}  "
              f"mask_cov={np.mean([x['mask_cov'] for x in s]):.2f}  "
              f"qok={sum(x['ok'] for x in s)}/{len(s)}")
    print(f"\nPNGs in {args.out}/  (top = raw+circles+mask, bottom = resulting strip)")


if __name__ == "__main__":
    main()
