"""
scan_mask_coverage.py — how much usable iris is actually in each CUVIRIS strip?

For every raw image it runs the CVRL pipeline and measures the fraction of the 64x512
unwrapped strip that the occlusion mask marks as VALID IRIS (mask_polar > 0). That strip-
space coverage is the number that matters for matching — unlike image-space mask_cov, it
tells us how much of what the encoder sees is real iris vs occlusion.

Outputs:
  - per-modality distribution (mean / median / percentiles) + a text histogram
  - count of images above coverage thresholds
  - USABLE IDENTITY count at each threshold: identities with >=1 VIS AND >=1 NIR above it
    (a deployment quality gate needs both an enrollable NIR and a probeable VIS)
  - mask_coverage.csv (path, modality, identity, strip_mask_cov, iris_r)

Decision:
  - most images high coverage  -> there IS clean iris; mask-as-channel + fine-tune worth building
  - most images low coverage   -> capture-quality wall; need better data / a capture gate

Usage:
  python tools/scan_mask_coverage.py --src ~/CUVIRIS --dataset cuviris --out eval_results/cuviris_cov
  python tools/scan_mask_coverage.py --src ~/PolyU_raw --dataset polyu --out eval_results/polyu_cov  # ref
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iris_norm import build_segmenter, to_mono                 # noqa: E402
from prepare_strips import IMG_EXTS, parse_meta                # noqa: E402


def strip_mask_cov(seg, mono):
    """Fraction of the 64x512 strip the mask marks as valid iris. None if seg fails quality."""
    pil = seg._fix_image(Image.fromarray(np.ascontiguousarray(mono), "L"))
    with torch.no_grad():
        mask = seg._segment(pil)
        pxyr, ixyr = seg._circ(pil)
        if not seg._quality_ok(pxyr, ixyr, mask):
            return None
        _, mask_polar = seg._cart_to_pol(pil, mask, pxyr, ixyr)
    return float((mask_polar > 0).mean())


def text_hist(vals, bins=10, width=40):
    if not vals:
        return
    counts, edges = np.histogram(vals, bins=bins, range=(0, 1))
    mx = max(counts.max(), 1)
    for i in range(bins):
        bar = "#" * int(width * counts[i] / mx)
        print(f"    {edges[i]:.1f}-{edges[i+1]:.1f} | {bar} {counts[i]}")


def main():
    ap = argparse.ArgumentParser(description="Scan strip-space iris mask coverage")
    ap.add_argument("--src", required=True)
    ap.add_argument("--dataset", default="cuviris", choices=["cuviris", "polyu"])
    ap.add_argument("--out", default="eval_results/cov")
    ap.add_argument("--mask_model", default="nestedsharedatrousresunet-006-0.028214-maskIoU-0.938446.pth")
    ap.add_argument("--circle_model", default="resnet18-027-0.008222-maskIoU-0.967159.pth")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    seg = build_segmenter("cvrl", device=args.device,
                          mask_model_path=args.mask_model, circle_model_path=args.circle_model)

    files = [f for f in Path(args.src).rglob("*") if f.suffix.lower() in IMG_EXTS]
    print(f"Found {len(files)} images under {args.src}\n")

    per_mod = defaultdict(list)                 # modality -> [cov]
    # identity -> {modality -> best cov}
    per_id = defaultdict(lambda: {"VIS": -1.0, "NIR": -1.0})
    rows = []
    for i, f in enumerate(sorted(files), 1):
        meta = parse_meta(f, args.dataset)
        if meta is None:
            continue
        subject, eye, modality, _ = meta
        idd = f"{subject}_{eye}"
        try:
            mono = to_mono(f, modality)
        except IOError:
            continue
        cov = strip_mask_cov(seg, mono)
        if cov is None:
            continue
        per_mod[modality].append(cov)
        per_id[idd][modality] = max(per_id[idd][modality], cov)
        rows.append((str(f), modality, idd, f"{cov:.4f}"))
        if i % 100 == 0:
            print(f"  {i}/{len(files)}")

    with open(os.path.join(args.out, "mask_coverage.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "modality", "identity", "strip_mask_cov"])
        w.writerows(rows)

    print("\n=== strip-space iris coverage (fraction of 64x512 that is valid iris) ===")
    for mod in ("VIS", "NIR"):
        v = np.array(per_mod[mod])
        if v.size == 0:
            continue
        print(f"\n{mod}: n={v.size}  mean={v.mean():.3f}  median={np.median(v):.3f}  "
              f"p10={np.percentile(v,10):.3f}  p90={np.percentile(v,90):.3f}")
        text_hist(v.tolist())

    print("\n=== images above coverage threshold ===")
    print(f"    {'thr':>5}  {'VIS':>12}  {'NIR':>12}")
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        vis_n = int((np.array(per_mod['VIS']) >= t).sum()) if per_mod['VIS'] else 0
        nir_n = int((np.array(per_mod['NIR']) >= t).sum()) if per_mod['NIR'] else 0
        vt = f"{vis_n}/{len(per_mod['VIS'])}" if per_mod['VIS'] else "-"
        nt = f"{nir_n}/{len(per_mod['NIR'])}" if per_mod['NIR'] else "-"
        print(f"    {t:>5.1f}  {vt:>12}  {nt:>12}")

    print("\n=== USABLE IDENTITIES at threshold (>=1 VIS AND >=1 NIR above thr) ===")
    total = len(per_id)
    print(f"    total identities with both spectra attempted: {total}")
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        usable = sum(1 for d in per_id.values() if d["VIS"] >= t and d["NIR"] >= t)
        print(f"    thr {t:.1f}: {usable}/{total} usable identities")

    print(f"\nCSV written to {args.out}/mask_coverage.csv")
    print("Read: if usable-identity count stays high at thr~0.4-0.5, there IS clean iris ->")
    print("mask-as-channel + fine-tune is worth it. If it collapses, it's a capture-quality wall.")


if __name__ == "__main__":
    main()
