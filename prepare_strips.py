"""
prepare_strips.py  —  RAW eye images  ->  64x512 normalized iris strips (CpGAN-ready)

Written against open-iris 1.11.1 (current GitHub / PyPI). PIN this version on BOTH
the training box and the FastAPI server (`pip install open-iris==1.11.1`). The same
normalization geometry MUST be used at train and at deploy, or stored embeddings drift.

Pipeline per image:
    raw  --> (VIS: RED channel | NIR: grayscale)          # red penetrates melanin best
         --> open-iris segmentation + normalization         # via IRISPipeline
         --> NormalizedIris.normalized_image + mask          # pulled from call_trace
         --> resize to 64 (radial) x 512 (angular)           # bilinear img / nearest mask
         --> background subtraction (8x8 block mean) + CLAHE  # Nigam order, on the STRIP
         --> mask soft-fill (occluded -> iris-region mean)    # don't encode occlusion
         --> save PNG

Normalization logic lives in iris_norm.py, which is also imported by the FastAPI server.
This guarantees identical preprocessing at train and deploy.

Output layout (identity = subject + eye):
    <dst>/VIS/<subject>_<eye>/<instance>.png
    <dst>/NIR/<subject>_<eye>/<instance>.png

Filename parsing assumes the following directory layouts (both are path-based, not
filename-based):

  PolyU : PolyU_Cross_Iris/<subject 001-209>/<L|R>/<VIS|NIR>/001_L_NIR_1.tiff
  CUVIRIS: <nir|vis>/<left|right>/<subject 001-049>/sub001_JPG_left_02.jpg
           (.bmp extension = NIR for CUVIRIS)

Adjust parse_meta() if the actual folder layout differs from the above.

Usage (on the Linux training box, with the open-iris HF seg model already cached):
    IRIS_ENV=SERVER python prepare_strips.py --src ~/PolyU_raw   --dst ~/PolyU_strips
    IRIS_ENV=SERVER python prepare_strips.py --src ~/CUVIRIS_raw --dst ~/CUVIRIS_strips --dataset cuviris

Smoke-test a single image first and inspect the output before running the full batch.
Originals are never modified.
"""

import argparse
import sys
import traceback
from pathlib import Path

from iris_norm import STRIP_H, STRIP_W, normalize_strip, to_mono  # noqa: F401 (re-exported)

IMG_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp"}
EYE_MAP = {"L": "left", "R": "right", "left": "left", "right": "right"}


# --------------------------------------------------------------------------- #
# Filename -> (subject, eye, modality, eye_side)
# --------------------------------------------------------------------------- #
def parse_meta(path: Path, dataset: str):
    """Return (subject, eye_letter, modality, eye_side) or None if unparseable.

    Derived from the DIRECTORY STRUCTURE (authoritative for both datasets), with the
    file extension as a fallback for modality. Real layouts as of this handover —
    VERIFY against the actual folders in the dev directory before batch-running:

      PolyU :  PolyU_Cross_Iris/<subject 001-209>/<L|R>/<VIS|NIR>/001_L_NIR_1.tiff
      CUVIRIS: <nir|vis>/<left|right>/<subject 001-049>/sub001_JPG_left_02.jpg (.bmp = NIR)

    If a real folder differs, fix THIS function.
    """
    low = [p.lower() for p in path.parts]

    # --- modality: from a path component, else fall back to extension (CUVIRIS NIR = .bmp)
    if "vis" in low:
        modality = "VIS"
    elif "nir" in low:
        modality = "NIR"
    else:
        modality = "NIR" if path.suffix.lower() == ".bmp" else "VIS"

    # --- eye: 'left'/'right' (CUVIRIS) or single-letter 'l'/'r' (PolyU)
    if "left" in low or "l" in low:
        eye, side = "L", "left"
    elif "right" in low or "r" in low:
        eye, side = "R", "right"
    else:
        return None

    # --- subject: first pure-digit path component (001..209 / 001..049)
    subject = next((p for p in path.parts if p.isdigit()), None)
    if subject is None:
        return None

    return subject, eye, modality, side


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="RAW iris -> 64x512 normalized strips (open-iris)")
    ap.add_argument("--src", required=True, help="root folder of RAW images (searched recursively)")
    ap.add_argument("--dst", required=True, help="output root for strips")
    ap.add_argument("--dataset", default="polyu", choices=["polyu", "cuviris"])
    args = ap.parse_args()

    import iris  # noqa: E402  (lazy import; needs the cached HF segmentation model)

    pipeline = iris.IRISPipeline()      # default conf: LinearNormalization node named "normalization"

    src_root, dst_root = Path(args.src), Path(args.dst)
    files = [f for f in src_root.rglob("*") if f.suffix.lower() in IMG_EXTS]
    print(f"Found {len(files)} candidate images under {src_root}")

    ok = seg_fail = parse_fail = read_fail = 0
    for i, path in enumerate(sorted(files), 1):
        meta = parse_meta(path, args.dataset)
        if meta is None:
            parse_fail += 1
            continue
        subject, eye, modality, eye_side = meta
        try:
            mono = to_mono(path, modality)
        except IOError:
            read_fail += 1
            continue

        strip, mask, success = normalize_strip(mono, pipeline, eye_side)
        if not success:
            seg_fail += 1
            continue

        # save: <dst>/<MODALITY>/<subject>_<eye>/<stem>.png
        out_dir = dst_root / modality / f"{subject}_{eye}"
        out_dir.mkdir(parents=True, exist_ok=True)
        import cv2
        cv2.imwrite(str(out_dir / f"{path.stem}.png"), strip)
        ok += 1

        if i % 200 == 0:
            print(f"  {i}/{len(files)}  ok={ok} seg_fail={seg_fail}")

    print("\n--- summary ---")
    print(f"  ok          : {ok}")
    print(f"  seg failures: {seg_fail}   (expected higher on VIS than NIR)")
    print(f"  parse skips : {parse_fail}")
    print(f"  read errors : {read_fail}")
    for mod in ("VIS", "NIR"):
        d = dst_root / mod
        if d.exists():
            ids = [p for p in d.iterdir() if p.is_dir()]
            imgs = list(d.rglob("*.png"))
            print(f"  {mod}: {len(ids)} identities, {len(imgs)} strips")
    print("\nNOTE: an identity is usable only if it has BOTH a VIS and an NIR folder.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
