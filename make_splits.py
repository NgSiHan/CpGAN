"""
make_splits.py — disjoint identity split for CpGAN training.

Collects identities present in BOTH VIS and NIR under <strips_root>, shuffles with a
fixed seed, then splits by identity (no identity appears in two splits).

Usage:
    # PolyU: 70/15/15
    python make_splits.py --strips_root ~/PolyU_strips --out splits_polyu.json

    # CUVIRIS fine-tune set: keep small test split
    python make_splits.py --strips_root ~/CUVIRIS_strips --out splits_cuviris.json \
        --train_frac 0.55 --val_frac 0.15 --test_frac 0.30

Output JSON:  {"train": [...], "val": [...], "test": [...]}
Each element is an identity folder name (e.g. "001_L").
"""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Disjoint identity split -> JSON")
    ap.add_argument("--strips_root", required=True,
                    help="Root containing VIS/ and NIR/ subdirectories of identity folders")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac",   type=float, default=0.15)
    ap.add_argument("--test_frac",  type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert abs(args.train_frac + args.val_frac + args.test_frac - 1.0) < 1e-6, \
        "train_frac + val_frac + test_frac must sum to 1.0"

    root = Path(args.strips_root)
    vis_dir = root / "VIS"
    nir_dir = root / "NIR"

    if not vis_dir.exists() or not nir_dir.exists():
        raise FileNotFoundError(
            f"Expected VIS/ and NIR/ under {root}. "
            f"Run prepare_strips.py first."
        )

    vis_ids = {p.name for p in vis_dir.iterdir() if p.is_dir()}
    nir_ids = {p.name for p in nir_dir.iterdir() if p.is_dir()}
    both = sorted(vis_ids & nir_ids)

    if not both:
        raise RuntimeError("No identities found in both VIS and NIR. Check strips_root.")

    print(f"VIS-only: {len(vis_ids - nir_ids)}  NIR-only: {len(nir_ids - vis_ids)}  "
          f"Both: {len(both)}")

    rng = random.Random(args.seed)
    rng.shuffle(both)

    n = len(both)
    n_train = int(n * args.train_frac)
    n_val   = int(n * args.val_frac)

    train = both[:n_train]
    val   = both[n_train:n_train + n_val]
    test  = both[n_train + n_val:]

    for split_name, split_ids in [("train", train), ("val", val), ("test", test)]:
        print(f"  {split_name}: {len(split_ids)} identities")
        if len(split_ids) < 10:
            print(f"  WARNING: {split_name} has fewer than 10 identities — "
                  f"metrics on this split will be noisy.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f, indent=2)
    print(f"\nSaved splits to {out}")


if __name__ == "__main__":
    main()
