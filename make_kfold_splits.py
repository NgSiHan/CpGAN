"""
make_kfold_splits.py — k-fold cross-validation splits for CUVIRIS (or any small dataset).

Discovers identities present in BOTH VIS and NIR, shuffles with a fixed seed,
then creates k fold JSONs.  Each fold holds out ~1/k of identities as test;
a small slice of the remaining identities becomes val (for checkpoint saving
during fine-tune); the rest is train.

Output:  <out_dir>/fold_00.json ... fold_{k-1:02d}.json
Each JSON:  {"train": [...], "val": [...], "test": [...]}

Val is intentionally small (~10% of non-test) — it's only used as an early-stopping
signal during fine-tune, not as a reporting metric.

Usage:
    python make_kfold_splits.py \\
        --strips_root ~/CUVIRIS_strips \\
        --k 5 --seed 42 \\
        --out_dir splits_kfold_cuviris
"""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="k-fold splits -> one JSON per fold")
    ap.add_argument("--strips_root", required=True,
                    help="Root containing VIS/ and NIR/ subdirectories of identity folders")
    ap.add_argument("--k",       type=int, default=5,    help="Number of folds")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.10,
                    help="Fraction of NON-test identities to use as val for "
                         "fine-tune early-stopping (default 0.10 ~= 1-2 val IDs per fold)")
    ap.add_argument("--out_dir", required=True, help="Directory to write fold JSON files")
    args = ap.parse_args()

    root    = Path(args.strips_root)
    vis_dir = root / "VIS"
    nir_dir = root / "NIR"

    if not vis_dir.exists() or not nir_dir.exists():
        raise FileNotFoundError(
            f"Expected VIS/ and NIR/ under {root}. Run prepare_strips.py first.")

    vis_ids = {p.name for p in vis_dir.iterdir() if p.is_dir()}
    nir_ids = {p.name for p in nir_dir.iterdir() if p.is_dir()}
    both    = sorted(vis_ids & nir_ids)

    print(f"VIS-only: {len(vis_ids - nir_ids)}  "
          f"NIR-only: {len(nir_ids - vis_ids)}  "
          f"Both: {len(both)}")

    if not both:
        raise RuntimeError("No identities with both VIS and NIR strips found.")

    n = len(both)
    if n < args.k * 2:
        raise RuntimeError(
            f"Only {n} identities — cannot make {args.k} folds with "
            f"at least 2 test IDs each. Reduce --k.")

    rng = random.Random(args.seed)
    shuffled = list(both)
    rng.shuffle(shuffled)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build fold boundaries (round-robin assignment)
    fold_sizes = [n // args.k + (1 if i < n % args.k else 0) for i in range(args.k)]
    starts = [sum(fold_sizes[:i]) for i in range(args.k)]

    print(f"\n{args.k}-fold split (n={n}, seed={args.seed}, val_frac={args.val_frac}):")
    print(f"{'Fold':>5}  {'Test':>6}  {'Val':>5}  {'Train':>6}  Test IDs")
    print("-" * 60)

    for fold in range(args.k):
        s, sz = starts[fold], fold_sizes[fold]
        test  = shuffled[s : s + sz]
        rest  = shuffled[:s] + shuffled[s + sz:]

        n_val = max(1, round(len(rest) * args.val_frac))
        val   = rest[:n_val]
        train = rest[n_val:]

        fold_data = {"train": train, "val": val, "test": test}
        out_path  = out_dir / f"fold_{fold:02d}.json"
        with open(out_path, "w") as f:
            json.dump(fold_data, f, indent=2)

        print(f"  {fold:3d}    {len(test):4d}   {len(val):4d}   {len(train):5d}   "
              f"{', '.join(test[:3])}{'…' if len(test) > 3 else ''}")

    print(f"\nWrote {args.k} fold JSONs to {out_dir}/")
    print("Next: python kfold_eval.py --kfold_dir <out_dir> ...")


if __name__ == "__main__":
    main()
