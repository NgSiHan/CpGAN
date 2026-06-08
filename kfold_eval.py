"""
kfold_eval.py — k-fold cross-validation for CUVIRIS fine-tune + eval.

For each fold JSON in --kfold_dir:
  1. Fine-tune from --checkpoint (PolyU cpgan_paper best.pt) on that fold's train IDs.
  2. Evaluate on that fold's test IDs with gallery_fusion=mean.
  3. Collect EER, GAR@1e-2, GAR@1e-3.

Prints a per-fold table and final mean ± std.  EERs are written to
<save_root>/kfold_results.json for later reference.

Usage:
    python kfold_eval.py \\
        --kfold_dir splits_kfold_cuviris \\
        --vis_root ~/CUVIRIS_strips/VIS \\
        --nir_root ~/CUVIRIS_strips/NIR \\
        --checkpoint checkpoints/cpgan_paper/best.pt \\
        --save_root checkpoints/kfold_cuviris \\
        --epochs 30 --batch_size 64 --lr 2e-5

Tip: run in a tmux window — each fold takes a few minutes.
     The script is restartable: if fold_XX/best.pt already exists it
     skips fine-tuning and goes straight to eval for that fold.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def run(cmd: list[str], tag: str) -> str:
    """Run a subprocess, stream stdout to console, and return full stdout."""
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"ERROR: command exited with code {result.returncode}", file=sys.stderr)
    return ""   # we streamed already; parse from a separate run if needed


def run_capture(cmd: list[str]) -> str:
    """Run a subprocess silently and return stdout."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout


def parse_eer(text: str) -> float | None:
    """Extract EER from eval.py stdout line: 'EER          : 0.3104  (31.04%)'"""
    m = re.search(r"EER\s*:\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


def parse_gar(text: str, label: str) -> float | None:
    """Extract GAR@FAR from eval.py stdout: 'GAR@FAR=1e-2 : 0.1234'"""
    m = re.search(rf"GAR@FAR={re.escape(label)}\s*:\s*([0-9.]+)", text)
    return float(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="k-fold fine-tune + eval for CUVIRIS cross-spectral iris")
    ap.add_argument("--kfold_dir",   required=True,
                    help="Directory of fold_XX.json files from make_kfold_splits.py")
    ap.add_argument("--vis_root",    required=True)
    ap.add_argument("--nir_root",    required=True)
    ap.add_argument("--checkpoint",  required=True,
                    help="Base checkpoint to fine-tune from (e.g. cpgan_paper/best.pt)")
    ap.add_argument("--save_root",   default="checkpoints/kfold_cuviris",
                    help="Root directory; each fold saved under <save_root>/fold_XX/")
    # fine-tune hyperparams (match what worked in the single fine-tune run)
    ap.add_argument("--epochs",       type=int,   default=30)
    ap.add_argument("--batch_size",   type=int,   default=64)
    ap.add_argument("--lr",           type=float, default=2e-5)
    ap.add_argument("--margin",       type=float, default=2.0)
    ap.add_argument("--lambda_gan",   type=float, default=0.3)
    ap.add_argument("--lambda_l2",    type=float, default=0.3)
    ap.add_argument("--lambda_perc",  type=float, default=0.1)
    ap.add_argument("--perc_warmup",  type=int,   default=5,
                    help="Shorter warmup than full training (30 epochs total)")
    ap.add_argument("--workers",      type=int,   default=4)
    ap.add_argument("--eval_every",   type=int,   default=5)
    ap.add_argument("--gallery_fusion", default="mean",
                    choices=["none", "mean"])
    # model type (match the checkpoint)
    ap.add_argument("--model_type",  default="unet",
                    choices=["unet", "encoder", "resnet"],
                    help="Must match the architecture of --checkpoint")
    ap.add_argument("--feat_dim",    type=int, default=128)
    args = ap.parse_args()

    kfold_dir  = Path(args.kfold_dir)
    save_root  = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    fold_files = sorted(kfold_dir.glob("fold_*.json"))
    if not fold_files:
        print(f"No fold_*.json files found in {kfold_dir}")
        sys.exit(1)

    print(f"Found {len(fold_files)} fold(s) in {kfold_dir}")
    print(f"Base checkpoint: {args.checkpoint}")
    print(f"Fine-tune: {args.epochs} epochs, lr={args.lr}, bs={args.batch_size}\n")

    results = []

    for fold_path in fold_files:
        fold_name = fold_path.stem          # e.g. "fold_00"
        fold_save = save_root / fold_name
        fold_save.mkdir(parents=True, exist_ok=True)
        best_pt   = fold_save / "best.pt"
        eval_dir  = fold_save / "eval"

        print(f"\n{'#'*60}")
        print(f"  {fold_name.upper()}  ({fold_path.name})")
        print(f"{'#'*60}")

        # ---- Fine-tune (skip if already done) ----
        if best_pt.exists():
            print(f"  {best_pt} already exists — skipping fine-tune")
        else:
            ft_cmd = [
                sys.executable, "train.py",
                "--vis_root",      args.vis_root,
                "--nir_root",      args.nir_root,
                "--splits",        str(fold_path),
                "--checkpoint",    args.checkpoint,
                "--reset_optimizer",
                "--independent_roll",
                "--epochs",        str(args.epochs),
                "--batch_size",    str(args.batch_size),
                "--lr",            str(args.lr),
                "--margin",        str(args.margin),
                "--lambda_gan",    str(args.lambda_gan),
                "--lambda_l2",     str(args.lambda_l2),
                "--lambda_perc",   str(args.lambda_perc),
                "--perc_warmup",   str(args.perc_warmup),
                "--workers",       str(args.workers),
                "--eval_every",    str(args.eval_every),
                "--save_dir",      str(fold_save),
            ]
            run(ft_cmd, f"Fine-tune {fold_name}")

        if not best_pt.exists():
            print(f"  ERROR: {best_pt} not found after fine-tune — skipping eval for this fold")
            results.append({
                "fold": fold_name,
                "eer": None, "gar_1e2": None, "gar_1e3": None,
                "error": "no checkpoint"
            })
            continue

        # ---- Eval on test split ----
        eval_cmd = [
            sys.executable, "eval.py",
            "--checkpoint",     str(best_pt),
            "--vis_root",       args.vis_root,
            "--nir_root",       args.nir_root,
            "--splits",         str(fold_path),
            "--split",          "test",
            "--model_type",     args.model_type,
            "--feat_dim",       str(args.feat_dim),
            "--gallery_fusion", args.gallery_fusion,
            "--out_dir",        str(eval_dir),
        ]
        print(f"\n  Running eval on test split...")
        out = run_capture(eval_cmd)
        print(out)

        eer    = parse_eer(out)
        gar_1e2 = parse_gar(out, "1e-2")
        gar_1e3 = parse_gar(out, "1e-3")

        if eer is None:
            print(f"  WARNING: could not parse EER from eval output for {fold_name}")

        results.append({
            "fold": fold_name,
            "eer": eer, "gar_1e2": gar_1e2, "gar_1e3": gar_1e3
        })
        print(f"  [{fold_name}] EER={eer:.4f}  GAR@1e-2={gar_1e2:.4f}  GAR@1e-3={gar_1e3:.4f}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  K-FOLD SUMMARY  ({len(results)} folds)")
    print(f"{'='*60}")
    print(f"  {'Fold':>8}  {'EER':>8}  {'GAR@1e-2':>10}  {'GAR@1e-3':>10}")
    print(f"  {'-'*42}")

    valid_eers  = []
    valid_gar12 = []
    valid_gar13 = []
    for r in results:
        eer_s = f"{r['eer']:.4f}" if r['eer'] is not None else "  ERROR"
        g12_s = f"{r['gar_1e2']:.4f}" if r['gar_1e2'] is not None else "  ERROR"
        g13_s = f"{r['gar_1e3']:.4f}" if r['gar_1e3'] is not None else "  ERROR"
        print(f"  {r['fold']:>8}  {eer_s:>8}  {g12_s:>10}  {g13_s:>10}")
        if r['eer'] is not None:
            valid_eers.append(r['eer'])
        if r['gar_1e2'] is not None:
            valid_gar12.append(r['gar_1e2'])
        if r['gar_1e3'] is not None:
            valid_gar13.append(r['gar_1e3'])

    if valid_eers:
        import statistics
        mean_eer = statistics.mean(valid_eers)
        std_eer  = statistics.stdev(valid_eers) if len(valid_eers) > 1 else 0.0
        mean_g12 = statistics.mean(valid_gar12) if valid_gar12 else float("nan")
        mean_g13 = statistics.mean(valid_gar13) if valid_gar13 else float("nan")

        print(f"  {'-'*42}")
        print(f"  {'mean':>8}  {mean_eer:.4f}    {mean_g12:.4f}      {mean_g13:.4f}")
        print(f"  {'std':>8}  {std_eer:.4f}")
        print(f"\n  Final: EER = {mean_eer*100:.2f}% ± {std_eer*100:.2f}%  "
              f"({len(valid_eers)}/{len(results)} folds)")

    # Save JSON results
    out_json = save_root / "kfold_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "folds": results,
            "mean_eer": mean_eer if valid_eers else None,
            "std_eer":  std_eer  if valid_eers else None,
            "mean_gar_1e2": mean_g12 if valid_gar12 else None,
            "mean_gar_1e3": mean_g13 if valid_gar13 else None,
        }, f, indent=2)
    print(f"\n  Results saved to {out_json}")


if __name__ == "__main__":
    main()
