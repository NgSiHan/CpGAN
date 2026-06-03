"""
eval.py — open-set EER evaluation for CpGAN cross-spectral iris verification.

Importable function:
    evaluate(net_vis, net_nir, vis_root, nir_root, ids, device) -> dict

CLI usage:
    python eval.py \
        --checkpoint checkpoints/cpgan/best.pt \
        --vis_root ~/PolyU_strips/VIS \
        --nir_root ~/PolyU_strips/NIR \
        --splits splits_polyu.json --split test \
        --out_dir eval_results/

Metrics reported: EER, GAR@FAR=1e-2, GAR@FAR=1e-3.
Outputs: genuine/impostor distance histogram + ROC curve PNG.
"""

import argparse
import glob
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_curve
import torchvision.transforms.functional as TF


# --------------------------------------------------------------------------- #
# Core metric
# --------------------------------------------------------------------------- #
def eer_from(distances: np.ndarray, labels: np.ndarray) -> float:
    """Compute Equal Error Rate. labels: 1=genuine, 0=impostor."""
    score = -np.asarray(distances)          # higher score = more genuine
    fpr, tpr, _ = roc_curve(labels, score)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))
    return float((fpr[i] + fnr[i]) / 2)


def gar_at_far(distances: np.ndarray, labels: np.ndarray, target_far: float) -> float:
    """Genuine Accept Rate at a given False Accept Rate."""
    score = -np.asarray(distances)
    fpr, tpr, _ = roc_curve(labels, score)
    # interpolate at target_far
    return float(np.interp(target_far, fpr, tpr))


# --------------------------------------------------------------------------- #
# Strip loader (no augmentation — deterministic embeddings)
# --------------------------------------------------------------------------- #
def _load_strip(path: str) -> torch.Tensor:
    img = Image.open(path).convert("L")
    t = TF.to_tensor(img)
    return TF.normalize(t, [0.5], [0.5])          # [1, 64, 512] in [-1, 1]


# --------------------------------------------------------------------------- #
# Evaluation function (importable from train_m0.py and train.py)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(net_vis, net_nir, vis_root: str, nir_root: str, ids: list,
             device: torch.device, n_impostor_multiplier: int = 5,
             batch_size: int = 128) -> dict:
    """Embed all strips, form genuine+impostor pairs, compute EER and GARs.

    Args:
        net_vis / net_nir: encoder models in eval mode (called as net(strip) -> emb).
        ids:               list of identity folder names to evaluate.
        n_impostor_multiplier: sample this many impostor pairs per genuine pair.

    Returns dict with keys: eer, gar_1e-2, gar_1e-3, distances, labels.
    """
    net_vis.eval()
    net_nir.eval()

    def embed_all(net, root, id_list):
        """Returns {identity: np.ndarray [N, feat_dim]}."""
        embs = {}
        for idd in id_list:
            paths = glob.glob(os.path.join(root, idd, "*.png"))
            if not paths:
                continue
            strips = torch.stack([_load_strip(p) for p in paths]).to(device)
            # process in sub-batches to avoid OOM on large identities
            parts = []
            for i in range(0, len(strips), batch_size):
                out = net(strips[i:i + batch_size])
                # UNet returns (reconstructed_image, embedding); IrisEncoder returns embedding only
                emb = out[1] if isinstance(out, tuple) else out
                # L2-normalize to match training (embeddings are unit-sphere vectors)
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                parts.append(emb.cpu().numpy())
            embs[idd] = np.concatenate(parts, axis=0)
        return embs

    vis_embs = embed_all(net_vis, vis_root, ids)
    nir_embs = embed_all(net_nir, nir_root, ids)

    shared_ids = [i for i in ids if i in vis_embs and i in nir_embs]

    distances = []
    labels = []

    # ALL genuine pairs: every (VIS_i, NIR_j) for each identity
    for idd in shared_ids:
        for v_emb in vis_embs[idd]:
            for n_emb in nir_embs[idd]:
                d = float(np.sum((v_emb - n_emb) ** 2))
                distances.append(d)
                labels.append(1)

    n_genuine = len(distances)

    # Random impostor pairs (different identity, VIS vs NIR)
    impostor_target = n_genuine * n_impostor_multiplier
    for _ in range(impostor_target):
        ida, idb = random.sample(shared_ids, 2)
        v_emb = random.choice(vis_embs[ida])
        n_emb = random.choice(nir_embs[idb])
        d = float(np.sum((v_emb - n_emb) ** 2))
        distances.append(d)
        labels.append(0)

    distances = np.array(distances, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    eer = eer_from(distances, labels)
    gar_1e2 = gar_at_far(distances, labels, 1e-2)
    gar_1e3 = gar_at_far(distances, labels, 1e-3)

    return {
        "eer": eer,
        "gar_1e-2": gar_1e2,
        "gar_1e-3": gar_1e3,
        "distances": distances,
        "labels": labels,
        "n_genuine": n_genuine,
        "n_impostor": impostor_target,
    }


def plot_results(results: dict, out_dir: str, tag: str = "eval"):
    """Save histogram and ROC PNG into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    dist = results["distances"]
    lbl  = results["labels"]
    gen_d = dist[lbl == 1]
    imp_d = dist[lbl == 0]

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, dist.max(), 80)
    ax.hist(gen_d, bins=bins, alpha=0.6, label="Genuine", color="steelblue", density=True)
    ax.hist(imp_d, bins=bins, alpha=0.6, label="Impostor", color="tomato", density=True)
    ax.axvline(gen_d.mean(), color="steelblue", linestyle="--", linewidth=1)
    ax.axvline(imp_d.mean(), color="tomato", linestyle="--", linewidth=1)
    ax.set_xlabel("Squared L2 distance")
    ax.set_ylabel("Density")
    ax.set_title(f"EER={results['eer']:.4f}  gen_mean={gen_d.mean():.2f}  imp_mean={imp_d.mean():.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_histogram.png"), dpi=120)
    plt.close(fig)

    # ROC
    score = -dist
    fpr, tpr, _ = roc_curve(lbl, score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    ax.set_xlabel("FAR")
    ax.set_ylabel("GAR")
    ax.set_title(f"ROC  EER={results['eer']:.4f}")
    ax.set_xscale("log")
    ax.set_xlim([1e-4, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{tag}_roc.png"), dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Evaluate CpGAN cross-spectral EER")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vis_root",   required=True)
    ap.add_argument("--nir_root",   required=True)
    ap.add_argument("--splits",     required=True, help="JSON from make_splits.py")
    ap.add_argument("--split",      default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_dir",    default="eval_results")
    ap.add_argument("--feat_dim",   type=int, default=128)
    ap.add_argument("--model_type", default="unet", choices=["unet", "encoder"])
    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(args.splits) as f:
        splits = json.load(f)
    ids = splits[args.split]
    print(f"Evaluating on {args.split} split: {len(ids)} identities")

    state = torch.load(args.checkpoint, map_location=device)

    if args.model_type == "encoder":
        from model import IrisEncoder
        net_vis = IrisEncoder(feat_dim=args.feat_dim).to(device)
        net_nir = IrisEncoder(feat_dim=args.feat_dim).to(device)
        net_vis.load_state_dict(state["net_vis"])
        net_nir.load_state_dict(state["net_nir"])
    else:
        from model import UNet
        net_vis = UNet(feat_dim=args.feat_dim).to(device)
        net_nir = UNet(feat_dim=args.feat_dim).to(device)
        net_vis.load_state_dict(state["net_vis"])
        net_nir.load_state_dict(state["net_nir"])

    results = evaluate(net_vis, net_nir, args.vis_root, args.nir_root, ids, device)

    print(f"\n--- Results ({args.split} split) ---")
    print(f"  EER          : {results['eer']:.4f}  ({results['eer']*100:.2f}%)")
    print(f"  GAR@FAR=1e-2 : {results['gar_1e-2']:.4f}")
    print(f"  GAR@FAR=1e-3 : {results['gar_1e-3']:.4f}")
    print(f"  Genuine pairs : {results['n_genuine']}")
    print(f"  Impostor pairs: {results['n_impostor']}")

    gen_d = results["distances"][results["labels"] == 1]
    imp_d = results["distances"][results["labels"] == 0]
    print(f"  Genuine dist  : mean={gen_d.mean():.3f}  std={gen_d.std():.3f}")
    print(f"  Impostor dist : mean={imp_d.mean():.3f}  std={imp_d.std():.3f}")

    tag = f"{args.split}_{Path(args.checkpoint).stem}"
    plot_results(results, args.out_dir, tag)
    print(f"\nPlots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
