"""
train_m0.py — Milestone 0: contrastive-only Siamese gate check.

Two IrisEncoder networks (one per spectrum), no decoder, no GAN.
Run this FIRST on PolyU before adding CpGAN complexity. Gate: genuine-mean distance
must be clearly below impostor-mean on the val split before proceeding to train.py.

Usage:
    python train_m0.py \
        --vis_root ~/PolyU_strips/VIS \
        --nir_root ~/PolyU_strips/NIR \
        --splits splits_polyu.json \
        --epochs 10 --batch_size 256 --margin 2.0 \
        --save_dir checkpoints/m0

If genuine and impostor distances don't separate after 10 epochs, the problem is in
data / normalization / pairing — fix those before adding GAN complexity.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CrossSpectralPairs
from eval import evaluate, plot_results
from model import IrisEncoder, ResNetIrisEncoder
from utils import AverageMeter


def contrastive_loss(emb_vis, emb_nir, lbl, margin):
    """lbl=1 genuine (pull together), lbl=0 impostor (push apart)."""
    dist = ((emb_vis - emb_nir) ** 2).sum(1)
    loss = lbl * dist + (1 - lbl) * F.relu(margin - dist)
    return loss.mean(), dist.detach()


def main():
    ap = argparse.ArgumentParser(description="Milestone-0 contrastive Siamese")
    ap.add_argument("--vis_root",   required=True)
    ap.add_argument("--nir_root",   required=True)
    ap.add_argument("--splits",     required=True, help="JSON from make_splits.py")
    ap.add_argument("--epochs",     type=int,   default=10)
    ap.add_argument("--batch_size", type=int,   default=256)
    ap.add_argument("--margin",     type=float, default=2.0)
    ap.add_argument("--lr",         type=float, default=2e-4)
    ap.add_argument("--feat_dim",    type=int,   default=128)
    ap.add_argument("--model_type",  default="conv", choices=["conv", "resnet"],
                    help="conv: original IrisEncoder (6-conv, 128-d default). "
                         "resnet: ResNetIrisEncoder (ResNet-18, 512-d default). "
                         "Pass --feat_dim 512 --lr 1e-4 with resnet.")
    ap.add_argument("--workers",    type=int,   default=8)
    ap.add_argument("--eval_every", type=int,   default=2, help="eval on val every N epochs")
    ap.add_argument("--save_dir",   default="checkpoints/m0")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available — check NVIDIA driver / torch build"
    device = torch.device("cuda:0")
    print(f"Training on: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.splits) as f:
        splits = json.load(f)

    train_ds = CrossSpectralPairs(
        args.vis_root, args.nir_root, splits["train"], train=True)
    val_ids = splits["val"]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.model_type == "resnet":
        net_vis = ResNetIrisEncoder(feat_dim=args.feat_dim).to(device)
        net_nir = ResNetIrisEncoder(feat_dim=args.feat_dim).to(device)
        print(f"Using ResNetIrisEncoder (feat_dim={args.feat_dim})")
    else:
        net_vis = IrisEncoder(feat_dim=args.feat_dim).to(device)
        net_nir = IrisEncoder(feat_dim=args.feat_dim).to(device)
        print(f"Using IrisEncoder (feat_dim={args.feat_dim})")

    optimizer = torch.optim.Adam(
        list(net_vis.parameters()) + list(net_nir.parameters()),
        lr=args.lr, betas=(0.5, 0.999))

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    best_eer = 1.0
    margin_t = torch.tensor(args.margin, device=device)

    for epoch in range(1, args.epochs + 1):
        net_vis.train()
        net_nir.train()

        loss_m   = AverageMeter()
        gen_dist = AverageMeter()
        imp_dist = AverageMeter()

        for vis, nir, lbl, *_ in train_loader:
            vis, nir, lbl = vis.to(device), nir.to(device), lbl.to(device)

            emb_vis = F.normalize(net_vis(vis), p=2, dim=1)
            emb_nir = F.normalize(net_nir(nir), p=2, dim=1)

            loss, dist = contrastive_loss(emb_vis, emb_nir, lbl, margin_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = vis.size(0)
            loss_m.update(loss.item(), bs)
            genuine_mask = lbl.bool()
            if genuine_mask.any():
                gen_dist.update(dist[genuine_mask].mean().item(), genuine_mask.sum().item())
            if (~genuine_mask).any():
                imp_dist.update(dist[~genuine_mask].mean().item(), (~genuine_mask).sum().item())

        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"loss={loss_m.avg:.4f}  "
              f"gen_dist={gen_dist.avg:.3f}  imp_dist={imp_dist.avg:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.eval_every == 0:
            results = evaluate(
                net_vis, net_nir, args.vis_root, args.nir_root,
                val_ids, device)
            eer = results["eer"]
            gd  = results["distances"][results["labels"] == 1].mean()
            id_ = results["distances"][results["labels"] == 0].mean()
            print(f"  [VAL] EER={eer:.4f}  gen_mean={gd:.3f}  imp_mean={id_:.3f}")

            if eer < best_eer:
                best_eer = eer
                ckpt_path = Path(args.save_dir) / "best.pt"
                torch.save({
                    "epoch":      epoch,
                    "net_vis":    net_vis.state_dict(),
                    "net_nir":    net_nir.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "val_eer":    eer,
                    "margin":     args.margin,
                    "feat_dim":   args.feat_dim,
                    "model_type": args.model_type,
                }, ckpt_path)
                print(f"  -> saved best checkpoint  (EER={best_eer:.4f})")

            # quick separation check
            if gd < id_:
                print("  [GATE] genuine-mean < impostor-mean: separation achieved")
            else:
                print("  [GATE] WARNING: genuine-mean >= impostor-mean — no separation yet")

    print(f"\nBest val EER: {best_eer:.4f}")
    print("If genuine-mean < impostor-mean consistently: GATE PASSED — proceed to train.py")
    print("Otherwise: debug data/normalization/pairing before adding GAN complexity.")


if __name__ == "__main__":
    main()
