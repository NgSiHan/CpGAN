"""
train.py — Full CpGAN cross-spectral iris matcher training.

Run ONLY after Milestone 0 (train_m0.py) confirms genuine/impostor distance separation.

Usage:
    python train.py \
        --vis_root ~/PolyU_strips/VIS \
        --nir_root ~/PolyU_strips/NIR \
        --splits splits_polyu.json \
        --epochs 50 --batch_size 256 --margin 2.0 \
        --lambda_gan 1.0 --lambda_l2 1.0 \
        --save_dir checkpoints/cpgan

Fine-tune on CUVIRIS (lower LR, may need smaller batch with GroupNorm swap):
    python train.py \
        --vis_root ~/CUVIRIS_strips/VIS \
        --nir_root ~/CUVIRIS_strips/NIR \
        --splits splits_cuviris.json \
        --checkpoint checkpoints/cpgan/best.pt \
        --epochs 20 --batch_size 64 --lr 2e-5 \
        --save_dir checkpoints/cpgan_cuviris
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import CrossSpectralPairs
from eval import evaluate, plot_results
from model import Discriminator, UNet
from utils import AverageMeter


def contrastive_loss(emb_vis, emb_nir, lbl, margin):
    dist = ((emb_vis - emb_nir) ** 2).sum(1)
    return (lbl * dist + (1 - lbl) * F.relu(margin - dist)).mean(), dist.detach()


def main():
    ap = argparse.ArgumentParser(description="CpGAN cross-spectral iris training")
    ap.add_argument("--vis_root",    required=True)
    ap.add_argument("--nir_root",    required=True)
    ap.add_argument("--splits",      required=True, help="JSON from make_splits.py")
    ap.add_argument("--checkpoint",  default=None,
                    help="Resume from or fine-tune from this checkpoint")
    ap.add_argument("--epochs",      type=int,   default=50)
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--margin",      type=float, default=2.0)
    ap.add_argument("--lr",          type=float, default=2e-4)
    ap.add_argument("--lambda_gan",  type=float, default=1.0)
    ap.add_argument("--lambda_l2",   type=float, default=1.0)
    ap.add_argument("--feat_dim",    type=int,   default=128)
    ap.add_argument("--workers",     type=int,   default=8)
    ap.add_argument("--eval_every",  type=int,   default=5)
    ap.add_argument("--shift_pixel", type=int,   default=14)
    ap.add_argument("--shift_prob",  type=float, default=0.5)
    ap.add_argument("--save_dir",    default="checkpoints/cpgan")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available — check NVIDIA driver / torch build"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0")
    print(f"Training on: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.splits) as f:
        splits = json.load(f)

    train_ds = CrossSpectralPairs(
        args.vis_root, args.nir_root, splits["train"], train=True,
        shift_pixel=args.shift_pixel, shift_prob=args.shift_prob)
    val_ids = splits["val"]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # --- models (single GPU, no DataParallel) ---
    net_vis  = UNet(feat_dim=args.feat_dim).to(device)
    net_nir  = UNet(feat_dim=args.feat_dim).to(device)
    disc_vis = Discriminator(in_channels=1).to(device)
    disc_nir = Discriminator(in_channels=1).to(device)

    optimizer_G = torch.optim.Adam(
        list(net_vis.parameters()) + list(net_nir.parameters()),
        lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(
        list(disc_vis.parameters()) + list(disc_nir.parameters()),
        lr=args.lr, betas=(0.5, 0.999))

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_G, T_max=args.epochs)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_D, T_max=args.epochs)

    start_epoch = 1
    best_eer = 1.0

    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        # load what's available — allows fine-tuning from an m0 encoder checkpoint too
        for key, net in [("net_vis", net_vis), ("net_nir", net_nir)]:
            if key in state:
                missing, unexpected = net.load_state_dict(state[key], strict=False)
                if missing:
                    print(f"  {key}: {len(missing)} missing keys (expected when loading encoder-only ckpt)")
        if "disc_vis" in state:
            disc_vis.load_state_dict(state["disc_vis"])
            disc_nir.load_state_dict(state["disc_nir"])
        if "optimizer_G" in state:
            optimizer_G.load_state_dict(state["optimizer_G"])
        if "optimizer_D" in state:
            optimizer_D.load_state_dict(state["optimizer_D"])
        start_epoch = state.get("epoch", 0) + 1
        best_eer    = state.get("val_eer", 1.0)
        print(f"  Resuming from epoch {start_epoch}, best EER so far: {best_eer:.4f}")

    adversarial_loss = torch.nn.MSELoss().to(device)
    l2_loss          = torch.nn.MSELoss().to(device)

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        net_vis.train(); net_nir.train()
        disc_vis.train(); disc_nir.train()

        loss_g_m = AverageMeter()
        loss_d_m = AverageMeter()
        loss_c_m = AverageMeter()

        for vis, nir, lbl in train_loader:
            bs = vis.size(0)
            vis, nir, lbl = vis.to(device), nir.to(device), lbl.to(device)

            valid = Variable(torch.ones(bs, 1, device=device),  requires_grad=False)
            fake  = Variable(torch.zeros(bs, 1, device=device), requires_grad=False)

            # ---- Generator step ----
            optimizer_G.zero_grad()
            with torch.autocast("cuda"):
                fake_vis, emb_vis = net_vis(vis)
                fake_nir, emb_nir = net_nir(nir)

                loss_c, dist = contrastive_loss(emb_vis, emb_nir, lbl, args.margin)

                loss_l2  = (l2_loss(fake_vis, vis) + l2_loss(fake_nir, nir)) / 2
                loss_gan = (adversarial_loss(disc_vis(fake_vis), valid) +
                            adversarial_loss(disc_nir(fake_nir), valid)) / 2

                loss_G = loss_c + args.lambda_gan * loss_gan + args.lambda_l2 * loss_l2

            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)

            # ---- Discriminator step ----
            optimizer_D.zero_grad()
            with torch.autocast("cuda"):
                d_loss = (adversarial_loss(disc_vis(vis), valid) +
                          adversarial_loss(disc_vis(fake_vis.detach()), fake) +
                          adversarial_loss(disc_nir(nir), valid) +
                          adversarial_loss(disc_nir(fake_nir.detach()), fake)) / 4

            scaler.scale(d_loss).backward()
            scaler.step(optimizer_D)
            scaler.update()

            loss_g_m.update(loss_G.item(), bs)
            loss_d_m.update(d_loss.item(), bs)
            loss_c_m.update(loss_c.item(), bs)

        scheduler_G.step()
        scheduler_D.step()

        print(f"Epoch {epoch:03d}  "
              f"G={loss_g_m.avg:.4f}  D={loss_d_m.avg:.4f}  "
              f"C={loss_c_m.avg:.4f}  "
              f"lr={scheduler_G.get_last_lr()[0]:.2e}")

        if epoch % args.eval_every == 0:
            results = evaluate(
                net_vis, net_nir, args.vis_root, args.nir_root,
                val_ids, device)
            eer = results["eer"]
            gd  = results["distances"][results["labels"] == 1].mean()
            id_ = results["distances"][results["labels"] == 0].mean()
            print(f"  [VAL] EER={eer:.4f}  "
                  f"GAR@1e-2={results['gar_1e-2']:.4f}  "
                  f"GAR@1e-3={results['gar_1e-3']:.4f}  "
                  f"gen={gd:.3f}  imp={id_:.3f}")

            if eer < best_eer:
                best_eer = eer
                ckpt_path = Path(args.save_dir) / "best.pt"
                torch.save({
                    "epoch":     epoch,
                    "net_vis":   net_vis.state_dict(),
                    "net_nir":   net_nir.state_dict(),
                    "disc_vis":  disc_vis.state_dict(),
                    "disc_nir":  disc_nir.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D": optimizer_D.state_dict(),
                    "val_eer":   eer,
                    "margin":    args.margin,
                    "feat_dim":  args.feat_dim,
                }, ckpt_path)
                print(f"  -> saved best checkpoint  (EER={best_eer:.4f})")

                plot_results(results, args.save_dir, tag=f"val_ep{epoch:03d}")

        # periodic checkpoint (every 10 epochs) to allow recovery
        if epoch % 10 == 0:
            torch.save({
                "epoch":     epoch,
                "net_vis":   net_vis.state_dict(),
                "net_nir":   net_nir.state_dict(),
                "disc_vis":  disc_vis.state_dict(),
                "disc_nir":  disc_nir.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "val_eer":   best_eer,
            }, Path(args.save_dir) / f"epoch_{epoch:03d}.pt")

    print(f"\nTraining complete. Best val EER: {best_eer:.4f}")
    print(f"Best checkpoint: {args.save_dir}/best.pt")
    print("Next: run eval.py --checkpoint best.pt --split test for final numbers.")


if __name__ == "__main__":
    main()
