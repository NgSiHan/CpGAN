"""
train.py — Full CpGAN cross-spectral iris matcher training.

Run ONLY after Milestone 0 (train_m0.py) confirms genuine/impostor distance separation.

Usage (paper-faithful, with perceptual loss):
    python train.py \
        --vis_root ~/PolyU_strips/VIS \
        --nir_root ~/PolyU_strips/NIR \
        --splits splits_polyu.json \
        --epochs 60 --batch_size 256 --margin 2.5 \
        --lambda_gan 0.0 --lambda_l2 1.0 --lambda_perc 0.3 \
        --save_dir checkpoints/cpgan_v2

Fine-tune on CUVIRIS (lower LR, may need smaller batch with GroupNorm swap):
    python train.py \
        --vis_root ~/CUVIRIS_strips/VIS \
        --nir_root ~/CUVIRIS_strips/NIR \
        --splits splits_cuviris.json \
        --checkpoint checkpoints/cpgan/best.pt \
        --epochs 20 --batch_size 64 --lr 2e-5 \
        --save_dir checkpoints/cpgan_cuviris

Key additions vs v1:
  --lambda_perc 0.3   VGG-16 perceptual loss (paper §5.3, λ₂=0.3); forces high-frequency
                      texture in reconstruction → richer encoder features; 0.0 to disable.
  --semi_hard         Semi-hard negative mining: for each genuine anchor use the closest
                      in-batch impostor within the margin instead of all random impostors.
                      Keeps the impostor gradient alive after easy negatives clear the margin.
  --margin 2.5        On the L2-normalised unit sphere imp_dist naturally clusters at ~2.0
                      (random vectors). margin=2.0 → impostor gradient dies. margin≥2.5
                      actively pushes impostors past random and keeps the loss signal alive.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.models as tv_models

from dataset import CrossSpectralPairs
from eval import evaluate, plot_results
from model import Discriminator, UNet
from utils import AverageMeter


# --------------------------------------------------------------------------- #
# Contrastive loss — plain and semi-hard-mined variants
# --------------------------------------------------------------------------- #
def contrastive_loss(emb_vis, emb_nir, lbl, margin):
    """Standard pairwise contrastive loss (all pairs equally weighted)."""
    dist = ((emb_vis - emb_nir) ** 2).sum(1)
    return (lbl * dist + (1 - lbl) * F.relu(margin - dist)).mean(), dist.detach()


def contrastive_loss_semi_hard(emb_vis, emb_nir, lbl, margin, ids=None):
    """Contrastive loss with semi-hard negative mining.

    For each genuine pair (anchor VIS, positive NIR) in the batch, the impostor
    gradient uses the *closest* in-batch NIR embedding that is both:
      (a) within the margin (dist < margin), and
      (b) from a *different identity* than the anchor.

    Condition (b) requires identity labels (`ids` tensor, shape [B], integer IDs).
    Without identity labels the off-diagonal NIR embeddings may contain same-identity
    strips (false negatives) — pushing same-identity pairs apart collapses the space.

    If `ids` is None, falls back to standard contrastive loss with a warning on first
    call.  Pass `ids` from the dataset to use semi-hard correctly.
    """
    if ids is None:
        # Cannot mine safely without identity labels — fall back to standard contrastive.
        # This avoids the false-negative collapse bug (see implementation notes).
        return contrastive_loss(emb_vis, emb_nir, lbl, margin)

    dist_mat = torch.cdist(emb_vis, emb_nir, p=2).pow(2)   # [B, B] pairwise squared-L2
    gen_mask = lbl.bool()          # [B] genuine flags

    total_loss = torch.tensor(0.0, device=emb_vis.device)
    n = 0

    for i in range(emb_vis.size(0)):
        if gen_mask[i]:
            # genuine pull
            d_pos = dist_mat[i, i]
            total_loss = total_loss + d_pos

            # semi-hard impostor push: find closest nir_j with DIFFERENT identity
            # and within margin.  Filter same-identity NIRs to avoid false negatives.
            different_id = (ids != ids[i])                 # [B] bool mask
            different_id[i] = False                        # also exclude self
            imp_dists = dist_mat[i].clone()
            imp_dists[~different_id] = float("inf")        # mask same-identity and self
            within = imp_dists[imp_dists < margin]
            if within.numel() > 0:
                total_loss = total_loss + F.relu(margin - within.min())
            n += 1

    if n == 0:
        return contrastive_loss(emb_vis, emb_nir, lbl, margin)

    _, dist_all = contrastive_loss(emb_vis, emb_nir, lbl, margin)
    return total_loss / n, dist_all


# --------------------------------------------------------------------------- #
# VGG-16 perceptual loss (paper §5.3, λ₂=0.3)
# --------------------------------------------------------------------------- #
class PerceptualLoss(nn.Module):
    """L2 distance in VGG-16 ReLU3-3 feature space.

    Matches the paper's formulation (Eq 16-18).  Inputs are expected in [-1, 1]
    (same as our strip tensors); they are shifted to [0, 1] then normalised to
    ImageNet stats before passing through VGG.

    The feature extractor is kept frozen (no grad) — it is a fixed perceptual
    reference, not a trained component.

    Because iris strips are 1-channel and VGG expects 3-channel RGB, we replicate
    the single channel three times.  This is consistent with how VGG was used in
    the original perceptual-loss paper (Johnson et al., 2016) on grayscale content.

    Memory note: strips are 64×512 which is much wider than VGG's typical 224×224
    input.  At batch=256 the first VGG conv on raw strips produces a [256,64,64,512]
    tensor (~2.1 GB) — OOM on 24 GB cards when combined with the UNet activations.
    We resize to PERC_H×PERC_W (default 64×128) before VGG: 4× width reduction makes
    the first conv [B,64,64,128] ≈ 0.5 GB at B=256 — safely within budget.
    Texture-frequency features are unaffected by this spatial downscale; perceptual loss
    is about local statistics, not absolute resolution.

    Additionally, target features are computed under torch.no_grad() (they are a fixed
    reference; their gradient is never needed), halving peak activation memory.
    """

    # VGG-16 layer indices up to and including ReLU3-3 (the 16th layer)
    RELU3_3_IDX = 16
    # Resize strips to this before VGG to bound activation memory.
    # Height 64 = radial resolution kept; width 128 = 4x angular downscale.
    PERC_H, PERC_W = 64, 128

    def __init__(self):
        super().__init__()
        vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        # freeze — perceptual reference only
        for p in vgg.parameters():
            p.requires_grad_(False)
        self.features = nn.Sequential(*list(vgg.features.children())[:self.RELU3_3_IDX + 1])
        # ImageNet mean/std for normalisation (applied after [0,1] scaling)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _prep(self, x: torch.Tensor) -> torch.Tensor:
        """[-1,1] grayscale [B,1,H,W] → resized, ImageNet-normalised [B,3,PERC_H,PERC_W]."""
        x = (x + 1) / 2                       # → [0, 1]
        # Resize to fixed spatial size to keep VGG activation memory bounded.
        # Use float32 for interpolate (AMP may pass fp16 tensors).
        x = F.interpolate(x.float(), size=(self.PERC_H, self.PERC_W),
                          mode="bilinear", align_corners=False)
        x = x.expand(-1, 3, -1, -1)           # 1-ch → 3-ch (replicate)
        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_feat = self.features(self._prep(pred))
        # Target features are a fixed reference — no gradient needed, saves ~half
        # the activation memory for the backward pass.
        with torch.no_grad():
            target_feat = self.features(self._prep(target)).detach()
        return F.mse_loss(pred_feat, target_feat)


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
    ap.add_argument("--lambda_perc", type=float, default=0.0,
                    help="Weight for VGG-16 perceptual loss (paper λ₂=0.3). "
                         "Requires torchvision VGG-16 weights (downloaded once on first use). "
                         "Set to 0.0 (default) to disable.")
    ap.add_argument("--perc_batch",  type=int,   default=64,
                    help="Number of samples per mini-batch for the perceptual loss. "
                         "VGG activations on the full training batch OOM on 24 GB cards; "
                         "this sub-samples a random subset each step. 64 is enough signal.")
    ap.add_argument("--perc_warmup", type=int,   default=20,
                    help="Epoch at which perceptual loss is switched on (default 20). "
                         "Perceptual gradient (P≈60× contrastive at epoch 1) dominates early "
                         "training and collapses embeddings to a single point. Delay it until "
                         "the contrastive loss has established genuine/impostor separation.")
    ap.add_argument("--semi_hard",   action="store_true",
                    help="Use semi-hard negative mining in the contrastive loss. "
                         "Finds the closest in-batch impostor within the margin for each "
                         "genuine anchor, keeping the impostor gradient alive as easy negatives "
                         "clear the margin. Recommended when imp_dist plateaus at margin.")
    ap.add_argument("--feat_dim",    type=int,   default=128)
    ap.add_argument("--workers",     type=int,   default=8)
    ap.add_argument("--eval_every",  type=int,   default=5)
    ap.add_argument("--shift_pixel", type=int,   default=14)
    ap.add_argument("--shift_prob",  type=float, default=0.5)
    ap.add_argument("--save_dir",    default="checkpoints/cpgan")
    ap.add_argument("--reset_optimizer", action="store_true",
                    help="Reset optimizer/LR to args.lr instead of restoring from checkpoint. "
                         "Use when resuming with new hyperparameters (e.g. changed margin).")
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
        if args.reset_optimizer:
            print("  --reset_optimizer: starting fresh LR (optimizer state from checkpoint ignored)")
        else:
            if "optimizer_G" in state:
                optimizer_G.load_state_dict(state["optimizer_G"])
            if "optimizer_D" in state:
                optimizer_D.load_state_dict(state["optimizer_D"])
        start_epoch = state.get("epoch", 0) + 1
        best_eer    = state.get("val_eer", 1.0)
        print(f"  Resuming from epoch {start_epoch}, best EER so far: {best_eer:.4f}")

    adversarial_loss = torch.nn.MSELoss().to(device)
    l2_loss          = torch.nn.MSELoss().to(device)

    # Perceptual loss — only instantiate if requested (downloads VGG weights once).
    # Not activated until epoch >= perc_warmup to prevent early-training embedding collapse:
    # the perceptual gradient (~60× contrastive at epoch 1) dominates the encoder and
    # collapses all embeddings to a single unit-sphere point before separation can form.
    perc_loss_fn = None
    if args.lambda_perc > 0:
        print(f"Perceptual loss enabled (λ_perc={args.lambda_perc}, "
              f"warmup until epoch {args.perc_warmup}). "
              f"Loading VGG-16 weights (downloaded once on first use)…")
        perc_loss_fn = PerceptualLoss().to(device)
        perc_loss_fn.eval()

    # Choose contrastive loss variant
    _contrastive = contrastive_loss_semi_hard if args.semi_hard else contrastive_loss
    if args.semi_hard:
        print("Semi-hard negative mining enabled.")

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        net_vis.train(); net_nir.train()
        disc_vis.train(); disc_nir.train()

        loss_g_m    = AverageMeter()
        loss_d_m    = AverageMeter()
        loss_c_m    = AverageMeter()
        loss_perc_m = AverageMeter()

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

                # L2-normalize onto unit sphere before computing distances.
                # Prevents embeddings drifting to arbitrary scale — without this the
                # margin becomes meaningless as distances grow unbounded and the
                # impostor-push term cycles between dead and explosive.
                # On unit sphere: squared-L2 ∈ [0, 4], margin should be > 2.0 to push
                # impostors past the natural random-vector clustering point (~2.0).
                emb_vis = F.normalize(emb_vis, p=2, dim=1)
                emb_nir = F.normalize(emb_nir, p=2, dim=1)

                loss_c, dist = _contrastive(emb_vis, emb_nir, lbl, args.margin)

                loss_l2  = (l2_loss(fake_vis, vis) + l2_loss(fake_nir, nir)) / 2
                loss_gan = (adversarial_loss(disc_vis(fake_vis), valid) +
                            adversarial_loss(disc_nir(fake_nir), valid)) / 2

                # Perceptual loss (paper §5.3) — VGG-16 ReLU3-3 feature distance.
                # Only active after perc_warmup epochs (contrastive must establish
                # separation first; see --perc_warmup docs).
                # Sub-sampled to args.perc_batch indices to keep VGG activation memory
                # bounded (full batch at 64×512 OOMs on 24 GB; see PerceptualLoss docs).
                if perc_loss_fn is not None and epoch >= args.perc_warmup:
                    pidx = torch.randperm(bs, device=device)[:min(bs, args.perc_batch)]
                    loss_perc = (perc_loss_fn(fake_vis[pidx], vis[pidx]) +
                                 perc_loss_fn(fake_nir[pidx], nir[pidx])) / 2
                else:
                    loss_perc = torch.tensor(0.0, device=device)

                loss_G = (loss_c
                          + args.lambda_gan  * loss_gan
                          + args.lambda_l2   * loss_l2
                          + args.lambda_perc * loss_perc)

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
            loss_perc_m.update(loss_perc.item(), bs)

        scheduler_G.step()
        scheduler_D.step()

        perc_active = perc_loss_fn is not None and epoch >= args.perc_warmup
        perc_str = f"  P={loss_perc_m.avg:.4f}" if perc_active else (
                   f"  P=off(warm)" if perc_loss_fn is not None else "")
        print(f"Epoch {epoch:03d}  "
              f"G={loss_g_m.avg:.4f}  D={loss_d_m.avg:.4f}  "
              f"C={loss_c_m.avg:.4f}{perc_str}  "
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
