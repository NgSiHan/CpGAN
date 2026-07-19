"""
train_arcface.py — ResNet-18 + ArcFace cross-spectral iris training.

Loss:
    L = lambda_arc * L_arcface + lambda_contrast * L_contrastive

    L_arcface:    shared ArcFaceHead applied to BOTH VIS and NIR embeddings
                  with their respective identity labels.  The shared weight matrix
                  forces both spectra toward the same identity directions.
    L_contrastive: cross-modal pairwise contrastive on L2-normalised embeddings.
                  Auxiliary; explicitly pulls VIS-NIR genuine pairs together and
                  pushes impostors apart.  Keep even when ArcFace is on.

GAN and decoder are dropped (per ablation: minimal EER benefit, added complexity).

Usage (PolyU from scratch):
    python train_arcface.py \\
        --vis_root ~/PolyU_strips/VIS \\
        --nir_root ~/PolyU_strips/NIR \\
        --splits splits_polyu.json \\
        --epochs 40 --batch_size 256 \\
        --save_dir checkpoints/arcface_polyu

Fine-tune on CUVIRIS (drop ArcFace head, resume encoders only):
    python train_arcface.py \\
        --vis_root ~/CUVIRIS_strips/VIS \\
        --nir_root ~/CUVIRIS_strips/NIR \\
        --splits splits_cuviris.json \\
        --checkpoint checkpoints/arcface_polyu/best.pt \\
        --reset_optimizer --independent_roll \\
        --lambda_arc 0.0 --lambda_contrast 1.0 \\
        --lr 5e-5 --epochs 20 --batch_size 64 \\
        --save_dir checkpoints/arcface_cuviris
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
from model import ArcFaceHead, ResNetIrisEncoder
from utils import AverageMeter


# --------------------------------------------------------------------------- #
# Cross-modal contrastive loss (auxiliary)
# --------------------------------------------------------------------------- #
def contrastive_loss(emb_vis_n, emb_nir_n, lbl, margin):
    """Standard contrastive on L2-normalised embeddings.
    lbl=1.0 genuine (pull), lbl=0.0 impostor (push past margin).
    """
    dist = ((emb_vis_n - emb_nir_n) ** 2).sum(1)
    loss = lbl * dist + (1 - lbl) * F.relu(margin - dist)
    return loss.mean(), dist.detach()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="ResNet-18 + ArcFace cross-spectral iris verification")
    ap.add_argument("--vis_root",    required=True)
    ap.add_argument("--nir_root",    required=True)
    ap.add_argument("--splits",      required=True,
                    help="JSON from make_splits.py")
    ap.add_argument("--checkpoint",  default=None,
                    help="Resume from this checkpoint. Encoders are always loaded. "
                         "ArcFace head is restored only if n_cls matches AND "
                         "--lambda_arc > 0 (e.g. NOT for CUVIRIS fine-tune).")
    ap.add_argument("--epochs",      type=int,   default=40)
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--feat_dim",    type=int,   default=512)
    ap.add_argument("--shared_encoder", action="store_true",
                    help="Tie VIS and NIR into ONE encoder (true Siamese). Forces a single "
                         "shared embedding space by construction — the fix for ArcFace's "
                         "two-encoders-drift-apart failure (see HANDOVER §3). Both strips are "
                         "1-channel, so one trunk is valid.")
    ap.add_argument("--margin",      type=float, default=2.0,
                    help="Contrastive margin (unit-sphere squared-L2, max=4.0). "
                         "2.5 if impostor distances plateau below 2.0.")
    ap.add_argument("--arc_s",       type=float, default=64.0,
                    help="ArcFace scale factor")
    ap.add_argument("--arc_m",       type=float, default=0.5,
                    help="ArcFace angular margin in radians (~28.6 deg)")
    ap.add_argument("--lambda_arc",  type=float, default=1.0,
                    help="Weight for ArcFace loss. "
                         "Set 0.0 for CUVIRIS fine-tune (n_cls changes, head dropped).")
    ap.add_argument("--lambda_contrast", type=float, default=0.1,
                    help="Weight for cross-modal contrastive loss. "
                         "Keep > 0 to explicitly pull VIS-NIR embeddings together.")
    ap.add_argument("--workers",     type=int,   default=8)
    ap.add_argument("--eval_every",  type=int,   default=2)
    ap.add_argument("--shift_pixel", type=int,   default=14)
    ap.add_argument("--shift_prob",  type=float, default=0.5)
    ap.add_argument("--independent_roll", action="store_true",
                    help="Non-co-registered data (CUVIRIS): roll genuine pairs "
                         "independently instead of jointly.")
    ap.add_argument("--reset_optimizer", action="store_true",
                    help="Ignore optimizer state from checkpoint. Required when switching "
                         "datasets or loss configuration (e.g. CUVIRIS fine-tune). "
                         "Also resets start_epoch=1 and best_eer=1.0.")
    ap.add_argument("--save_dir",    default="checkpoints/arcface")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda:0")
    print(f"Training on: {torch.cuda.get_device_name(0)}")
    os.makedirs(args.save_dir, exist_ok=True)

    with open(args.splits) as f:
        splits = json.load(f)
    train_ids = splits["train"]
    val_ids   = splits["val"]

    train_ds = CrossSpectralPairs(
        args.vis_root, args.nir_root, train_ids, train=True,
        shift_pixel=args.shift_pixel, shift_prob=args.shift_prob,
        independent_roll=args.independent_roll)

    # n_cls MUST come from train_ds.ids (post VIS-intersect-NIR filter).
    # Using len(splits["train"]) would overcount if any identity lacks strips
    # in one spectrum, causing ArcFace scatter_ to throw an index-out-of-range.
    n_cls = len(train_ds.ids)
    print(f"Training identities (VIS intersect NIR): {n_cls}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    # --- Models ---
    net_vis  = ResNetIrisEncoder(feat_dim=args.feat_dim).to(device)
    if args.shared_encoder:
        net_nir = net_vis          # same object -> one shared space by construction
        print("Shared encoder: ONE ResNetIrisEncoder for both VIS and NIR (tied weights)")
    else:
        net_nir = ResNetIrisEncoder(feat_dim=args.feat_dim).to(device)
    arc_head = ArcFaceHead(feat_dim=args.feat_dim, n_cls=n_cls,
                           s=args.arc_s, m=args.arc_m).to(device)

    # --- Optimizer ---
    # When lambda_arc == 0 (CUVIRIS fine-tune) the ArcFace head receives no
    # gradient — exclude it from the optimizer to avoid weight decay on a dead head.
    # Shared encoder: net_nir IS net_vis, so listing both would double-count every
    # param (double-step in Adam). Use net_vis's params only when tied.
    if args.shared_encoder:
        enc_params = list(net_vis.parameters())
    else:
        enc_params = list(net_vis.parameters()) + list(net_nir.parameters())
    if args.lambda_arc > 0:
        opt_params = enc_params + list(arc_head.parameters())
    else:
        opt_params = enc_params

    optimizer = torch.optim.Adam(
        opt_params, lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    best_eer    = 1.0

    # --- Checkpoint loading ---
    if args.checkpoint is not None:
        print(f"Loading checkpoint: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)

        for key, net in [("net_vis", net_vis), ("net_nir", net_nir)]:
            if key in state:
                missing, unexpected = net.load_state_dict(state[key], strict=False)
                if missing:
                    print(f"  {key}: {len(missing)} missing key(s) "
                          f"(ok if loading from different arch)")
                if unexpected:
                    print(f"  {key}: {len(unexpected)} unexpected key(s) (ignored)")
            else:
                print(f"  WARNING: '{key}' not found in checkpoint")

        # Restore ArcFace head only if n_cls matches and we're actually using it.
        # On CUVIRIS fine-tune, n_cls changes (different #ids) so we always reinit.
        if "arc_head" in state and args.lambda_arc > 0:
            saved_n_cls = state["arc_head"]["weight"].shape[0]
            if saved_n_cls == n_cls:
                arc_head.load_state_dict(state["arc_head"])
                print(f"  arc_head restored (n_cls={n_cls})")
            else:
                print(f"  arc_head NOT restored: saved n_cls={saved_n_cls} != "
                      f"current n_cls={n_cls} — reinitialising head.")
        elif args.lambda_arc == 0:
            print("  arc_head skipped (lambda_arc=0.0)")

        if args.reset_optimizer:
            print("  --reset_optimizer: fresh optimizer state, start_epoch=1, best_eer=1.0")
            start_epoch = 1
            best_eer    = 1.0
        else:
            if "optimizer" in state:
                try:
                    optimizer.load_state_dict(state["optimizer"])
                except (ValueError, KeyError) as e:
                    print(f"  optimizer restore failed ({e}) — starting fresh")
            start_epoch = state.get("epoch", 0) + 1
            best_eer    = state.get("val_eer", 1.0)
        print(f"  Resuming from epoch {start_epoch}, best EER: {best_eer:.4f}")

    margin_t = torch.tensor(args.margin, device=device)

    print(f"\nConfig: feat_dim={args.feat_dim}  lambda_arc={args.lambda_arc}  "
          f"lambda_contrast={args.lambda_contrast}  margin={args.margin}")
    print(f"        arc_s={args.arc_s}  arc_m={args.arc_m}  "
          f"lr={args.lr}  epochs={args.epochs}  n_cls={n_cls}\n")

    # --------------------------------------------------------------------------- #
    # Training loop
    # --------------------------------------------------------------------------- #
    for epoch in range(start_epoch, start_epoch + args.epochs):
        net_vis.train()
        net_nir.train()
        if args.lambda_arc > 0:
            arc_head.train()
        else:
            arc_head.eval()  # keep in eval mode when not training

        loss_m     = AverageMeter()
        loss_arc_m = AverageMeter()
        loss_c_m   = AverageMeter()
        gen_dist_m = AverageMeter()
        imp_dist_m = AverageMeter()

        for batch in train_loader:
            # Dataset returns (vis, nir, lbl, vis_id_idx, nir_id_idx)
            vis, nir, lbl, vis_ids, nir_ids = batch
            vis     = vis.to(device)
            nir     = nir.to(device)
            lbl     = lbl.to(device)
            vis_ids = vis_ids.to(device)   # LongTensor [B]
            nir_ids = nir_ids.to(device)   # LongTensor [B]
            bs      = vis.size(0)

            optimizer.zero_grad()

            with torch.autocast("cuda"):
                # Raw (un-normalised) embeddings — ArcFaceHead normalises internally
                emb_vis_raw = net_vis(vis)   # [B, feat_dim]
                emb_nir_raw = net_nir(nir)   # [B, feat_dim]

                # --- ArcFace loss ---
                # ArcFaceHead.forward casts to float32 for trig stability.
                # If NaN appears in epoch 1, add: with torch.autocast("cuda", enabled=False)
                if args.lambda_arc > 0:
                    loss_arc = (arc_head(emb_vis_raw, vis_ids) +
                                arc_head(emb_nir_raw, nir_ids)) / 2
                else:
                    loss_arc = torch.tensor(0.0, device=device)

                # --- Cross-modal contrastive (auxiliary) ---
                if args.lambda_contrast > 0:
                    emb_vis_n = F.normalize(emb_vis_raw, p=2, dim=1)
                    emb_nir_n = F.normalize(emb_nir_raw, p=2, dim=1)
                    loss_c, dist = contrastive_loss(emb_vis_n, emb_nir_n, lbl, margin_t)
                else:
                    loss_c = torch.tensor(0.0, device=device)
                    dist   = torch.zeros(bs, device=device)

                loss = args.lambda_arc * loss_arc + args.lambda_contrast * loss_c

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_m.update(loss.item(), bs)
            loss_arc_m.update(loss_arc.item(), bs)
            loss_c_m.update(loss_c.item(), bs)

            genuine_mask = lbl.bool()
            if genuine_mask.any():
                gen_dist_m.update(dist[genuine_mask].mean().item(),
                                  genuine_mask.sum().item())
            if (~genuine_mask).any():
                imp_dist_m.update(dist[~genuine_mask].mean().item(),
                                  (~genuine_mask).sum().item())

        scheduler.step()

        print(f"Epoch {epoch:03d}/{start_epoch + args.epochs - 1:03d}  "
              f"loss={loss_m.avg:.4f}  arc={loss_arc_m.avg:.4f}  "
              f"contrast={loss_c_m.avg:.4f}  "
              f"gen={gen_dist_m.avg:.3f}  imp={imp_dist_m.avg:.3f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.eval_every == 0:
            results = evaluate(
                net_vis, net_nir,
                args.vis_root, args.nir_root,
                val_ids, device)
            eer = results["eer"]
            gd  = results["distances"][results["labels"] == 1].mean()
            id_ = results["distances"][results["labels"] == 0].mean()
            print(f"  [VAL] EER={eer:.4f}  "
                  f"GAR@1e-2={results['gar_1e-2']:.4f}  "
                  f"GAR@1e-3={results['gar_1e-3']:.4f}  "
                  f"gen={gd:.3f}  imp={id_:.3f}")

            if gd < id_:
                print("  [GATE] genuine-mean < impostor-mean: separation achieved")
            else:
                print("  [GATE] WARNING: no separation yet")

            if eer < best_eer:
                best_eer  = eer
                ckpt_path = Path(args.save_dir) / "best.pt"
                torch.save({
                    "epoch":      epoch,
                    "net_vis":    net_vis.state_dict(),
                    "net_nir":    net_nir.state_dict(),
                    "arc_head":   arc_head.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                    "val_eer":    eer,
                    "feat_dim":   args.feat_dim,
                    "n_cls":      n_cls,
                    "arc_s":      args.arc_s,
                    "arc_m":      args.arc_m,
                    "shared_encoder": args.shared_encoder,
                }, ckpt_path)
                print(f"  -> saved best checkpoint (EER={best_eer:.4f})")
                plot_results(results, args.save_dir, tag=f"val_ep{epoch:03d}")

        # Periodic save every 10 epochs (recovery from power outage etc.)
        if epoch % 10 == 0:
            torch.save({
                "epoch":    epoch,
                "net_vis":  net_vis.state_dict(),
                "net_nir":  net_nir.state_dict(),
                "arc_head": arc_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_eer":  best_eer,
                "feat_dim": args.feat_dim,
                "n_cls":    n_cls,
                "shared_encoder": args.shared_encoder,
            }, Path(args.save_dir) / f"epoch_{epoch:03d}.pt")

    print(f"\nTraining complete. Best val EER: {best_eer:.4f}")
    print("Next: python eval.py --checkpoint .../best.pt --model_type resnet "
          "--feat_dim 512 --gallery_fusion mean --split test")


if __name__ == "__main__":
    main()
