"""
pretrain_ubiris.py — VIS-only ArcFace pretraining of the ResNet iris encoder (HANDOVER Phase 2).

Goal: give a ResNet-18 encoder a strong iris prior from UBIRIS.v2's many VIS identities so it
stops overfitting the 292 PolyU identities (the Lever-1 failure). Single-modality classification:
each strip -> encoder -> ArcFaceHead -> cross-entropy over identity.

Then fine-tune cross-modal on PolyU with SEPARATE encoders (weight-tying was refuted, HANDOVER §3):
    python train_arcface.py \
        --vis_root ~/PolyU_strips/VIS --nir_root ~/PolyU_strips/NIR --splits splits_cvrl.json \
        --checkpoint checkpoints/ubiris_pretrain/best.pt --reset_optimizer \
        --lambda_arc 0.0 --lambda_contrast 1.0 --margin 2.0 --feat_dim 512 \
        --lr 5e-5 --epochs 30 --batch_size 256 --save_dir checkpoints/phase2_polyu
    # NOTE: no --shared_encoder. Pure contrastive fine-tune (lambda_arc 0) of the pretrained encoders.

Usage:
    python pretrain_ubiris.py --strips_root ~/UBIRIS_strips/VIS \
        --epochs 40 --batch_size 256 --save_dir checkpoints/ubiris_pretrain
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import ArcFaceHead, ResNetIrisEncoder
from utils import AverageMeter


def _gray_loader(path):
    return Image.open(path).convert("L")          # strips are single-channel 64x512


def main():
    ap = argparse.ArgumentParser(description="UBIRIS VIS-only ArcFace pretraining")
    ap.add_argument("--strips_root", required=True,
                    help="UBIRIS strips VIS dir; ImageFolder layout <root>/<id>/*.png")
    ap.add_argument("--epochs",      type=int,   default=40)
    ap.add_argument("--batch_size",  type=int,   default=256)
    ap.add_argument("--lr",          type=float, default=1e-4)
    ap.add_argument("--feat_dim",    type=int,   default=512)
    ap.add_argument("--arc_s",       type=float, default=64.0)
    ap.add_argument("--arc_m",       type=float, default=0.5)
    ap.add_argument("--roll_aug",    type=int,   default=32,
                    help="max random angular (width) roll in px for augmentation; 0 disables. "
                         "Iris strips are rotation-ambiguous — roll aug makes the prior shift-tolerant.")
    ap.add_argument("--workers",     type=int,   default=8)
    ap.add_argument("--save_dir",    default="checkpoints/ubiris_pretrain")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda:0")
    print(f"Training on: {torch.cuda.get_device_name(0)}")
    os.makedirs(args.save_dir, exist_ok=True)

    tf = transforms.Compose([
        transforms.ToTensor(),                    # [1, 64, 512] in [0,1]
        transforms.Normalize([0.5], [0.5]),       # -> [-1, 1], matches eval._load_strip
    ])
    ds = datasets.ImageFolder(args.strips_root, loader=_gray_loader, transform=tf)
    n_cls = len(ds.classes)
    print(f"UBIRIS pretrain: {len(ds)} strips over {n_cls} identities")
    assert n_cls > 1, "need >1 identity to train ArcFace"

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    net  = ResNetIrisEncoder(feat_dim=args.feat_dim).to(device)
    head = ArcFaceHead(feat_dim=args.feat_dim, n_cls=n_cls,
                       s=args.arc_s, m=args.arc_m).to(device)

    optimizer = torch.optim.Adam(
        list(net.parameters()) + list(head.parameters()),
        lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    def save(tag):
        # Save the encoder under BOTH net_vis/net_nir so train_arcface.py warm-starts
        # both spectra from the UBIRIS VIS prior (iris texture transfers as an init).
        sd = net.state_dict()
        torch.save({
            "epoch": tag, "net_vis": sd, "net_nir": sd,
            "arc_head": head.state_dict(), "feat_dim": args.feat_dim, "n_cls": n_cls,
        }, Path(args.save_dir) / ("best.pt" if tag == "best" else f"epoch_{tag:03d}.pt"))

    for epoch in range(1, args.epochs + 1):
        net.train(); head.train()
        loss_m, acc_m = AverageMeter(), AverageMeter()

        for img, label in loader:
            img, label = img.to(device), label.to(device)
            if args.roll_aug > 0:
                img = torch.roll(img, shifts=int(torch.randint(-args.roll_aug, args.roll_aug + 1, (1,))), dims=-1)

            optimizer.zero_grad()
            with torch.autocast("cuda"):
                emb = net(img)
                loss = head(emb, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

            # cheap train accuracy: argmax of cosine logits == label
            with torch.no_grad():
                cos = F.normalize(emb.float(), dim=1) @ F.normalize(head.weight.float(), dim=1).t()
                acc = (cos.argmax(1) == label).float().mean().item()
            loss_m.update(loss.item(), img.size(0))
            acc_m.update(acc, img.size(0))

        scheduler.step()
        print(f"Epoch {epoch:03d}/{args.epochs}  loss={loss_m.avg:.4f}  "
              f"train_acc={acc_m.avg:.3f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % 10 == 0 or epoch == args.epochs:
            save("best")   # no cross-modal val here; latest = best prior. Overwrites each time.
            print(f"  -> saved encoder prior (epoch {epoch})")

    print(f"\nPretrain complete. Encoder prior at {args.save_dir}/best.pt")
    print("Next: fine-tune on PolyU (see header — NO --shared_encoder, pure contrastive).")


if __name__ == "__main__":
    main()
