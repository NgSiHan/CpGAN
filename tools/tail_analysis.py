"""
tail_analysis.py — find the genuine-pair tail that's holding EER up, and tie it to
capture-time image-quality metrics so we can decide whether an (identity-blind) quality
filter is justified.

IMPORTANT honesty rule:
  - Embedding distance is used here ONLY to SURFACE suspects (diagnostic).
  - Any actual deployment filter must use INTRINSIC quality (computable at capture without
    knowing identity): segmentation success, iris visibility, contrast, blur, specular.
  - We never delete "genuine pairs that didn't match" to lower EER — that is circular.

Outputs (under --out):
  worst_pairs.csv         rank, dist, vis_path, nir_path
  worst_vis_images.csv    rank, best_genuine_dist, n_tail_pairs, + quality metrics, path
  worst_nir_images.csv    same for NIR
  pair_<rank>_<dist>.png  montage (VIS strip / NIR strip) of the worst genuine pairs
  visimg_<rank>.png       montage of a bad VIS image vs its BEST same-identity NIR match
  console summary         Pareto (what % of images cause the tail) + quality correlation

Usage:
  python tools/tail_analysis.py \
      --checkpoint checkpoints/m0_cvrl_raw/best.pt --model_type encoder \
      --vis_root ~/PolyU_strips_cvrl_raw/VIS --nir_root ~/PolyU_strips_cvrl_raw/NIR \
      --splits splits_cvrl.json --split test --top 40 --out eval_results/tail
"""

import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path

# make repo root importable when run as `python tools/tail_analysis.py`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


# --------------------------------------------------------------------------- #
def load_strip_tensor(path):
    img = Image.open(path).convert("L")
    t = TF.to_tensor(img)
    return TF.normalize(t, [0.5], [0.5])          # [1,64,512] in [-1,1]


def quality_metrics(path, _cache={}):
    """Identity-blind, capture-time strip-quality signals (these could gate enrollment)."""
    if path in _cache:
        return _cache[path]
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    m = {
        "mean": float(g.mean()),                                  # near-black VIS -> low
        "contrast": float(g.std()),                               # flat/washed -> low
        "blur": float(cv2.Laplacian(g, cv2.CV_32F).var()),        # blurry -> low (less texture)
        "dark_frac": float((g < 10).mean()),                      # occlusion / black VIS -> high
        "sat_frac": float((g > 245).mean()),                      # specular -> high
    }
    _cache[path] = m
    return m


@torch.no_grad()
def embed_dir(net, root, ids, device, batch=128):
    """Return {identity: [(path, emb np[D]) ...]} with L2-normalized embeddings."""
    out = {}
    for idd in ids:
        paths = sorted(glob.glob(os.path.join(root, idd, "*.png")))
        if not paths:
            continue
        strips = torch.stack([load_strip_tensor(p) for p in paths]).to(device)
        embs = []
        for i in range(0, len(strips), batch):
            o = net(strips[i:i + batch])
            e = o[1] if isinstance(o, tuple) else o
            e = torch.nn.functional.normalize(e, p=2, dim=1)
            embs.append(e.cpu().numpy())
        embs = np.concatenate(embs, axis=0)
        out[idd] = list(zip(paths, embs))
    return out


def montage(vis_path, nir_path, label, scale=2):
    v = cv2.imread(vis_path, cv2.IMREAD_GRAYSCALE)
    n = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
    gap = np.zeros((4, v.shape[1]), np.uint8)
    stack = np.vstack([v, gap, n])
    stack = cv2.cvtColor(stack, cv2.COLOR_GRAY2BGR)
    stack = cv2.resize(stack, (stack.shape[1] * scale, stack.shape[0] * scale),
                       interpolation=cv2.INTER_NEAREST)
    cv2.putText(stack, "VIS", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(stack, "NIR", (6, (v.shape[0] + 4) * scale + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(stack, label, (6, stack.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return stack


def main():
    ap = argparse.ArgumentParser(description="Surface the genuine-pair tail + quality metrics")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--vis_root", required=True)
    ap.add_argument("--nir_root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--model_type", default="encoder", choices=["encoder", "unet"])
    ap.add_argument("--feat_dim", type=int, default=128)
    ap.add_argument("--top", type=int, default=40, help="how many worst pairs/images to dump as PNG")
    ap.add_argument("--out", default="eval_results/tail")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.splits) as f:
        ids = json.load(f)[args.split]

    state = torch.load(args.checkpoint, map_location=device)
    if args.model_type == "encoder":
        from model import IrisEncoder
        net_vis = IrisEncoder(feat_dim=args.feat_dim).to(device)
        net_nir = IrisEncoder(feat_dim=args.feat_dim).to(device)
    else:
        from model import UNet
        net_vis = UNet(feat_dim=args.feat_dim).to(device)
        net_nir = UNet(feat_dim=args.feat_dim).to(device)
    net_vis.load_state_dict(state["net_vis"]); net_vis.eval()
    net_nir.load_state_dict(state["net_nir"]); net_nir.eval()

    vis = embed_dir(net_vis, args.vis_root, ids, device)
    nir = embed_dir(net_nir, args.nir_root, ids, device)
    shared = [i for i in ids if i in vis and i in nir]
    print(f"{args.split}: {len(shared)} identities with both spectra")

    # ---- all genuine pairs ----
    pairs = []                                  # (dist, idd, vis_path, nir_path)
    vis_best = {}                               # vis_path -> best genuine dist
    vis_tailcount = {}                          # vis_path -> # tail pairs
    nir_best = {}
    nir_tailcount = {}
    for idd in shared:
        for vp, ve in vis[idd]:
            for npath, ne in nir[idd]:
                d = float(np.sum((ve - ne) ** 2))
                pairs.append((d, idd, vp, npath))
                if d < vis_best.get(vp, 1e9): vis_best[vp] = d
                if d < nir_best.get(npath, 1e9): nir_best[npath] = d

    dists = np.array([p[0] for p in pairs], dtype=np.float32)
    # operating threshold ~ genuine median + impostor floor midpoint; use a robust proxy:
    thr = float(np.quantile(dists, 0.85))       # "tail" = worst 15% of genuine distances
    for d, idd, vp, npath in pairs:
        if d > thr:
            vis_tailcount[vp] = vis_tailcount.get(vp, 0) + 1
            nir_tailcount[npath] = nir_tailcount.get(npath, 0) + 1

    print(f"genuine pairs: {len(pairs)}   mean={dists.mean():.3f}   "
          f"tail threshold (85th pct)={thr:.3f}")

    # ---- worst PAIRS ----
    pairs.sort(key=lambda x: -x[0])
    with open(os.path.join(args.out, "worst_pairs.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["rank", "dist", "identity", "vis_path", "nir_path"])
        for r, (d, idd, vp, npath) in enumerate(pairs, 1):
            w.writerow([r, f"{d:.4f}", idd, vp, npath])
    for r, (d, idd, vp, npath) in enumerate(pairs[:args.top], 1):
        cv2.imwrite(os.path.join(args.out, f"pair_{r:03d}_d{d:.2f}.png"),
                    montage(vp, npath, f"{idd}  dist={d:.3f}"))

    # ---- worst IMAGES (per-image best-genuine-distance) + quality ----
    vis_by_path = {p: e for idd in shared for p, e in vis[idd]}
    nir_by_path = {p: e for idd in shared for p, e in nir[idd]}

    def best_opp_match(emb, opp_by_id, idd):
        """Nearest same-identity opposite-modality strip: (path, emb) or None."""
        cands = opp_by_id.get(idd, [])
        if not cands:
            return None
        return min(cands, key=lambda pe: float(np.sum((emb - pe[1]) ** 2)))

    def dump_images(best, tailcount, self_by_path, opp_by_id, tag):
        rows = []
        for path, bd in best.items():
            idd = Path(path).parent.name
            rows.append((bd, tailcount.get(path, 0), idd, path, quality_metrics(path)))
        rows.sort(key=lambda x: -x[0])
        with open(os.path.join(args.out, f"worst_{tag}_images.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "best_genuine_dist", "n_tail_pairs", "identity",
                        "mean", "contrast", "blur", "dark_frac", "sat_frac", "path"])
            for r, (bd, tc, idd, path, q) in enumerate(rows, 1):
                w.writerow([r, f"{bd:.4f}", tc, idd, f"{q['mean']:.1f}", f"{q['contrast']:.1f}",
                            f"{q['blur']:.1f}", f"{q['dark_frac']:.3f}", f"{q['sat_frac']:.3f}", path])
        # montage: worst image vs its BEST same-id opposite-modality match
        for r, (bd, tc, idd, path, q) in enumerate(rows[:args.top], 1):
            opp = best_opp_match(self_by_path[path], opp_by_id, idd)
            if opp is None:
                continue
            lbl = f"{idd} bestGenDist={bd:.3f} dark={q['dark_frac']:.2f}"
            m = montage(path, opp[0], lbl) if tag == "vis" else montage(opp[0], path, lbl)
            cv2.imwrite(os.path.join(args.out, f"{tag}img_{r:03d}.png"), m)
        return rows

    vis_rows = dump_images(vis_best, vis_tailcount, vis_by_path, nir, "vis")
    nir_rows = dump_images(nir_best, nir_tailcount, nir_by_path, vis, "nir")

    # ---- Pareto: do a few images cause most of the tail? ----
    def pareto(tailcount, total_tail):
        if total_tail == 0:
            return
        counts = sorted(tailcount.values(), reverse=True)
        cum = np.cumsum(counts)
        n10 = max(1, len(counts) // 10)
        print(f"    top {n10} imgs ({100*n10/len(counts):.0f}% of imgs with tail) "
              f"account for {100*cum[n10-1]/total_tail:.0f}% of tail pairs")

    total_tail = sum(vis_tailcount.values())
    print("\n--- tail concentration ---")
    print(f"  VIS images implicated in tail: {len(vis_tailcount)}")
    pareto(vis_tailcount, total_tail)
    print(f"  NIR images implicated in tail: {len(nir_tailcount)}")
    pareto(nir_tailcount, sum(nir_tailcount.values()))

    # ---- quality correlation: are tail images intrinsically worse? ----
    def split_quality(rows):
        k = max(1, len(rows) // 5)
        worst = rows[:k]; best = rows[-k:]
        def avg(rs, key): return np.mean([r[4][key] for r in rs])
        print(f"    {'metric':10s} {'worst20%':>10s} {'best20%':>10s}")
        for key in ("mean", "contrast", "blur", "dark_frac", "sat_frac"):
            print(f"    {key:10s} {avg(worst,key):10.2f} {avg(best,key):10.2f}")

    print("\n--- VIS quality: tail (worst 20%) vs clean (best 20%) ---")
    split_quality(vis_rows)
    print("\n--- NIR quality: tail vs clean ---")
    split_quality(nir_rows)

    print(f"\nWrote CSVs + {args.top} pair/image montages to {args.out}/")
    print("Eyeball pair_*.png and visimg_*.png. If tail images are intrinsically bad")
    print("(low contrast / high dark_frac / low blur), an identity-blind quality gate is justified.")


if __name__ == "__main__":
    main()
