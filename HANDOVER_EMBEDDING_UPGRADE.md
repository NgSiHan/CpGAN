# Handover — CpGAN shared-embedding cross-spectral iris, next phase

> **STATUS (2026-07-19): architecture investigation COMPLETE — bottleneck is DATA, not model.**
> Phases 0–3 all ran. The tiny 6-conv contrastive champion (**PolyU test EER 0.1248**) is unbeaten by
> every architecture lever tried: ResNet+ArcFace (0.307), shared/tied encoder (0.385), UBIRIS-pretrained
> ResNet+contrastive (0.1443). **Weight-tying and big encoders do not help — §3's premise is refuted.**
> Real-world CUVIRIS 5-fold = **0.3172 ± 2.45%, GAR@1e-3≈0** (not deployable at strict FAR). The only
> lever left is MORE paired VIS-NIR data. Next experiment: add PolyU Session 2 identities (§5a) and test
> whether EER moves. STOP tuning architecture. Champion `best.pt` is the model to carry forward.

**Read this first. This repo is the `Coupled-GAN` (CpGAN) cross-spectral iris matcher.**
The goal: match a **VIS (visible, phone) probe** against an **NIR (near-infrared, scanner) gallery**
by embedding both spectra into one shared vector space and comparing by distance — **not** by
translating images and **not** using open-iris/Hamming. Matching is in embedding space.

You (the assistant reading this) only have this repo, not the wider machine. All external paths on
the user's Windows PC that you need are listed explicitly below.

---

## 0. TL;DR of where we are and what to do

- **Best result so far: EER 0.121 on PolyU** with the *tiny* 6-conv `IrisEncoder` + plain contrastive
  loss (`train_m0.py`). The full CpGAN (GAN + perceptual) tied at 0.1197 — **the GAN adds ~nothing.**
- **The encoder upgrade was already attempted and REGRESSED:** ResNet-18 + ArcFace scored **0.307**
  (worse). Root cause is diagnosed and fixable (see §3). Do **not** just "make the encoder bigger" and
  expect a win — the previous attempt already disproves that without the fix below.
- **The real deployment number is bad and data-bound:** CUVIRIS (real phone-VIS vs scanner-NIR),
  honest 5-fold CV = **EER 0.397 ± 3.5%, GAR@1e-3 = 0** (accepts ~no genuine users at a real threshold).
- **The plan:** (1) re-confirm the 0.121 baseline so the harness is trusted; (2) **Lever 1** — fix the
  cross-modal alignment bug that sank ArcFace (shared/tied encoder + strong cross-modal loss); (3)
  **Lever 2** — pretrain the VIS encoder on UBIRIS.v2 to fight data scarcity; (4) get the honest CUVIRIS
  k-fold number. Details in §3–§7.
- **User has NO real data yet** (and it'll only be ~30 people when they do — too few to train on, only
  fine-tune/eval). So we ride public datasets: **PolyU** (primary), **UBIRIS.v2** (VIS pretrain),
  **CUVIRIS** (realistic proxy).
- **Assume the Linux GPU box is wiped** — set up conda/tmux/PyTorch from scratch (§4). `SETUP.md` in this
  repo has the canonical setup guide; §4 is the delta + gotchas.

---

## 1. Why this approach (plain version — the user is re-learning ML)

Two ways to do VIS↔NIR matching:
- **Translate** a VIS image into a fake NIR image, then match (the *other* project, "IPGAN"). This hit a
  hard ceiling (~24–35% EER): generating an image that a fixed matcher finds *matchable* is very hard,
  and it needs the two spectra pixel-aligned.
- **Shared embedding (this repo):** learn a network that turns any strip — VIS or NIR — into a 128-d
  vector, trained so the **same eye's VIS and NIR vectors sit close** and **different eyes sit far**.
  Match = distance between vectors. **No image generated, no pixel registration needed.** This is how
  modern face recognition works. It's the more promising path and already beat every translation attempt.

**Key terms:**
- *Embedding* — the vector a strip is turned into.
- *Contrastive loss* — training signal that pulls genuine pairs together / pushes impostors apart (works
  on pairs).
- *ArcFace* — a stronger face-recognition loss that treats each identity as a class with a hard angular
  margin. Usually beats contrastive **when there's enough data.**
- *EER* — Equal Error Rate; lower is better; the single headline number.
- *GAR@FAR=1e-3* — fraction of genuine users accepted at a strict security threshold; the *deployment*
  metric. Ours is currently ~0 on real data = unusable, purely from data scarcity.

---

## 2. Full scoreboard (re-evaluation of what's been done — don't re-derive)

| Experiment | Dataset | EER | Takeaway |
|---|---|---|---|
| `IrisEncoder` (6-conv, 128-d) + contrastive (`train_m0.py`) | PolyU | **0.121** | **champion; still unbeaten** |
| Full CpGAN (+GAN +perceptual) (`train.py`) | PolyU | 0.1197 | GAN/decoder add ~nothing |
| PolyU-provided norm strips (unenh / enh) | PolyU | 0.197 / 0.199 | our CVRL segmentation beats PolyU's own strips |
| ResNet-18 encoder, plain M0 | PolyU | 0.246 | undertrained (wrong betas, too few epochs) |
| **ResNet-18 + ArcFace** (`train_arcface.py`) | PolyU | **0.307** | **regressed — alignment bug (§3)** |
| CpGAN zero-shot | CUVIRIS | 0.44 | pure sensor-domain gap |
| CpGAN fine-tuned, single split | CUVIRIS | 0.31 | lucky split |
| CpGAN fine-tuned, honest 5-fold | CUVIRIS | 0.397 ± 3.5% | superseded — see reconfirm below |
| **Shared/tied encoder (Lever 1)** | PolyU | **0.385** | 2026-07-19: tying HURTS — refutes §3 |
| **UBIRIS-pretrained ResNet + contrastive (Phase 2)** | PolyU | **0.1443** | 2026-07-19: still loses to champion 0.1248 |
| **CpGAN 5-fold, reconfirmed (rebuilt harness)** | CUVIRIS | **0.3172 ± 2.45%**, GAR@1e-3≈0 | 2026-07-19: current honest number |

**Two conclusions carried forward:**
1. **0.121 is a *data* ceiling on PolyU** (292 train identities), not obviously an architecture ceiling —
   but the one architecture attempt (ArcFace) failed for a *fixable* reason, so it's worth one clean shot.
2. **CUVIRIS is data-starved** (47 subj, ~2 NIR/eye). No architecture trick substitutes for more paired
   phone-VIS/scanner-NIR data. Manage expectations accordingly.

**Winning strip recipe (locked, don't re-litigate):** CVRL segmenter, **raw rubber-sheet, NO CLAHE, NO
soft-fill** (`--backend cvrl --no_enhance --no_soft_fill`). VIS → **red channel** only; NIR → grayscale.
64 (radial) × 512 (angular) strips. Both CLAHE and soft-fill were ablated and *hurt* cross-spectral
matching. Margin on the L2-normalized unit sphere must exceed 2.0 (impostors cluster at 2.0); use
**2.5** for full runs, 2.0 for the M0 gate.

---

## 3. Why ArcFace regressed — and the fix (this is Lever 1, the main modification)

> **REFUTED (2026-07-19).** Lever 1's central claim below — that tying the encoder weights is "the
> biggest lever" — was tested and is **wrong**. Weight-tying HURTS at both capacities:
> tiny-conv non-shared 0.135 vs shared 0.204; ResNet shared 0.385 vs two-encoder ArcFace 0.307.
> The shared space comes from the contrastive **loss**, not shared **weights** — VIS and NIR need
> different low-level filters. Keep **separate** encoders. `--shared_encoder` exists but is a dead end.
> The remaining real lever is Phase 2 (pretrain a *separate* VIS encoder on UBIRIS). Read §3 as history.

**What happened:** we use **two independent encoders** (one VIS, one NIR). ArcFace made *each* encoder
good at classifying *its own* spectrum, but the two networks drifted to **different regions of the 512-d
space** and both still scored well on ArcFace. The cross-modal glue (`--lambda_contrast 0.1`) was far
too weak to force the same person's VIS and NIR vectors to actually *meet*. Evidence: training genuine
distance fell to 1.13 but **validation genuine distance stayed at 1.88** — barely below impostors (2.0).
"Two good spaces that don't talk to each other."

**The fix — force one shared space. Modifications to make (in priority order):**

1. **Tie the encoder weights (biggest lever).** Use a **single** encoder that processes both VIS and NIR
   (true Siamese), instead of `net_vis` and `net_nir` being separate networks. One space by construction.
   - Add `--shared_encoder` to `train_arcface.py` (and `train_m0.py`): when set, instantiate **one**
     `ResNetIrisEncoder` and run both modalities through it. Save a flag in the checkpoint so `eval.py`
     rebuilds correctly.
   - Note: `ResNetIrisEncoder` already takes 1-channel input; both spectra are 1-channel strips, so a
     shared trunk is valid. (If you want a small modality-specific first conv but shared rest, that's a
     fallback — try fully-shared first.)
2. **Strong cross-modal supervision.** Raise `--lambda_contrast` from 0.1 to **~1.0**, keep ArcFace
   (`--lambda_arc 1.0`). Optionally add a **cross-modal center loss** (pull each identity's VIS and NIR
   embeddings toward a shared per-class center) — a ~15-line addition; only add if tied-weights +
   strong contrastive still under-aligns.
3. **Correct optimizer for ResNet.** `train_arcface.py` already uses `betas=(0.9, 0.999)`, wd 5e-4, 40
   epochs — good. Do NOT reuse `train_m0.py`'s GAN betas `(0.5, 0.999)` for ResNet (that caused the 0.246
   undertrain). If you gate ResNet through M0, patch its betas first, or just skip M0 for ResNet and let
   `train_arcface.py` be the test.

**Gate for Lever 1:** on PolyU, **beat 0.121**. If tied-weights + strong contrastive clears it, the
embedding architecture is alive → go to Lever 2. If it *still* can't beat 0.121, that's strong evidence
the ceiling is data, not model → stop tuning architecture, go to Lever 2/3 (data).

---

## 4. Linux setup from scratch (box is wiped)

`SETUP.md` in this repo is the canonical guide — follow its env/tmux/data sections. Delta + gotchas:

```bash
# 1. conda env
conda create -n iris-cpgan python=3.10 -y
conda activate iris-cpgan

# 2. PyTorch matching the box's CUDA (check `nvidia-smi` top-right). Most 4090 boxes: cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"  # expect True, RTX 4090

# 3. repo deps
cd ~/Coupled-GAN && pip install -r requirements.txt   # numpy pillow scikit-learn matplotlib tqdm opencv-python open-iris==1.11.1

# 4. tmux for long runs
tmux new -s cpgan      # detach Ctrl+B then D ; reattach: tmux attach -t cpgan
watch -n5 nvidia-smi   # in a second pane (Ctrl+B then %)
```

Gotchas (seen before): PyTorch 2.6 needs `weights_only=False` on `torch.load` (code already handles it);
if `onnxruntime`/`pycuda` complains, it's only for open-iris server mode — irrelevant here, ignore. CVRL
segmentation uses the two `.pth` in the repo root (they're **git-tracked**, so `git pull` brings them).

---

## 5. Data — exact locations on the user's Windows PC and how to move it

**Transfer rule:** code/docs via **git push/pull**; big binaries (datasets, `best.pt`) via **scp**.
Datasets are NOT in git. Segmentation weights ARE in git (repo root).

### 5a. PolyU — PRIMARY paired VIS/NIR dataset (the 0.121 source)
- **PC path (raw, Session 1):**
  `C:\dev\polyu_iris_database\PolyU_Cross_Submit\PolyU_Cross_Session_1\PolyU_Cross_Iris`
  - Layout (matches `prepare_strips.py --dataset polyu`): `<subj>/<L|R>/<VIS|NIR>/<subj>_<L|R>_<VIS|NIR>_<n>.tiff`
  - 209 subjects × 2 eyes ≈ 418 identities, ~15 instances each.
- **Session 2 also on PC** (extra identities — consider adding for more training data):
  `...\PolyU_Cross_Submit\PolyU_Cross_Session_2\PolyU_Cross_Iris_Session_2`
- **Do NOT use** the provided norm strips (`PolyU_Cross_Norm_*`) for training — they scored 0.197–0.199
  vs our 0.121. Always re-segment the **raw** `PolyU_Cross_Iris` with CVRL.
- **scp to Linux:**
  ```bash
  # from Windows (adjust user@host):
  scp -r "C:\dev\polyu_iris_database\PolyU_Cross_Submit\PolyU_Cross_Session_1\PolyU_Cross_Iris" user@LINUX:~/PolyU_raw
  ```

### 5b. UBIRIS.v2 — VIS-only, for VIS-encoder pretraining (Lever 2)
- **PC path:** `C:\dev\ubiris2_1\CLASSES_400_300_Part1\` (filenames `C<class>_S<sess>_I<img>.tiff`,
  ~261 classes). Second half is still zipped: `C:\dev\ubiris2_2.zip` → extract for the full ~500+ classes.
- Full-eye 400×300 VIS images → need CVRL segmentation into strips. `prepare_strips.py` has **no UBIRIS
  parser yet** — add `--dataset ubiris` that derives identity from the `C<class>` token and eye/side as
  needed (UBIRIS labels are per-eye already). Small parser addition in `parse_meta()`.
- **scp:** `scp -r "C:\dev\ubiris2_1" user@LINUX:~/ubiris2_1`

### 5c. CUVIRIS — real phone-VIS / scanner-NIR (the realistic proxy for the eventual 30-person set)
- **PC path:** `C:\dev\CUVIRIS\{nir,vis}\{left,right}\<subj>\...` (`.bmp`=NIR). 47 subjects.
  Matches `prepare_strips.py --dataset cuviris`.
- **scp:** `scp -r "C:\dev\CUVIRIS" user@LINUX:~/CUVIRIS_raw`

### 5d. (Optional/bonus) UTIRIS — extra paired VIS+NIR (~79 eyes)
- Already-prepared **512×512** ImageFolder at `C:\dev\UTIRIS_GAN\{VIS,NIR}\<subj>_<L|R>\` — but that's the
  *old translation-project* format, NOT 64×512 CVRL strips. To use here you'd re-segment UTIRIS *raw* into
  strips. Low priority; only if you want more cross-spectral alignment pairs.

### 5e. The champion checkpoint (for baseline sanity + warm-start)
- **PC path:** `C:\dev\Coupled-GAN\checkpoints\best.pt` (~1.1 GB, the PolyU 0.121 model). Untracked → scp:
  ```bash
  scp "C:\dev\Coupled-GAN\checkpoints\best.pt" user@LINUX:~/Coupled-GAN/checkpoints/best.pt
  ```
- **NIR pretraining corpus (Lever 2, NIR side):** none on PC. Optional download (CASIA-Iris-Thousand or
  Notre Dame ND-CrossSensor) if you decide to pretrain the NIR encoder too. VIS pretrain (UBIRIS) is the
  higher-value one — do that first.

---

## 6. Build strips + splits (do this once per dataset, on Linux, in tmux)

```bash
cd ~/Coupled-GAN && conda activate iris-cpgan

# ---- PolyU (primary) — the winning recipe: CVRL, raw, no CLAHE, no soft-fill ----
# smoke-test one subject first:
python prepare_strips.py --dataset polyu --backend cvrl --no_enhance --no_soft_fill \
    --src ~/PolyU_raw/001/L/NIR --dst /tmp/smoke   # eyeball /tmp/smoke/NIR/001_L/*.png => 64x512 iris band
# full build:
python prepare_strips.py --dataset polyu --backend cvrl --no_enhance --no_soft_fill \
    --src ~/PolyU_raw --dst ~/PolyU_strips \
    --mask_model nestedsharedatrousresunet-006-0.028214-maskIoU-0.938446.pth \
    --circle_model resnet18-027-0.008222-maskIoU-0.967159.pth
# DO NOT regenerate the split. The canonical PolyU split is splits_cvrl.json (292/62/64),
# already in the repo — it is the split best.pt and every documented result used.
# Regenerating makes a fresh random partition that leaks vs best.pt (see WARNING in §7).

# ---- CUVIRIS (realistic) ----
python prepare_strips.py --dataset cuviris --backend cvrl --no_enhance --no_soft_fill \
    --src ~/CUVIRIS_raw --dst ~/CUVIRIS_strips
python make_kfold_splits.py --strips_root ~/CUVIRIS_strips --k 5 --seed 42 --out_dir splits_kfold_cuviris

# ---- UBIRIS (VIS pretrain) — after adding the --dataset ubiris parser ----
# python prepare_strips.py --dataset ubiris --backend cvrl --no_enhance --no_soft_fill \
#     --src ~/ubiris2_1 --dst ~/UBIRIS_strips
```

---

## 7. The training plan (phased — gate before advancing)

### Phase 0 — trust the harness (½ hour)
Rebuild PolyU strips + splits (§6), then re-evaluate the champion to confirm the pipeline reproduces:
```bash
python eval.py --checkpoint checkpoints/best.pt --model_type unet --feat_dim 128 \
    --vis_root ~/PolyU_strips/VIS --nir_root ~/PolyU_strips/NIR \
    --splits splits_cvrl.json --split test --gallery_fusion mean \
    --out_dir eval_results/baseline_repro
```
Expect ≈ **0.12 EER**. If it's way off, the strips/splits differ from the original run — fix that before
anything else (don't chase model changes on a broken harness — that wasted a month last time).

> **WARNING — split trap (hit & resolved 2026-07-19).** Use **`splits_cvrl.json`** for ALL PolyU work.
> `best.pt` was trained on the cvrl partition (292/62/64). A stray `splits_polyu.json` (282/60/61) is a
> *different* random partition whose test set overlaps best.pt's train/val by 48/61 ids — evaluating
> best.pt on it gives a fake 0.06 EER. On `splits_cvrl.json` best.pt reproduces at **0.1248** (correct).
> `splits_polyu.json` is discarded; ignore any earlier instruction in this doc to generate or use it.

### Phase 1 — Lever 1: fix cross-modal alignment (the main experiment)
Implement `--shared_encoder` (§3) + strong contrastive, then:
```bash
python train_arcface.py \
    --vis_root ~/PolyU_strips/VIS --nir_root ~/PolyU_strips/NIR --splits splits_cvrl.json \
    --shared_encoder --feat_dim 512 \
    --arc_s 64.0 --arc_m 0.5 --lambda_arc 1.0 --lambda_contrast 1.0 --margin 2.0 \
    --epochs 40 --batch_size 256 --lr 1e-4 --eval_every 2 \
    --save_dir checkpoints/arcface_shared_polyu
```
Watch: `arc` loss falls 5–6 → ~1; **val `gen` distance must drop well below 1.88** (that was the failure
signature); EER beats 0.121 by epoch ~20–25.
- **Beats 0.121** → architecture path alive → Phase 2.
- **Doesn't** → ceiling is data → skip to Phase 3 and the data levers; stop tuning architecture.
Try, in order, if the first under-performs: (a) tied encoder + `lambda_contrast 1.0` (above); (b) add
cross-modal center loss; (c) `lambda_arc 0.5 / lambda_contrast 1.0` (lean more on cross-modal).

### Phase 2 — Lever 2: pretrain to fight data scarcity (only if Phase 1 helped)
1. Add UBIRIS parser, build `~/UBIRIS_strips` (§6).
2. Pretrain the shared encoder on UBIRIS (VIS-only, big identity count) with ArcFace to learn a strong
   iris prior.
3. Fine-tune the *cross-modal alignment* on PolyU paired strips (resume encoder, lower LR ~5e-5).
   Re-eval on PolyU test. (Optional: also pretrain the NIR side if you downloaded a NIR corpus — §5e.)

### Phase 3 — the honest real-world number (CUVIRIS)
```bash
python kfold_eval.py --kfold_dir splits_kfold_cuviris \
    --vis_root ~/CUVIRIS_strips/VIS --nir_root ~/CUVIRIS_strips/NIR \
    --checkpoint checkpoints/<best_polyu_model>/best.pt \
    --save_root checkpoints/kfold_cuviris \
    --epochs 30 --batch_size 64 --lr 2e-5 --model_type <match_your_encoder> --feat_dim <512 or 128>
```
Reports `EER mean ± std` over all 47 subjects — the defensible CUVIRIS number. Current baseline to beat:
**0.397 ± 3.5%**. Expect this to stay high until real paired data exists; that's the honest state.

---

## 8. Git / transfer workflow (per user preference)

- **Code + this doc + splits JSON:** commit on Windows → `git push`; on Linux `git pull`. Segmentation
  `.pth` are already tracked and travel this way.
- **Datasets + `.pt` checkpoints:** `scp` (they're gitignored / too big). Commands in §5.
- **Do not** add Claude as commit author / do not push on the user's behalf without asking — user commits.
- After a Linux run, `scp` the checkpoint + `eval_results/` back to Windows to inspect plots.

---

## 9. Guardrails (mistakes made before — don't repeat)

- **Don't trust a single test split** on small data — CUVIRIS 0.31 was luck; k-fold said 0.40. Always
  k-fold on the small sets.
- **Don't make the encoder bigger without the alignment fix** — that path already failed (0.307).
- **Don't re-add CLAHE/soft-fill** to strips — ablated, both hurt cross-spectral matching.
- **Don't chase the paper's 1.02% by tuning** — it's very likely a different/larger data regime; our
  subject-disjoint split is harder and honest. Report our protocol, don't fake theirs.
- **The eventual ~30-person real set is for fine-tune/eval only**, never training from scratch.
- Sanity-check the harness (Phase 0) before any model change.

---

## 10. Data-lever findings — the actionable conclusion (2026-07-19)

After architecture was exhausted (see status banner + §2/§3), we tested the real lever: **more paired
data.** Protocol = append new identities to the champion's TRAIN split only, keep val/test = `splits_cvrl.json`,
train the tiny-conv champion, eval on the SAME fixed 64-id PolyU test. Results:

| Training pool | added ids | Test EER | GAR@1e-3 | Impostor dist |
|---|---|---|---|---|
| S1 only | — | 0.1531 | 0.110 | 1.977 |
| S1 + PolyU **Session 2** (same sensor, co-registered) | +22 | **0.1358** | 0.113 | 1.983 |
| S1 + S2 + **UTIRIS** (diff sensor, NON-co-registered) | +126 | 0.1478 | 0.033 | 1.905 |

**The curve is non-monotonic — this is the key result:**
- **Matched-domain data helps.** PolyU S2 (same rig, co-registered) improved EER ~1.7pt.
- **Mismatched-domain data HURTS.** UTIRIS, despite 6× more identities, made EER *worse* and collapsed
  GAR@1e-3. Signature: impostor distance contracted (1.98→1.90) — cross-domain data compressed the space
  and eroded separation.

**→ The lever is paired data from the MATCHED sensor + registration protocol, not raw iris count.**
For the eventual product this is the whole ballgame: **the capture campaign must match the deployment rig**
(phone-VIS + scanner-NIR, same co-registration handling). Do NOT pad training with public iris sets from
other sensors — it can actively degrade the deployment metric.

Caveats / open thread: single-seed (repeat for a reportable number); UTIRIS confounds domain with
non-co-registration (`train_m0` uses joint-roll; UTIRIS needs independent roll) — decomposing needs
per-source roll, a real `dataset.py`/`train_m0.py` change, only worth it to answer "does ANY aligned
extra data help." The `--dataset utiris` parser also had ~20% parse-skips (326/1612) — fix before relying
on UTIRIS. New code this phase: `prepare_strips.py --id_prefix` + `--dataset ubiris/utiris`,
`pretrain_ubiris.py`, `--shared_encoder` (dead end), `test_shared_encoder.py`. Split records:
`splits_cvrl_s1s2.json`, `splits_cvrl_s1s2_ut.json`.

---

*Origin: consolidated 2026-07-18 from the Coupled-GAN project history; extended 2026-07-19 with the
architecture-exhaustion + data-lever results. The wider `C:\dev` folder is NOT attached to your session —
all external paths you need are in §5. If a path is missing, ask the user to confirm it rather than guessing.*
