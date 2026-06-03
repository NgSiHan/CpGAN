# Setup & Training Guide

## 1. Transfer files from Windows to Linux

### One-time initial copy (from Windows PowerShell)
```powershell
# Replace user@linux-box with your actual username and hostname/IP
scp -r C:\dev\Coupled-GAN user@linux-box:~/Coupled-GAN
```

### Ongoing updates (recommended — push on Windows, pull on Linux)

On Windows after making changes:
```powershell
cd C:\dev\Coupled-GAN
git add .
git commit -m "your message here"
git push
```

On Linux:
```bash
cd ~/Coupled-GAN
git pull
```

If you don't have a remote set up yet, re-run `scp` for each update, or:
```powershell
# Sync only changed files (faster than full copy)
scp iris_norm.py prepare_strips.py dataset.py model.py train.py train_m0.py eval.py make_splits.py user@linux-box:~/Coupled-GAN/
```

---

## 2. Environment setup on Linux

### 2.1 Check your CUDA version first
```bash
nvidia-smi
# Note the "CUDA Version" in the top-right corner (e.g. 12.1, 12.4)
```

### 2.2 Create the conda environment
```bash
conda create -n iris-cpgan python=3.10 -y
conda activate iris-cpgan
```

Add this to your `~/.bashrc` so you don't forget to activate it:
```bash
echo 'conda activate iris-cpgan' >> ~/.bashrc
```

### 2.3 Install PyTorch with CUDA support

Pick the line matching your CUDA version from `nvidia-smi`:

```bash
# CUDA 12.1 (most common for RTX 4090 on modern drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 11.8 (older drivers)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is visible to PyTorch:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Should print: True  and  NVIDIA GeForce RTX 4090
```

### 2.4 Install remaining dependencies
```bash
cd ~/Coupled-GAN
pip install -r requirements.txt
```

### 2.5 Download the open-iris segmentation model

The model is fetched from HuggingFace on first use. Run this once while you have internet access so it's cached before training:
```bash
IRIS_ENV=SERVER python -c "import iris; iris.IRISPipeline(); print('Model cached OK')"
```

If HuggingFace is slow or blocked, set a local cache dir:
```bash
export HF_HOME=~/hf_cache   # add to ~/.bashrc too
```

### 2.6 Verify everything
```bash
python -c "
import torch
from model import UNet, IrisEncoder, Discriminator

assert torch.cuda.is_available(), 'No CUDA'
print('GPU:', torch.cuda.get_device_name(0))

x = torch.zeros(2, 1, 64, 512)
m = UNet()
img, emb = m(x)
assert img.shape == (2, 1, 64, 512)
assert emb.shape == (2, 128)
print('UNet OK:', img.shape, emb.shape)

enc = IrisEncoder()
assert enc(x).shape == (2, 128)
print('IrisEncoder OK')

disc = Discriminator()
assert disc(x).shape == (2, 1)
print('Discriminator OK')
print('All checks passed.')
"
```

---

## 3. Training with tmux (sessions survive SSH disconnect)

### Start a named session before running anything long
```bash
tmux new-session -s iris
```

Inside the tmux session, run your commands normally:
```bash
conda activate iris-cpgan
cd ~/Coupled-GAN
```

### Detach without killing the session
```
Ctrl+B, then D
```
Your SSH connection can now drop safely — training keeps running.

### Reattach later
```bash
tmux attach -t iris
# or if you forgot the name:
tmux ls                  # list all sessions
tmux attach -t 0         # attach to session 0
```

### Useful tmux shortcuts
| Keys | Action |
|---|---|
| `Ctrl+B, D` | Detach (leave session running) |
| `Ctrl+B, [` | Scroll mode (use arrow keys / PgUp to scroll output) |
| `q` | Exit scroll mode |
| `Ctrl+B, C` | New window within session |
| `Ctrl+B, N` | Next window |
| `Ctrl+B, "` | Split pane horizontally |
| `Ctrl+B, %` | Split pane vertically |

### Recommended: one window per long job
```bash
tmux new-session -s iris           # main session
# Ctrl+B, C → new window for preprocessing
# Ctrl+B, C → new window for training
# Ctrl+B, N → switch between windows
```

---

## 4. Data preparation

### 4.1 Smoke-test on ONE image before batch-running
```bash
# Pick any single image from the dataset
IRIS_ENV=SERVER python prepare_strips.py \
    --src ~/PolyU_raw/001/L/NIR \
    --dst /tmp/smoke_test

# Inspect the result — should be a clean 64×512 grayscale PNG
ls -la /tmp/smoke_test/NIR/001_L/
python -c "
import cv2, sys
img = cv2.imread('/tmp/smoke_test/NIR/001_L/<filename>.png', 0)
print('Shape:', img.shape)   # (64, 512)
print('Min/Max:', img.min(), img.max())
"
```

### 4.2 Batch-run PolyU (inside tmux)
```bash
IRIS_ENV=SERVER python prepare_strips.py \
    --src ~/PolyU_raw \
    --dst ~/PolyU_strips
# Watch seg_fail counts — expect more failures on VIS than NIR
```

### 4.3 Make identity splits
```bash
python make_splits.py \
    --strips_root ~/PolyU_strips \
    --out splits_polyu.json
# Prints: VIS-only N, NIR-only N, Both N, train/val/test counts
```

---

## 5. Training pipeline

### Step 1 — Milestone 0 (run this first, gate before full CpGAN)
```bash
python train_m0.py \
    --vis_root ~/PolyU_strips/VIS \
    --nir_root ~/PolyU_strips/NIR \
    --splits splits_polyu.json \
    --epochs 10 --batch_size 256 --margin 2.0 \
    --save_dir checkpoints/m0
```

**Gate:** look for `[GATE] genuine-mean < impostor-mean: separation achieved` in the output.
If you don't see it after 10 epochs, there's a data/normalization problem — don't proceed to step 2.

Monitor GPU utilisation in a separate tmux pane:
```bash
watch -n1 nvidia-smi
# GPU util should be well above 50% — if it's near 0%, raise --workers
```

### Step 2 — Full CpGAN (only after Milestone 0 gate passes)
```bash
python train.py \
    --vis_root ~/PolyU_strips/VIS \
    --nir_root ~/PolyU_strips/NIR \
    --splits splits_polyu.json \
    --epochs 50 --batch_size 256 --margin 2.0 \
    --save_dir checkpoints/cpgan
```

### Step 3 — Evaluate
```bash
python eval.py \
    --checkpoint checkpoints/cpgan/best.pt \
    --vis_root ~/PolyU_strips/VIS \
    --nir_root ~/PolyU_strips/NIR \
    --splits splits_polyu.json --split test \
    --out_dir eval_results/cpgan
```

---

## 6. Copying results back to Windows

```powershell
# From Windows PowerShell — copy checkpoint and eval plots
scp -r user@linux-box:~/Coupled-GAN/checkpoints C:\dev\Coupled-GAN\
scp -r user@linux-box:~/Coupled-GAN/eval_results C:\dev\Coupled-GAN\
```
