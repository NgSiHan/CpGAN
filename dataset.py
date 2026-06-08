"""
dataset.py — CrossSpectralPairs dataset for CpGAN training.

Returns (vis_strip, nir_strip, label) where label=1 for genuine (same identity)
and label=0 for impostor (different identity), 50/50 balanced.

Augmentation: circular horizontal roll only (simulates eye/head rotation on the angular
axis). RandomRotation and RandomHorizontalFlip are intentionally absent — both are
semantically wrong for iris strips and destroy identity information.
"""

import glob
import os
import random

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


def _index(root: str) -> dict:
    """Build {identity: [file_paths]} from <root>/<identity>/*.png."""
    d = {}
    for idd in sorted(os.listdir(root)):
        paths = glob.glob(os.path.join(root, idd, "*.png"))
        if paths:
            d[idd] = paths
    return d


class CrossSpectralPairs(Dataset):
    """Returns (vis_strip [1,64,512], nir_strip [1,64,512], label float).

    label = 1.0 for genuine (same identity), 0.0 for impostor (different identity).
    """

    def __init__(self, vis_root: str, nir_root: str, ids: list,
                 train: bool = True, shift_pixel: int = 14, shift_prob: float = 0.5,
                 independent_roll: bool = False):
        self.vis = _index(vis_root)
        self.nir = _index(nir_root)
        # only keep identities that have strips in BOTH spectra and belong to this split
        self.ids = [i for i in ids if i in self.vis and i in self.nir]
        assert self.ids, (
            f"No identities found in both VIS ({vis_root}) and NIR ({nir_root}) "
            f"for the provided split. Check paths and make_splits.py output."
        )
        self.train = train
        self.shift_pixel = shift_pixel
        self.shift_prob = shift_prob
        # Stable integer index for each identity (used by ArcFaceHead label input)
        self.id_to_idx: dict = {idd: i for i, idd in enumerate(self.ids)}
        # False (default): co-registered data (PolyU) -> roll a genuine pair JOINTLY to keep
        #   the simultaneously-captured alignment.
        # True: non-co-registered data (CUVIRIS, captured at different times/devices) -> roll
        #   genuine pairs INDEPENDENTLY to learn rotation-invariance, since no shared
        #   alignment exists to preserve.
        self.independent_roll = independent_roll

    def __len__(self) -> int:
        # nominal epoch size = total number of VIS strips across split identities
        return sum(len(self.vis[i]) for i in self.ids)

    def _load(self, path: str, shift: int = 0) -> torch.Tensor:
        """Load a strip PNG and return a [1, 64, 512] tensor in [-1, 1].

        Args:
            shift: pre-computed angular shift (cols) to apply.  Caller is responsible
                   for passing the *same* shift to both strips of a genuine pair so that
                   co-registered VIS/NIR strips stay aligned after augmentation.
                   For impostor pairs, shifts are independent (different identities,
                   no co-registration assumption).
        """
        img = Image.open(path).convert("L")
        t = TF.to_tensor(img)                              # [1, 64, 512] in [0, 1]
        if shift != 0:
            t = torch.roll(t, shifts=shift, dims=-1)
        return TF.normalize(t, [0.5], [0.5])               # -> [-1, 1]

    def _sample_shift(self) -> int:
        """Return a random angular shift, or 0 if augmentation is disabled/skipped."""
        if self.train and random.random() < self.shift_prob:
            return random.randint(-self.shift_pixel, self.shift_pixel)
        return 0

    def __getitem__(self, _):
        genuine = random.random() < 0.5
        ida = random.choice(self.ids)

        if genuine:
            idb = ida   # set here so nir_id_idx can be computed uniformly below
            if self.independent_roll:
                # Non-co-registered genuine pair (CUVIRIS): independent shifts -> rotation-invariance.
                vis = self._load(random.choice(self.vis[ida]), shift=self._sample_shift())
                nir = self._load(random.choice(self.nir[ida]), shift=self._sample_shift())
            else:
                # Co-registered genuine pair (PolyU): SAME shift to preserve the alignment.
                shift = self._sample_shift()
                vis = self._load(random.choice(self.vis[ida]), shift=shift)
                nir = self._load(random.choice(self.nir[ida]), shift=shift)
        else:
            # Impostor pair: different identities, no co-registration constraint —
            # independent shifts are fine and add extra diversity.
            vis = self._load(random.choice(self.vis[ida]), shift=self._sample_shift())
            idb = random.choice([i for i in self.ids if i != ida])
            nir = self._load(random.choice(self.nir[idb]), shift=self._sample_shift())

        label      = torch.tensor(1.0 if genuine else 0.0)
        vis_id_idx = torch.tensor(self.id_to_idx[ida], dtype=torch.long)
        nir_id_idx = torch.tensor(self.id_to_idx[idb], dtype=torch.long)
        return vis, nir, label, vis_id_idx, nir_id_idx
