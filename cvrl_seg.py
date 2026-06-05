"""
cvrl_seg.py — CVRL/Notre Dame iris segmentation + normalization backend.

Lifted from the OpenSourceIrisRecognition repo (Khan, Flynn, Czajka — IREX open-source
iris paper, arXiv:2605.20735). We keep ONLY the front-end:

    fix_image  ->  segment (pixel mask, NestedSharedAtrousResUNet)
               ->  circApprox (pupil/iris circles, ResNet18)
               ->  cartToPol_torch (Daugman rubber-sheet -> polar image + polar mask)

The HDBIF/ArcIris *matchers* are intentionally NOT ported — we only want segmentation +
normalization. Output contract matches iris_norm: a STRIP_H x STRIP_W uint8 strip plus a
boolean iris mask (True = valid iris pixel).

Why this over open-iris: open-iris (Worldcoin Orb) is NIR-only and fails ~33% of VIS
images; this segmenter's training corpus includes UBIRIS v2 (visible-light) and shows
near-zero failure-to-enroll on off-distribution data in the paper. It also emits the polar
strip natively at the configured size (we set 64x512), so no post-resize is needed.

torch is imported at module top, so this file is only importable on the training box.
iris_norm.py imports it lazily (Windows dev machine has no torch).

Weights (download from the Notre Dame Box link in the repo's models/readme.txt, then place
in the repo root):
    mask  : nestedsharedatrousresunet-*-maskIoU-*.pth   (NestedSharedAtrousResUNet, width=32)
    circle: resnet18-*-maskIoU-*.pth                    (resnet18 + conv/fclayer heads)
"""

import math
from math import pi

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision.transforms import Compose, Normalize, ToTensor


# --------------------------------------------------------------------------- #
# Network architectures (verbatim from OpenSourceIrisRecognition/.../network.py)
# --------------------------------------------------------------------------- #
class SharedAtrousConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(int(out_channels / 2), in_channels, 3, 3))
        nn.init.kaiming_uniform_(self.weights, mode="fan_out", nonlinearity="relu")
        if bias:
            self.bias1 = nn.Parameter(torch.zeros(int(out_channels / 2)))
            self.bias2 = nn.Parameter(torch.zeros(int(out_channels / 2)))
        else:
            self.bias1 = None
            self.bias2 = None

    def forward(self, x):
        x1 = nn.functional.conv2d(x, self.weights, stride=1, padding="same", bias=self.bias1)
        x2 = nn.functional.conv2d(x, self.weights, stride=1, padding="same", dilation=2, bias=self.bias2)
        return torch.cat([x1, x2], 1)


class SharedAtrousResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.net = nn.Sequential(
            SharedAtrousConv2d(in_channels, middle_channels, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            SharedAtrousConv2d(middle_channels, out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        return self.relu(x)


class Resize(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode="nearest", align_corners=None,
                 recompute_scale_factor=None, antialias=False):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
            align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor,
            antialias=self.antialias)


class NestedSharedAtrousResUNet(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        nb = [width, width * 2, width * 4, width * 8, width * 16]
        self.pool = Resize(scale_factor=0.5, mode="bilinear")
        self.up = Resize(scale_factor=2, mode="bilinear")
        self.conv0_0 = SharedAtrousResBlock(num_channels, nb[0], nb[0])
        self.conv1_0 = SharedAtrousResBlock(nb[0], nb[1], nb[1])
        self.conv2_0 = SharedAtrousResBlock(nb[1], nb[2], nb[2])
        self.conv3_0 = SharedAtrousResBlock(nb[2], nb[3], nb[3])
        self.conv4_0 = SharedAtrousResBlock(nb[3], nb[4], nb[4])
        self.conv0_1 = SharedAtrousResBlock(nb[0] + nb[1], nb[0], nb[0])
        self.conv1_1 = SharedAtrousResBlock(nb[1] + nb[2], nb[1], nb[1])
        self.conv2_1 = SharedAtrousResBlock(nb[2] + nb[3], nb[2], nb[2])
        self.conv3_1 = SharedAtrousResBlock(nb[3] + nb[4], nb[3], nb[3])
        self.conv0_2 = SharedAtrousResBlock(nb[0] * 2 + nb[1], nb[0], nb[0])
        self.conv1_2 = SharedAtrousResBlock(nb[1] * 2 + nb[2], nb[1], nb[1])
        self.conv2_2 = SharedAtrousResBlock(nb[2] * 2 + nb[3], nb[2], nb[2])
        self.conv0_3 = SharedAtrousResBlock(nb[0] * 3 + nb[1], nb[0], nb[0])
        self.conv1_3 = SharedAtrousResBlock(nb[1] * 3 + nb[2], nb[1], nb[1])
        self.conv0_4 = SharedAtrousResBlock(nb[0] * 4 + nb[1], nb[0], nb[0])
        self.final = nn.Conv2d(nb[0] * 4, num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        return self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))


class fclayer(nn.Module):
    def __init__(self, in_h=8, in_w=10, out_n=6):
        super().__init__()
        self.in_h, self.in_w, self.out_n = in_h, in_w, out_n
        self.fc_list = nn.ModuleList([nn.Linear(in_h * in_w, 6) for _ in range(out_n)])
        self.act = nn.GELU()
        self.fc2 = nn.Linear(36, 6)

    def forward(self, x):
        x = x.reshape(-1, self.out_n, self.in_h, self.in_w)
        outs = [self.fc_list[i](x[:, i, :, :].reshape(-1, self.in_h * self.in_w)) for i in range(self.out_n)]
        out = torch.cat(outs, 1)
        return self.fc2(self.act(out))


class conv(nn.Module):
    def __init__(self, in_channels=512, out_n=6):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_n, kernel_size=1, stride=1, padding="same")

    def forward(self, x):
        return self.conv(x)


# --------------------------------------------------------------------------- #
# Segmenter wrapper (segmentation + rubber-sheet normalization only)
# --------------------------------------------------------------------------- #
class CVRLSegmenter:
    """Drop-in replacement for an open-iris IRISPipeline in iris_norm.normalize_strip.

    .run(mono_uint8) -> (strip uint8 [polar_h, polar_w], mask bool [polar_h, polar_w], ok)
    """

    is_cvrl = True                     # marker used by iris_norm.normalize_strip dispatch
    NET_INPUT_SIZE = (320, 240)        # (w, h) for the seg/circle nets
    ISO_RES = (640, 480)               # (w, h) ISO-compliant working resolution

    def __init__(self, mask_model_path, circle_model_path, device="cuda",
                 polar_h=64, polar_w=512, min_pupil_r=12, min_iris_r=16):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.polar_h = polar_h
        self.polar_w = polar_w
        self.min_pupil_r = min_pupil_r
        self.min_iris_r = min_iris_r

        # NB: load under no_grad (NOT inference_mode) so the weights are normal tensors.
        # inference_mode-created tensors poison any later autograd-tracked call (e.g. when the
        # diagnostic invokes the inner methods directly). We freeze params instead.
        with torch.no_grad():
            self.circle_model = models.resnet18()
            self.circle_model.avgpool = conv(in_channels=512, out_n=6)
            self.circle_model.fc = fclayer(out_n=6)
            self.circle_model.load_state_dict(torch.load(circle_model_path, map_location=self.device))
            self.circle_model = self.circle_model.float().to(self.device).eval()

            self.mask_model = NestedSharedAtrousResUNet(1, 1, width=32, resolution=(240, 320))
            self.mask_model.load_state_dict(torch.load(mask_model_path, map_location=self.device))
            self.mask_model = self.mask_model.float().to(self.device).eval()
        for p in list(self.circle_model.parameters()) + list(self.mask_model.parameters()):
            p.requires_grad_(False)

        self._tf = Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))])

    # -- ISO 4:3 / 640x480 (pad short axis, then resize) ------------------- #
    def _fix_image(self, image):
        w, h = image.size
        ar = float(w) / float(h)
        if 1.333 <= ar <= 1.334:
            return image.copy().resize(self.ISO_RES)
        if ar < 1.333:
            w_new = h * (4.0 / 3.0)
            out = Image.new(image.mode, (int(w_new), h), 127)
            out.paste(image, (int((w_new - w) / 2), 0))
        else:
            h_new = w * (3.0 / 4.0)
            out = Image.new(image.mode, (w, int(h_new)), 127)
            out.paste(image, (0, int((h_new - h) / 2)))
        return out.resize(self.ISO_RES)

    def _segment(self, image):
        w, h = image.size
        im = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
        logit = self.mask_model(self._tf(im).unsqueeze(0).to(self.device))[0]
        mask = torch.where(torch.sigmoid(logit) > 0.5, 255, 0).cpu().numpy()[0]
        return cv2.resize(np.uint8(mask), (w, h), interpolation=cv2.INTER_NEAREST_EXACT)

    def _circ(self, image):
        w, h = image.size
        im = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
        xyr = self.circle_model(self._tf(im).unsqueeze(0).repeat(1, 3, 1, 1).to(self.device)).tolist()[0]
        diag = math.sqrt(w ** 2 + h ** 2)
        pupil = np.array([xyr[0] * w, xyr[1] * h, xyr[2] * 0.5 * 0.8 * diag])
        iris = np.array([xyr[3] * w, xyr[4] * h, xyr[5] * 0.5 * diag])
        return pupil, iris

    def _grid_sample(self, inp, grid, mode):
        N, C, H, W = inp.shape
        gx = grid[:, :, :, 0]
        gy = grid[:, :, :, 1]
        gx = ((gx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
        gy = ((gy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
        newgrid = torch.stack([gx, gy], dim=-1)
        return torch.nn.functional.grid_sample(inp, newgrid, mode=mode, align_corners=True, padding_mode="border")

    def _cart_to_pol(self, image, mask, pupil_xyr, iris_xyr, interpolation="bilinear"):
        img = torch.tensor(np.array(image)).float().unsqueeze(0).unsqueeze(0).to(self.device)
        msk = torch.tensor(np.array(mask)).float().unsqueeze(0).unsqueeze(0).to(self.device)
        width, height = img.shape[3], img.shape[2]
        ph, pw = self.polar_h, self.polar_w
        pupil_xyr = torch.tensor(pupil_xyr).unsqueeze(0).float().to(self.device)
        iris_xyr = torch.tensor(iris_xyr).unsqueeze(0).float().to(self.device)

        theta = (2 * pi * torch.linspace(0, pw - 1, pw) / pw).to(self.device)
        pxc = (pupil_xyr[:, 0].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, pw))
        pyc = (pupil_xyr[:, 1].reshape(-1, 1) + pupil_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, pw))
        ixc = (iris_xyr[:, 0].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.cos(theta).reshape(1, pw))
        iyc = (iris_xyr[:, 1].reshape(-1, 1) + iris_xyr[:, 2].reshape(-1, 1) @ torch.sin(theta).reshape(1, pw))

        radius = (torch.linspace(1, ph, ph) / ph).reshape(-1, 1).to(self.device)
        px = torch.matmul((1 - radius), pxc.reshape(-1, 1, pw))
        py = torch.matmul((1 - radius), pyc.reshape(-1, 1, pw))
        ix = torch.matmul(radius, ixc.reshape(-1, 1, pw))
        iy = torch.matmul(radius, iyc.reshape(-1, 1, pw))

        x_norm = (((px + ix).float() - 1) / (width - 1)) * 2 - 1
        y_norm = (((py + iy).float() - 1) / (height - 1)) * 2 - 1
        grid = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1).to(self.device)

        img_polar = torch.clamp(torch.round(self._grid_sample(img, grid, interpolation)), 0, 255)
        mask_polar = (self._grid_sample(msk, grid, "nearest") > 0.5).long() * 255
        return (img_polar[0][0].cpu().numpy()).astype(np.uint8), mask_polar[0][0].cpu().numpy().astype(np.uint8)

    def _quality_ok(self, pupil_xyr, iris_xyr, mask):
        """Biological sanity checks (paper §V-A3). Returns False -> treat as seg failure."""
        px, py, pr = pupil_xyr
        ix, iy, ir = iris_xyr
        if ir <= pr:                                            # iris must be larger than pupil
            return False
        if pr < self.min_pupil_r or ir < self.min_iris_r:       # insufficient radii
            return False
        alpha = pr / ir
        if not (0.1 <= alpha <= 0.8):                           # abnormal pupil/iris ratio
            return False
        if math.hypot(px - ix, py - iy) / ir > 0.5:             # excessive concentric deviation
            return False
        visible = (mask > 0).sum()
        denom = pi * (ir + pr) * (ir - pr)
        if denom <= 0 or (visible / denom) < 0.1:               # too little iris visible
            return False
        return True

    @torch.inference_mode()
    def run(self, mono):
        """mono: uint8 HxW grayscale (output of iris_norm.to_mono)."""
        pil = Image.fromarray(np.ascontiguousarray(mono), "L")
        pil = self._fix_image(pil)
        try:
            mask = self._segment(pil)
            pupil_xyr, iris_xyr = self._circ(pil)
        except Exception:
            return None, None, False
        if not self._quality_ok(pupil_xyr, iris_xyr, mask):
            return None, None, False
        img_polar, mask_polar = self._cart_to_pol(pil, mask, pupil_xyr, iris_xyr)
        if img_polar is None:
            return None, None, False
        return img_polar, (mask_polar > 127), True
