import math

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function


# https://github.com/znxlwm/pytorch-pix2pix/blob/master/network.py
class IrisEncoder(nn.Module):
    """Encoder-only path for Milestone-0 contrastive Siamese.

    Mirrors the UNet encoder exactly (conv1-conv6 + fc1) but instantiates no decoder,
    so forward() returns only the 128-d embedding. Useful for the gate check before
    adding GAN complexity.

    Input:  [B, 1, 64, 512]  — single-channel 64×512 grayscale strip in [-1, 1]
    Output: [B, feat_dim]    — L2-normalizable embedding
    """

    def __init__(self, d: int = 64, feat_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # fc1: 6 stride-2 convs on 64×512 → spatial 1×8, channels 512 → flat 4096
        self.fc1 = nn.Linear(4096, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        e = self.conv1(x)
        e = self.conv2_bn(self.conv2(torch.nn.functional.leaky_relu(e, 0.2)))
        e = self.conv3_bn(self.conv3(torch.nn.functional.leaky_relu(e, 0.2)))
        e = self.conv4_bn(self.conv4(torch.nn.functional.leaky_relu(e, 0.2)))
        e = self.conv5_bn(self.conv5(torch.nn.functional.leaky_relu(e, 0.2)))
        e = self.conv6(torch.nn.functional.leaky_relu(e, 0.2))
        return self.fc1(e.view(bs, -1))


class UNet(nn.Module):
    # initializers
    def __init__(self, d=64, feat_dim=128):
        super(UNet, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)    # 1-channel grayscale strip input
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # self.conv8_bn = nn.BatchNorm2d(d * 8)
        self.fc1 = nn.Linear(4096, feat_dim)
        # Unet decoder
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(768, d * 8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(640, d * 4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(320, 1, 4, 2, 1)   # 1-channel output
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

    # forward method
    def forward(self, input):
        bs = input.size(0)
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6(F.leaky_relu(e5, 0.2))

        # e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        # e8 = self.conv8(F.leaky_relu(e7, 0.2))
        # # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        bottleneck = e6.view(bs, -1)
        bottleneck = self.fc1(bottleneck)

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e6))), 0.5, training=True)
        d1 = torch.cat([d1, e5], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e4], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e3], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))

        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5, training=True)
        d4 = torch.cat([d4, e2], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e1], 1)
        d6 = self.deconv6(F.relu(d5))

        # d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        # d6 = torch.cat([d6, e2], 1)
        # d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        # d7 = torch.cat([d7, e1], 1)
        # d8 = self.deconv8(F.relu(d7))
        o = torch.tanh(d6)

        return o, bottleneck


# conv_output = (W-K + 2P/S +1)
# input_3, output_64, kernel_4, stride_2, padding_1


class UNetV2(nn.Module):
    # initializers
    def __init__(self, d=64, feat_dim=128):
        super(UNetV2, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # self.conv8_bn = nn.BatchNorm2d(d * 8)
        self.fc1 = nn.Linear(2048, feat_dim)
        # Unet decoder
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

    # forward method
    def forward(self, input):
        bs = input.size(0)
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        # e8 = self.conv8(F.leaky_relu(e7, 0.2))
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        bottleneck = e7.view(bs, -1)
        bottleneck = self.fc1(bottleneck)

        # d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        # d1 = torch.cat([d1, e7], 1)
        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(e7))), 0.5, training=True)
        d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        o = torch.tanh(d8)

        return o, bottleneck


class UNetV3(nn.Module):
    # initializers
    def __init__(self, d=64, feat_dim=128):
        super(UNetV3, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # self.conv8_bn = nn.BatchNorm2d(d * 8)
        self.fc1 = nn.Linear(8192, feat_dim)
        # Unet decoder
        # self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        # self.deconv1_bn = nn.BatchNorm2d(d * 8)
        # self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        # self.deconv2_bn = nn.BatchNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

    # forward method
    def forward(self, input):
        bs = input.size(0)
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        # e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        # e8 = self.conv8(F.leaky_relu(e7, 0.2))
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        bottleneck = e6.view(bs, -1)
        bottleneck = self.fc1(bottleneck)

        # d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        # d1 = torch.cat([d1, e7], 1)
        # d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(e7))), 0.5, training=True)
        # d2 = torch.cat([d2, e6], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(e6))), 0.5, training=True)
        d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        o = torch.tanh(d8)

        return o, bottleneck


class UNetV4(nn.Module):
    # initializers
    def __init__(self, d=64, feat_dim=128):
        super(UNetV4, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.BatchNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv7_bn = nn.BatchNorm2d(d * 8)
        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        # self.conv8_bn = nn.BatchNorm2d(d * 8)
        self.fc1 = nn.Linear(32768, feat_dim)
        # Unet decoder
        # self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        # self.deconv1_bn = nn.BatchNorm2d(d * 8)
        # self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        # self.deconv2_bn = nn.BatchNorm2d(d * 8)
        # self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        # self.deconv3_bn = nn.BatchNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d * 8)
        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

    # forward method
    def forward(self, input):
        bs = input.size(0)
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        # e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        # e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))
        # e8 = self.conv8(F.leaky_relu(e7, 0.2))
        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))
        bottleneck = e5.view(bs, -1)
        bottleneck = self.fc1(bottleneck)

        # d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)
        # d1 = torch.cat([d1, e7], 1)
        # d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(e7))), 0.5, training=True)
        # d2 = torch.cat([d2, e6], 1)
        # d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(e6))), 0.5, training=True)
        # d3 = torch.cat([d3, e5], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(e5)))
        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)
        d4 = torch.cat([d4, e4], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        o = torch.tanh(d8)

        return o, bottleneck


class UNetV5(nn.Module):
    # initializers
    def __init__(self, d=64, feat_dim=128):
        super(UNetV5, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)

        self.fc1 = nn.Linear(131072, feat_dim)
        # Unet decoder

        self.deconv5 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

    # forward method
    def forward(self, input):
        bs = input.size(0)
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))

        bottleneck = e4.view(bs, -1)
        bottleneck = self.fc1(bottleneck)

        d5 = self.deconv5_bn(self.deconv5(F.relu(e4)))
        d5 = torch.cat([d5, e3], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        o = torch.tanh(d8)

        return o, bottleneck


class UNetV6(nn.Module):
    # initializers
    def __init__(self, d=64, feat_dim=128):
        super(UNetV6, self).__init__()
        # Unet encoder
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        # self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        # self.conv4_bn = nn.BatchNorm2d(d * 8)

        self.fc1 = nn.Linear(262144, feat_dim)
        # Unet decoder

        # self.deconv5 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        # self.deconv5_bn = nn.BatchNorm2d(d * 4)
        self.deconv6 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv6_bn = nn.BatchNorm2d(d * 2)
        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv7_bn = nn.BatchNorm2d(d)
        self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

    # forward method
    def forward(self, input):
        bs = input.size(0)
        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))

        bottleneck = e3.view(bs, -1)
        bottleneck = self.fc1(bottleneck)

        d6 = self.deconv6_bn(self.deconv6(F.relu(e3)))
        d6 = torch.cat([d6, e2], 1)
        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))
        d7 = torch.cat([d7, e1], 1)
        d8 = self.deconv8(F.relu(d7))
        o = torch.tanh(d8)

        return o, bottleneck


class Mapper(nn.Module):
    def __init__(self, prenet='resnet50', outdim=128, image_channel=3):
        super(Mapper, self).__init__()

        model = getattr(models, prenet)(pretrained=False)

        self.model = list(model.children())[:-1]

        self.backbone = nn.Sequential(*self.model)

        if prenet == 'resnet18':
            nfc = 512
        elif prenet == 'resnet50':
            nfc = 2048

        self.fc1 = nn.Linear(nfc, outdim)
        # self.fc2 = nn.Linear(1024, outdim)
        # Upsample
        decoder = []
        in_channels = nfc
        out_channels = in_channels // 2
        # Range should be 3 if using resnet18, and 5 if using resnet50
        for i in range(8):
            decoder += [nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)]
            in_channels = out_channels
            out_channels = in_channels // 2

        decoder += [nn.ReflectionPad2d(3),
                    nn.Conv2d(out_channels * 2, image_channel, 7),
                    nn.Tanh()]
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        bs = x.size(0)
        y = self.backbone(x)
        copy = y
        y = y.view(bs, -1)
        output = self.fc1(y)
        img = self.decoder(copy)
        return img, output

    def EncodeImage(self, x):
        y = self.backbone(x)
        return y

    def DecodeImage(self, x):
        img = self.decoder(x)
        return img


class Discriminator(nn.Module):
    """Simple global discriminator for LSGAN on 1-channel 64×512 iris strips.

    4 stride-2 conv blocks: 64×512 → 4×32, 128 channels → flat 16384.
    """

    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            block.extend([nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)])
            return block

        self.shared_conv = nn.Sequential(
            *discriminator_block(in_channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        # 4 stride-2 blocks on 64×512: H→4, W→32, C=128 → flat = 16384
        self.D1 = nn.Linear(16384, 1)

    def forward(self, img):
        out = self.shared_conv(img)
        return self.D1(out.view(out.shape[0], -1))
