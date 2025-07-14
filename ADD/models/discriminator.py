# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Projected discriminator architecture from
"StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis".
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import SpectralNorm
from torchvision.transforms import RandomCrop, Normalize
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

#from th_utils import misc
from ADD.models.shared import ResidualBlock, FullyConnectedLayer
from ADD.models.vit_utils import make_vit_backbone, forward_vit
from ADD.models.DiffAugment import DiffAugment
from ADD.utils.util_net import reload_model_

class SpectralConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 3, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups.
        G = np.ceil(x.size(0)/self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats.
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size = kernel_size,
            padding = kernel_size//2,
            padding_mode = 'circular',
        ),
        nn.GroupNorm(4, channels),
        nn.LeakyReLU(0.2, True),
    )


class DiscHead_f(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=7),
            ResidualBlock(make_block(channels, kernel_size=7))
        )

        if self.c_dim > 0:
            self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=7, padding=3)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = x.permute(2,1,0)
        h = self.main(x)
        out = self.cls(h)
        self.c_dim = 0
        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)
            cmap = cmap.permute(2,1,0)
        
            out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return out

class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim

        self.main = nn.Sequential(
            make_block(channels, kernel_size=1),
            ResidualBlock(make_block(channels, kernel_size=9))
        )

        if self.c_dim > 0:
            self.cmapper = FullyConnectedLayer(self.c_dim, cmap_dim)
            self.cls = SpectralConv1d(channels, cmap_dim, kernel_size=1, padding=0)
        else:
            self.cls = SpectralConv1d(channels, 1, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.main(x)
        out = self.cls(h)
        self.c_dim = 0 
        if self.c_dim > 0:
            cmap = self.cmapper(c).unsqueeze(-1)
            out = (out * cmap).sum(1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))
        return out


class DINO(torch.nn.Module):
    def __init__(self, hooks: list[int] = [2,5,8,11], hook_patch: bool = True):
        super().__init__()
        self.n_hooks = len(hooks) + int(hook_patch)

        self.model = make_vit_backbone(
        timm.create_model('vit_small_patch16_224_dino', pretrained=False),
            patch_size=[16,16], hooks=hooks, hook_patch=hook_patch,
        )
        reload_model_(self.model, torch.load('ADD/dino_deitsmall16_pretrain.pth')) 
        self.model = self.model.eval().requires_grad_(False)


        self.img_resolution = self.model.model.patch_embed.img_size[0]
        self.embed_dim = self.model.model.embed_dim
        self.norm = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ''' input: x in [0, 1]; output: dict of activations '''
        x = F.interpolate(x, self.img_resolution, mode='area')
        x = self.norm(x)
        features = forward_vit(self.model, x)
        return features

from einops import rearrange
#from torch import rearrange
class SubPixelConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super(SubPixelConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

from PIL import Image
cnt = 0 
import torch
import torch.nn.functional as F
class ProjectedDiscriminator(nn.Module):
    def __init__(self, c_dim: int, diffaug: bool = True, p_crop: float = 0.5):
        super().__init__()
        self.c_dim = c_dim
        self.diffaug = diffaug
        self.p_crop = p_crop
        self.unet = None
        self.conv = nn.Conv2d(2560, 1280, 1)

        self.dino = DINO() 
        self.up = SubPixelConvLayer(in_channels=16,out_channels=3,upscale_factor=4)
        heads = []
        heads_f = []
        c_dim = 384 #384
        for i in range(self.dino.n_hooks):
            heads += [str(i), DiscHead(self.dino.embed_dim, c_dim)],
        for i in range(self.dino.n_hooks):
            heads_f += [str(i), DiscHead_f(self.dino.embed_dim, c_dim)],
        self.heads = nn.ModuleDict(heads)
        self.heads_f = nn.ModuleDict(heads_f)

    def train(self, mode: bool = True):
        self.dino = self.dino.train(False)
        self.heads = self.heads.train(True)
        self.heads_f = self.heads_f.train(True)
        self.up = self.up.train(True)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:

        x = self.up(x)

        # Forward pass through DINO ViT.
        features = self.dino(x)

        # Apply discriminator heads.
        logits = []
        logits_f = []
        for k, head in self.heads.items():
            features[k].requires_grad_(True)
            logits.append(head(features[k], c).view(x.size(0), -1))
        for k, head in self.heads_f.items():
            features[k].requires_grad_(True)
            out = head(features[k], c)
            logits_f.append(out.view(out.size(0), -1))

        return logits, logits_f, features
