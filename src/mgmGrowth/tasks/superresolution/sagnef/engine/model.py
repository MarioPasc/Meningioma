# -*- coding: utf-8 -*-
"""
SAGNEF model: light 3D U-Net gate producing softmax weight maps over experts.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- small 3D U-Net gate ----

def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, 3, padding=1, bias=False),
        nn.GroupNorm(8, c_out),
        nn.LeakyReLU(inplace=True),
        nn.Conv3d(c_out, c_out, 3, padding=1, bias=False),
        nn.GroupNorm(8, c_out),
        nn.LeakyReLU(inplace=True),
    )

class GatingUNet3D(nn.Module):
    """
    Produces per-voxel softmax weights over M experts.
    Input channels: C = (#experts) + (1 if include_lr).
    """
    def __init__(self, in_ch: int, n_experts: int, wf: int = 24):
        super().__init__()
        self.enc1 = conv_block(in_ch, wf)
        self.down1 = nn.Conv3d(wf, wf*2, 2, stride=2)
        self.enc2 = conv_block(wf*2, wf*2)
        self.down2 = nn.Conv3d(wf*2, wf*4, 2, stride=2)
        self.bott = conv_block(wf*4, wf*4)
        self.up2 = nn.ConvTranspose3d(wf*4, wf*2, 2, stride=2)
        self.dec2 = conv_block(wf*4, wf*2)
        self.up1 = nn.ConvTranspose3d(wf*2, wf, 2, stride=2)
        self.dec1 = conv_block(wf*2, wf)
        self.head = nn.Conv3d(wf, n_experts, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.leaky_relu(self.down1(e1), inplace=True))
        b  = self.bott(F.leaky_relu(self.down2(e2), inplace=True))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.head(d1)
        return F.softmax(logits, dim=1)  # (B, M, D, H, W)

class SAGNEF(nn.Module):
    """
    Spatially-Adaptive Gated Nonlinear Ensemble Fusion.
    Blends expert volumes with gate weights. Experts are inputs, not trainable here.
    """
    def __init__(self, in_ch: int, n_experts: int, wf: int = 24):
        super().__init__()
        self.gate = GatingUNet3D(in_ch=in_ch, n_experts=n_experts, wf=wf)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, D, H, W), where C = (#experts) + (optionally LR)
        Returns:
          y_hat: (B, 1, D, H, W)
          W:     (B, M, D, H, W) softmax weights
        """
        # Experts are last M channels
        # If LR is included, convention: channel 0=LR, then experts 1..M
        # Else, experts 0..M-1
        B, C, D, H, W = x.shape
        # Detect expert slice
        # Here we assume we always know M by module
        Wm = self.gate(x)  # (B, M, D, H, W)
        # Extract expert stack from x
        M = Wm.shape[1]
        experts = x[:, -M:, ...]  # (B, M, D, H, W)
        y_hat = (Wm * experts).sum(dim=1, keepdim=True)  # convex mix
        return y_hat, Wm
