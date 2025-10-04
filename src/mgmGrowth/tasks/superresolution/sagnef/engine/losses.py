# -*- coding: utf-8 -*-
"""
Losses for SAGNEF: MSE + (1-SSIM) + TV(weights) + entropy(weights).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class LossConfig:
    w_mse: float
    w_ssim: float
    w_tv: float
    w_entropy: float
    ssim_window: int
    ssim_sigma: float
    ssim_K1: float
    ssim_K2: float

class MSE3D(nn.Module):
    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(y_hat, y)

class SSIM3D(nn.Module):
    """
    Differentiable SSIM for 3D using Gaussian window.
    Returns: 1 - mean SSIM in [0,1].
    """
    def __init__(self, win: int = 7, sigma: float = 1.5, K1: float = 0.01, K2: float = 0.03):
        super().__init__()
        self.win = win
        self.sigma = sigma
        self.K1 = K1
        self.K2 = K2

        # 3D separable Gaussian
        ax = torch.arange(win, dtype=torch.float32) - (win - 1) / 2.0
        gauss_1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        gauss_1d = (gauss_1d / gauss_1d.sum()).view(1, 1, win, 1, 1)
        self.register_buffer("gD", gauss_1d)
        self.register_buffer("gH", gauss_1d.transpose(2, 3))
        self.register_buffer("gW", gauss_1d.transpose(2, 4))

    def _filt(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv3d(x, self.gD, padding=(self.win//2,0,0), groups=x.shape[1])
        x = F.conv3d(x, self.gH, padding=(0,self.win//2,0), groups=x.shape[1])
        x = F.conv3d(x, self.gW, padding=(0,0,self.win//2), groups=x.shape[1])
        return x

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        C1 = (self.K1 ** 2)
        C2 = (self.K2 ** 2)
        mu_x = self._filt(y_hat)
        mu_y = self._filt(y)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = self._filt(y_hat * y_hat) - mu_x2
        sigma_y2 = self._filt(y * y) - mu_y2
        sigma_xy = self._filt(y_hat * y) - mu_xy
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
        loss = 1.0 - ssim_map.mean()
        return loss

class TVWeights(nn.Module):
    """Isotropic TV on gating weights W: sum over expert channels."""
    def forward(self, W: torch.Tensor) -> torch.Tensor:
        dz = W[..., 1:, :, :] - W[..., :-1, :, :]
        dy = W[..., :, 1:, :] - W[..., :, :-1, :]
        dx = W[..., :, :, 1:] - W[..., :, :, :-1]
        tv = (dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()).sqrt()
        return tv

class EntropyWeights(nn.Module):
    """Encourage confident but not peaky gates. Lower entropy penalty."""
    def forward(self, W: torch.Tensor) -> torch.Tensor:
        eps = 1e-8
        H = -(W * (W + eps).log()).sum(dim=1).mean()
        return H

class SAGNEFLoss(nn.Module):
    """
    Combine terms; returns dict with components for logging.
    """
    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.mse = MSE3D()
        self.ssim = SSIM3D(cfg.ssim_window, cfg.ssim_sigma, cfg.ssim_K1, cfg.ssim_K2)
        self.tv = TVWeights()
        self.ent = EntropyWeights()
        self.w_mse = cfg.w_mse
        self.w_ssim = cfg.w_ssim
        self.w_tv = cfg.w_tv
        self.w_entropy = cfg.w_entropy

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor, W: torch.Tensor) -> Dict[str, torch.Tensor]:
        mse = self.mse(y_hat, y)
        ssim = self.ssim(y_hat, y)
        tv = self.tv(W)
        ent = self.ent(W)
        total = self.w_mse * mse + self.w_ssim * ssim + self.w_tv * tv + self.w_entropy * ent
        return {"total": total, "mse": mse, "ssim": ssim, "tv": tv, "entropy": ent}
