# -*- coding: utf-8 -*-
"""
Validation metrics.
"""
from __future__ import annotations
from typing import Dict
import torch
import torch.nn.functional as F
from src.mgmGrowth.tasks.superresolution.sagnef.engine.losses import SSIM3D

def psnr(y_hat: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    mse = F.mse_loss(y_hat, y).item()
    if mse == 0: return 99.0
    import math
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)

def rmse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return (F.mse_loss(y_hat, y).sqrt().item())

def ssim3d(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return (1.0 - SSIM3D()(y_hat, y)).item()
