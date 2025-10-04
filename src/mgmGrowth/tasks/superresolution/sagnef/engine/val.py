# -*- coding: utf-8 -*-
"""
Full-volume validation utilities.
"""
from __future__ import annotations
from typing import Dict, Tuple, List
import os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.mgmGrowth.tasks.superresolution.sagnef.utils.io import load_nii, to_np, normalize, save_like, SubjectEntry
from src.mgmGrowth.tasks.superresolution.sagnef.utils.patch import sliding_windows, reconstruct_overlap_add
from src.mgmGrowth.tasks.superresolution.sagnef.engine.metrics import psnr, rmse, ssim3d

def predict_volume(model, vol_stack: np.ndarray,
                   patch: Tuple[int,int,int], stride: Tuple[int,int,int],
                   hann: bool, device: torch.device) -> np.ndarray:
    """
    vol_stack: (C, D, H, W) numpy
    returns y_hat: (1, D, H, W) numpy
    """
    C, D, H, W = vol_stack.shape
    coords, patches = [], []
    Dz, Hy, Wx = patch
    for sl in sliding_windows((D, H, W), patch, stride):
        z, y, x = sl[0].start, sl[1].start, sl[2].start
        crop = vol_stack[:, sl[0], sl[1], sl[2]][None]  # (1,C,Dp,Hp,Wp)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            inp = torch.from_numpy(crop).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                y_hat, _ = model(inp)
        patches.append(y_hat.detach().cpu().numpy()[0])
        coords.append((z, y, x))
    patches = np.stack(patches, axis=0)  # (N, 1, Dp, Hp, Wp)
    coords = np.asarray(coords, dtype=np.int32)
    recon = reconstruct_overlap_add(patches, coords, (D, H, W), hann=hann)
    return recon

def evaluate_fold(model, entries: List[SubjectEntry], cfg, out_dir: str,
                  include_lr: bool, experts_sorted: List[str], device: torch.device) -> Dict:
    """
    Save test predictions; compute metrics vs HR.
    """
    os.makedirs(out_dir, exist_ok=True)
    stats = []
    for s in entries:
        hr_img = load_nii(s.hr_path)
        hr = to_np(hr_img)
        # build input stack
        chans = []
        if include_lr and s.lr_path:
            chans.append(normalize(to_np(load_nii(s.lr_path)), cfg.data["normalization"],
                                   cfg.data["norm_percentiles"][0], cfg.data["norm_percentiles"][1])[None])
        for e in experts_sorted:
            chans.append(normalize(to_np(load_nii(s.expert_paths[e])), cfg.data["normalization"],
                                   cfg.data["norm_percentiles"][0], cfg.data["norm_percentiles"][1])[None])
        X = np.concatenate(chans, axis=0)  # (C,D,H,W)
        y_hat = predict_volume(model, X, tuple(cfg.patch["patch_size"]),
                               tuple(cfg.patch["stride"]), cfg.patch["hann_blend"], device)
        # reverse normalization of HR’s scale? Here metrics use normalized space.
        y_hat_t = torch.from_numpy(y_hat[None]).float()
        y_t = torch.from_numpy(hr[None, None]).float()
        ps, rm, ss = psnr(y_hat_t, y_t), rmse(y_hat_t, y_t), ssim3d(y_hat_t, y_t)
        # save prediction with HR geometry
        pred_path = os.path.join(out_dir, os.path.basename(s.hr_path))
        save_like(y_hat[0], hr_img, pred_path)
        stats.append(dict(pid=s.pid, psnr=ps, rmse=rm, ssim=ss, pred_path=pred_path))
    return {"records": stats}
