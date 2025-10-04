# -*- coding: utf-8 -*-
"""
Lightweight dataset that samples 3D patches given manifest entries.
"""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from .io import load_nii, to_np, normalize, SubjectEntry

Array = np.ndarray

class PatchDataset(Dataset):
    """
    Samples random 3D patches from subjects.
    Inputs are concatenated channels: [LR?] + experts(4).
    Target is HR.
    """
    def __init__(self,
                 entries: List[SubjectEntry],
                 patch_size: Tuple[int, int, int],
                 patches_per_volume: int,
                 include_lr: bool,
                 norm_mode: str,
                 norm_percentiles: Tuple[float, float]):
        self.entries = entries
        self.patch_size = patch_size
        self.ppv = patches_per_volume
        self.include_lr = include_lr
        self.norm_mode = norm_mode
        self.p_lo, self.p_hi = norm_percentiles

        # preload as arrays for speed
        self._vols: List[Dict[str, Array]] = []
        for s in self.entries:
            hr = to_np(load_nii(s.hr_path))
            hr = normalize(hr, mode=self.norm_mode, p_lo=self.p_lo, p_hi=self.p_hi)
            chans: List[Array] = []
            if self.include_lr and s.lr_path:
                lr = to_np(load_nii(s.lr_path))
                lr = normalize(lr, mode=self.norm_mode, p_lo=self.p_lo, p_hi=self.p_hi)
                chans.append(lr[None])
            for e in sorted(s.expert_paths.keys()):
                ex = to_np(load_nii(s.expert_paths[e]))
                ex = normalize(ex, mode=self.norm_mode, p_lo=self.p_lo, p_hi=self.p_hi)
                chans.append(ex[None])
            X = np.concatenate(chans, axis=0)  # (C, D, H, W)
            self._vols.append(dict(X=X, Y=hr[None], shape=hr.shape)) #type: ignore

        # index mapping: (vol_idx, rnd_coords)
        self._index = []
        for vi, v in enumerate(self._vols):
            D, H, W = v["shape"]
            Dz, Hy, Wx = self.patch_size
            for _ in range(self.ppv):
                z = np.random.randint(0, max(D - Dz + 1, 1))
                y = np.random.randint(0, max(H - Hy + 1, 1))
                x = np.random.randint(0, max(W - Wx + 1, 1))
                self._index.append((vi, (z, y, x)))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        vi, (z, y, x) = self._index[idx]
        X = self._vols[vi]["X"]
        Y = self._vols[vi]["Y"]
        Dz, Hy, Wx = self.patch_size
        xp = X[:, z:z+Dz, y:y+Hy, x:x+Wx]
        yp = Y[:, z:z+Dz, y:y+Hy, x:x+Wx]
        return torch.from_numpy(xp.copy()), torch.from_numpy(yp.copy())
