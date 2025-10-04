# -*- coding: utf-8 -*-
"""
3D patching and overlap-add reconstruction with optional Hann blending.
"""
from __future__ import annotations
from typing import Tuple, Iterator, Optional
import numpy as np

Array = np.ndarray

def _hann3d(sz: Tuple[int, int, int]) -> Array:
    def _hann(n):  # inclusive hann
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / max(n - 1, 1))
    wx, wy, wz = (_hann(sz[0]), _hann(sz[1]), _hann(sz[2]))
    w = wx[:, None, None] * wy[None, :, None] * wz[None, None, :]
    w = w / (w.max() + 1e-8)
    return w.astype(np.float32)

def sliding_windows(shape: Tuple[int, int, int],
                    patch: Tuple[int, int, int],
                    stride: Tuple[int, int, int]) -> Iterator[Tuple[slice, slice, slice]]:
    D, H, W = shape
    pD, pH, pW = patch
    sD, sH, sW = stride
    for z in range(0, max(D - pD + 1, 1), sD):
        for y in range(0, max(H - pH + 1, 1), sH):
            for x in range(0, max(W - pW + 1, 1), sW):
                yield (slice(z, z + pD), slice(y, y + pH), slice(x, x + pW))

def reconstruct_overlap_add(patches: Array,
                            coords: np.ndarray,
                            vol_shape: Tuple[int, int, int],
                            hann: bool = True) -> Array:
    """
    Overlap-add using optional 3D Hann weights.
    patches:  (N, C, Dp, Hp, Wp)
    coords:   (N, 3) start indices (z, y, x) for each patch
    returns:  (C, D, H, W)
    """
    N, C, Dp, Hp, Wp = patches.shape
    out = np.zeros((C,) + vol_shape, dtype=np.float32)
    acc = np.zeros((1,) + vol_shape, dtype=np.float32)
    win = _hann3d((Dp, Hp, Wp))[None, ...] if hann else np.ones((1, Dp, Hp, Wp), np.float32)

    for i in range(N):
        z, y, x = coords[i]
        out[:, z:z+Dp, y:y+Hp, x:x+Wp] += patches[i] * win
        acc[:, z:z+Dp, y:y+Hp, x:x+Wp] += win

    out /= np.maximum(acc, 1e-6)
    return out
