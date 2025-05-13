# file: src/mgmGrowth/tasks/superresolution/tools/downsample.py
"""Gaussian blur + block-average down-sampling in z (SimpleITK)."""
from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import numpy as np
import SimpleITK as sitk

from src.mgmGrowth.tasks.superresolution import LOGGER as _L
from .nifti_io import change_spacing_z


@lru_cache(maxsize=8)
def _sigma_phys(target_dz: float) -> float:
    """σ in **mm** such that FWHM ≈ target_dz (FWHM = 2.355 σ)."""
    return target_dz / 2.355


def downsample_z(
    vol: np.ndarray,
    spacing: Tuple[float, float, float],
    target_dz_mm: float,
    *,
    antialias: bool = True,
) -> tuple[np.ndarray, Tuple[float, float, float]]:
    """
    Return (downsampled_volume, new_spacing) where only z-spacing becomes *target_dz_mm*.
    """
    dx, dy, dz = spacing
    factor = int(round(target_dz_mm / dz))
    if factor < 1 or abs(factor * dz - target_dz_mm) > 1e-5:
        raise ValueError("target dz must be an integer multiple of current dz")

    if antialias:
        sigma_mm = _sigma_phys(target_dz_mm)
        _L.debug("RecursiveGaussian σ=%.3f mm along z", sigma_mm)
        img = sitk.GetImageFromArray(vol)
        img_blur = sitk.RecursiveGaussian(
            img, sigma=sigma_mm, direction=2  # 0=x, 1=y, 2=z
        )
        vol = sitk.GetArrayFromImage(img_blur)

    z_new = vol.shape[0] // factor
    trimmed = vol[: z_new * factor, :, :]
    vol_ds = trimmed.reshape(z_new, factor, *trimmed.shape[1:]).mean(axis=1)

    return vol_ds.astype(vol.dtype, copy=False), change_spacing_z(spacing, target_dz_mm)
