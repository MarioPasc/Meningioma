# file: src/mgmGrowth/tasks/superresolution/tools/metrics.py
"""PSNR / SSIM utilities with ROI support."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray
from pathlib import Path


# ---------- helpers --------------------------------------------------

def matching_gt_seg(
    lr_path: Path,
    orig_root: Path,
) -> tuple[Path, Path | None]:
    """
    Given the *low-res* volume path, return the *high-res* path and seg mask
    inside *orig_root*.

    Both trees share patient IDs and filenames.
    """
    patient = lr_path.parent.name                # BraTS-MEN-XXXXX-000
    gt_path = orig_root / patient / lr_path.name
    seg_path = orig_root / patient / (patient + "-seg.nii.gz")
    return gt_path, seg_path if seg_path.exists() else None


def _load(path: Path) -> Tuple[NDArray[np.float32], Tuple[float, float, float]]:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.float32), img.GetSpacing()


def _norm(arr: NDArray[np.float32]) -> NDArray[np.float32]:
    lo, hi = np.percentile(arr, (0.5, 99.5))
    return ((arr - lo) / (hi - lo + 1e-8)).clip(0.0, 1.0)


# ---------- PSNR -----------------------------------------------------


def _psnr(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    mse = np.mean((a - b) ** 2, dtype=np.float64)
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


# ---------- SSIM (vector version) ------------------------------------


def _ssim_vec(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    K1, K2, L = 0.01, 0.03, 1.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    mu_a, mu_b = a.mean(dtype=np.float64), b.mean(dtype=np.float64)
    sigma_a2 = ((a - mu_a) ** 2).mean(dtype=np.float64)
    sigma_b2 = ((b - mu_b) ** 2).mean(dtype=np.float64)
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean(dtype=np.float64)
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return float(num / den) if den else 1.0


# ---------- public API -----------------------------------------------


def psnr_ssim_regions(
    gt_path: Path,
    sr_path: Path,
    seg_path: Path | None,
) -> np.ndarray:
    """
    Return a (4, 2) array = [[PSNR, SSIM] for
    (whole, tumour_core, edema, surrounding_tumour)].

    If *seg_path* is None or a region is empty → NaN.
    """
    gt, _ = _load(gt_path)
    sr, _ = _load(sr_path)
    gt_n, sr_n = _norm(gt), _norm(sr)

    out = np.full((4, 2), np.nan, dtype=np.float32)

    # region ids None / 1 / 2 / 3
    for idx, rid in enumerate([None, 1, 2, 3]):
        if rid is None:
            mask = np.ones_like(gt, dtype=bool)
        else:
            if seg_path is None:
                continue
            seg = sitk.GetArrayFromImage(sitk.ReadImage(str(seg_path)))
            mask = seg == rid
            if mask.sum() == 0:
                continue

        a, b = gt_n[mask], sr_n[mask]
        out[idx, 0] = _psnr(a, b)
        out[idx, 1] = _ssim_vec(a, b)

    return out  # shape (4, 2)
