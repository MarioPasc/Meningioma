# file: src/mgmGrowth/tasks/superresolution/tools/metrics.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

# ---------------- I/O & normalisation ---------------------------------
def _load(path: Path) -> NDArray[np.float32]:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def _norm(v: NDArray[np.float32]) -> NDArray[np.float32]:
    lo, hi = np.percentile(v, (0.5, 99.5))
    return ((v - lo) / (hi - lo + 1e-8)).clip(0.0, 1.0)


# ---------------- metrics ---------------------------------------------
def _psnr(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    mse = np.mean((a - b) ** 2, dtype=np.float64)
    return float("inf") if mse == 0 else 10.0 * np.log10(1.0 / mse)


def _ssim_vec(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    K1, K2, L = 0.01, 0.03, 1.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a2 = ((a - mu_a) ** 2).mean()
    sigma_b2 = ((b - mu_b) ** 2).mean()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
    den = (mu_a**2 + mu_b**2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return float(num / den) if den else 1.0


def _mi(a: NDArray[np.float32], b: NDArray[np.float32], bins: int = 64) -> float:
    ha, _ = np.histogram(a, bins=bins, range=(0, 1), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(0, 1), density=True)
    hab, _, _ = np.histogram2d(a, b, bins=bins, range=((0, 1), (0, 1)), density=True)
    ha += 1e-12
    hb += 1e-12
    hab += 1e-12
    return float(np.sum(hab * np.log(hab / (ha[:, None] * hb[None, :]))))


# ---------------- public API ------------------------------------------
def metrics_regions(
    gt: Path,
    sr: Path,
    seg: Path | None,
) -> NDArray[np.float32]:
    """
    Compute [[PSNR, SSIM, MI] x 4]  (whole + 3 tumour ROIs).
    """
    gt_n, sr_n = map(_norm, (_load(gt), _load(sr)))
    out = np.full((4, 3), np.nan, dtype=np.float32)

    seg_arr = None
    if seg and seg.exists():
        seg_arr = _load(seg).astype(np.int16)

    for i, rid in enumerate([None, 1, 2, 3]):
        if rid is None:
            mask = np.ones_like(gt_n, dtype=bool)
        else:
            if seg_arr is None:
                continue
            mask = seg_arr == rid
            if not np.any(mask):
                continue

        a, b = gt_n[mask], sr_n[mask]
        out[i, 0] = _psnr(a, b)
        out[i, 1] = _ssim_vec(a, b)
        out[i, 2] = _mi(a, b)

    return out  # (4,3)


# ---------------- helper to map LR to GT & SEG ------------------------
def matching_gt_seg(lr_path: Path, orig_root: Path) -> Tuple[Path, Path | None]:
    patient = lr_path.parent.name
    gt = orig_root / patient / lr_path.name
    seg = orig_root / patient / f"{patient}-seg.nii.gz"
    return gt, seg if seg.exists() else None
