#!/usr/bin/env python3
"""
compute_sr_metrics.py
=====================

Quantify SR reconstruction quality (PSNR, SSIM, Bhattacharyya coefficient)
against the native high-resolution (HR) images.

Folder layout (example)
-----------------------
  HR_ROOT/
      BraTS-MEN-00018-000/
          BraTS-MEN-00018-000-t1c.nii.gz
          BraTS-MEN-00018-000-t2w.nii.gz
          BraTS-MEN-00018-000-seg.nii.gz
      ...
  RESULTS_ROOT/
      SMORE/
          3mm/output_volumes/*.nii.gz
          5mm/output_volumes/*.nii.gz
          7mm/output_volumes/*.nii.gz
      BSpline/
          3mm/output_volumes/*.nii.gz
          ...

Output
------
`metrics.npz` containing

    metrics      (P, 2, 3, M, 3, 4) float64   # NaN for missing cases
    patient_ids  list[str]
    pulses       ["t1c", "t2w"]
    resolutions  [3, 5, 7]                    # mm
    models       list[str]
    metric_names ["PSNR", "SSIM", "BC"]
    roi_labels   ["all", "core", "edema", "surround"]
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import pathlib
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import warnings
warnings.filterwarnings("ignore",
                        message="Mean of empty slice")
warnings.filterwarnings("ignore",
                        message="invalid value encountered in divide")


import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim

from mgmGrowth.tasks.superresolution import LOGGER

# ------------------------------------------------------------- constants ---
ROI_LABELS = ("all", "core", "edema", "surround")   # 0,1,2,3
METRIC_NAMES = ("PSNR", "SSIM", "BC")
STAT_NAMES   = ("mean", "std")          # NEW axis length = 2
PULSES = ("t1c", "t2w")
RESOLUTIONS = (3, 5, 7)                             # mm
HR_RE = re.compile(r"^(?P<pid>[^/]+)-(?P<pulse>t1c|t2w)\.nii\.gz$")

# ---------------------------------------------------------- dataclasses ---
@dataclass(frozen=True)
class VolumePaths:
    hr: pathlib.Path
    seg: pathlib.Path
    sr: pathlib.Path


# ----------------------------------------------------------- utils --------
def read_image(path: pathlib.Path) -> sitk.Image:
    """SimpleITK reader with float64 output."""
    img = sitk.ReadImage(str(path))
    return sitk.Cast(img, sitk.sitkFloat64)


def sitk_to_np(img: sitk.Image) -> np.ndarray:
    """Preserve ordering (z, y, x) for numpy."""
    arr = sitk.GetArrayFromImage(img)        # returns z,y,x
    return np.asanyarray(arr, dtype=np.float64)


def match_slices(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    If *a* and *b* differ by ≤2 slices along *any* axis, pad the smaller
    one with zeros so both arrays share the same shape.
    """
    if a.shape == b.shape:
        return a, b
    diffs = np.subtract(a.shape, b.shape)
    if np.any(np.abs(diffs) > 2):
        raise ValueError(f"Images differ by >2 voxels {a.shape} vs {b.shape}")
    # pad smaller
    pad_a = tuple((0, max(0, -d)) for d in diffs)
    pad_b = tuple((0, max(0,  d)) for d in diffs)
    return np.pad(a, pad_a, constant_values=0), np.pad(b, pad_b, constant_values=0)


def exclude_z_slices(arr: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """
    Return *arr* with the Z-slices listed in *idx* removed.

    Any index < 0 or ≥ arr.shape[0] is silently ignored to avoid IndexError.
    """
    if not idx:                                   # empty / None → no-op
        return arr

    z = arr.shape[0]
    idx = np.asarray(idx, dtype=int)
    in_bounds = (idx >= 0) & (idx < z)
    if not in_bounds.all():                       # tell the user once
        bad = idx[~in_bounds]
        LOGGER.debug("Ignoring %d slice indices outside [0,%d]: %s",
                     bad.size, z - 1, bad.tolist())
    idx = idx[in_bounds]

    mask = np.ones(z, dtype=bool)
    mask[idx] = False
    return arr[mask, ...]

# --------------------------------------- metric helpers (robust to NaN/Inf)
def _finite(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def psnr(hr: np.ndarray, sr: np.ndarray, data_range: float | None = None) -> float:
    hr, sr = _finite(hr, sr)
    if data_range is None:
        data_range = np.ptp(hr)
    mse = np.mean((hr - sr) ** 2)
    if mse == 0:                                            # SAFE-SSIM
        return np.inf
    if data_range == 0:
        return np.nan
    return 20 * np.log10(data_range) - 10 * np.log10(mse)

# SKIMAGE-SSIM ────────────────────────────────────────────────────────────
def safe_ssim_slice(hr2d: np.ndarray,
                    sr2d: np.ndarray,
                    mask2d: np.ndarray) -> float:
    """
    SSIM on a *single axial slice* using skimage.

    * Chooses the largest odd ``win_size`` ≤ min(height, width).
    * When the side is < 3 pixels the function returns **NaN** instead of
      raising `ValueError`.
    """
    # nothing to compare?
    if not mask2d.any():
        return np.nan

    h, w = hr2d.shape
    m = min(h, w)

    if m < 3:                            # too small for any legal window
        LOGGER.debug("ROI too small for SSIM (%dx%d) → NaN", h, w)
        return np.nan

    # choose an odd win_size that fits
    ws = 7
    if m < 7:
        ws = m if m % 2 == 1 else m - 1

    try:
        return float(
            ssim(hr2d,
                 sr2d,
                 data_range=float(np.ptp(hr2d)),
                 gaussian_weights=True,
                 win_size=ws,
                 mask=mask2d)             # ← only voxels inside ROI
        )
    except ValueError as err:
        LOGGER.debug("SSIM failed (%s) → NaN", err)
        return np.nan


def bhattacharyya(hr: np.ndarray, sr: np.ndarray, bins: int = 256) -> float:
    hr, sr = _finite(hr, sr)
    if hr.size == 0 or sr.size == 0:
        return np.nan
    h_hist, _ = np.histogram(hr, bins=bins, density=True)
    s_hist, _ = np.histogram(sr, bins=bins, density=True)
    return float(np.sum(np.sqrt(h_hist * s_hist)))


# ------------------------------------------------------ per-ROI metrics ---
def roi_mask(seg: np.ndarray, label: int | None) -> np.ndarray:
    if label is None:
        return np.ones_like(seg, dtype=bool)
    return seg == label


def compute_metrics(hr: np.ndarray, sr: np.ndarray, seg: np.ndarray) -> np.ndarray: # (3,4,2)   # SLICE-WISE
    """
    Slice-wise metrics – returns array (3 metrics, 4 ROIs, 2 stats).

    * stats[0] = mean across retained slices
    * stats[1] = std  across retained slices
    """
    out = np.full((len(METRIC_NAMES),
                   len(ROI_LABELS),
                   len(STAT_NAMES)), np.nan, dtype=np.float64)

    z_slices = hr.shape[0]
    for ridx, label in enumerate((None, 1, 2, 3)):
        # gather per-slice values
        psnr_list, ssim_list, bc_list = [], [], []
        for z in range(z_slices):
            roi = roi_mask(seg[z], label)               # SKIMAGE-SSIM
            if roi.sum() == 0:
                continue

            # PSNR & Bhattacharyya on masked voxels (1-D vectors)
            h_vec, s_vec = hr[z][roi], sr[z][roi]
            psnr_list.append(psnr(h_vec, s_vec))
            bc_list.append(bhattacharyya(h_vec, s_vec))

            # SSIM on the full 2-D slice with mask
            ssim_list.append(
                safe_ssim_slice(hr[z], sr[z], roi)
            )

        for vals, midx in zip((psnr_list, ssim_list, bc_list),
                              range(len(METRIC_NAMES))):
            if vals:                      # at least one valid slice
                out[midx, ridx, 0] = np.nanmean(vals)
                out[midx, ridx, 1] = np.nanstd(vals)
    return out


# ----------------------------------------------------- filesystem walker --
def collect_paths(hr_root: pathlib.Path,
                  results_root: pathlib.Path,
                  pulse: str,
                  model: str,
                  resolution_mm: int) -> Iterable[VolumePaths]:
    """
    Yield VolumePaths for every patient that has both HR and SR volumes.
    """
    res_dir = results_root / model / f"{resolution_mm}mm" / "output_volumes"
    for patient_dir in sorted(hr_root.iterdir()):
        pid = patient_dir.name
        hr_path = patient_dir / f"{pid}-{pulse}.nii.gz"
        seg_path = patient_dir / f"{pid}-seg.nii.gz"
        sr_path = res_dir / f"{pid}-{pulse}.nii.gz"
        if hr_path.exists() and seg_path.exists() and sr_path.exists():
            yield VolumePaths(hr=hr_path, seg=seg_path, sr=sr_path)


# ------------------------------------------------ worker (per patient) ----
def process_patient(vpaths: VolumePaths,
                    exclude: Sequence[int]) -> np.ndarray:
    hr_img = read_image(vpaths.hr)
    sr_img = read_image(vpaths.sr)
    seg_img = read_image(vpaths.seg)

    hr_arr = sitk_to_np(hr_img)
    sr_arr = sitk_to_np(sr_img)
    seg_arr = sitk_to_np(seg_img).astype(int)

    hr_arr, sr_arr = match_slices(hr_arr, sr_arr)
    hr_arr = exclude_z_slices(hr_arr, exclude)
    sr_arr = exclude_z_slices(sr_arr, exclude)
    seg_arr = exclude_z_slices(seg_arr, exclude)



    return compute_metrics(hr_arr, sr_arr, seg_arr)   # (3,4)


# ------------------------------------------------------------- main -------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute PSNR/SSIM/Bhattacharyya for SR volumes.")
    ap.add_argument("--hr_root",  type=pathlib.Path,
                    default=pathlib.Path(
                    "/home/mariopasc/Python/Datasets/Meningiomas/BraTS/SR/subset"))
    ap.add_argument("--results_root", type=pathlib.Path,
                    default=pathlib.Path(
                    "/home/mariopasc/Python/Results/Meningioma/super_resolution/models"))

    # -------- pulse ----------------------------------------------------------
    ap.add_argument("--pulse",  choices=("t1c", "t2w", "both"),
                    default="both",
                    help="Evaluate given pulse or both (default)")

    # -------- slice window ---------------------------------------------------
    ap.add_argument("--slice-window", nargs=2, type=int, metavar=("MIN", "MAX"),
                    default=(10, 140),
                    help="Only consider slices MIN..MAX inclusive "
                        "(default 10-140); outside range is ignored.")

    ap.add_argument("--out",   type=pathlib.Path,
                    default=pathlib.Path(
                    "/home/mariopasc/Python/Results/Meningioma/super_resolution/metrics/metrics.npz"))
    ap.add_argument("--workers", type=int, default=mp.cpu_count()-1)
    args = ap.parse_args()


    # discover models dynamically
    model_dirs = [d.name for d in (args.results_root).iterdir() if d.is_dir()]
    model_dirs.sort()

    # convert tuple → range of indices to drop
    lo, hi = args.slice_window
    exclude = list(range(0, lo))               # [0, lo-1]
    exclude += list(range(hi+1, 10**6))        # (hi, ∞) -- upper bound trimmed later

    # ---------------------------------------------------------------- engine --
    patients = sorted([d.name for d in args.hr_root.iterdir()])
    # SLICE-WISE – added STAT dimension
    metrics_arr = np.full((len(patients),             # P
                           len(PULSES),               # 2
                           len(RESOLUTIONS),          # 3
                           len(model_dirs),
                           len(METRIC_NAMES),         # 3
                           len(ROI_LABELS),           # 4
                           len(STAT_NAMES)),          # 2
                          np.nan, dtype=np.float64)

    patient_idx = {pid: i for i, pid in enumerate(patients)}

    pulse_list = PULSES if args.pulse == "both" else [args.pulse]

    for m_idx, model in enumerate(model_dirs):
        for pulse in pulse_list:
            for r_idx, res in enumerate(RESOLUTIONS):

                items = list(collect_paths(args.hr_root,
                                        args.results_root,
                                        pulse,
                                        model,
                                        res))
                if not items:
                    continue
                LOGGER.info("Model %s | %d patients | 1x1x%d mm", model, len(items), res)

                with mp.Pool(args.workers) as pool:
                    res_metrics = pool.starmap(process_patient,
                                            [(vp, exclude) for vp in items])

                for vp, metr in zip(items, res_metrics):
                    p = patient_idx[vp.hr.parent.name]
                    pulse_idx = PULSES.index(pulse)
                    metrics_arr[p, pulse_idx, r_idx, m_idx, :, :, :] = metr   # SLICE-WISE
                    LOGGER.info("Done %s -> %.2f±%.2f dB (PSNR, all-slices)", vp.hr.parent.name, metr[0, 0, 0], metr[0, 0, 1])

    np.savez(args.out,
             metrics=metrics_arr,
             patient_ids=patients,
             pulses=PULSES,
             resolutions_mm=RESOLUTIONS,
             models=model_dirs,
             metric_names=METRIC_NAMES,
             roi_labels=ROI_LABELS,
             stat_names=STAT_NAMES)                         # SLICE-WISE

    LOGGER.info("Saved metrics to %s", args.out)

"""
metrics_arr.shape
# (P, 2, 3, M, 3, 4, 2)
#  ↑  ↑  ↑  ↑  ↑  ↑  └── mean / std  (STAT_NAMES)
#  |  |  |  |  |  └──── ROI_LABELS
#  |  |  |  |  └────── METRIC_NAMES
#  |  |  |  └───────── model index
#  |  |  └──────────── resolution {3,5,7} mm
#  |  └─────────────── pulse {t1c,t2w}
#  └────────────────── patient

"""

if __name__ == "__main__":
    main()
