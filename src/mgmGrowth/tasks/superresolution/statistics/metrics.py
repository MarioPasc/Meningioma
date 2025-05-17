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

import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim

# ------------------------------------------------------------- constants ---
ROI_LABELS = ("all", "core", "edema", "surround")   # 0,1,2,3
METRIC_NAMES = ("PSNR", "SSIM", "BC")
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
    """Drop slices along z-axis given a list of indices."""
    mask = np.ones(arr.shape[0], dtype=bool)
    mask[list(idx)] = False
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
    if mse == 0:
        return np.inf
    return 20 * np.log10(data_range) - 10 * np.log10(mse)


def ssim_3d(hr: np.ndarray, sr: np.ndarray) -> float:
    hr, sr = _finite(hr, sr)
    # reshape to pseudo 2-D image (z, y*x) because skimage SSIM is 2-D
    z, y, x = hr.shape
    return float(
        ssim(hr.reshape(z, y * x),
             sr.reshape(z, y * x),
             data_range=np.ptp(hr), gaussian_weights=True)
    )


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


def compute_metrics(hr: np.ndarray, sr: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Return metrics array shape (3 metrics, 4 ROIs)."""
    metrics = np.empty((len(METRIC_NAMES), len(ROI_LABELS)), dtype=np.float64)
    for ridx, label in enumerate((None, 1, 2, 3)):       # None → full vol
        m = roi_mask(seg, label)
        hr_roi, sr_roi = hr[m], sr[m]
        metrics[0, ridx] = psnr(hr_roi, sr_roi)
        metrics[1, ridx] = ssim_3d(hr_roi.reshape(-1, 1, 1),
                                   sr_roi.reshape(-1, 1, 1))
        metrics[2, ridx] = bhattacharyya(hr_roi, sr_roi)
    return metrics


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
    ap.add_argument("hr_root", type=pathlib.Path)
    ap.add_argument("results_root", type=pathlib.Path)
    ap.add_argument("--pulse", choices=PULSES, required=True)
    ap.add_argument("--exclude-slices", nargs="*", type=int, default=())
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("metrics.npz"))
    ap.add_argument("--workers", type=int, default=mp.cpu_count() - 1)
    args = ap.parse_args()

    # discover models dynamically
    model_dirs = [d.name for d in (args.results_root).iterdir() if d.is_dir()]
    model_dirs.sort()

    patients = sorted([d.name for d in args.hr_root.iterdir()])
    metrics_arr = np.full((len(patients),            # P
                           len(PULSES),              # 2
                           len(RESOLUTIONS),         # 3
                           len(model_dirs),          # M
                           len(METRIC_NAMES),        # 3
                           len(ROI_LABELS)),         # 4
                          np.nan, dtype=np.float64)

    patient_idx = {pid: i for i, pid in enumerate(patients)}

    for m_idx, model in enumerate(model_dirs):
        for r_idx, res in enumerate(RESOLUTIONS):
            items = list(collect_paths(args.hr_root,
                                       args.results_root,
                                       args.pulse,
                                       model,
                                       res))
            if not items:
                continue

            with mp.Pool(args.workers) as pool:
                results = pool.starmap(process_patient,
                                       [(vp, args.exclude_slices) for vp in items])

            for vp, metr in zip(items, results):
                p = patient_idx[vp.hr.parent.name]   # map to array index
                pulse_idx = PULSES.index(args.pulse)
                metrics_arr[p, pulse_idx, r_idx, m_idx, :, :] = metr

    np.savez(args.out,
             metrics=metrics_arr,
             patient_ids=patients,
             pulses=PULSES,
             resolutions_mm=RESOLUTIONS,
             models=model_dirs,
             metric_names=METRIC_NAMES,
             roi_labels=ROI_LABELS)

    print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
