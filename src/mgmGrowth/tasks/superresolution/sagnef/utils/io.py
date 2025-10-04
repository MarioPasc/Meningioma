# -*- coding: utf-8 -*-
"""
I/O and dataset utilities for SAGNEF.
"""
from __future__ import annotations
import json, os, shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import nibabel as nib

Array = np.ndarray

@dataclass(frozen=True)
class SubjectEntry:
    """Paths for one subject at fixed spacing and pulse."""
    pid: str
    expert_paths: Dict[str, str]      # {expert: nifti_path}
    lr_path: Optional[str]            # may be None
    hr_path: str

def load_manifest(cv_json: str) -> Dict[str, Dict[str, Dict[str, Dict]]]:
    """Load K-fold manifest JSON."""
    with open(cv_json, "r") as f:
        return json.load(f)

def entries_for_split(manifest: dict, fold: str, split: str,
                      spacing: str, pulse: str,
                      experts: List[str]) -> List[SubjectEntry]:
    """
    Extract SubjectEntry list for (fold, split, spacing, pulse).
    """
    out: List[SubjectEntry] = []
    for pid, blocks in manifest[fold][split].items():
        # experts block (e.g., "ECLARE": {"3mm": {"t1c": "..."}, ...})
        expert_paths = {}
        for e in experts:
            try:
                expert_paths[e] = blocks[e][spacing][pulse]
            except KeyError:
                raise KeyError(f"Missing path for {pid} {e} {spacing} {pulse}")
        # LR/HR blocks
        lr_path = blocks.get("LR", {}).get(spacing, {}).get(pulse, None)
        hr_path = blocks["HR"][spacing][pulse]
        out.append(SubjectEntry(pid=pid, expert_paths=expert_paths,
                                lr_path=lr_path, hr_path=hr_path))
    return out

def load_nii(path: str) -> nib.Nifti1Image:
    """Load NIfTI with mmap off for speed-safety when slicing."""
    return nib.load(path, mmap=False) # type: ignore

def save_like(data: Array, like_img: nib.Nifti1Image, out_path: str) -> None:
    """Save ndarray as NIfTI using 'like' header+affine."""
    nib.Nifti1Image(data, like_img.affine, like_img.header).to_filename(out_path)

def resample_like(src: nib.Nifti1Image, like: nib.Nifti1Image, order: int = 1) -> nib.Nifti1Image:
    """
    Resample src to match like (shape+affine). Linear by default.
    Requires nibabel[scipy].
    """
    try:
        from nibabel.processing import resample_from_to
    except Exception as e:
        raise RuntimeError("Resampling requires nibabel[scipy].") from e
    return resample_from_to(src, (like.shape, like.affine), order=order)

def nearly_same_shape(a: Tuple[int, ...], b: Tuple[int, ...], tol: int = 3) -> bool:
    """Shape match within L-infty voxel tolerance."""
    if len(a) != len(b): return False
    return max(abs(i - j) for i, j in zip(a, b)) <= tol

def dataset_sanity_fix(entries: List[SubjectEntry],
                       max_vox_diff: int = 3) -> None:
    """
    One-time pass: if any expert/LR deviates <= max_vox_diff voxels from HR,
    resample in-place to match HR grid. Overwrite the file safely.
    """
    for s in entries:
        hr = load_nii(s.hr_path)
        # Experts
        for e, p in s.expert_paths.items():
            img = load_nii(p)
            if not nearly_same_shape(img.shape, hr.shape, max_vox_diff):
                img_r = resample_like(img, hr, order=1)
                tmp = p + ".tmp.nii.gz"
                nib.save(img_r, tmp)
                shutil.move(tmp, p)
        # LR
        if s.lr_path:
            lr = load_nii(s.lr_path)
            if not nearly_same_shape(lr.shape, hr.shape, max_vox_diff):
                lr_r = resample_like(lr, hr, order=1)
                tmp = s.lr_path + ".tmp.nii.gz"
                nib.save(lr_r, tmp)
                shutil.move(tmp, s.lr_path)

def to_np(img: nib.Nifti1Image, dtype=np.float32) -> Array:
    """Get ndarray copy as dtype C-order."""
    return np.asarray(img.get_fdata(), dtype=dtype).copy(order="C")

def normalize(vol: Array, mode: str = "zscore",
              p_lo: float = 0.5, p_hi: float = 99.5,
              eps: float = 1e-6) -> Array:
    """Simple per-volume normalization."""
    if mode == "zscore":
        m, s = float(vol.mean()), float(vol.std() + eps)
        return (vol - m) / s
    elif mode == "percentile":
        lo, hi = np.percentile(vol, [p_lo, p_hi])
        vol = np.clip(vol, lo, hi)
        return (vol - lo) / (hi - lo + eps)
    else:
        return vol
