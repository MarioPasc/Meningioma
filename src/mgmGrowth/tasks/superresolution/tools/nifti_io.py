# file: src/mgmGrowth/tasks/superresolution/tools/nifti_io.py
"""SimpleITK-based NIfTI I/O with spacing helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import SimpleITK as sitk
import numpy as np

from src.mgmGrowth.tasks.superresolution import LOGGER as _L


def load_nifti(path: Path) -> tuple[np.ndarray, Tuple[float, float, float]]:
    """Return *array* and voxel spacing (dx, dy, dz)."""
    img = sitk.ReadImage(str(path))
    spacing = img.GetSpacing()  # (dx, dy, dz)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    return np.asarray(arr), spacing


def save_nifti(
    arr: np.ndarray,
    spacing: Tuple[float, float, float],
    out_path: Path,
    reference: Path | None = None,
) -> None:
    """
    Save *arr* with *spacing* to *out_path*.

    If *reference* is given, use its origin/direction to preserve geometry.
    """
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    if reference:
        ref_img = sitk.ReadImage(str(reference))
        img.SetDirection(ref_img.GetDirection())
        img.SetOrigin(ref_img.GetOrigin())
    sitk.WriteImage(img, str(out_path), True)
    _L.debug("Wrote %s", out_path)


def change_spacing_z(
    spacing: Tuple[float, float, float],
    new_dz: float,
) -> Tuple[float, float, float]:
    dx, dy, _ = spacing
    return dx, dy, new_dz
