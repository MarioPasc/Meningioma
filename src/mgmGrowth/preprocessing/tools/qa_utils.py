#!/usr/bin/env python3
"""
Minimal quality-assurance helpers used to validate preprocessing steps.

The functions return *dicts* so calling code can decide whether to log,
raise, or ignore issues.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

import numpy as np
import SimpleITK as sitk


__all__ = [
    "IntensityStats",
    "intensity_summary",
    "geometry_summary",
    "transform_sanity_check",
]


@dataclass(frozen=True)
class IntensityStats:
    mean: float
    std: float
    p01: float
    p99: float
    min: float
    max: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def intensity_summary(image: sitk.Image) -> IntensityStats:
    """Return common statistics for *image* (whole volume)."""
    arr = sitk.GetArrayFromImage(image).astype(np.float64)
    arr = arr[arr != 0]  # ignore padding
    return IntensityStats(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        p01=float(np.percentile(arr, 1)),
        p99=float(np.percentile(arr, 99)),
        min=float(arr.min()),
        max=float(arr.max()),
    )


def geometry_summary(image: sitk.Image) -> Dict[str, Any]:
    """Return size, spacing, direction cosines."""
    return {
        "size": image.GetSize(),
        "spacing": image.GetSpacing(),
        "direction": tuple(round(v, 3) for v in image.GetDirection()),
        "origin": tuple(round(o, 2) for o in image.GetOrigin()),
    }


def transform_sanity_check(transform_path: str | Path) -> Dict[str, Any]:
    """
    Perform a crude sanity check on an ANTs **text** affine transform
    (…`GenericAffine.mat`). Binary composite files (`.h5`) are skipped.

    Currently: parse the 12 numbers and ensure scale / shear are
    within ±10 and translation below ±200 mm.  Flags `ok = False`
    if any crude limit is violated.
    """
    import numpy as np

    transform_path = Path(transform_path)
    if transform_path.suffix.lower() == ".h5":
        # Binary composite → cannot parse; mark as "unchecked".
        return {"ok": None, "reason": "binary_composite"}

    mat = np.loadtxt(transform_path, comments="#")
    # ANTs affines are 3x4 (last col is translation)
    scale_shear = mat[:, :3].ravel()
    translation = mat[:, 3]

    ok = (
        np.all(np.abs(scale_shear) < 10.0)
        and np.all(np.abs(translation) < 200.0)
    )

    return {
        "scale_shear_max": float(np.abs(scale_shear).max()),
        "translation_max_mm": float(np.abs(translation).max()),
        "ok": bool(ok),
    }
