#!/usr/bin/env python3
"""
MRI intensity-normalisation utilities.

Two complementary strategies are provided:

1. `zscore_normalise` – per-scan z-score normalisation using either a
   binary mask (brain voxels) or all non-zero voxels.

2. `histogram_match` – histogram matching of a *moving* image to a
   *reference* image using SimpleITK's built-in filter.

Both functions return a **new** SimpleITK image (the input is not
modified) and a dictionary with the parameters applied – useful for
logging or downstream QA.
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import numpy as np
import SimpleITK as sitk


__all__ = [
    "zscore_normalise",
    "histogram_match",
]


def _get_foreground_array(
    image: sitk.Image, mask: Optional[sitk.Image] = None
) -> np.ndarray:
    """
    Extract voxel intensities belonging to the foreground mask.

    If *mask* is ``None`` every non-zero voxel in *image* is considered
    foreground.

    Returns a 1-D NumPy array.
    """
    data = sitk.GetArrayFromImage(image).astype(np.float64)

    if mask is None:
        return data[data != 0]

    mask_arr = sitk.GetArrayFromImage(mask).astype(bool)
    return data[mask_arr]


def zscore_normalise(
    image: sitk.Image,
    mask: Optional[sitk.Image] = None,
    clip: bool = True,
    clip_range: Tuple[float, float] = (-3.0, 3.0),
) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Per-scan z-score normalisation.

    Parameters
    ----------
    image
        Input MRI volume.
    mask
        Binary mask defining foreground voxels (typically a brain mask).
        If ``None`` uses all non-zero voxels.
    clip
        If ``True`` clamp normalised intensities to *clip_range*.
    clip_range
        Lower / upper bounds (in z-score units) for clipping.

    Returns
    -------
    (normalised_image, info_dict)
        *normalised_image* is a new ``sitk.Image`` (type ``sitkFloat32``);
        *info_dict* holds the ``mean`` and ``std`` applied.
    """
    fg = _get_foreground_array(image, mask)

    if fg.size == 0:
        raise ValueError("No foreground voxels found for normalisation.")

    mu, sigma = float(fg.mean()), float(fg.std(ddof=0))
    if sigma == 0.0:
        raise RuntimeError("Zero intensity standard-deviation – cannot normalise.")

    image_arr = sitk.GetArrayFromImage(image).astype(np.float32)
    norm_arr = (image_arr - mu) / sigma

    if clip:
        lo, hi = clip_range
        norm_arr = np.clip(norm_arr, lo, hi)

    norm_img = sitk.GetImageFromArray(norm_arr)
    norm_img.CopyInformation(image)

    return norm_img, {"mean": mu, "std": sigma, "clipped": clip}


def histogram_match(
    moving_image: sitk.Image,
    reference_image: sitk.Image,
    number_of_histogram_levels: int = 256,
    number_of_match_points: int = 50,
) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Histogram match *moving_image* to *reference_image*.

    Parameters mirror SimpleITK's ``HistogramMatchingImageFilter``.

    Returns the matched image (float32) and a dict describing the
    parameters used.
    """
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(number_of_histogram_levels)
    matcher.SetNumberOfMatchPoints(number_of_match_points)
    matcher.ThresholdAtMeanIntensityOn()

    matched = matcher.Execute(moving_image, reference_image)
    matched = sitk.Cast(matched, sitk.sitkFloat32)

    return matched, {
        "histogram_levels": number_of_histogram_levels,
        "match_points": number_of_match_points,
    }
