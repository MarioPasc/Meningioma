#!/usr/bin/env python3
"""
Brain-mask utility helpers aimed at:

* Ensuring atlas masks have foreground = 1 and background = 0
  (the existing pipeline inverted this).

* Optional dilation of the mask so that extra-axial meningiomas are
  unlikely to be cropped away.

All functions are pure and side-effect-free.
"""

from __future__ import annotations

from typing import Tuple

import SimpleITK as sitk


__all__ = [
    "ensure_binary_polarity",
    "dilate_mask",
]


def ensure_binary_polarity(
    mask: sitk.Image, brain_is_one: bool = True
) -> sitk.Image:
    """
    Invert *mask* if its foreground / background polarity does not match
    the expected ``brain_is_one``.

    Detection heuristic: count of zero voxels vs. non-zero voxels.
    """
    arr = sitk.GetArrayFromImage(mask)
    zeros = (arr == 0).sum()
    nonzeros = arr.size - zeros
    brain_should_be_one = brain_is_one

    # Heuristic: assume the brain occupies < 60 % of the volume.
    # If that assumption fails, user must supply explicit polarity.
    brain_is_big = nonzeros > zeros  # then 1s > 0s

    needs_inversion = (brain_is_big and brain_should_be_one is False) or (
        not brain_is_big and brain_should_be_one is True
    )

    if needs_inversion:
        inverted = sitk.BinaryNot(mask)
        inverted.CopyInformation(mask)
        return inverted

    return mask  # unchanged


def dilate_mask(mask: sitk.Image, radius_mm: float = 2.0) -> sitk.Image:
    """
    Morphologically dilate *mask* by *radius_mm* in physical units.

    Useful to ensure extra-axial tumours just outside the BET brain
    boundary are retained for further processing.
    """
    spacing = mask.GetSpacing()
    radius_vox: Tuple[int, ...] = tuple(
        max(1, int(round(radius_mm / sp))) for sp in spacing
    )
    dilated = sitk.BinaryDilate(mask, radius_vox, sitk.sitkBall)
    dilated.CopyInformation(mask)
    return dilated
