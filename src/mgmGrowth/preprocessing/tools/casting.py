#!/usr/bin/env python3
"""
Utilities to cast MRI volumes / masks to standard data-types.

Changes compared to the original implementation:
* The volume is **always** cast to ``float32`` (regardless of mask).
* The mask is cast to ``uint8`` only if provided.
* Mask values are clamped into {0, 1} before casting.
"""

from __future__ import annotations

from typing import Tuple, Optional

import SimpleITK as sitk


__all__ = ["cast_volume_and_optional_mask"]


def cast_volume_and_optional_mask(
    volume_img: sitk.Image, mask_img: Optional[sitk.Image] = None
) -> Tuple[sitk.Image, Optional[sitk.Image]]:
    """
    Cast *volume_img* → ``sitkFloat32``; *mask_img* (if any) → ``sitkUInt8``.

    Returns a tuple ``(cast_volume, cast_mask_or_None)``.
    """
    cast_volume = sitk.Cast(volume_img, sitk.sitkFloat32)

    if mask_img is None:
        return cast_volume, None

    clamp = sitk.ClampImageFilter()
    clamp.SetLowerBound(0)
    clamp.SetUpperBound(1)
    mask_img = clamp.Execute(mask_img)

    cast_mask = sitk.Cast(mask_img, sitk.sitkUInt8)
    return cast_volume, cast_mask
