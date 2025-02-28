#!/usr/bin/env python3
"""
resample.py
Resamples a volume and mask (both sitk.Images) to a specified isotropic spacing.
"""

import SimpleITK as sitk
import numpy as np
from typing import Tuple


def resample_images(
    volume_img: sitk.Image,
    mask_img: sitk.Image,
    new_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Resamples a volume and mask to a specified voxel spacing, preserving alignment.

    Args:
        volume_img (sitk.Image):
            The input volume in sitk.Image format.
        mask_img (sitk.Image):
            The corresponding mask in sitk.Image format.
        new_spacing (tuple, optional):
            Desired spacing in (x, y, z). Defaults to (1.0, 1.0, 1.0).

    Returns:
        Tuple[sitk.Image, sitk.Image]:
            (resampled_volume, resampled_mask), both as sitk.Images
            with the specified voxel spacing.

    Raises:
        RuntimeError:
            If any unexpected error occurs during resampling.

    Comments:
        More info on available interpolators from the SimpleITK library in
        https://simpleitk.org/doxygen/v2_4/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5
    """
    original_spacing = np.array(volume_img.GetSpacing())
    original_size = np.array(volume_img.GetSize())
    if any(sp <= 0 for sp in new_spacing):
        raise ValueError(f"Invalid new_spacing {new_spacing}. Must be > 0.")

    # Compute new size
    spacing_ratio = original_spacing / np.array(new_spacing)
    new_size_float = original_size * spacing_ratio
    new_size = tuple(int(round(sz)) for sz in new_size_float)

    # --- Resample volume ---
    vol_resampler = sitk.ResampleImageFilter()
    vol_resampler.SetOutputSpacing(new_spacing)
    vol_resampler.SetSize(new_size)
    vol_resampler.SetOutputOrigin(volume_img.GetOrigin())
    vol_resampler.SetOutputDirection(volume_img.GetDirection())
    # Higher-order interpolation for intensity images
    vol_resampler.SetInterpolator(
        sitk.sitkBSpline
    )  # Modify to accept more interpolators
    try:
        resampled_vol = vol_resampler.Execute(volume_img)
    except Exception as e:
        raise RuntimeError(f"Resampling volume failed: {e}") from e

    # --- Resample mask ---
    mask_resampler = sitk.ResampleImageFilter()
    mask_resampler.SetOutputSpacing(new_spacing)
    mask_resampler.SetSize(new_size)
    mask_resampler.SetOutputOrigin(mask_img.GetOrigin())
    mask_resampler.SetOutputDirection(mask_img.GetDirection())
    # Use nearest-neighbor for segmentation masks
    mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    try:
        resampled_mask = mask_resampler.Execute(mask_img)
    except Exception as e:
        raise RuntimeError(f"Resampling mask failed: {e}") from e

    return resampled_vol, resampled_mask


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python resample.py <volume.nii.gz> <mask.nrrd> <spacing_x> <spacing_y> <spacing_z>"
        )
        sys.exit(1)

    vol_path = sys.argv[1]
    mask_path = sys.argv[2]
    spacing_tuple = tuple(float(s) for s in sys.argv[3:6])

    try:
        vol_in = sitk.ReadImage(vol_path)
        mask_in = sitk.ReadImage(mask_path)

        vol_out, mask_out = resample_images(vol_in, mask_in, spacing_tuple)  # type: ignore
        sitk.WriteImage(vol_out, "resampled_volume.nii.gz")
        sitk.WriteImage(mask_out, "resampled_mask.nrrd")
        print("Resampling completed.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
