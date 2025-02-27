#!/usr/bin/env python3
"""
match_geometry.py
Ensures a mask has the same geometry (size, spacing, origin, direction)
as the given volume. Returns both as sitk.Images.
"""

from typing import Tuple, Optional
import SimpleITK as sitk


def match_mask_and_volume_dimensions(
    volume_img: sitk.Image, mask_img: sitk.Image
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Ensures that a mask has the same geometry (size, spacing, origin, direction)
    as a given volume. If they differ, the mask is resampled onto the volume's grid.

    Args:
        volume_img (sitk.Image):
            The reference volume (SimpleITK Image).
        mask_img (sitk.Image):
            The mask (SimpleITK Image) that may need resampling.

    Returns:
        Tuple[sitk.Image, sitk.Image]:
            (volume_out, mask_out), where:
              - volume_out is the original volume_img (unchanged)
              - mask_out is either the original mask_img (if geometry matched)
                or a resampled version on the volume grid.
    """
    # Check if size, origin, spacing, and direction match
    same_size = volume_img.GetSize() == mask_img.GetSize()
    same_spacing = volume_img.GetSpacing() == mask_img.GetSpacing()
    same_origin = volume_img.GetOrigin() == mask_img.GetOrigin()
    same_direction = volume_img.GetDirection() == mask_img.GetDirection()

    if all([same_size, same_spacing, same_origin, same_direction]):
        # No resampling needed
        return volume_img, mask_img
    else:
        # Resample the mask onto the volume's grid
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(volume_img)
        # Use NearestNeighbor for label images to preserve discrete values
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        matched_mask = resampler.Execute(mask_img)

        return volume_img, matched_mask


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python match_geometry.py <volume.nii.gz> <mask.nrrd>")
        sys.exit(1)

    vol_path = sys.argv[1]
    mask_path = sys.argv[2]

    try:
        vol_in = sitk.ReadImage(vol_path)
        mask_in = sitk.ReadImage(mask_path)

        vol_out, mask_out = match_mask_and_volume_dimensions(vol_in, mask_in)

        sitk.WriteImage(vol_out, "volume_matched.nii.gz")
        sitk.WriteImage(mask_out, "mask_matched.nrrd")

        print("Geometry matching completed. Saved matched volume and mask.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
