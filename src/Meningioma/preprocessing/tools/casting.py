#!/usr/bin/env python3

import SimpleITK as sitk
from typing import Tuple


def cast_volume_and_mask(
    volume_img: sitk.Image, mask_img: sitk.Image
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Casts the input volume to float32 and the input mask to uint8 entirely using
    SimpleITK filters.

    Args:
        volume_img (sitk.Image):
            The input volume (any float or integer type).
        mask_img (sitk.Image):
            The input mask (integer type). If squeeze=True, we clamp to [0,255].

    Returns:
        Tuple[sitk.Image, sitk.Image]:
            (cast_volume, cast_mask), both as SimpleITK Images in the new types.

    Example usage:
        >>> import SimpleITK as sitk
        >>> vol = sitk.ReadImage('volume.nii.gz')
        >>> msk = sitk.ReadImage('mask.nrrd')
        >>> vol_out, msk_out = cast_volume_and_mask_sitk(vol, msk)
        >>> sitk.WriteImage(vol_out, 'volume_float32.nii.gz')
        >>> sitk.WriteImage(msk_out, 'mask_uint8.nrrd')
    """

    # 1) Clamp the mask to [0,1]
    clamp_filter = sitk.ClampImageFilter()
    clamp_filter.SetLowerBound(0)
    clamp_filter.SetUpperBound(1)
    mask_img = clamp_filter.Execute(mask_img)

    # 2) Cast the volume to float32
    cast_volume = sitk.Cast(volume_img, sitk.sitkFloat32)

    # 3) Cast the mask to uint8
    cast_mask = sitk.Cast(mask_img, sitk.sitkUInt8)

    return cast_volume, cast_mask


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python cast_sitk_only.py <volume.nii.gz> <mask.nrrd>")
        sys.exit(1)

    vol_path = sys.argv[1]
    mask_path = sys.argv[2]

    # Read input images
    try:
        volume_in = sitk.ReadImage(vol_path)
        mask_in = sitk.ReadImage(mask_path)
    except Exception as e:
        print(f"Error reading input files: {e}")
        sys.exit(1)

    # Cast entirely in SimpleITK
    cast_vol, cast_msk = cast_volume_and_mask(volume_in, mask_in)

    # Write outputs
    sitk.WriteImage(cast_vol, "cast_volume.nii.gz")
    sitk.WriteImage(cast_msk, "cast_mask.nrrd")
    print("Casting completed successfully.")
