#!/usr/bin/env python3
"""
Casts a volume to float32 and a mask to uint8, optionally squeezing the intensities into a new range
if needed.

Usage:
    python casting.py \
        --volume /path/to/volume.nii.gz \
        --mask /path/to/mask.nrrd \
        --output_volume /path/to/cast_volume.nii.gz \
        --output_mask /path/to/cast_mask.nrrd \
        [--squeeze]
"""

import argparse
import sys
from typing import Tuple

import SimpleITK as sitk
import numpy as np


def cast_volume_and_mask(
    volume_path: str,
    mask_path: str,
    output_volume_path: str,
    output_mask_path: str,
    squeeze: bool = False,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Casts the input volume to float32 and the input mask to uint8. Optionally
    rescales intensities to fit within the allowable data range if needed.

    :param volume_path: Path to the input NIfTI volume (any float or integer type).
    :param mask_path: Path to the input NRRD mask (often uint8 or int16).
    :param output_volume_path: Where the cast volume will be saved as NIfTI.
    :param output_mask_path: Where the cast mask will be saved as NRRD.
    :param squeeze: If True, will min-max normalize the volume intensities to [0, 1]
                    before casting to float32, and will also ensure mask labels
                    fit in [0, 255] for uint8.
    :return: A tuple (cast_volume, cast_mask) as SimpleITK Image objects.
    :raises RuntimeError: If reading or writing fails, or if mask labels exceed [0,255].
    """

    try:
        volume_img = sitk.ReadImage(volume_path)
    except Exception as e:
        raise RuntimeError(f"Could not read volume from {volume_path}: {e}") from e

    try:
        mask_img = sitk.ReadImage(mask_path)
    except Exception as e:
        raise RuntimeError(f"Could not read mask from {mask_path}: {e}") from e

    volume_array = sitk.GetArrayFromImage(volume_img).astype(np.float64)  # for safety
    mask_array = sitk.GetArrayFromImage(mask_img).astype(np.int64)

    # Squeeze intensities if requested
    if squeeze:
        # ====== Squeeze volume to [0, 1] ====== #
        vmin = np.min(volume_array)
        vmax = np.max(volume_array)
        if vmax - vmin != 0:
            volume_array = (volume_array - vmin) / (vmax - vmin)
        else:
            volume_array = np.zeros_like(volume_array)  # all pixels same -> 0

        # ====== Ensure mask fits in [0, 255] ====== #
        mask_min = np.min(mask_array)
        mask_max = np.max(mask_array)
        if mask_min < 0 or mask_max > 255:
            mask_array = np.clip(mask_array, 0, 255)

    # Cast to final data types
    volume_array = volume_array.astype(np.float32)
    mask_array = mask_array.astype(
        np.uint8
    )  # also checks if negative or >255 if not squeezed

    cast_volume_img = sitk.GetImageFromArray(volume_array)
    cast_volume_img.CopyInformation(volume_img)  # preserve origin, spacing, direction

    cast_mask_img = sitk.GetImageFromArray(mask_array)
    cast_mask_img.CopyInformation(mask_img)  # preserve origin, spacing, direction

    try:
        sitk.WriteImage(cast_volume_img, output_volume_path)
        sitk.WriteImage(cast_mask_img, output_mask_path)
    except Exception as e:
        raise RuntimeError(f"Failed to write cast images: {e}") from e

    return cast_volume_img, cast_mask_img


def main():
    parser = argparse.ArgumentParser(
        description="Cast volume to float32 and mask to uint8."
    )
    parser.add_argument("--volume", required=True, help="Path to input NIfTI volume")
    parser.add_argument("--mask", required=True, help="Path to input NRRD mask")
    parser.add_argument(
        "--output_volume", required=True, help="Path for cast volume (NIfTI)"
    )
    parser.add_argument(
        "--output_mask", required=True, help="Path for cast mask (NRRD)"
    )
    parser.add_argument(
        "--squeeze",
        action="store_true",
        help="Rescale volume intensities to [0,1] and ensure mask fits [0,255].",
    )

    args = parser.parse_args()

    try:
        cast_volume_and_mask(
            volume_path=args.volume,
            mask_path=args.mask,
            output_volume_path=args.output_volume,
            output_mask_path=args.output_mask,
            squeeze=args.squeeze,
        )
        print(
            f"Casting completed. Saved volume to {args.output_volume}, mask to {args.output_mask}."
        )
    except Exception as e:
        print(f"Error during casting: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
