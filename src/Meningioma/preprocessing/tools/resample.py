#!/usr/bin/env python3
"""
Resamples a NIfTI volume and an NRRD mask to a specified voxel spacing (default = 1x1x1 mm).

Usage:
    python resample.py \
        --volume /path/to/volume.nii.gz \
        --mask /path/to/mask.nrrd \
        --spacing 1.0 1.0 1.0 \
        --output_volume /path/to/resampled_volume.nii.gz \
        --output_mask /path/to/resampled_mask.nrrd
"""

import argparse
import sys
from typing import Tuple

import numpy as np
import SimpleITK as sitk


def resample_images(
    volume_path: str,
    mask_path: str,
    output_volume_path: str,
    output_mask_path: str,
    new_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Resamples a volume and its corresponding mask to a specified voxel spacing,
    preserving their alignment.

    :param volume_path: Path to the input NIfTI volume.
    :param mask_path: Path to the input NRRD mask.
    :param output_volume_path: Path where the resampled volume will be saved.
    :param output_mask_path: Path where the resampled mask will be saved.
    :param new_spacing: Desired spacing in (x, y, z), default (1.0, 1.0, 1.0).
    :return: A tuple (resampled_volume, resampled_mask) as SimpleITK Image objects.
    :raises RuntimeError: If reading or writing fails, or if dimension mismatch occurs.
    """

    try:
        volume_img = sitk.ReadImage(volume_path)
    except Exception as e:
        raise RuntimeError(f"Could not read volume from {volume_path}: {e}") from e

    try:
        mask_img = sitk.ReadImage(mask_path)
    except Exception as e:
        raise RuntimeError(f"Could not read mask from {mask_path}: {e}") from e

    # ========== Resample the Volume ========== #
    original_spacing = np.array(volume_img.GetSpacing())
    original_size = np.array(volume_img.GetSize())
    # Determine the new size based on the ratio of old/new spacing
    new_size = (original_size * (original_spacing / new_spacing)).astype(np.int32)

    # Setup resample filter for the volume
    volume_resampler = sitk.ResampleImageFilter()
    volume_resampler.SetOutputSpacing(new_spacing)
    volume_resampler.SetSize(new_size.tolist())
    volume_resampler.SetOutputDirection(volume_img.GetDirection())
    volume_resampler.SetOutputOrigin(volume_img.GetOrigin())
    # Use a higher-order interpolation for the volume
    volume_resampler.SetInterpolator(sitk.sitkBSpline)
    resampled_volume = volume_resampler.Execute(volume_img)

    # ========== Resample the Mask ========== #
    # We replicate the same size, spacing, direction, and origin,
    # but use nearest-neighbor to preserve label integrity.
    mask_resampler = sitk.ResampleImageFilter()
    mask_resampler.SetOutputSpacing(new_spacing)
    mask_resampler.SetSize(new_size.tolist())
    mask_resampler.SetOutputDirection(mask_img.GetDirection())
    mask_resampler.SetOutputOrigin(mask_img.GetOrigin())
    mask_resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampled_mask = mask_resampler.Execute(mask_img)

    # Save outputs
    try:
        sitk.WriteImage(resampled_volume, output_volume_path)
        sitk.WriteImage(resampled_mask, output_mask_path)
    except Exception as e:
        raise RuntimeError(f"Could not write resampled images: {e}") from e

    return resampled_volume, resampled_mask


def main():
    parser = argparse.ArgumentParser(
        description="Resample volume and mask to new voxel spacing."
    )
    parser.add_argument("--volume", required=True, help="Path to input NIfTI volume")
    parser.add_argument("--mask", required=True, help="Path to input NRRD mask")
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Target voxel spacing (x y z), default is 1.0 1.0 1.0",
    )
    parser.add_argument(
        "--output_volume", required=True, help="Path for resampled volume"
    )
    parser.add_argument("--output_mask", required=True, help="Path for resampled mask")

    args = parser.parse_args()

    spacing_tuple = tuple(args.spacing)

    try:
        resample_images(
            volume_path=args.volume,
            mask_path=args.mask,
            output_volume_path=args.output_volume,
            output_mask_path=args.output_mask,
            new_spacing=spacing_tuple,
        )
        print(
            f"Resampling completed. Saved volume to {args.output_volume}, mask to {args.output_mask}."
        )
    except Exception as e:
        print(f"Error during resampling: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
