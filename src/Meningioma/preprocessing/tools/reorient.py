#!/usr/bin/env python3
"""
reorient.py
Re-orients a NIfTI volume and NRRD mask to a specified orientation (e.g. RAS).

Usage:
    python reorient.py \
        --volume /path/to/volume.nii.gz \
        --mask /path/to/mask.nrrd \
        --orientation RAS \
        --output_volume /path/to/reoriented_volume.nii.gz \
        --output_mask /path/to/reoriented_mask.nrrd
"""

import argparse
import sys
from typing import Tuple

import SimpleITK as sitk


def reorient_images(
    volume_path: str,
    mask_path: str,
    output_volume_path: str,
    output_mask_path: str,
    orientation: str = "RAS",
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Re-orients a NIfTI volume and an NRRD mask into the specified orientation.

    :param volume_path: Path to the input NIfTI volume.
    :param mask_path: Path to the input NRRD mask.
    :param output_volume_path: Path where the reoriented volume will be saved.
    :param output_mask_path: Path where the reoriented mask will be saved.
    :param orientation: Desired orientation string, e.g. 'RAS', 'LPS', 'RAI', etc.
    :return: A tuple (reoriented_volume, reoriented_mask) as SimpleITK Image objects.
    :raises ValueError: If the specified orientation is invalid.
    :raises RuntimeError: If image loading or saving fails.
    """

    # Valid orientation codes for SimpleITK's DICOMOrientImageFilter
    # For example: 'RAS', 'LPS', 'RAI', etc.
    # We'll trust the user to supply a valid code recognized by SimpleITK.
    # (SimpleITK typically supports standard 3-letter codes).
    if len(orientation) != 3:
        raise ValueError(
            f"Orientation '{orientation}' must be a 3-letter string (e.g. 'RAS')."
        )

    try:
        # Load the volume (NIfTI)
        volume_img = sitk.ReadImage(volume_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read volume from {volume_path}: {e}") from e

    try:
        # Load the mask (NRRD)
        mask_img = sitk.ReadImage(mask_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read mask from {mask_path}: {e}") from e

    # Create the orientation filter
    orient_filter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation(orientation)

    # Apply re-orientation
    reoriented_volume = orient_filter.Execute(volume_img)
    reoriented_mask = orient_filter.Execute(mask_img)

    try:
        # Save outputs
        sitk.WriteImage(reoriented_volume, output_volume_path)
        sitk.WriteImage(reoriented_mask, output_mask_path)
    except Exception as e:
        raise RuntimeError(f"Failed to write reoriented images: {e}") from e

    return reoriented_volume, reoriented_mask


def main():
    parser = argparse.ArgumentParser(
        description="Re-orient volume and mask to specified orientation."
    )
    parser.add_argument("--volume", required=True, help="Path to input NIfTI volume")
    parser.add_argument("--mask", required=True, help="Path to input NRRD mask")
    parser.add_argument(
        "--orientation", default="RAS", help="Target orientation (e.g. RAS, LPS, etc.)"
    )
    parser.add_argument(
        "--output_volume", required=True, help="Output path for reoriented NIfTI volume"
    )
    parser.add_argument(
        "--output_mask", required=True, help="Output path for reoriented NRRD mask"
    )

    args = parser.parse_args()

    try:
        reorient_images(
            volume_path=args.volume,
            mask_path=args.mask,
            output_volume_path=args.output_volume,
            output_mask_path=args.output_mask,
            orientation=args.orientation,
        )
        print(
            f"Re-orientation completed. Saved volume to {args.output_volume}, mask to {args.output_mask}."
        )
    except Exception as e:
        print(f"Error during re-orientation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
