#!/usr/bin/env python3
"""
denoise_susan.py

Applies FSL SUSAN denoising to a 3D neuroimaging volume using Nipype.
This script is particularly helpful when you want to remain in Python,
but leverage FSL's SUSAN for denoising.

Requirements:
    - Nipype (pip install nipype)
    - FSL installed and configured (FSLDIR environment variable, etc.)
    - nibabel (for NIfTI I/O)
    - SimpleITK

Usage (Command-line):
    python denoise_susan.py <input_nii_or_nrrd> [--output out_file] [--brightness_threshold float] ...
    
Example:
    python denoise_susan.py my_volume.nii.gz --output denoised.nii.gz --brightness_threshold 0.1
"""

import argparse
import os
import tempfile
from typing import Optional

import SimpleITK as sitk
import nibabel as nib
import numpy as np

# Nipype FSL
try:
    from nipype.interfaces.fsl import SUSAN
except ImportError as e:
    raise ImportError(
        "Nipype with FSL support is required for this script. "
        "Install via `pip install nipype` and ensure FSL is installed."
    ) from e


def denoise_susan(
    image_sitk: sitk.Image,
    brightness_threshold: float = 0.05,
    fwhm: float = 1.0,
    dimension: int = 3,
    mask_sitk: Optional[sitk.Image] = None,
    verbose: bool = False,
) -> sitk.Image:
    """
    Denoise an MRI volume using FSL's SUSAN (via Nipype).

    SUSAN parameters reference:
        - brightness_threshold (float):
            The intensity threshold below which changes in intensity are considered noise.
            Typically a fraction of the robust range of the image.
        - fwhm (float):
            The full-width-at-half-maximum of the smoothing kernel in mm.

    Args:
        image_sitk (sitk.Image):
            The input volume as a SimpleITK image.
        brightness_threshold (float, optional):
            Intensity threshold for SUSAN. Default is 0.05 (i.e., 5% of max).
        fwhm (float, optional):
            Smoothing kernel FWHM in mm. Default 1.0 mm.
        dimension (int, optional):
            Dimensionality for SUSAN (should be 3 for 3D volumes). Default 3.
        mask_sitk (sitk.Image, optional):
            Optional brain or region mask for denoising. If provided, will be passed to SUSAN.
        verbose (bool, optional):
            Print extra info about intermediate steps.

    Returns:
        sitk.Image: The denoised volume in SimpleITK format.

    Raises:
        RuntimeError: If the SUSAN command fails or if FSL is not installed properly.

    Example:
        >>> import SimpleITK as sitk
        >>> from denoise_susan import denoise_susan
        >>> img = sitk.ReadImage('input.nii.gz')
        >>> denoised_img = denoise_susan(img, brightness_threshold=0.05, fwhm=1.0)
        >>> sitk.WriteImage(denoised_img, 'denoised.nii.gz')
    """
    # Step 1: Convert SITK image to temporary NIfTI
    original_spacing = image_sitk.GetSpacing()
    original_direction = image_sitk.GetDirection()
    original_origin = image_sitk.GetOrigin()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_nii_path = os.path.join(tmpdir, "input_susan.nii.gz")
        sitk.WriteImage(image_sitk, input_nii_path)

        mask_nii_path = ""
        if mask_sitk is not None:
            mask_nii_path = os.path.join(tmpdir, "mask_susan.nii.gz")
            sitk.WriteImage(mask_sitk, mask_nii_path)

        # Step 2: Configure Nipype SUSAN interface
        susan = SUSAN()
        susan.inputs.in_file = input_nii_path
        susan.inputs.brightness_threshold = brightness_threshold
        susan.inputs.fwhm = fwhm
        if mask_nii_path:
            susan.inputs.mask_file = mask_nii_path
        susan.inputs.dimension = dimension
        # The output file will be in the same tmpdir
        output_nii_path = os.path.join(tmpdir, "output_susan.nii.gz")
        susan.inputs.out_file = output_nii_path
        if verbose:
            print(
                f"[SUSAN] Running SUSAN with threshold={brightness_threshold}, fwhm={fwhm}"
            )

        # Step 3: Execute
        try:
            res = susan.run()
            if verbose:
                print(f"[SUSAN] Completed. Outputs: {res.outputs}")
        except Exception as e:
            raise RuntimeError(f"FSL SUSAN failed: {e}") from e

        # Step 4: Read back the denoised result as a SITK image
        denoised_sitk = sitk.ReadImage(output_nii_path)

    # Step 5: Re-apply original spacing/origin/direction
    # SUSAN doesn't typically alter geometry, but let's be safe ...
    denoised_sitk.SetSpacing(original_spacing)
    denoised_sitk.SetOrigin(original_origin)
    denoised_sitk.SetDirection(original_direction)

    return denoised_sitk


def main():
    parser = argparse.ArgumentParser(
        description="Denoise a 3D MRI volume using FSL's SUSAN via Nipype."
    )
    parser.add_argument(
        "input_file", type=str, help="Path to input volume (NIfTI or NRRD)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="denoised_susan.nii.gz",
        help="Path to save the denoised output (NIfTI).",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default="",
        help="Optional brain or region mask (NIfTI or NRRD).",
    )
    parser.add_argument(
        "--brightness_threshold",
        type=float,
        default=0.05,
        help="SUSAN brightness threshold (fraction of robust range). Default=0.05",
    )
    parser.add_argument(
        "--fwhm",
        type=float,
        default=1.0,
        help="SUSAN smoothing kernel FWHM in mm. Default=1.0 mm.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra debugging info.",
    )
    args = parser.parse_args()

    # Load input volume with SITK
    vol_sitk = sitk.ReadImage(args.input_file)

    mask_sitk = None
    if args.mask:
        mask_sitk = sitk.ReadImage(args.mask)

    # Run SUSAN
    denoised = denoise_susan(
        vol_sitk,
        brightness_threshold=args.brightness_threshold,
        fwhm=args.fwhm,
        mask_sitk=mask_sitk,
        verbose=args.verbose,
    )
    # Save output
    sitk.WriteImage(denoised, args.output)
    if args.verbose:
        print(f"Denoised volume saved to {args.output}")


if __name__ == "__main__":
    main()
