#!/usr/bin/env python3
"""
fsl_bet_brain_extraction.py

Provides a function to run FSL's BET (Brain Extraction Tool) via Nipype,
returning a SimpleITK image (brain-extracted). Also includes a command-line main.

Requires:
  - Nipype
  - FSL installed on your system
  - Python 3, SimpleITK

Example usage:
  python fsl_bet_brain_extraction.py input.nii.gz --frac 0.5 --output_brain bet_brain.nii.gz --output_mask bet_mask.nii.gz
"""

import os
import argparse
import tempfile
from typing import Optional, Tuple

import SimpleITK as sitk
from nipype.interfaces.fsl import BET  # type:ignore


def fsl_bet_brain_extraction(
    input_image_sitk: sitk.Image,
    frac: float = 0.5,
    robust: bool = False,
    vertical_gradient: float = 0.0,
    skull: bool = False,
    verbose: bool = False,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Run FSL's BET on the given input_image_sitk via Nipype, returning
    the extracted brain and mask as SITK images.

    Args:
        input_image_sitk (sitk.Image):
            The input 3D MRI volume as a SimpleITK image.
        frac (float, optional):
            Fractional intensity threshold (0->1); default=0.5.
            Smaller values -> smaller brain outline, larger values -> bigger brain outline.
        robust (bool, optional):
            If True, uses robust brain centre estimation (via -R). Slower but might be more accurate.
        vertical_gradient (float, optional):
            Vertical gradient (-g option) to deal with e.g. neck or top slices.
        skull (bool, optional):
            If True, extracts the entire skull+brain rather than just brain (using -s).
        verbose (bool, optional):
            If True, prints debug info.

    Returns:
        (extracted_brain_image, extracted_mask):
            - extracted_brain_image: SITK image with non-brain voxels set to 0.
            - extracted_mask: The binary brain mask as a SITK image.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) Save input as NIfTI
        in_nii_path = os.path.join(tmpdir, "input_bet.nii.gz")
        sitk.WriteImage(input_image_sitk, in_nii_path)

        # 2) Configure Nipype BET
        bet = BET()
        bet.inputs.in_file = in_nii_path
        out_file = os.path.join(tmpdir, "bet_out.nii.gz")
        bet.inputs.out_file = out_file
        bet.inputs.frac = frac
        bet.inputs.robust = robust
        bet.inputs.vertical_gradient = vertical_gradient
        bet.inputs.mask = True  # Output mask file as well
        # If user wants to extract the skull + brain
        if skull:
            bet.inputs.skull = True

        if verbose:
            print(
                f"[FSL BET] Running BET with frac={frac}, robust={robust}, vertical_gradient={vertical_gradient}, skull={skull}"
            )

        # 3) Run BET
        res = bet.run()
        if verbose:
            print("[FSL BET] Completed. Outputs:")
            print(res.outputs)

        # Nipype BET output:
        #  - out_file => the extracted brain or skull+brain
        #  - mask_file => the binary mask

        extracted_brain_path = res.outputs.out_file
        extracted_mask_path = res.outputs.mask_file

        # 4) Convert results back to SITK
        extracted_brain_sitk = sitk.ReadImage(extracted_brain_path)
        extracted_mask_sitk = sitk.ReadImage(extracted_mask_path)

        # Keep geometry consistent
        extracted_brain_sitk.CopyInformation(input_image_sitk)
        extracted_mask_sitk.CopyInformation(input_image_sitk)

        return extracted_brain_sitk, extracted_mask_sitk


def main():
    parser = argparse.ArgumentParser(
        description="Run FSL BET on a 3D volume, output extracted brain + mask."
    )
    parser.add_argument("input_image", type=str, help="Input 3D volume (NIfTI).")
    parser.add_argument(
        "--output_brain",
        type=str,
        default="bet_brain.nii.gz",
        help="Output brain-extracted volume path. Default=bet_brain.nii.gz",
    )
    parser.add_argument(
        "--output_mask",
        type=str,
        default="bet_mask.nii.gz",
        help="Output mask path. Default=bet_mask.nii.gz",
    )
    parser.add_argument(
        "--frac",
        type=float,
        default=0.5,
        help="Fractional intensity threshold (0->1). Default=0.5",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Robust brain centre estimation (-R). Slower but might be more accurate.",
    )
    parser.add_argument(
        "--vertical_gradient",
        type=float,
        default=0.0,
        help="Vertical gradient (-g option). Default=0.0",
    )
    parser.add_argument(
        "--skull",
        action="store_true",
        help="Extract brain+skull instead of just brain (-s option).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug info.")
    args = parser.parse_args()

    # Read input as SITK
    input_sitk = sitk.ReadImage(args.input_image)

    # Run BET
    extracted_brain_sitk, extracted_mask_sitk = fsl_bet_brain_extraction(
        input_sitk,
        frac=args.frac,
        robust=args.robust,
        vertical_gradient=args.vertical_gradient,
        skull=args.skull,
        verbose=args.verbose,
    )

    # Save
    sitk.WriteImage(extracted_brain_sitk, args.output_brain)
    sitk.WriteImage(extracted_mask_sitk, args.output_mask)

    if args.verbose:
        print(f"[DONE] Brain-extracted volume saved to {args.output_brain}")
        print(f"[DONE] Brain mask saved to {args.output_mask}")


if __name__ == "__main__":
    main()
