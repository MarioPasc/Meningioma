#!/usr/bin/env python3
"""
ants_brain_extraction.py

Provides a function to run ANTs-based brain extraction via Nipype's BrainExtraction interface,
returning a SimpleITK image (brain-extracted). Also includes a command-line 'main' for usage.

Requires:
  - Nipype
  - ANTs installed on your system
  - Python 3, SimpleITK

Example usage:
  python ants_brain_extraction.py input.nii.gz --out_mask extracted_brain_mask.nii.gz --template T1_template.nii.gz --prob_mask T1_BrainProbMask.nii.gz
"""

import os
import argparse
import tempfile
from typing import Optional, Tuple

import SimpleITK as sitk
import nibabel as nib
from nipype.interfaces.ants.segmentation import BrainExtraction  # type: ignore


def ants_brain_extraction(
    input_image_sitk: sitk.Image,
    anatomical_template: str,
    extraction_registration_mask: Optional[str] = None,
    dimension: int = 3,
    verbose: bool = False,
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Run the ANTs BrainExtraction interface on the given input_image_sitk.

    BrainExtraction requires:
      - A structural template (anatomical_template)
      - A probability mask or prior, typically specified via the .inputs.brain_probability_mask

    NOTE: For a standard T1-based brain extraction, you typically pass:
      - 'anatomical_template' as a T1 template
      - 'extraction_registration_mask' as the T1 brain probability map or mask

    Args:
        input_image_sitk (sitk.Image):
            The input 3D MRI volume as a SimpleITK image.
        anatomical_template (str):
            Path to a reference template (e.g., T1 anatomical template).
        extraction_registration_mask (str, optional):
            A brain probability mask or prior. If provided, we set
            BrainExtraction.inputs.brain_probability_mask = that path.
        dimension (int, optional):
            Dimensionality for the interface (3 for 3D). Default=3.
        verbose (bool, optional):
            If True, prints extra debugging info.

    Returns:
        (extracted_brain_image, extracted_mask):
            - extracted_brain_image: SITK image of the input with non-brain voxels removed.
            - extracted_mask: The binary brain mask as a SITK image.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) Save input_image_sitk as NIfTI
        in_nii_path = os.path.join(tmpdir, "input.nii.gz")
        sitk.WriteImage(input_image_sitk, in_nii_path)

        # 2) Configure Nipype BrainExtraction
        brain_extraction = BrainExtraction()
        brain_extraction.inputs.dimension = dimension
        brain_extraction.inputs.anatomical_image = in_nii_path
        brain_extraction.inputs.brain_template = anatomical_template
        if extraction_registration_mask:
            brain_extraction.inputs.brain_probability_mask = (
                extraction_registration_mask
            )
        # Output files
        out_prefix = os.path.join(tmpdir, "ants_be_")
        brain_extraction.inputs.out_prefix = out_prefix

        # BrainExtraction will produce:
        #   ants_be_BrainExtractionBrain.nii.gz (brain-extracted volume)
        #   ants_be_BrainExtractionMask.nii.gz  (binary mask)

        if verbose:
            print("[ANTS BrainExtraction] Running with:")
            print(
                f"  dimension={dimension}, template={anatomical_template}, prob_mask={extraction_registration_mask}"
            )

        # 3) Run
        res = brain_extraction.run()
        if verbose:
            print("[ANTS BrainExtraction] Completed. Outputs:")
            print(res.outputs)

        # 4) Convert results back to SITK
        extracted_brain_path = res.outputs.BrainExtractionBrain
        extracted_mask_path = res.outputs.BrainExtractionMask

        extracted_brain_sitk = sitk.ReadImage(extracted_brain_path)
        extracted_mask_sitk = sitk.ReadImage(extracted_mask_path)

        # (Optional) copy geometry from input if needed
        extracted_brain_sitk.CopyInformation(input_image_sitk)
        extracted_mask_sitk.CopyInformation(input_image_sitk)

        return extracted_brain_sitk, extracted_mask_sitk


def main():
    parser = argparse.ArgumentParser(
        description="Run ANTs Brain Extraction on a 3D volume, output extracted brain + mask."
    )
    parser.add_argument("input_image", type=str, help="Input 3D volume (NIfTI).")
    parser.add_argument(
        "--template",
        type=str,
        required=True,
        help="Path to T1 structural template for BrainExtraction.",
    )
    parser.add_argument(
        "--prob_mask",
        type=str,
        default="",
        help="Path to brain probability mask (e.g., T1_BrainProbabilityMask).",
    )
    parser.add_argument(
        "--output_brain",
        type=str,
        default="extracted_brain.nii.gz",
        help="Output path for the brain-extracted volume. Default=extracted_brain.nii.gz",
    )
    parser.add_argument(
        "--output_mask",
        type=str,
        default="extracted_brain_mask.nii.gz",
        help="Output path for the extracted brain mask. Default=extracted_brain_mask.nii.gz",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=3,
        help="Dimensionality (2D or 3D). Default=3 for 3D volumes.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug info.")
    args = parser.parse_args()

    # Read input as SITK
    input_sitk = sitk.ReadImage(args.input_image)

    # Run extraction
    extracted_brain_sitk, extracted_mask_sitk = ants_brain_extraction(
        input_sitk,
        anatomical_template=args.template,
        extraction_registration_mask=args.prob_mask if args.prob_mask else None,
        dimension=args.dimension,
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
