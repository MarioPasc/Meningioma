#!/usr/bin/env python3
"""
Provides functions to perform registration (affine and nonlinear) to the SRI24 atlas 
via Nipype's ANTS interface, returning SimpleITK images. Supports registering both 
image volumes and their corresponding masks. Also includes a command-line main.

Requires:
  - Nipype
  - ANTs installed on your system
  - Python 3, SimpleITK

Example usage:
  python sri24_registration.py moving_image.nii.gz --atlas_path /path/to/SRI24.nii.gz 
  --mask_path subject_mask.nii.gz --output_registered registered_output.nii.gz 
  --output_mask registered_mask.nii.gz --output_transform transform_
"""

import os
import argparse
import json
import time
from typing import Optional, Tuple, Dict, Any

import SimpleITK as sitk # type:ignore
from nipype.interfaces.ants import Registration, ApplyTransforms  # type:ignore


def register_to_sri24(
    moving_image_sitk: sitk.Image,
    fixed_image_sitk: sitk.Image,
    output_dir: str,
    output_transform_prefix: str,
    output_image_path: str,
    initial_moving_transform: Optional[str] = None,
    dimension: int = 3,
    use_histogram_matching: bool = True,
    winsorize_lower_quantile: float = 0.005,
    winsorize_upper_quantile: float = 0.995,
    number_threads: int = 1,
    verbose: bool = False,
) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Register a moving image (T1 or T2) to a reference atlas (SRI24)
    using ANTs via Nipype, returning the registered image as a SimpleITK image
    along with the transformation parameters.

    Args:
        moving_image_sitk (sitk.Image):
            The input 3D MRI volume (T1 or T2) as a SimpleITK image to be registered.
        fixed_image_sitk (sitk.Image):
            The reference atlas (SRI24) as a SimpleITK image.
        output_dir (str):
            Directory to save all output files.
        output_transform_prefix (str):
            Prefix for the output transformation files.
        output_image_path (str):
            Path to save the registered image.
        initial_moving_transform (str, optional):
            Initial transform to apply. Can be used to initialize from a previous transform.
        dimension (int, optional):
            Dimension of the image registration problem. Default is 3.
        use_histogram_matching (bool, optional):
            Whether to use histogram matching prior to registration. Default is True.
        winsorize_lower_quantile (float, optional):
            Lower quantile to winsorize the intensities. Default is 0.005.
        winsorize_upper_quantile (float, optional):
            Upper quantile to winsorize the intensities. Default is 0.995.
        verbose (bool, optional):
            If True, prints debug info.

    Returns:
        (registered_image, transform_params):
            - registered_image: SITK image of the moving image registered to the fixed image.
            - transform_params: Dictionary containing paths to the transformation files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    if verbose:
        print("[ANTS REGISTRATION] Starting registration process")
        print(f"[ANTS REGISTRATION] Output directory: {output_dir}")

    # 1) Save input images as NIfTI
    moving_nii_path = os.path.join(output_dir, "moving_input.nii.gz")
    fixed_nii_path = os.path.join(output_dir, "fixed_atlas.nii.gz")

    if verbose:
        print(
            f"[ANTS REGISTRATION] Saving input images to: {moving_nii_path} and {fixed_nii_path}"
        )

    sitk.WriteImage(moving_image_sitk, moving_nii_path)
    sitk.WriteImage(fixed_image_sitk, fixed_nii_path)

    # 2) Configure Nipype ANTS Registration
    if verbose:
        print("[ANTS REGISTRATION] Configuring ANTS registration parameters")

    reg = Registration()
    reg.inputs.fixed_image = fixed_nii_path
    reg.inputs.moving_image = moving_nii_path

    # Set registration parameters
    reg.inputs.dimension = dimension
    reg.inputs.use_histogram_matching = use_histogram_matching
    reg.inputs.winsorize_lower_quantile = winsorize_lower_quantile
    reg.inputs.winsorize_upper_quantile = winsorize_upper_quantile

    # Performance parameters
    reg.inputs.num_threads = number_threads

    # Output paths
    transform_prefix = os.path.join(output_dir, output_transform_prefix)

    if verbose:
        print(f"[ANTS REGISTRATION] Transform prefix set to: {transform_prefix}")

    reg.inputs.output_transform_prefix = transform_prefix
    reg.inputs.output_warped_image = output_image_path

    # Initial transform if provided
    if initial_moving_transform:
        if verbose:
            print(
                f"[ANTS REGISTRATION] Using initial transform: {initial_moving_transform}"
            )
        reg.inputs.initial_moving_transform = initial_moving_transform

    # Registration stages: rigid, affine, and SyN (nonlinear)
    reg.inputs.transforms = ["Rigid", "Affine", "SyN"]

    # Number of iterations for each stage
    reg.inputs.number_of_iterations = [
        [1000, 500, 250, 100],
        [1000, 500, 250, 100],
        [100, 70, 50, 20],
    ]

    # Shrink factors for multi-resolution optimization
    reg.inputs.shrink_factors = [[8, 4, 2, 1], [8, 4, 2, 1], [8, 4, 2, 1]]

    # Smoothing sigmas for each level (in mm)
    reg.inputs.smoothing_sigmas = [[3, 2, 1, 0], [3, 2, 1, 0], [3, 2, 1, 0]]

    # Similarity metric for each stage
    reg.inputs.metric = ["MI", "MI", "CC"]

    # Fixed image for each stage
    reg.inputs.metric_weight = [1.0, 1.0, 1.0]

    # Sampling strategy and percentage
    reg.inputs.sampling_strategy = ["Regular", "Regular", "None"]
    reg.inputs.sampling_percentage = [0.25, 0.25, None]

    # Radius or number of bins
    reg.inputs.radius_or_number_of_bins = [32, 32, 4]

    # Transform parameters
    reg.inputs.transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]

    # Convergence threshold and window size
    reg.inputs.convergence_threshold = [1e-6, 1e-6, 1e-6]
    reg.inputs.convergence_window_size = [10, 10, 10]

    # Additional parameters for SyN
    reg.inputs.sigma_units = ["vox", "vox", "vox"]
    reg.inputs.write_composite_transform = True

    if verbose:
        print(f"[ANTS REGISTRATION] Registration configuration complete")
        print(f"[ANTS REGISTRATION] Running with transforms: {reg.inputs.transforms}")
        print(f"[ANTS REGISTRATION] Using histogram matching: {use_histogram_matching}")
        reg.terminal_output = "stream"

    # 3) Run Registration
    if verbose:
        print(
            "[ANTS REGISTRATION] Starting registration execution - this may take some time..."
        )
        stage_start = time.time()

    res = reg.run()

    if verbose:
        stage_end = time.time()
        print(
            f"[ANTS REGISTRATION] Registration completed in {stage_end - stage_start:.2f} seconds"
        )

    # 4) Convert results back to SITK
    registered_path = res.outputs.warped_image

    if verbose:
        print(f"[ANTS REGISTRATION] Loading registered image from: {registered_path}")

    registered_sitk = sitk.ReadImage(registered_path)

    # Keep original metadata
    registered_sitk.CopyInformation(fixed_image_sitk)

    # Collect transform files
    transform_params = {
        "composite_transform": res.outputs.composite_transform,
        "inverse_composite_transform": res.outputs.inverse_composite_transform,
        "fixed_image_path": fixed_nii_path,
        "moving_image_path": moving_nii_path,
    }

    # Get individual transform files if they exist
    transform_params["rigid_transform"] = (
        transform_prefix + "0GenericAffine.mat"
        if os.path.exists(transform_prefix + "0GenericAffine.mat")
        else None
    )
    transform_params["affine_transform"] = (
        transform_prefix + "1GenericAffine.mat"
        if os.path.exists(transform_prefix + "1GenericAffine.mat")
        else None
    )
    transform_params["warp_transform"] = (
        transform_prefix + "2Warp.nii.gz"
        if os.path.exists(transform_prefix + "2Warp.nii.gz")
        else None
    )
    transform_params["inverse_warp_transform"] = (
        transform_prefix + "2InverseWarp.nii.gz"
        if os.path.exists(transform_prefix + "2InverseWarp.nii.gz")
        else None
    )

    # Save transform_params as JSON for future reference
    transform_params_path = os.path.join(output_dir, "transform_params.json")
    with open(transform_params_path, "w") as f:
        json.dump(transform_params, f, indent=2)

    if verbose:
        end_time = time.time()
        print(
            f"[ANTS REGISTRATION] Total registration process completed in {end_time - start_time:.2f} seconds"
        )
        print(
            f"[ANTS REGISTRATION] Transform parameters saved to: {transform_params_path}"
        )
        print("[ANTS REGISTRATION] Transform files:")
        for k, v in transform_params.items():
            print(f"[ANTS REGISTRATION]   {k}: {v}")

    return registered_sitk, transform_params


def register_to_sri24_with_mask(
    moving_image_sitk: sitk.Image,
    moving_mask_sitk: sitk.Image,
    fixed_image_sitk: sitk.Image,
    output_dir: str,
    output_transform_prefix: str,
    output_image_prefix: str,
    output_mask_prefix: str,
    initial_moving_transform: Optional[str] = None,
    dimension: int = 3,
    use_histogram_matching: bool = True,
    winsorize_lower_quantile: float = 0.005,
    winsorize_upper_quantile: float = 0.995,
    num_threads: int = 1,
    verbose: bool = False,
) -> Tuple[sitk.Image, sitk.Image, Dict[str, Any]]:
    """
    Register a moving image (T1 or T2) and its corresponding mask to a reference atlas (SRI24)
    using ANTs via Nipype, returning the registered image and mask as SimpleITK images
    along with the transformation parameters.

    Args:
        moving_image_sitk (sitk.Image):
            The input 3D MRI volume (T1 or T2) as a SimpleITK image to be registered.
        moving_mask_sitk (sitk.Image):
            The corresponding mask for the moving image as a SimpleITK image.
        fixed_image_sitk (sitk.Image):
            The reference atlas (SRI24) as a SimpleITK image.
        output_dir (str):
            Directory to save all output files.
        output_transform_prefix (str):
            Prefix for the output transformation files.
        output_image_path (str):
            Path to save the registered image.
        output_mask_path (str):
            Path to save the registered mask.
        initial_moving_transform (str, optional):
            Initial transform to apply. Can be used to initialize from a previous transform.
        dimension (int, optional):
            Dimension of the image registration problem. Default is 3.
        use_histogram_matching (bool, optional):
            Whether to use histogram matching prior to registration. Default is True.
        winsorize_lower_quantile (float, optional):
            Lower quantile to winsorize the intensities. Default is 0.005.
        winsorize_upper_quantile (float, optional):
            Upper quantile to winsorize the intensities. Default is 0.995.
        verbose (bool, optional):
            If True, prints debug info.

    Returns:
        (registered_image, registered_mask, transform_params):
            - registered_image: SITK image of the moving image registered to the fixed image.
            - registered_mask: SITK image of the moving mask registered to the fixed image.
            - transform_params: Dictionary containing paths to the transformation files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("[ANTS REGISTRATION] Creating output directory for entire process")
        print(f"[ANTS REGISTRATION] Output directory path: {output_dir}")

    # Save the mask input to the output directory
    mask_path = os.path.join(output_dir, "moving_mask_input.nii.gz")
    sitk.WriteImage(moving_mask_sitk, mask_path)

    # Build output paths
    output_image_path = (
        os.path.join(output_dir, output_image_prefix)
        if output_image_prefix.endswith(".nii.gz")
        else os.path.join(output_dir, output_image_prefix + ".nii.gz")
    )

    output_mask_path = (
        os.path.join(output_dir, output_mask_prefix)
        if output_mask_prefix.endswith(".nii.gz")
        else os.path.join(output_dir, output_mask_prefix + ".nii.gz")
    )

    if verbose:
        print(f"[ANTS REGISTRATION] Saved input mask to: {mask_path}")
        print("[ANTS REGISTRATION] Starting registration of image volume")

    # First, register the image
    registered_image, transform_params = register_to_sri24(
        moving_image_sitk=moving_image_sitk,
        fixed_image_sitk=fixed_image_sitk,
        output_dir=output_dir,
        output_transform_prefix=output_transform_prefix,
        output_image_path=output_image_path,
        initial_moving_transform=initial_moving_transform,
        dimension=dimension,
        use_histogram_matching=use_histogram_matching,
        winsorize_lower_quantile=winsorize_lower_quantile,
        winsorize_upper_quantile=winsorize_upper_quantile,
        number_threads=num_threads,
        verbose=verbose,
    )

    # Now, apply the same transformation to the mask
    if verbose:
        print("[ANTS REGISTRATION] Starting registration of mask volume")
        start_time = time.time()

    # Use ANTs ApplyTransforms to transform the mask
    if verbose:
        print("[ANTS REGISTRATION] Configuring ApplyTransforms for mask")
        print(
            f"[ANTS REGISTRATION] Using composite transform: {transform_params['composite_transform']}"
        )

    at = ApplyTransforms()
    at.inputs.dimension = dimension
    at.inputs.input_image = mask_path
    at.inputs.reference_image = transform_params["fixed_image_path"]
    at.inputs.output_image = output_mask_path
    at.inputs.transforms = [transform_params["composite_transform"]]
    at.inputs.interpolation = (
        "NearestNeighbor"  # Important for masks to preserve label values
    )

    if verbose:
        print(
            "[ANTS REGISTRATION] Applying transformation to mask - this may take a moment..."
        )
        stage_start = time.time()
        at.terminal_output = "stream"

    at_result = at.run()

    if verbose:
        stage_end = time.time()
        print(
            f"[ANTS REGISTRATION] Mask transformation completed in {stage_end - stage_start:.2f} seconds"
        )
        print(
            f"[ANTS REGISTRATION] Registered mask saved to: {at_result.outputs.output_image}"
        )

    # Load the transformed mask
    registered_mask = sitk.ReadImage(at_result.outputs.output_image)
    registered_mask.CopyInformation(fixed_image_sitk)

    # Update transform_params to include mask paths
    transform_params["input_mask_path"] = mask_path
    transform_params["output_mask_path"] = output_mask_path

    # Save updated transform_params as JSON
    transform_params_path = os.path.join(output_dir, "transform_params.json")
    with open(transform_params_path, "w") as f:
        json.dump(transform_params, f, indent=2)

    if verbose:
        end_time = time.time()
        print(
            f"[ANTS REGISTRATION] Total mask registration process completed in {end_time - start_time:.2f} seconds"
        )
        print(
            f"[ANTS REGISTRATION] Updated transform parameters saved to: {transform_params_path}"
        )

    return registered_image, registered_mask, transform_params


def main():
    """Command-line interface for the registration function."""
    parser = argparse.ArgumentParser(
        description="Register a T1/T2 image to SRI24 atlas."
    )
    parser.add_argument(
        "input_image", type=str, help="Path to the input image file (NIfTI format)."
    )
    parser.add_argument(
        "--atlas_path",
        type=str,
        required=True,
        help="Path to the SRI24 atlas image (NIfTI format).",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="Path to the mask file corresponding to the input image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./registration_output",
        help="Directory to save all output files.",
    )
    parser.add_argument(
        "--output_registered",
        type=str,
        default=None,
        help="Path to save the registered output image.",
    )
    parser.add_argument(
        "--output_mask",
        type=str,
        default=None,
        help="Path to save the registered output mask (if mask_path is provided).",
    )
    parser.add_argument(
        "--output_transform_prefix",
        type=str,
        default="transform_",
        help="Prefix for the output transformation files.",
    )
    parser.add_argument(
        "--initial_transform",
        type=str,
        default=None,
        help="Optional initial transform to apply.",
    )
    parser.add_argument(
        "--histogram_matching",
        action="store_true",
        default=True,
        help="Use histogram matching prior to registration.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=1,
        help="Number of threads to use for processing.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed progress information."
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set default output paths if not provided
    if args.output_registered is None:
        args.output_registered = os.path.join(args.output_dir, "registered.nii.gz")

    if args.mask_path is not None and args.output_mask is None:
        args.output_mask = os.path.join(args.output_dir, "registered_mask.nii.gz")

    # Load input images
    if args.verbose:
        print(f"[ANTS REGISTRATION] Loading input image: {args.input_image}")
        print(f"[ANTS REGISTRATION] Loading atlas image: {args.atlas_path}")

    moving_image = sitk.ReadImage(args.input_image)
    fixed_image = sitk.ReadImage(args.atlas_path)

    if args.verbose:
        print(f"[ANTS REGISTRATION] Moving image size: {moving_image.GetSize()}")
        print(f"[ANTS REGISTRATION] Fixed image size: {fixed_image.GetSize()}")
        print(f"[ANTS REGISTRATION] Output directory: {args.output_dir}")
        print(f"[ANTS REGISTRATION] Output registered image: {args.output_registered}")

    # Check if we're processing a mask as well
    if args.mask_path:
        if args.verbose:
            print(f"[ANTS REGISTRATION] Loading mask image: {args.mask_path}")
            print(f"[ANTS REGISTRATION] Output registered mask: {args.output_mask}")

        moving_mask = sitk.ReadImage(args.mask_path)

        if args.verbose:
            print(f"[ANTS REGISTRATION] Mask image size: {moving_mask.GetSize()}")
            print("[ANTS REGISTRATION] Performing registration with mask...")

        # Perform registration with mask
        registered_image, registered_mask, transform_params = (
            register_to_sri24_with_mask(
                moving_image_sitk=moving_image,
                moving_mask_sitk=moving_mask,
                fixed_image_sitk=fixed_image,
                output_dir=args.output_dir,
                output_transform_prefix=args.output_transform_prefix,
                output_image_path=args.output_registered,
                output_mask_path=args.output_mask,
                initial_moving_transform=args.initial_transform,
                use_histogram_matching=args.histogram_matching,
                num_threads=args.num_threads,
                verbose=args.verbose,
            )
        )

    else:
        # Perform registration without mask
        if args.verbose:
            print("[ANTS REGISTRATION] Performing registration without mask...")

        registered_image, transform_params = register_to_sri24(
            moving_image_sitk=moving_image,
            fixed_image_sitk=fixed_image,
            output_dir=args.output_dir,
            output_transform_prefix=args.output_transform_prefix,
            output_image_path=args.output_registered,
            initial_moving_transform=args.initial_transform,
            use_histogram_matching=args.histogram_matching,
            number_threads=args.num_threads,
            verbose=args.verbose,
        )

    if args.verbose:
        print("[ANTS REGISTRATION] Registration completed successfully")
        print(f"[ANTS REGISTRATION] All outputs saved to directory: {args.output_dir}")
        print(
            f"[ANTS REGISTRATION] Transform files saved with prefix: {args.output_transform_prefix}"
        )
        print(
            f"[ANTS REGISTRATION] Transform parameters saved to: {os.path.join(args.output_dir, 'transform_params.json')}"
        )


if __name__ == "__main__":
    main()
