#!/usr/bin/env python3
"""
Provides functions to perform registration (affine and nonlinear) to the SRI24 atlas 
via Nipype's ANTS interface, returning SimpleITK images. Supports registering both 
image volumes and their corresponding masks.
"""

import os
import argparse
import json
import time
import yaml
from typing import Optional, Tuple, Dict, Any, List, Union

from mgmGrowth.preprocessing import LOGGER

import SimpleITK as sitk
from nipype.interfaces.ants import Registration, ApplyTransforms 

def apply_composed_transforms(
    input_image: sitk.Image,
    t1_to_atlas_transform_params: Dict[str, Any],
    secondary_to_t1_transform_params: Dict[str, Any],
    output_path: str,
    interpolation: str = "Linear",
    dimension: int = 3,
    invert_secondary_to_t1: bool = False,
    number_threads: int = 1,
    verbose: bool = False,
    cleanup: bool = True
) -> sitk.Image:
    """
    Apply a composition of transforms to register a secondary modality (T2, SUSC, etc.) 
    to the atlas space in a two-step process:
    1. Apply the transform from secondary modality to T1 in subject's native space
    2. Apply the transform from T1 to atlas space
    
    This approach preserves the inherent alignment advantages of subject-level multimodal 
    registration before a global warp to template space.
    
    Args:
        input_image (sitk.Image):
            The secondary modality image (T2, SUSC, etc.) to be transformed.
        t1_to_atlas_transform_params (Dict[str, Any]):
            Transform parameters from T1 to atlas, typically from register_to_sri24().
        secondary_to_t1_transform_params (Dict[str, Any]):
            Transform parameters from secondary modality to T1.
        output_path (str):
            Path to save the transformed image.
        interpolation (str, optional):
            Interpolation method. Default is "Linear". Options include:
            - "Linear": For intensity images (default)
            - "NearestNeighbor": For label/mask images
            - "BSpline": For smoother results
            - "GenericLabel": For label images with smoother results
        dimension (int, optional):
            Dimension of the image transformation problem. Default is 3.
        invert_secondary_to_t1 (bool, optional):
            Whether to invert the secondary-to-T1 transform. Default is False.
        number_threads (int, optional):
            Number of CPU threads to use. Default is 1.
        verbose (bool, optional):
            If True, LOGGER.infos debug info.
        cleanup (bool, optional):
            If True, removes temporary files after processing.
            
    Returns:
        sitk.Image: The transformed secondary modality image in atlas space
    """
    if verbose:
        LOGGER.info("[ANTS TRANSFORM] Starting composed transform application process")
        start_time = time.time()
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save input image temporarily
    temp_input_path = os.path.join(output_dir, "temp_secondary_input.nii.gz")
    sitk.WriteImage(input_image, temp_input_path)
    
    if verbose:
        LOGGER.info(f"[ANTS TRANSFORM] Input image saved to {temp_input_path}")
        LOGGER.info(f"[ANTS TRANSFORM] Applying composition of transforms to {output_path}")
    
    # Get transform files
    if invert_secondary_to_t1:
        secondary_to_t1_transform = secondary_to_t1_transform_params["inverse_composite_transform"]
        if verbose:
            LOGGER.info(f"[ANTS TRANSFORM] Using inverted secondary-to-T1 transform: {secondary_to_t1_transform}")
    else:
        secondary_to_t1_transform = secondary_to_t1_transform_params["composite_transform"]
        if verbose:
            LOGGER.info(f"[ANTS TRANSFORM] Using secondary-to-T1 transform: {secondary_to_t1_transform}")
    
    t1_to_atlas_transform = t1_to_atlas_transform_params["composite_transform"]
    
    if verbose:
        LOGGER.info(f"[ANTS TRANSFORM] Using T1-to-atlas transform: {t1_to_atlas_transform}")
        LOGGER.info(f"[ANTS TRANSFORM] Using interpolation method: {interpolation}")
    
    # Set up Nipype ApplyTransforms interface
    at = ApplyTransforms()
    at.inputs.dimension = dimension
    at.inputs.input_image = temp_input_path
    at.inputs.reference_image = t1_to_atlas_transform_params["fixed_image_path"]
    at.inputs.output_image = output_path
    
    # ANTs applies transforms in reverse order, so the second in the list
    # (secondary_to_t1_transform) is applied FIRST, and the first in the list
    # (t1_to_atlas_transform) is applied SECOND.
    at.inputs.transforms = [t1_to_atlas_transform, secondary_to_t1_transform]
    at.inputs.interpolation = interpolation
    at.inputs.num_threads = number_threads
    
    if verbose:
        LOGGER.info("[ANTS TRANSFORM] Applying transforms - this may take a moment...")
        stage_start = time.time()
        at.terminal_output = "stream"
    
    # Run the transform application
    at_result = at.run()
    
    if verbose:
        stage_end = time.time()
        LOGGER.info(f"[ANTS TRANSFORM] Transform application completed in {stage_end - stage_start:.2f} seconds")
        LOGGER.info(f"[ANTS TRANSFORM] Transformed image saved to: {at_result.outputs.output_image}")
    
    # Load the transformed image
    registered_image = sitk.ReadImage(at_result.outputs.output_image)
    atlas_image = sitk.ReadImage(t1_to_atlas_transform_params["fixed_image_path"])
    registered_image.CopyInformation(atlas_image)
    
    # Clean up temporary files
    if cleanup and verbose:
        LOGGER.info("[ANTS TRANSFORM] Cleaning up temporary files...")
    
    if cleanup:
        temp_files = [temp_input_path]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    if verbose:
                        LOGGER.info(f"[ANTS TRANSFORM] Removed temporary file: {temp_file}")
                except Exception as e:
                    if verbose:
                        LOGGER.info(f"[ANTS TRANSFORM] Failed to remove temporary file {temp_file}: {e}")
    
    if verbose:
        end_time = time.time()
        LOGGER.info(f"[ANTS TRANSFORM] Total transform process completed in {end_time - start_time:.2f} seconds")
    
    return registered_image

def register_to_sri24(
    moving_image_sitk: sitk.Image,
    atlas_path: str,
    output_dir: str,
    output_transform_prefix: str,
    output_image_path: str,
    initial_moving_transform: Optional[str] = None,
    dimension: int = 3,
    use_histogram_matching: bool = True,
    winsorize_lower_quantile: float = 0.005,
    winsorize_upper_quantile: float = 0.995,
    number_threads: int = 1,
    transforms: Optional[List[str]] = None,
    transform_parameters: Optional[List[Tuple]] = None,
    iterations: Optional[List[List[int]]] = None,
    shrink_factors: Optional[List[List[int]]] = None,
    smoothing_sigmas: Optional[List[List[float]]] = None,
    metrics: Optional[List[str]] = None,
    metric_weights: Optional[List[float]] = None,
    sampling_strategy: Optional[List[str]] = None, 
    sampling_percentage: Optional[List[float]] = None,
    radius_or_number_of_bins: Optional[List[int]] = None,
    convergence_threshold: Optional[List[float]] = None,
    convergence_window_size: Optional[List[int]] = None,
    sigma_units: Optional[List[str]] = None,
    write_composite_transform: bool = True,
    verbose: bool = False,
    cleanup: bool = True,
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
        number_threads (int, optional):
            Number of CPU threads to use. Default is 1.
        transforms (List[str], optional):
            Registration stages to use. Default is ["Rigid", "Affine", "SyN"].
        transform_parameters (List[Tuple], optional):
            Parameters for each transform stage. Default is [(0.1,), (0.1,), (0.1, 3.0, 0.0)].
        iterations (List[List[int]], optional):
            Number of iterations for each stage. Default is [[1000,500,250,100], [1000,500,250,100], [100,70,50,20]].
        shrink_factors (List[List[int]], optional):
            Shrink factors for each stage. Default is [[8,4,2,1], [8,4,2,1], [8,4,2,1]].
        smoothing_sigmas (List[List[float]], optional):
            Smoothing sigmas for each stage. Default is [[3,2,1,0], [3,2,1,0], [3,2,1,0]].
        metrics (List[str], optional):
            Similarity metrics for each stage. Default is ["MI", "MI", "CC"].
        metric_weights (List[float], optional):
            Weights for each metric. Default is [1.0, 1.0, 1.0].
        sampling_strategy (List[str], optional):
            Sampling strategy for each stage. Default is ["Regular", "Regular", "None"].
        sampling_percentage (List[float], optional):
            Sampling percentage for each stage. Default is [0.25, 0.25, None].
        radius_or_number_of_bins (List[int], optional):
            Radius or number of bins for each stage. Default is [32, 32, 4].
        convergence_threshold (List[float], optional):
            Convergence threshold for each stage. Default is [1e-6, 1e-6, 1e-6].
        convergence_window_size (List[int], optional):
            Convergence window size for each stage. Default is [10, 10, 10].
        sigma_units (List[str], optional):
            Sigma units for each stage. Default is ["vox", "vox", "vox"].
        write_composite_transform (bool, optional):
            Whether to write composite transform. Default is True.
        verbose (bool, optional):
            If True, LOGGER.infos debug info.

    Returns:
        (registered_image, transform_params):
            - registered_image: SITK image of the moving image registered to the fixed image.
            - transform_params: Dictionary containing paths to the transformation files.
    """
    # Default registration parameters if not provided
    if transforms is None:
        transforms = ["Rigid", "Affine", "SyN"]
    
    if transform_parameters is None:
        transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]
    
    if iterations is None:
        iterations = [[1000, 500, 250, 100], [1000, 500, 250, 100], [100, 70, 50, 20]]
    
    if shrink_factors is None:
        shrink_factors = [[8, 4, 2, 1], [8, 4, 2, 1], [8, 4, 2, 1]]
    
    if smoothing_sigmas is None:
        smoothing_sigmas = [[3, 2, 1, 0], [3, 2, 1, 0], [3, 2, 1, 0]]
    
    if metrics is None:
        metrics = ["MI", "MI", "CC"]
    
    if metric_weights is None:
        metric_weights = [1.0, 1.0, 1.0]
        
    if sampling_strategy is None:
        sampling_strategy = ["Regular", "Regular", "None"]
        
    if sampling_percentage is None:
        sampling_percentage = [0.25, 0.25, 0.0]
        
    if radius_or_number_of_bins is None:
        radius_or_number_of_bins = [32, 32, 4]
        
    if convergence_threshold is None:
        convergence_threshold = [1e-6, 1e-6, 1e-6]
        
    if convergence_window_size is None:
        convergence_window_size = [10, 10, 10]
        
    if sigma_units is None:
        sigma_units = ["vox", "vox", "vox"]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    if verbose:
        LOGGER.info("[ANTS REGISTRATION] Starting registration process")
        LOGGER.info(f"[ANTS REGISTRATION] Output directory: {output_dir}")

    # 1) Save input images as NIfTI
    moving_nii_path = os.path.abspath(os.path.join(output_dir, "moving_input.nii.gz"))
    fixed_nii_path = atlas_path

    if verbose:
        LOGGER.info(
            f"[ANTS REGISTRATION] Temporarily Saving input images to: {moving_nii_path}"
        )

    sitk.WriteImage(moving_image_sitk, moving_nii_path)

    # 2) Configure Nipype ANTS Registration
    if verbose:
        LOGGER.info("[ANTS REGISTRATION] Configuring ANTS registration parameters")

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

    # Set other registration parameters
    reg.inputs.transforms = transforms
    reg.inputs.transform_parameters = transform_parameters
    reg.inputs.number_of_iterations = iterations
    reg.inputs.shrink_factors = shrink_factors
    reg.inputs.smoothing_sigmas = smoothing_sigmas
    reg.inputs.metric = metrics
    reg.inputs.metric_weight = metric_weights

    # Set advanced registration parameters
    reg.inputs.sampling_strategy = sampling_strategy
    reg.inputs.sampling_percentage = sampling_percentage
    reg.inputs.radius_or_number_of_bins = radius_or_number_of_bins
    reg.inputs.convergence_threshold = convergence_threshold
    reg.inputs.convergence_window_size = convergence_window_size
    reg.inputs.sigma_units = sigma_units
    reg.inputs.write_composite_transform = write_composite_transform

    if verbose:
        LOGGER.info(f"[ANTS REGISTRATION] Transform prefix set to: {transform_prefix}")

    reg.inputs.output_transform_prefix = transform_prefix
    reg.inputs.output_warped_image = output_image_path

    # Initial transform if provided
    if initial_moving_transform:
        if verbose:
            LOGGER.info(
                f"[ANTS REGISTRATION] Using initial transform: {initial_moving_transform}"
            )
        reg.inputs.initial_moving_transform = initial_moving_transform

    if verbose:
        LOGGER.info(f"[ANTS REGISTRATION] Registration configuration complete")
        LOGGER.info(f"[ANTS REGISTRATION] Running with transforms: {reg.inputs.transforms}")
        LOGGER.info(f"[ANTS REGISTRATION] Using histogram matching: {use_histogram_matching}")
        reg.terminal_output = "stream"

    # 3) Run Registration
    if verbose:
        LOGGER.info(
            "[ANTS REGISTRATION] Starting registration execution - this may take some time..."
        )
        stage_start = time.time()

    res = reg.run()

    if verbose:
        stage_end = time.time()
        LOGGER.info(
            f"[ANTS REGISTRATION] Registration completed in {stage_end - stage_start:.2f} seconds"
        )

    # 4) Convert results back to SITK
    registered_path = res.outputs.warped_image

    if verbose:
        LOGGER.info(f"[ANTS REGISTRATION] Loading registered image from: {registered_path}")

    registered_sitk = sitk.ReadImage(registered_path)
    fixed_image = sitk.ReadImage(atlas_path)
    registered_sitk.CopyInformation(fixed_image)

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
        LOGGER.info(
            f"[ANTS REGISTRATION] Total registration process completed in {end_time - start_time:.2f} seconds"
        )
        LOGGER.info(
            f"[ANTS REGISTRATION] Transform parameters saved to: {transform_params_path}"
        )
        LOGGER.info("[ANTS REGISTRATION] Transform files:")
        for k, v in transform_params.items():
            LOGGER.info(f"[ANTS REGISTRATION]   {k}: {v}")
    # Clean up temporary files
    if cleanup and verbose:
        LOGGER.info("[ANTS REGISTRATION] Cleaning up temporary files...")
        
    if cleanup:
        temp_files = [moving_nii_path]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    if verbose:
                        LOGGER.info(f"[ANTS REGISTRATION] Removed temporary file: {temp_file}")
                except Exception as e:
                    if verbose:
                        LOGGER.info(f"[ANTS REGISTRATION] Failed to remove temporary file {temp_file}: {e}")
    return registered_sitk, transform_params

def register_to_sri24_with_mask(
    moving_image_sitk: sitk.Image,
    moving_mask_sitk: sitk.Image,
    atlas_path: str,
    output_dir: str,
    output_transform_prefix: str,
    output_image_prefix: str,
    output_mask_path: str,
    initial_moving_transform: Optional[str] = None,
    dimension: int = 3,
    use_histogram_matching: bool = True,
    winsorize_lower_quantile: float = 0.005,
    winsorize_upper_quantile: float = 0.995,
    number_threads: int = 1,
    transforms: Optional[List[str]] = None,
    transform_parameters: Optional[List[Tuple]] = None,
    iterations: Optional[List[List[int]]] = None,
    shrink_factors: Optional[List[List[int]]] = None,
    smoothing_sigmas: Optional[List[List[float]]] = None,
    metrics: Optional[List[str]] = None,
    metric_weights: Optional[List[float]] = None,
    sampling_strategy: Optional[List[str]] = None, 
    sampling_percentage: Optional[List[float]] = None,
    radius_or_number_of_bins: Optional[List[int]] = None,
    convergence_threshold: Optional[List[float]] = None,
    convergence_window_size: Optional[List[int]] = None,
    sigma_units: Optional[List[str]] = None,
    write_composite_transform: bool = True,
    verbose: bool = False,
    cleanup: bool = True
) -> Tuple[sitk.Image, sitk.Image, Dict[str, Any]]:
    """
    Register a moving image (T1) and its corresponding mask to a reference atlas (SRI24)
    using ANTs via Nipype, returning the registered image and mask as SimpleITK images
    along with the transformation parameters.

    Args:
        moving_image_sitk (sitk.Image):
            The input 3D MRI volume (T1) as a SimpleITK image to be registered.
        moving_mask_sitk (sitk.Image):
            The corresponding mask for the moving image as a SimpleITK image.
        atlas_path (str):
            The reference atlas (SRI24) as a path.
        output_dir (str):
            Directory to save all output files.
        output_transform_prefix (str):
            Prefix for the output transformation files.
        output_image_prefix (str):
            Path or filename for the registered image output.
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
        number_threads (int, optional):
            Number of CPU threads to use. Default is 1.
        transforms (List[str], optional):
            Registration stages to use. Default is ["Rigid", "Affine", "SyN"].
        transform_parameters (List[Tuple], optional):
            Parameters for each transform stage. Default is [(0.1,), (0.1,), (0.1, 3.0, 0.0)].
        iterations (List[List[int]], optional):
            Number of iterations for each stage. Default is [[1000,500,250,100], [1000,500,250,100], [100,70,50,20]].
        shrink_factors (List[List[int]], optional):
            Shrink factors for each stage. Default is [[8,4,2,1], [8,4,2,1], [8,4,2,1]].
        smoothing_sigmas (List[List[float]], optional):
            Smoothing sigmas for each stage. Default is [[3,2,1,0], [3,2,1,0], [3,2,1,0]].
        metrics (List[str], optional):
            Similarity metrics for each stage. Default is ["MI", "MI", "CC"].
        metric_weights (List[float], optional):
            Weights for each metric. Default is [1.0, 1.0, 1.0].
        sampling_strategy (List[str], optional):
            Sampling strategy for each stage. Default is ["Regular", "Regular", "None"].
        sampling_percentage (List[float], optional):
            Sampling percentage for each stage. Default is [0.25, 0.25, None].
        radius_or_number_of_bins (List[int], optional):
            Radius or number of bins for each stage. Default is [32, 32, 4].
        convergence_threshold (List[float], optional):
            Convergence threshold for each stage. Default is [1e-6, 1e-6, 1e-6].
        convergence_window_size (List[int], optional):
            Convergence window size for each stage. Default is [10, 10, 10].
        sigma_units (List[str], optional):
            Sigma units for each stage. Default is ["vox", "vox", "vox"].
        write_composite_transform (bool, optional):
            Whether to write composite transform. Default is True.
        verbose (bool, optional):
            If True, LOGGER.infos debug info.

    Returns:
        (registered_image, registered_mask, transform_params):
            - registered_image: SITK image of the moving image registered to the fixed image.
            - registered_mask: SITK image of the moving mask registered to the fixed image.
            - transform_params: Dictionary containing paths to the transformation files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        LOGGER.info("[ANTS REGISTRATION] Creating output directory for entire process")
        LOGGER.info(f"[ANTS REGISTRATION] Output directory path: {output_dir}")

    # Save the mask input to the output directory
    mask_path = os.path.join(output_dir, "moving_mask_input.nii.gz")
    sitk.WriteImage(moving_mask_sitk, mask_path)

    # Build output paths
    output_image_path = (
        os.path.join(output_dir, output_image_prefix)
        if output_image_prefix.endswith(".nii.gz")
        else os.path.join(output_dir, output_image_prefix + ".nii.gz")
    )

    if verbose:
        LOGGER.info(f"[ANTS REGISTRATION] Saved input mask to: {mask_path}")
        LOGGER.info("[ANTS REGISTRATION] Starting registration of image volume")

    # First, register the image
    registered_image, transform_params = register_to_sri24(
        moving_image_sitk=moving_image_sitk,
        atlas_path=atlas_path,
        output_dir=output_dir,
        output_transform_prefix=output_transform_prefix,
        output_image_path=output_image_path,
        initial_moving_transform=initial_moving_transform,
        dimension=dimension,
        use_histogram_matching=use_histogram_matching,
        winsorize_lower_quantile=winsorize_lower_quantile,
        winsorize_upper_quantile=winsorize_upper_quantile,
        number_threads=number_threads,
        transforms=transforms,
        transform_parameters=transform_parameters,
        iterations=iterations,
        shrink_factors=shrink_factors,
        smoothing_sigmas=smoothing_sigmas,
        metrics=metrics,
        metric_weights=metric_weights,
        sampling_strategy=sampling_strategy,
        sampling_percentage=sampling_percentage,
        radius_or_number_of_bins=radius_or_number_of_bins,
        convergence_threshold=convergence_threshold,
        convergence_window_size=convergence_window_size,
        sigma_units=sigma_units,
        write_composite_transform=write_composite_transform,
        verbose=verbose,
        cleanup=cleanup
    )

    # Now, apply the same transformation to the mask
    if verbose:
        LOGGER.info("[ANTS REGISTRATION] Starting registration of mask volume")
        start_time = time.time()

    # Use ANTs ApplyTransforms to transform the mask
    if verbose:
        LOGGER.info("[ANTS REGISTRATION] Configuring ApplyTransforms for mask")
        LOGGER.info(
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
        LOGGER.info(
            "[ANTS REGISTRATION] Applying transformation to mask - this may take a moment..."
        )
        stage_start = time.time()
        at.terminal_output = "stream"

    at_result = at.run()

    if verbose:
        stage_end = time.time()
        LOGGER.info(
            f"[ANTS REGISTRATION] Mask transformation completed in {stage_end - stage_start:.2f} seconds"
        )
        LOGGER.info(
            f"[ANTS REGISTRATION] Registered mask saved to: {at_result.outputs.output_image}"
        )

    # Load the transformed mask
    registered_mask = sitk.ReadImage(at_result.outputs.output_image)
    fixed_image = sitk.ReadImage(atlas_path)  # Read the atlas image
    registered_mask.CopyInformation(fixed_image)

    # Update transform_params to include mask paths
    transform_params["input_mask_path"] = mask_path
    transform_params["output_mask_path"] = output_mask_path

    # Save updated transform_params as JSON
    transform_params_path = os.path.join(output_dir, "transform_params.json")
    with open(transform_params_path, "w") as f:
        json.dump(transform_params, f, indent=2)

    if verbose:
        end_time = time.time()
        LOGGER.info(
            f"[ANTS REGISTRATION] Total mask registration process completed in {end_time - start_time:.2f} seconds"
        )
        LOGGER.info(
            f"[ANTS REGISTRATION] Updated transform parameters saved to: {transform_params_path}"
        )
    # Clean up temporary files
    if cleanup and verbose:
        LOGGER.info("[ANTS REGISTRATION] Cleaning up temporary files...")
        
    if cleanup:
        temp_files = [mask_path]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    if verbose:
                        LOGGER.info(f"[ANTS REGISTRATION] Removed temporary file: {temp_file}")
                except Exception as e:
                    if verbose:
                        LOGGER.info(f"[ANTS REGISTRATION] Failed to remove temporary file {temp_file}: {e}")
    return registered_image, registered_mask, transform_params

def register_image_to_sri24(
    moving_image: sitk.Image,
    moving_mask: Optional[sitk.Image] = None,
    fixed_image: Optional[sitk.Image] = None,
    config_path: str = "../../configs/registration_sri24.yaml",
    verbose: bool = False,
    cleanup:bool = True,
) -> Union[Tuple[sitk.Image, Dict[str, Any]], Tuple[sitk.Image, sitk.Image, Dict[str, Any]]]:
    """
    Register a SimpleITK image to the SRI24 atlas with optional mask using a YAML configuration file.
    
    Args:
        moving_image (sitk.Image):
            The input image to be registered as a SimpleITK image
        moving_mask (Optional[sitk.Image]):
            The input mask as a SimpleITK image (optional)
        config_path (str):
            Path to YAML config file containing registration parameters
        verbose (bool):
            Enable verbose output
    
    Returns:
        If mask is provided:
            Tuple[sitk.Image, sitk.Image, Dict[str, Any]]: Registered image, registered mask, and transform parameters
        Otherwise:
            Tuple[sitk.Image, Dict[str, Any]]: Registered image and transform parameters
    """
    # Load configuration file
    if not config_path:
        raise ValueError("Missing required parameter: config_path")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading configuration file: {e}")
    
    # Override verbose setting if provided
    config['verbose'] = verbose
    
    # Create output directory
    output_dir = config.get('output_dir', './registration_output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Here, if an ATLAS is passed, then we temporarily save it to the output directory
    # if an ATLAS is not passed, then we are going to use the ATLAS stored in the config
    # path

    if fixed_image is None:
        # Load atlas
        atlas_path = config.get('atlas_path')
        if not atlas_path:
            raise ValueError("Missing required parameter in config: atlas_path, or input you atlas sitk.Image object")
    else:
        # Save fixed image temporarily
        atlas_path = os.path.join(output_dir, "fixed_image.nii.gz")
        sitk.WriteImage(fixed_image, atlas_path)

    try:
        fixed_image = sitk.ReadImage(str(atlas_path))
    except Exception as e:
        raise RuntimeError(f"Error loading atlas image: {e}")
    
    # Set up output paths
    output_registered_path = config.get('output_registered')
    if not output_registered_path:
        output_registered_path = os.path.join(output_dir, "registered.nii.gz")
    elif not os.path.isabs(str(output_registered_path)):
        output_registered_path = os.path.join(output_dir, str(output_registered_path))
    
    # Convert convergence_threshold from strings to floats 
    if config.get('convergence_threshold'):
        convergence_threshold = [float(threshold) for threshold in config.get('convergence_threshold')]
    else:
        convergence_threshold = None

    # Extract registration parameters from config
    reg_params = {
        'output_dir': output_dir,
        'output_transform_prefix': config.get('output_transform_prefix', 'transform_'),
        'initial_moving_transform': config.get('initial_transform'),
        'use_histogram_matching': config.get('histogram_matching', True),
        'dimension': config.get('dimension', 3),
        'winsorize_lower_quantile': config.get('winsorize_lower_quantile', 0.005),
        'winsorize_upper_quantile': config.get('winsorize_upper_quantile', 0.995),
        'number_threads': config.get('number_threads', 1),
        'transforms': config.get('transforms'),
        'transform_parameters': config.get('transform_parameters'),
        'iterations': config.get('iterations'),
        'shrink_factors': config.get('shrink_factors'),
        'smoothing_sigmas': config.get('smoothing_sigmas'),
        'metrics': config.get('metrics'),
        'metric_weights': config.get('metric_weights'),
        'sampling_strategy': config.get('sampling_strategy'),
        'sampling_percentage': config.get('sampling_percentage'),
        'radius_or_number_of_bins': config.get('radius_or_number_of_bins'),
        'convergence_threshold': convergence_threshold,
        'convergence_window_size': config.get('convergence_window_size'),
        'sigma_units': config.get('sigma_units'),
        'write_composite_transform': config.get('write_composite_transform', True),
        'verbose': verbose,
        'cleanup': cleanup
    }

    if verbose:
        LOGGER.info(f"[ANTS REGISTRATION] Moving image size: {moving_image.GetSize()}")
        LOGGER.info(f"[ANTS REGISTRATION] Fixed image size: {fixed_image.GetSize()}")
        LOGGER.info(f"[ANTS REGISTRATION] Output directory: {output_dir}")
        LOGGER.info(f"[ANTS REGISTRATION] Output registered image: {output_registered_path}")
    
    # Process with mask if provided
    if moving_mask is not None:
        output_mask_path = config.get('output_mask')
        if not output_mask_path:
            output_mask_path = os.path.join(output_dir, "registered_mask.nii.gz")
        elif not os.path.isabs(str(output_mask_path)):
            output_mask_path = os.path.join(output_dir, str(output_mask_path))
        
        # Save mask temporarily
        mask_nii_path = os.path.join(output_dir, "moving_mask_input.nii.gz")
        sitk.WriteImage(moving_mask, mask_nii_path)
        
        if verbose:
            LOGGER.info(f"[ANTS REGISTRATION] Mask image size: {moving_mask.GetSize()}")
            LOGGER.info(f"[ANTS REGISTRATION] Output registered mask: {output_mask_path}")
            LOGGER.info("[ANTS REGISTRATION] Performing registration with mask...")
        
        # Perform registration with mask
        registered_image, registered_mask, transform_params = register_to_sri24_with_mask(
            moving_image_sitk=moving_image,
            moving_mask_sitk=moving_mask,
            atlas_path=atlas_path,
            output_image_prefix=os.path.basename(str(output_registered_path)),
            output_mask_path=output_mask_path,
            **reg_params
        )
        
        return registered_image, registered_mask, transform_params
    else:
        # Perform registration without mask
        if verbose:
            LOGGER.info("[ANTS REGISTRATION] Performing registration without mask...")
        output_image_prefix=os.path.basename(str(output_registered_path))
        output_image_path = (
            os.path.join(output_dir, output_image_prefix)
            if output_image_prefix.endswith(".nii.gz")
            else os.path.join(output_dir, output_image_prefix + ".nii.gz")
        )

        registered_image, transform_params = register_to_sri24(
            moving_image_sitk=moving_image,
            atlas_path=atlas_path,
            output_image_path=output_image_path,
            **reg_params
        )
        
        return registered_image, transform_params

def main():
    """
    Command-line interface for the registration function supporting both YAML configuration
    and direct CLI arguments, with the latter taking precedence.
    """
    parser = argparse.ArgumentParser(
        description="Register a T1/T2 image to SRI24 atlas using YAML config or CLI arguments."
    )
    
    # Primary input/output arguments (can override YAML)
    parser.add_argument("-i", "--input_image", type=str, help="Path to the input image")
    parser.add_argument("-m", "--mask_path", type=str, help="Path to the input mask (optional)")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory path")
    parser.add_argument("-r", "--output_registered", type=str, help="Output registered image path")
    parser.add_argument("-k", "--output_mask", type=str, help="Output registered mask path")
    
    # Optional YAML config
    parser.add_argument("-c", "--config", type=str, help="Path to YAML configuration file")
    
    # Other common parameters
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use")
    
    args = parser.parse_args()
    
    # Initialize config with defaults
    config = {
        'verbose': args.verbose or False,
        'num_threads': args.threads or 1
    }
    
    # Load YAML config if provided (as base configuration)
    if args.config:
        try:
            with open(args.config, 'r') as f:
                yaml_config = yaml.safe_load(f)
                config.update(yaml_config)  # Update with YAML values
        except Exception as e:
            LOGGER.info(f"Error loading configuration file: {e}")
            return
    
    # Override with CLI arguments if provided
    if args.input_image:
        config['input_image'] = args.input_image
    if args.mask_path:
        config['mask_path'] = args.mask_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.output_registered:
        config['output_registered'] = args.output_registered
    if args.output_mask:
        config['output_mask'] = args.output_mask
    if args.verbose:
        config['verbose'] = args.verbose
    if args.threads:
        config['num_threads'] = args.threads
    
    # Check required parameters
    if not config.get('input_image') or not config.get('atlas_path'):
        LOGGER.info("Error: Missing required parameters. Please provide input_image and atlas_path.")
        parser.LOGGER.info_help()
        return
    
    # Set up output directory
    output_dir = config.get('output_dir', './registration_output')
    os.makedirs(output_dir, exist_ok=True)
    
    verbose = config.get('verbose', False)
    
    # Load input images
    if verbose:
        LOGGER.info(f"[ANTS REGISTRATION] Loading input image: {config['input_image']}")
        LOGGER.info(f"[ANTS REGISTRATION] Loading atlas image: {config['atlas_path']}")
    
    try:
        moving_image = sitk.ReadImage(config['input_image'])
        fixed_image = sitk.ReadImage(config['atlas_path'])
    except Exception as e:
        LOGGER.info(f"Error loading images: {e}")
        return
    
    # Extract registration parameters from config
    reg_params = {
        'output_dir': output_dir,
        'output_transform_prefix': config.get('output_transform_prefix', 'transform_'),
        'initial_moving_transform': config.get('initial_transform'),
        'use_histogram_matching': config.get('histogram_matching', True),
        'dimension': config.get('dimension', 3),
        'winsorize_lower_quantile': config.get('winsorize_lower_quantile', 0.005),
        'winsorize_upper_quantile': config.get('winsorize_upper_quantile', 0.995),
        'number_threads': config.get('num_threads', 1),
        'transforms': config.get('transforms'),
        'transform_parameters': config.get('transform_parameters'),
        'iterations': config.get('iterations'),
        'shrink_factors': config.get('shrink_factors'),
        'smoothing_sigmas': config.get('smoothing_sigmas'),
        'metrics': config.get('metrics'),
        'metric_weights': config.get('metric_weights'),
        'verbose': verbose
    }
    
    # Set output paths
    output_registered = config.get('output_registered')
    if not output_registered:
        output_registered = os.path.join(output_dir, "registered.nii.gz")
    elif not os.path.isabs(output_registered):
        output_registered = os.path.join(output_dir, output_registered)
    
    reg_params['output_image_path'] = output_registered
    
    if verbose:
        LOGGER.info(f"[ANTS REGISTRATION] Moving image size: {moving_image.GetSize()}")
        LOGGER.info(f"[ANTS REGISTRATION] Fixed image size: {fixed_image.GetSize()}")
        LOGGER.info(f"[ANTS REGISTRATION] Output directory: {output_dir}")
        LOGGER.info(f"[ANTS REGISTRATION] Output registered image: {output_registered}")
    
    # Check if we're processing a mask as well
    mask_path = config.get('mask_path')
    if mask_path:
        output_mask = config.get('output_mask')
        if not output_mask:
            output_mask = os.path.join(output_dir, "registered_mask.nii.gz")
        elif not os.path.isabs(output_mask):
            output_mask = os.path.join(output_dir, output_mask)
        
        if verbose:
            LOGGER.info(f"[ANTS REGISTRATION] Loading mask image: {mask_path}")
            LOGGER.info(f"[ANTS REGISTRATION] Output registered mask: {output_mask}")
        
        try:
            moving_mask = sitk.ReadImage(mask_path)
        except Exception as e:
            LOGGER.info(f"Error loading mask image: {e}")
            return
        
        if verbose:
            LOGGER.info(f"[ANTS REGISTRATION] Mask image size: {moving_mask.GetSize()}")
            LOGGER.info("[ANTS REGISTRATION] Performing registration with mask...")
        
        # Perform registration with mask
        registered_image, registered_mask, transform_params = register_to_sri24_with_mask(
            moving_image_sitk=moving_image,
            moving_mask_sitk=moving_mask,
            fixed_image_sitk=fixed_image,
            output_image_prefix=os.path.basename(output_registered),
            output_mask_path=output_mask,
            **reg_params
        )
    
    else:
        # Perform registration without mask
        if verbose:
            LOGGER.info("[ANTS REGISTRATION] Performing registration without mask...")
        
        registered_image, transform_params = register_to_sri24(
            moving_image_sitk=moving_image,
            fixed_image_sitk=fixed_image,
            **reg_params
        )
    
    if verbose:
        LOGGER.info("[ANTS REGISTRATION] Registration completed successfully")
        LOGGER.info(f"[ANTS REGISTRATION] All outputs saved to directory: {output_dir}")
        LOGGER.info(f"[ANTS REGISTRATION] Transform files saved with prefix: {reg_params['output_transform_prefix']}")
        LOGGER.info(f"[ANTS REGISTRATION] Transform parameters saved to: {os.path.join(output_dir, 'transform_params.json')}")
    
    return registered_image, transform_params
if __name__ == "__main__":
    main()
