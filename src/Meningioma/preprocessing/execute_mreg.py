import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import SimpleITK as sitk
import os
import json
import argparse
import numpy as np

from Meningioma.preprocessing.tools.remove_extra_channels import remove_first_channel
from Meningioma.preprocessing.tools.nrrd_to_nifti import nifti_write_3d
from Meningioma.preprocessing.tools.casting import cast_volume_and_mask
from Meningioma.preprocessing.tools.denoise_susan import denoise_susan
from Meningioma.preprocessing.tools.bias_field_corr_n4 import generate_brain_mask_sitk, n4_bias_field_correction
from Meningioma.preprocessing.tools.skull_stripping.fsl_bet import fsl_bet_brain_extraction
from Meningioma.preprocessing.tools.registration.ants_registration import register_image_to_sri24, apply_composed_transforms

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('meningioma_preprocessing.log')
    ]
)
logger = logging.getLogger('meningioma_preprocessing')


def load_plan(json_path):
    """Load preprocessing plan from JSON file."""
    with open(json_path, 'r') as f:
        plan = json.load(f)
    return plan


def extract_patient_data(plan, patient_ids):
    """
    Extract volume and mask paths for each patient's pulse sequences.
    
    Returns the data with T1 sequences ordered first for each patient.
    """
    results = {}
    
    # Iterate through requested patients
    for patient_id in patient_ids:
        patient_key = f"P{patient_id}"
        
        # Check if patient exists in the plan
        if patient_key not in plan["data"]:
            logger.warning(f"Patient {patient_id} not found in the plan")
            continue
        
        patient_data = plan["data"][patient_key]
        pulses_data = {}
        
        # Separate T1 and other pulses to ensure T1 is processed first
        t1_pulses = {}
        other_pulses = {}
        
        # Iterate through each pulse sequence (SUSC, T1, T2, TC)
        for pulse_name, pulse_info in patient_data["pulses"].items():
            # Skip if there's an error with this pulse
            if pulse_info.get("error", True):
                logger.info(f"Skipping {patient_key} {pulse_name} due to error flag")
                continue
            
            # Extract volume and mask paths
            volume_path = pulse_info.get("volume", {}).get("route", "")
            mask_path = pulse_info.get("segmentation", {}).get("route", "")
            
            # Only include if we have valid paths
            if volume_path and mask_path:
                pulse_data = {
                    "volume_path": volume_path,
                    "mask_path": mask_path,
                    "modality": pulse_info.get("modality", ""),
                }
                
                # Verify files exist
                if not os.path.exists(volume_path):
                    logger.warning(f"Volume file not found: {volume_path}")
                if not os.path.exists(mask_path):
                    logger.warning(f"Mask file not found: {mask_path}")
                
                # Separate T1 and other pulses
                if pulse_name.upper() == "T1":
                    t1_pulses[pulse_name] = pulse_data
                else:
                    other_pulses[pulse_name] = pulse_data
        
        # Combine with T1 first, then other pulses
        pulses_data.update(t1_pulses)
        pulses_data.update(other_pulses)
        
        # Add patient data to results if we have any valid pulses
        if pulses_data:
            results[patient_key] = pulses_data
    
    return results


def create_output_directories(output_dir, patient_ids):
    """Create output directory structure for patients."""
    base_dir = Path(output_dir) / "meningioma"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for patient_id in patient_ids:
        patient_dir = base_dir / f"P{patient_id}"
        patient_dir.mkdir(exist_ok=True)
        
    return base_dir


def rm_pipeline(
    pulse_data: Dict[str, Any],
    preprocessing_plan: Dict[str, Any],
    patient_output_dir: Path,
    patient_id: str,
    pulse_name: str,
    t1_info: Dict[str, Any],  # New parameter to store T1 paths and transform info
    verbose: bool = True,
    save_intermediate: bool = False
) -> Dict[str, Any]:
    """
    Apply RM-specific preprocessing steps to a pulse sequence.
    
    Implements the full RM preprocessing pipeline with steps including:
    - Channel removal
    - NIfTI export
    - Volume casting
    - Image reorientation
    - Image resampling
    - Denoising (optional)
    - Brain masking
    - Bias field correction (N4)
    - Brain extraction (FSL BET)
    - Registration (T1 -> SRI24 or T2/SUSC -> T1 -> SRI24)
    
    Args:
        pulse_data: Dictionary containing pulse data and metadata
        preprocessing_plan: Dictionary with RM preprocessing configuration
        patient_output_dir: Output directory for the patient's processed files
        patient_id: Patient identifier (e.g., 'P1')
        pulse_name: Name of the pulse sequence (e.g., 'T1', 'T2', 'SUSC')
        t1_info: Dictionary containing T1 paths and transform info
        verbose: Whether to print detailed processing information
        save_intermediate: If True, save intermediate processing files to 'others' directory
    
    Returns:
        Dictionary containing the processed pulse data with updated file paths
    """
    # Create a working dictionary for this pulse
    processed_pulse = pulse_data.copy()
    current_image = None
    current_header = None
    current_mask = None
    brain_mask = None  # Initialize brain_mask to None
    
    # Create main patient output directory and "others" subdirectory for intermediate files
    patient_output_dir.mkdir(exist_ok=True)
    
    # Create "others" directory for intermediate files if needed
    if save_intermediate:
        others_dir = patient_output_dir / "others"
        others_dir.mkdir(exist_ok=True)
    else:
        others_dir = patient_output_dir  # Default to main dir
    
    if verbose:
        logger.info(f"\n[RM / {pulse_name}] Starting RM preprocessing pipeline")
    
    # 1. remove_channel (if it exists in the plan)
    if "remove_channel" in preprocessing_plan:
        if verbose:
            logger.info(f"[RM / CHANNEL REMOVAL] Processing {Path(pulse_data['volume_path']).name}")
        
        try:
            channel = preprocessing_plan["remove_channel"].get("channel", 0)
            if verbose:
                logger.info(f"[RM / CHANNEL REMOVAL] Extracting channel {channel}")
            
            # Apply remove_channel function to the volume
            current_image, current_header = remove_first_channel(
                pulse_data["volume_path"], 
                channel, 
                verbose=verbose
            )
            
            if verbose:
                logger.info(f"[RM / CHANNEL REMOVAL] Completed: size={current_image.GetSize()}, "
                      f"dimensions={current_image.GetDimension()}, "
                      f"spacing={current_image.GetSpacing()}")
            
        except Exception as e:
            logger.error(f"[RM / CHANNEL REMOVAL] Error: {str(e)}")
            
    
    # 2. export_nifti (if it exists in the plan)
    if "export_nifti" in preprocessing_plan and preprocessing_plan["export_nifti"]:
        if verbose:
            logger.info(f"[RM / NIFTI EXPORT] Converting to NIfTI format")
        
        try:
            # Output path for the NIfTI file - This is a main output file, not an intermediate
            nifti_filename = f"{pulse_name}_{patient_id.replace('P', '')}.nii.gz"
            nifti_path = patient_output_dir / nifti_filename
            
            # If we have a current_image from a previous step, use it
            # Otherwise, use the original volume path
            input_data = (current_image, current_header) if current_image is not None else pulse_data["volume_path"]
            
            # Convert to NIfTI
            output_path = nifti_write_3d(
                volume=input_data, #type: ignore
                out_file=str(nifti_path),
                verbose=verbose
            )
            
            # Update the processed data with the new file path
            processed_pulse["nifti_path"] = output_path
            
            # If this is T1, store this path for later use by other modalities
            if pulse_name.upper() == "T1":
                t1_info["t1_native_path"] = output_path
            
            if verbose:
                logger.info(f"[RM / NIFTI EXPORT] Saved to {output_path}")
            
            # Load the exported NIfTI for further processing
            current_image = sitk.ReadImage(output_path)
            
        except Exception as e:
            logger.error(f"[RM / NIFTI EXPORT] Error: {str(e)}")
            # Continue even if NIfTI export fails
    
    # 3. Load the segmentation mask if we have a current_image
    if current_image is not None:
        try:
            if verbose:
                logger.info(f"[RM / MASK LOADING] Loading segmentation mask")
            
            current_mask = sitk.ReadImage(pulse_data["mask_path"])
            
            if verbose:
                logger.info(f"[RM / MASK LOADING] Mask loaded: size={current_mask.GetSize()}, "
                      f"dimensions={current_mask.GetDimension()}")
                
        except Exception as e:
            logger.error(f"[RM / MASK LOADING] Error loading mask: {str(e)}")
            # Continue processing even if mask loading fails
    
    # 4. cast_volume (if it exists in the plan)
    if current_image is not None and current_mask is not None and "cast_volume" in preprocessing_plan and preprocessing_plan["cast_volume"]:
        if verbose:
            logger.info(f"[RM / CAST VOLUME] Casting volume to float32 and mask to uint8")
        
        try:
            # Cast volume to float32 and mask to uint8
            current_image, current_mask = cast_volume_and_mask(
                current_image, 
                current_mask
            )
            
            # Save the cast volume and mask to the appropriate directory
            if save_intermediate:
                cast_volume_path = others_dir / f"{pulse_name}_{patient_id.replace('P', '')}_cast.nii.gz"
                cast_mask_path = others_dir / f"{pulse_name}_{patient_id.replace('P', '')}_mask_cast.nii.gz"
                
                sitk.WriteImage(current_image, str(cast_volume_path))
                sitk.WriteImage(current_mask, str(cast_mask_path))
                
                # Update the processed data with the new file paths
                processed_pulse["cast_volume_path"] = str(cast_volume_path)
                processed_pulse["cast_mask_path"] = str(cast_mask_path)
                
                if verbose:
                    logger.info(f"[RM / CAST VOLUME] Volume cast to {current_image.GetPixelID()} and saved to {cast_volume_path}")
                    logger.info(f"[RM / CAST VOLUME] Mask cast to {current_mask.GetPixelID()} and saved to {cast_mask_path}")
            
        except Exception as e:
            logger.error(f"[RM / CAST VOLUME] Error: {str(e)}")
            # Continue even if casting fails
    
    # 7. denoise (if it exists and is enabled in the plan)
    if current_image is not None and "denoise" in preprocessing_plan:
        # Check if denoising is enabled
        denoise_enabled = preprocessing_plan["denoise"].get("enable", False)
        
        if denoise_enabled:
            if verbose:
                logger.info(f"[RM / DENOISE] Applying SUSAN denoising to volume")
            
            try:
                # Get denoise parameters
                susan_params = preprocessing_plan["denoise"].get("susan", {})
                brightness_threshold = susan_params.get("brightness_threshold", 0.001)
                fwhm = susan_params.get("fwhm", 0.5)
                dimension = susan_params.get("dimension", 3)
                
                if verbose:
                    logger.info(f"[RM / DENOISE] SUSAN parameters: brightness_threshold={brightness_threshold}, fwhm={fwhm}, dimension={dimension}")
                
                # Apply denoising to the volume
                current_image = denoise_susan(
                    current_image,
                    brightness_threshold=brightness_threshold,
                    fwhm=fwhm,
                    dimension=dimension,
                    mask_sitk=current_mask,  # Optional: Use mask to constrain denoising
                    verbose=verbose
                )
                
                # Save the denoised volume if requested
                if save_intermediate:
                    denoised_volume_path = others_dir / f"{pulse_name}_{patient_id.replace('P', '')}_denoised.nii.gz"
                    sitk.WriteImage(current_image, str(denoised_volume_path))
                    
                    # Update the processed data with the new file path
                    processed_pulse["denoised_volume_path"] = str(denoised_volume_path)
                    
                    if verbose:
                        logger.info(f"[RM / DENOISE] Volume denoised and saved to {denoised_volume_path}")
                
            except Exception as e:
                logger.error(f"[RM / DENOISE] Error: {str(e)}")
                # Continue even if denoising fails
        else:
            if verbose:
                logger.info(f"[RM / DENOISE] Denoising is disabled in the preprocessing plan, skipping")
    
    # 8. brain_mask (if it exists in the plan)
    if current_image is not None and "brain_mask" in preprocessing_plan:
        if verbose:
            logger.info(f"[RM / BRAIN MASK] Generating brain mask")
        
        try:
            # Get brain mask parameters
            brain_mask_params = preprocessing_plan["brain_mask"]
            threshold_method = brain_mask_params.get("threshold_method", "li")
            structure_size_2d = brain_mask_params.get("structure_size_2d", 7)
            iterations_2d = brain_mask_params.get("iterations_2d", 3)
            structure_size_3d = brain_mask_params.get("structure_size_3d", 3)
            iterations_3d = brain_mask_params.get("iterations_3d", 1)
            
            if verbose:
                logger.info(f"[RM / BRAIN MASK] Parameters: threshold_method={threshold_method}, "
                      f"structure_size_2d={structure_size_2d}, iterations_2d={iterations_2d}, "
                      f"structure_size_3d={structure_size_3d}, iterations_3d={iterations_3d}")
            
            # Generate brain mask using the current image
            _, brain_mask = generate_brain_mask_sitk(
                current_image,
                threshold_method=threshold_method,
                structure_size_2d=structure_size_2d,
                iterations_2d=iterations_2d,
                structure_size_3d=structure_size_3d,
                iterations_3d=iterations_3d
            )
            
            # Save the brain mask if requested
            if save_intermediate:
                brain_mask_path = others_dir / f"{pulse_name}_{patient_id.replace('P', '')}_brain_mask.nii.gz"
                sitk.WriteImage(brain_mask, str(brain_mask_path))
                
                # Update the processed data with the new file path
                processed_pulse["brain_mask_path"] = str(brain_mask_path)
                
                if verbose:
                    logger.info(f"[RM / BRAIN MASK] Brain mask generated and saved to {brain_mask_path}")
            
        except Exception as e:
            logger.error(f"[RM / BRAIN MASK] Error: {str(e)}")
            # Continue even if brain mask generation fails
    
    # 9. bias_field_correction (if it exists in the plan and we have a brain mask)
    if current_image is not None and "bias_field_correction" in preprocessing_plan and brain_mask is not None:
        if "n4" in preprocessing_plan["bias_field_correction"]:
            if verbose:
                logger.info(f"[RM / BIAS CORRECTION] Applying N4 bias field correction")
            
            try:
                # Get N4 parameters
                n4_params = preprocessing_plan["bias_field_correction"]["n4"]
                shrink_factor = n4_params.get("shrink_factor", 4)
                max_iterations = n4_params.get("max_iterations", 100)
                control_points = n4_params.get("control_points", 6)
                bias_field_fwhm = n4_params.get("bias_field_fwhm", 0.1)
                
                if verbose:
                    logger.info(f"[RM / BIAS CORRECTION] N4 parameters: shrink_factor={shrink_factor}, "
                          f"max_iterations={max_iterations}, control_points={control_points}, "
                          f"bias_field_fwhm={bias_field_fwhm}")
                
                # Apply N4 bias field correction
                corrected_image = n4_bias_field_correction(
                    current_image,
                    mask_sitk=brain_mask,
                    shrink_factor=shrink_factor,
                    max_iterations=max_iterations,
                    bias_field_fwhm=bias_field_fwhm,
                    control_points=control_points,
                    verbose=verbose
                )
                
                # Update the current image with the bias-corrected one
                current_image = corrected_image
                
                # Save the bias-corrected image if requested
                if save_intermediate:
                    bias_corrected_path = others_dir / f"{pulse_name}_{patient_id.replace('P', '')}_bias_corrected.nii.gz"
                    sitk.WriteImage(current_image, str(bias_corrected_path))
                    
                    # Update the processed data with the new file path
                    processed_pulse["bias_corrected_path"] = str(bias_corrected_path)
                    
                    if verbose:
                        logger.info(f"[RM / BIAS CORRECTION] Image bias-corrected and saved to {bias_corrected_path}")
                
            except Exception as e:
                logger.error(f"[RM / BIAS CORRECTION] Error: {str(e)}")
                # Continue even if bias correction fails
        else:
            if verbose:
                logger.info(f"[RM / BIAS CORRECTION] Only N4 bias field correction is supported, skipping")
    
    # 10. brain_extraction (if it exists in the plan)
    if current_image is not None and "brain_extraction" in preprocessing_plan:
        # Check for fsl_bet method
        if "fsl_bet" in preprocessing_plan["brain_extraction"]:
            if verbose:
                logger.info(f"[RM / BRAIN EXTRACTION] Applying FSL BET brain extraction")
            
            try:
                # Get FSL BET parameters
                bet_params = preprocessing_plan["brain_extraction"]["fsl_bet"]
                frac = bet_params.get("frac", 0.5)
                robust = bet_params.get("robust", True)
                vertical_gradient = bet_params.get("vertical_gradient", 0.0)
                skull = bet_params.get("skull", False)
                
                if verbose:
                    logger.info(f"[RM / BRAIN EXTRACTION] FSL BET parameters: frac={frac}, "
                          f"robust={robust}, vertical_gradient={vertical_gradient}, "
                          f"skull={skull}")
                
                # Apply FSL BET brain extraction
                extracted_brain, brain_mask = fsl_bet_brain_extraction(
                    current_image,
                    frac=frac,
                    robust=robust,
                    vertical_gradient=vertical_gradient,
                    skull=skull,
                    verbose=verbose
                )
                
                # Update the current image with the brain-extracted one
                current_image = extracted_brain
                
                # Save the brain-extracted image and mask if requested
                if save_intermediate:
                    brain_extracted_path = others_dir / f"{pulse_name}_{patient_id.replace('P', '')}_brain_extracted.nii.gz"
                    brain_extraction_mask_path = others_dir / f"{pulse_name}_{patient_id.replace('P', '')}_brain_extraction_mask.nii.gz"
                    
                    sitk.WriteImage(current_image, str(brain_extracted_path))
                    sitk.WriteImage(brain_mask, str(brain_extraction_mask_path))
                    
                    # Update the processed data with the new file paths
                    processed_pulse["brain_extracted_path"] = str(brain_extracted_path)
                    processed_pulse["brain_extraction_mask_path"] = str(brain_extraction_mask_path)
                    
                    if verbose:
                        logger.info(f"[RM / BRAIN EXTRACTION] Brain extracted and saved to {brain_extracted_path}")
                        logger.info(f"[RM / BRAIN EXTRACTION] Brain extraction mask saved to {brain_extraction_mask_path}")
                
            except Exception as e:
                logger.error(f"[RM / BRAIN EXTRACTION] Error: {str(e)}")
                # Continue even if brain extraction fails
        else:
            if verbose:
                logger.info(f"[RM / BRAIN EXTRACTION] Only FSL BET brain extraction is supported, skipping")
    
    # 11. registration (if it exists in the plan)
    if current_image is not None and "registration" in preprocessing_plan:
        if "sri24" in preprocessing_plan["registration"]:
            sri24_config = preprocessing_plan["registration"]["sri24"]
            
            if sri24_config.get("enable", False):
                if verbose:
                    logger.info(f"[RM / REGISTRATION] Checking if we need T1->SRI24 or T2/SUSC->T1->SRI24")
                
                try:
                    # CHANGES HERE -- We see if we are T1 or T2/SUSC
                    if pulse_name.upper() == "T1":
                        # =======================
                        # (A) T1 -> SRI24
                        # =======================
                        if verbose:
                            logger.info(f"[RM / REGISTRATION] Pulse is T1, registering directly to SRI24.")
                        
                        config_path = sri24_config.get("config_path", None)
                        
                        # Create intermediate directories if needed
                        if save_intermediate:
                            registration_output_dir = others_dir / f"{pulse_name}_registration"
                            registration_output_dir.mkdir(exist_ok=True)
                        else:
                            registration_output_dir = patient_output_dir
                        
                        # Register T1 -> SRI24
                        t1_reg_image, t1_reg_mask, t1_transform_params = register_image_to_sri24(
                            moving_image=current_image,
                            moving_mask=current_mask,
                            config_path=config_path,
                            verbose=verbose,
                        )
                        
                        # Update current_image to the atlas-registered T1
                        current_image = t1_reg_image
                        current_mask = t1_reg_mask
                        
                        # Final registered T1 - This is a main output file, not intermediate
                        reg_image_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_registered_sri24.nii.gz"
                        reg_mask_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_mask_registered_sri24.nii.gz"
                        
                        sitk.WriteImage(current_image, str(reg_image_path))
                        sitk.WriteImage(current_mask, str(reg_mask_path))
                        
                        # Store transform params for usage by T2 or SUSC
                        transform_json_path = patient_output_dir / "transform_params_T1.json"
                        with open(transform_json_path, "w") as f:
                            json.dump(t1_transform_params, f, indent=2)
                        
                        # Store the transform params in t1_info for use by other modalities
                        t1_info["t1_transform_params"] = t1_transform_params
                        t1_info["t1_transform_json_path"] = str(transform_json_path)
                        
                        if verbose:
                            logger.info(f"[RM / REGISTRATION] T1->SRI24 done. Transform saved to {transform_json_path}")
                        
                        processed_pulse["registered_image_path"] = str(reg_image_path)
                        processed_pulse["registered_mask_path"] = str(reg_mask_path)
                        processed_pulse["registration_transform_params"] = t1_transform_params
                    else:
                        # =======================
                        # (B) T2/SUSC/else -> T1 in native space -> Atlas
                        # =======================
                        if verbose:
                            logger.info(f"[RM / REGISTRATION] Pulse is {pulse_name}, registering to T1 (native space) and then to atlas")
                        
                        # Check if we have T1 information
                        if "t1_native_path" not in t1_info or "t1_transform_params" not in t1_info:
                            logger.error(f"[RM / REGISTRATION] T1 information not available. Process T1 first.")
                            raise ValueError("T1 must be processed before other modalities for registration")
                        
                        # Create intermediate directories if needed
                        if save_intermediate:
                            secondary_reg_output_dir = others_dir / f"{pulse_name}_to_T1_registration"
                            secondary_reg_output_dir.mkdir(exist_ok=True)
                        else:
                            secondary_reg_output_dir = patient_output_dir
                        
                        # 2) Register secondary modality to T1 native space
                        if verbose:
                            logger.info(f"[RM / REGISTRATION] Registering {pulse_name} to T1 in native space")
                        
                        # Save current images for registration if needed
                        if save_intermediate:
                            moving_path = secondary_reg_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_forReg.nii.gz"
                            moving_mask_path = secondary_reg_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_mask_forReg.nii.gz"
                            sitk.WriteImage(current_image, str(moving_path))
                            sitk.WriteImage(current_mask, str(moving_mask_path))
                        
                        # Load T1 native for registration - we'll use it as the "fixed" image
                        t1_native_image = sitk.ReadImage(t1_info["t1_native_path"])
                        
                        # Register secondary modality to T1
                        secondary_to_t1_image, secondary_to_t1_mask, secondary_transform_params = register_image_to_sri24(
                            moving_image=current_image,
                            moving_mask=current_mask,
                            fixed_image=t1_native_image, 
                            config_path=sri24_config.get("config_path_for_T2_to_T1", sri24_config.get("config_path")),
                            verbose=verbose,
                        )
                        
                        # Save the registered image and mask in T1 space if requested
                        if save_intermediate:
                            secondary_in_t1_path = secondary_reg_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_registered_to_T1.nii.gz"
                            secondary_in_t1_mask_path = secondary_reg_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_mask_registered_to_T1.nii.gz"
                            sitk.WriteImage(secondary_to_t1_image, str(secondary_in_t1_path))
                            sitk.WriteImage(secondary_to_t1_mask, str(secondary_in_t1_mask_path))
                            
                            # Save transform parameters for secondary->T1
                            secondary_transform_json_path = secondary_reg_output_dir / f"transform_params_{pulse_name}_to_T1.json"
                            with open(secondary_transform_json_path, "w") as f:
                                json.dump(secondary_transform_params, f, indent=2)
                            
                            if verbose:
                                logger.info(f"[RM / REGISTRATION] {pulse_name}->T1 registration complete")
                                logger.info(f"[RM / REGISTRATION] Transform saved to {secondary_transform_json_path}")
                        
                        # 4) Compose transforms secondary->T1->Atlas using the new function
                        if verbose:
                            logger.info(f"[RM / REGISTRATION] Composing transforms: {pulse_name}->T1, then T1->Atlas => {pulse_name}->Atlas")
                        
                        # Define output paths for the final registered images
                        # These are main output files, not intermediates
                        final_atlas_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_registered_sri24.nii.gz"
                        final_atlas_mask_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_mask_registered_sri24.nii.gz"
                        
                        # Apply composed transform to the image
                        final_reg_img = apply_composed_transforms(
                            input_image=current_image,
                            t1_to_atlas_transform_params=t1_info["t1_transform_params"],
                            secondary_to_t1_transform_params=secondary_transform_params,
                            output_path=str(final_atlas_path),
                            interpolation="Linear",  # for intensity images
                            verbose=verbose
                        )
                        
                        # Apply composed transform to the mask using NearestNeighbor interpolation
                        final_reg_mask = apply_composed_transforms(
                            input_image=current_mask,
                            t1_to_atlas_transform_params=t1_info["t1_transform_params"],
                            secondary_to_t1_transform_params=secondary_transform_params,
                            output_path=str(final_atlas_mask_path),
                            interpolation="NearestNeighbor",  # for mask/label images
                            verbose=verbose
                        )
                        
                        # Update the pipeline with the final images
                        current_image = final_reg_img
                        current_mask = final_reg_mask
                        
                        # Update the processed pulse data
                        processed_pulse["registered_image_path"] = str(final_atlas_path)
                        processed_pulse["registered_mask_path"] = str(final_atlas_mask_path)
                        
                        # Also store transform params for reference
                        processed_pulse["t1_to_atlas_transform_params"] = t1_info["t1_transform_params"]
                        processed_pulse["secondary_to_t1_transform_params"] = secondary_transform_params
                        
                        if verbose:
                            logger.info(f"[RM / REGISTRATION] Done: {pulse_name} is now registered to the SRI24 space")
                            logger.info(f"[RM / REGISTRATION] Registered image saved to: {final_atlas_path}")
                            logger.info(f"[RM / REGISTRATION] Registered mask saved to: {final_atlas_mask_path}")
                
                except Exception as e:
                    logger.error(f"[RM / REGISTRATION] Error during registration: {str(e)}")

    return processed_pulse


def apply_preprocessing_steps(
    plan: Dict[str, Any], 
    patient_data: Dict[str, Dict[str, Dict[str, Any]]], 
    output_dir: Path,
    verbose: bool = True,
    save_intermediate: bool = False
) -> None:
    """
    Apply preprocessing steps to patient data according to the preprocessing plan.
    
    The function processes each patient's pulse sequences, applying the appropriate
    preprocessing steps based on the modality (RM or TC). It delegates to specific
    pipeline functions for each modality.
    
    Args:
        plan: The preprocessing plan dictionary containing steps for each modality
        patient_data: Dictionary of patient data containing pulse sequences and their metadata
        output_dir: Base output directory for storing preprocessed files
        verbose: Whether to print detailed processing information
        save_intermediate: If True, save intermediate processing files to 'others' directory
    
    Returns:
        Dictionary containing the updated patient data with processed file paths
    """
    # Get preprocessing steps for RM and TC
    rm_preprocessing = plan.get("preprocessing_plan", {}).get("RM", {})
    tc_preprocessing = plan.get("preprocessing_plan", {}).get("TC", {})
    
    # Iterate through patients and their pulses
    for patient_id, pulses in patient_data.items():
        if verbose:
            logger.info(f"\n[PATIENT] Processing {patient_id}")
        
        patient_output_dir = output_dir / patient_id
        
        # Dictionary to store T1 information for this patient
        t1_info: Dict[Any, Any] = {}
        
        for pulse_name, pulse_data in pulses.items():
            modality = pulse_data.get("modality", "")
            
            if verbose:
                logger.info(f"\n[{modality} / {pulse_name}] Starting preprocessing")
            
            # Process based on modality
            if modality == "RM":
                # Apply RM preprocessing pipeline
                processed_pulse = rm_pipeline(
                    pulse_data,
                    rm_preprocessing,
                    patient_output_dir,
                    patient_id,
                    pulse_name,
                    t1_info,  # Pass T1 info dictionary
                    verbose,
                    save_intermediate
                )
                
            elif modality == "TC":
                # In the future, we would call a tc_pipeline function here
                # For now, just add the original pulse data
                
                if verbose:
                    logger.info(f"[TC / {pulse_name}] TC processing not yet implemented")
            
            else:
                if verbose:
                    logger.info(f"[UNKNOWN / {pulse_name}] Unknown modality: {modality}")


def main():
    """Main function to process the preprocessing plan."""
    parser = argparse.ArgumentParser(description='Process meningioma preprocessing plan')
    parser.add_argument('--plan', 
                        default="/home/mariopasc/Python/Datasets/Meningiomas/jsons/plan_meningioma.json", 
                        type=str, help='Path to preprocessing plan JSON')
    parser.add_argument('--patients', 
                        default="1",
                        type=str,  
                        help='Comma-separated list of patient IDs (e.g., "1,2,3")')
    parser.add_argument('--output', 
                        default="/home/mariopasc/Python/Datasets/Meningiomas",
                        type=str,  
                        help='Output directory for preprocessed files')
    parser.add_argument('--verbose', 
                        action='store_true',
                        help='Print verbose output')
    parser.add_argument('--save-intermediate', 
                        action='store_true',
                        help='Save intermediate processing files')
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
        
    logger.info(f"Starting meningioma preprocessing with plan: {args.plan}")
    
    # Convert patient IDs to list of integers
    patient_ids = [int(p.strip()) for p in args.patients.split(',')]
    
    # Load preprocessing plan
    plan = load_plan(args.plan)
    
    # Extract data for specified patients with T1 ordered first
    patient_data = extract_patient_data(plan, patient_ids)
    
    # Create output directories
    output_base_dir = create_output_directories(args.output, patient_ids)
    logger.info(f"\nCreated output directories in: {output_base_dir}")
    
    # Apply preprocessing steps
    apply_preprocessing_steps(
        plan, 
        patient_data, 
        output_base_dir, 
        args.verbose, 
        args.save_intermediate
    )
    
    logger.info("Preprocessing completed successfully")


if __name__ == "__main__":
    main()