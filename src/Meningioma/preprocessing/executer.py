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
from Meningioma.preprocessing.tools.reorient import reorient_images
from Meningioma.preprocessing.tools.resample import resample_images
from Meningioma.preprocessing.tools.denoise_susan import denoise_susan
from Meningioma.preprocessing.tools.bias_field_corr_n4 import generate_brain_mask_sitk, n4_bias_field_correction
from Meningioma.preprocessing.tools.skull_stripping.fsl_bet import fsl_bet_brain_extraction
from Meningioma.preprocessing.tools.registration.ants_sri24_reg import register_image_to_sri24




def load_plan(json_path):
    """Load preprocessing plan from JSON file."""
    with open(json_path, 'r') as f:
        plan = json.load(f)
    return plan


def extract_patient_data(plan, patient_ids):
    """Extract volume and mask paths for each patient's pulse sequences."""
    results = {}
    
    # Iterate through requested patients
    for patient_id in patient_ids:
        patient_key = f"P{patient_id}"
        
        # Check if patient exists in the plan
        if patient_key not in plan["data"]:
            print(f"Warning: Patient {patient_id} not found in the plan")
            continue
        
        patient_data = plan["data"][patient_key]
        pulses_data = {}
        
        # Iterate through each pulse sequence (SUSC, T1, T2, TC)
        for pulse_name, pulse_info in patient_data["pulses"].items():
            # Skip if there's an error with this pulse
            if pulse_info.get("error", True):
                print(f"Skipping {patient_key} {pulse_name} due to error flag")
                continue
            
            # Extract volume and mask paths
            volume_path = pulse_info.get("volume", {}).get("route", "")
            mask_path = pulse_info.get("segmentation", {}).get("route", "")
            
            # Only include if we have valid paths
            if volume_path and mask_path:
                pulses_data[pulse_name] = {
                    "volume_path": volume_path,
                    "mask_path": mask_path,
                    "modality": pulse_info.get("modality", ""),
                }
                
                # Verify files exist
                if not os.path.exists(volume_path):
                    print(f"Warning: Volume file not found: {volume_path}")
                if not os.path.exists(mask_path):
                    print(f"Warning: Mask file not found: {mask_path}")
        
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
    verbose: bool = True
) -> None:
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
    
    Args:
        pulse_data: Dictionary containing pulse data and metadata
        preprocessing_plan: Dictionary with RM preprocessing configuration
        patient_output_dir: Output directory for the patient's processed files
        patient_id: Patient identifier (e.g., 'P1')
        pulse_name: Name of the pulse sequence (e.g., 'T1', 'T2', 'SUSC')
        verbose: Whether to print detailed processing information
    
    Returns:
        Tuple containing:
            - The processed pulse data with updated file paths
            - A boolean indicating success (True) or failure (False)
    """
    # Create a working dictionary for this pulse
    processed_pulse = pulse_data.copy()
    current_image = None
    current_header = None
    current_mask = None
    
    if verbose:
        print(f"\n[RM / {pulse_name}] Starting RM preprocessing pipeline")
    
    # 1. remove_channel (if it exists in the plan)
    if "remove_channel" in preprocessing_plan:
        if verbose:
            print(f"[RM / CHANNEL REMOVAL] Processing {Path(pulse_data['volume_path']).name}")
        
        try:
            channel = preprocessing_plan["remove_channel"].get("channel", 0)
            if verbose:
                print(f"[RM / CHANNEL REMOVAL] Extracting channel {channel}")
            
            # Apply remove_channel function to the volume
            current_image, current_header = remove_first_channel(
                pulse_data["volume_path"], 
                channel, 
                verbose=verbose
            )
            
            if verbose:
                print(f"[RM / CHANNEL REMOVAL] Completed: size={current_image.GetSize()}, "
                      f"dimensions={current_image.GetDimension()}, "
                      f"spacing={current_image.GetSpacing()}")
            
        except Exception as e:
            print(f"[RM / CHANNEL REMOVAL] Error: {str(e)}")
            
    
    # 2. export_nifti (if it exists in the plan)
    if "export_nifti" in preprocessing_plan and preprocessing_plan["export_nifti"]:
        if verbose:
            print(f"[RM / NIFTI EXPORT] Converting to NIfTI format")
        
        try:
            # Output path for the NIfTI file
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
            
            if verbose:
                print(f"[RM / NIFTI EXPORT] Saved to {output_path}")
            
            # Load the exported NIfTI for further processing
            current_image = sitk.ReadImage(output_path)
            
        except Exception as e:
            print(f"[RM / NIFTI EXPORT] Error: {str(e)}")
            # Continue even if NIfTI export fails
    
    # 3. Load the segmentation mask if we have a current_image
    if current_image is not None:
        try:
            if verbose:
                print(f"[RM / MASK LOADING] Loading segmentation mask")
            
            current_mask = sitk.ReadImage(pulse_data["mask_path"])
            
            if verbose:
                print(f"[RM / MASK LOADING] Mask loaded: size={current_mask.GetSize()}, "
                      f"dimensions={current_mask.GetDimension()}")
                
        except Exception as e:
            print(f"[RM / MASK LOADING] Error loading mask: {str(e)}")
            # Continue processing even if mask loading fails
    
    # 4. cast_volume (if it exists in the plan)
    if current_image is not None and current_mask is not None and "cast_volume" in preprocessing_plan and preprocessing_plan["cast_volume"]:
        if verbose:
            print(f"[RM / CAST VOLUME] Casting volume to float32 and mask to uint8")
        
        try:
            # Cast volume to float32 and mask to uint8
            current_image, current_mask = cast_volume_and_mask(
                current_image, 
                current_mask
            )
            
            # Save the cast volume and mask
            cast_volume_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_cast.nii.gz"
            cast_mask_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_mask_cast.nii.gz"
            
            sitk.WriteImage(current_image, str(cast_volume_path))
            sitk.WriteImage(current_mask, str(cast_mask_path))
            
            # Update the processed data with the new file paths
            processed_pulse["cast_volume_path"] = str(cast_volume_path)
            processed_pulse["cast_mask_path"] = str(cast_mask_path)
            
            if verbose:
                print(f"[RM / CAST VOLUME] Volume cast to {current_image.GetPixelID()} and saved to {cast_volume_path}")
                print(f"[RM / CAST VOLUME] Mask cast to {current_mask.GetPixelID()} and saved to {cast_mask_path}")
            
        except Exception as e:
            print(f"[RM / CAST VOLUME] Error: {str(e)}")
            # Continue even if casting fails
    
    # 5. reorientation (if it exists in the plan)
    if current_image is not None and current_mask is not None and "reorientation" in preprocessing_plan:
        if verbose:
            print(f"[RM / REORIENTATION] Reorienting volume and mask")
        
        try:
            # Get the desired orientation
            orientation = preprocessing_plan["reorientation"].get("orientation", "LPS")
            
            if verbose:
                print(f"[RM / REORIENTATION] Target orientation: {orientation}")
            
            # Reorient the volume and mask
            current_image, current_mask = reorient_images(
                current_image, 
                current_mask, 
                orientation
            )
            
            # Save the reoriented volume and mask
            reorient_volume_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_reoriented.nii.gz"
            reorient_mask_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_mask_reoriented.nii.gz"
            
            sitk.WriteImage(current_image, str(reorient_volume_path))
            sitk.WriteImage(current_mask, str(reorient_mask_path))
            
            # Update the processed data with the new file paths
            processed_pulse["reoriented_volume_path"] = str(reorient_volume_path)
            processed_pulse["reoriented_mask_path"] = str(reorient_mask_path)
            
            if verbose:
                print(f"[RM / REORIENTATION] Volume reoriented to {orientation} and saved to {reorient_volume_path}")
                print(f"[RM / REORIENTATION] Mask reoriented to {orientation} and saved to {reorient_mask_path}")
            
        except Exception as e:
            print(f"[RM / REORIENTATION] Error: {str(e)}")
            # Continue even if reorientation fails
    
    # 6. resample (if it exists in the plan)
    if current_image is not None and current_mask is not None and "resample" in preprocessing_plan:
        if verbose:
            print(f"[RM / RESAMPLE] Resampling volume and mask")
            
        try:
            # Get the desired spacing
            spacing = tuple(preprocessing_plan["resample"].get("spacing", (1.0, 1.0, 1.0)))
            
            if verbose:
                print(f"[RM / RESAMPLE] Target spacing: {spacing}")
            
            # Resample the volume and mask
            current_image, current_mask = resample_images(
                current_image,
                current_mask,
                new_spacing=spacing
            )
            
            # Save the resampled volume and mask
            resample_volume_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_resampled.nii.gz"
            resample_mask_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_mask_resampled.nii.gz"
            
            sitk.WriteImage(current_image, str(resample_volume_path))
            sitk.WriteImage(current_mask, str(resample_mask_path))
            
            # Update the processed data with the new file paths
            processed_pulse["resampled_volume_path"] = str(resample_volume_path)
            processed_pulse["resampled_mask_path"] = str(resample_mask_path)
            
            if verbose:
                print(f"[RM / RESAMPLE] Volume resampled to spacing {current_image.GetSpacing()} and saved to {resample_volume_path}")
                print(f"[RM / RESAMPLE] Mask resampled and saved to {resample_mask_path}")
            
        except Exception as e:
            print(f"[RM / RESAMPLE] Error: {str(e)}")
            # Continue even if resampling fails
    
    # 7. denoise (if it exists and is enabled in the plan)
    if current_image is not None and "denoise" in preprocessing_plan:
        # Check if denoising is enabled
        denoise_enabled = preprocessing_plan["denoise"].get("enable", False)
        
        if denoise_enabled:
            if verbose:
                print(f"[RM / DENOISE] Applying SUSAN denoising to volume")
            
            try:
                # Get denoise parameters
                susan_params = preprocessing_plan["denoise"].get("susan", {})
                brightness_threshold = susan_params.get("brightness_threshold", 0.001)
                fwhm = susan_params.get("fwhm", 0.5)
                dimension = susan_params.get("dimension", 3)
                
                if verbose:
                    print(f"[RM / DENOISE] SUSAN parameters: brightness_threshold={brightness_threshold}, fwhm={fwhm}, dimension={dimension}")
                
                # Apply denoising to the volume
                current_image = denoise_susan(
                    current_image,
                    brightness_threshold=brightness_threshold,
                    fwhm=fwhm,
                    dimension=dimension,
                    mask_sitk=current_mask,  # Optional: Use mask to constrain denoising
                    verbose=verbose
                )
                
                # Save the denoised volume
                denoised_volume_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_denoised.nii.gz"
                sitk.WriteImage(current_image, str(denoised_volume_path))
                
                # Update the processed data with the new file path
                processed_pulse["denoised_volume_path"] = str(denoised_volume_path)
                
                if verbose:
                    print(f"[RM / DENOISE] Volume denoised and saved to {denoised_volume_path}")
                
            except Exception as e:
                print(f"[RM / DENOISE] Error: {str(e)}")
                # Continue even if denoising fails
        else:
            if verbose:
                print(f"[RM / DENOISE] Denoising is disabled in the preprocessing plan, skipping")
    
    # 8. brain_mask (if it exists in the plan)
    if current_image is not None and "brain_mask" in preprocessing_plan:
        if verbose:
            print(f"[RM / BRAIN MASK] Generating brain mask")
        
        try:
            # Get brain mask parameters
            brain_mask_params = preprocessing_plan["brain_mask"]
            threshold_method = brain_mask_params.get("threshold_method", "li")
            structure_size_2d = brain_mask_params.get("structure_size_2d", 7)
            iterations_2d = brain_mask_params.get("iterations_2d", 3)
            structure_size_3d = brain_mask_params.get("structure_size_3d", 3)
            iterations_3d = brain_mask_params.get("iterations_3d", 1)
            
            if verbose:
                print(f"[RM / BRAIN MASK] Parameters: threshold_method={threshold_method}, "
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
            
            # Save the brain mask
            brain_mask_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_brain_mask.nii.gz"
            sitk.WriteImage(brain_mask, str(brain_mask_path))
            
            # Update the processed data with the new file path
            processed_pulse["brain_mask_path"] = str(brain_mask_path)
            
            if verbose:
                print(f"[RM / BRAIN MASK] Brain mask generated and saved to {brain_mask_path}")
            
        except Exception as e:
            print(f"[RM / BRAIN MASK] Error: {str(e)}")
            # Continue even if brain mask generation fails
    
    # 9. bias_field_correction (if it exists in the plan and we have a brain mask)
    if current_image is not None and "bias_field_correction" in preprocessing_plan and brain_mask is not None:
        if "n4" in preprocessing_plan["bias_field_correction"]:
            if verbose:
                print(f"[RM / BIAS CORRECTION] Applying N4 bias field correction")
            
            try:
                # Get N4 parameters
                n4_params = preprocessing_plan["bias_field_correction"]["n4"]
                shrink_factor = n4_params.get("shrink_factor", 4)
                max_iterations = n4_params.get("max_iterations", 100)
                control_points = n4_params.get("control_points", 6)
                bias_field_fwhm = n4_params.get("bias_field_fwhm", 0.1)
                
                if verbose:
                    print(f"[RM / BIAS CORRECTION] N4 parameters: shrink_factor={shrink_factor}, "
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
                
                # Save the bias-corrected image
                bias_corrected_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_bias_corrected.nii.gz"
                sitk.WriteImage(current_image, str(bias_corrected_path))
                
                # Update the processed data with the new file path
                processed_pulse["bias_corrected_path"] = str(bias_corrected_path)
                
                if verbose:
                    print(f"[RM / BIAS CORRECTION] Image bias-corrected and saved to {bias_corrected_path}")
                
            except Exception as e:
                print(f"[RM / BIAS CORRECTION] Error: {str(e)}")
                # Continue even if bias correction fails
        else:
            if verbose:
                print(f"[RM / BIAS CORRECTION] Only N4 bias field correction is supported, skipping")
    
    # 10. brain_extraction (if it exists in the plan)
    if current_image is not None and "brain_extraction" in preprocessing_plan:
        # Check for fsl_bet method
        if "fsl_bet" in preprocessing_plan["brain_extraction"]:
            if verbose:
                print(f"[RM / BRAIN EXTRACTION] Applying FSL BET brain extraction")
            
            try:
                # Get FSL BET parameters
                bet_params = preprocessing_plan["brain_extraction"]["fsl_bet"]
                frac = bet_params.get("frac", 0.5)
                robust = bet_params.get("robust", True)
                vertical_gradient = bet_params.get("vertical_gradient", 0.0)
                skull = bet_params.get("skull", False)
                
                if verbose:
                    print(f"[RM / BRAIN EXTRACTION] FSL BET parameters: frac={frac}, "
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
                
                # Save the brain-extracted image and mask
                brain_extracted_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_brain_extracted.nii.gz"
                brain_extraction_mask_path = patient_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_brain_extraction_mask.nii.gz"
                
                sitk.WriteImage(current_image, str(brain_extracted_path))
                sitk.WriteImage(brain_mask, str(brain_extraction_mask_path))
                
                # Update the processed data with the new file paths
                processed_pulse["brain_extracted_path"] = str(brain_extracted_path)
                processed_pulse["brain_extraction_mask_path"] = str(brain_extraction_mask_path)
                
                if verbose:
                    print(f"[RM / BRAIN EXTRACTION] Brain extracted and saved to {brain_extracted_path}")
                    print(f"[RM / BRAIN EXTRACTION] Brain extraction mask saved to {brain_extraction_mask_path}")
                
            except Exception as e:
                print(f"[RM / BRAIN EXTRACTION] Error: {str(e)}")
                # Continue even if brain extraction fails
        else:
            if verbose:
                print(f"[RM / BRAIN EXTRACTION] Only FSL BET brain extraction is supported, skipping")
    
    # 11. registration (if it exists in the plan)
    if current_image is not None and "registration" in preprocessing_plan:
        # Check for sri24 registration
        if "sri24" in preprocessing_plan["registration"]:
            sri24_config = preprocessing_plan["registration"]["sri24"]
            
            # Check if registration is enabled
            if sri24_config.get("enable", False):
                if verbose:
                    print(f"[RM / REGISTRATION] Applying SRI24 atlas registration")
                
                try:
                    # Get configuration file path
                    config_path = sri24_config.get("config_path")
                    
                    if not config_path or not os.path.exists(config_path):
                        print(f"[RM / REGISTRATION] Warning: Config file not found at {config_path}")
                        print(f"[RM / REGISTRATION] Using default registration parameters")
                    
                    # Create registration output subdirectory
                    registration_output_dir = patient_output_dir / f"{pulse_name}_registration"
                    registration_output_dir.mkdir(exist_ok=True)
                    
                    # Prepare output paths
                    reg_image_path = registration_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_registered.nii.gz"
                    reg_mask_path = registration_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_mask_registered.nii.gz"
                    
                    if verbose:
                        print(f"[RM / REGISTRATION] Config path: {config_path}")
                        print(f"[RM / REGISTRATION] Registration output directory: {registration_output_dir}")
                        print(f"[RM / REGISTRATION] Output registered image: {reg_image_path}")
                        print(f"[RM / REGISTRATION] Output registered mask: {reg_mask_path}")
                    
                    # Save current image and mask to temporary files for registration
                    temp_img_path = registration_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_for_registration.nii.gz"
                    temp_mask_path = registration_output_dir / f"{pulse_name}_{patient_id.replace('P', '')}_mask_for_registration.nii.gz"
                    
                    sitk.WriteImage(current_image, str(temp_img_path))
                    sitk.WriteImage(current_mask, str(temp_mask_path)) # type: ignore
                    
                    # Call registration function with mask
                    registered_image, registered_mask, transform_params = register_image_to_sri24( #type: ignore
                        moving_image=current_image,
                        moving_mask=current_mask,
                        config_path=config_path,
                        verbose=verbose
                    )

                    # Update current image and mask
                    current_image = registered_image
                    current_mask = registered_mask
                    
                    # Update the processed data with the new file paths
                    processed_pulse["registered_image_path"] = str(reg_image_path)
                    processed_pulse["registered_mask_path"] = str(reg_mask_path)
                    processed_pulse["registration_transform_params"] = transform_params
                    
                    if verbose:
                        print(f"[RM / REGISTRATION] Volume registered to SRI24 atlas and saved to {reg_image_path}")
                        print(f"[RM / REGISTRATION] Mask registered to SRI24 atlas and saved to {reg_mask_path}")
                        print(f"[RM / REGISTRATION] Transform parameters saved to {registration_output_dir / 'transform_params.json'}")
                
                except Exception as e:
                    print(f"[RM / REGISTRATION] Error: {str(e)}")
                    # Continue even if registration fails
            else:
                if verbose:
                    print(f"[RM / REGISTRATION] SRI24 registration is disabled in the preprocessing plan, skipping")
        else:
            if verbose:
                print(f"[RM / REGISTRATION] Only SRI24 atlas registration is supported, skipping")
    
    # 12. transpose (if it exists in the plan)
    if current_image is not None and current_mask is not None and "transpose" in preprocessing_plan:
        if verbose:
            print(f"[RM / TRANSPOSE] Applying transpose to volume and mask")
        
        try:
            # Get the axis order for transposing
            axis_order = preprocessing_plan["transpose"].get("axis", [2, 1, 0])
            
            if verbose:
                print(f"[RM / TRANSPOSE] Transposing with axis order: {axis_order}")
            
            # Convert SimpleITK images to numpy arrays for transposing
            volume_array = sitk.GetArrayFromImage(current_image)
            mask_array = sitk.GetArrayFromImage(current_mask)
            
            # Transpose the arrays
            transposed_volume = np.transpose(volume_array, axis_order)
            transposed_mask = np.transpose(mask_array, axis_order)
            
            # Create new SimpleITK images from the transposed arrays
            # Note: We need to handle spacing and direction appropriately
            original_spacing = current_image.GetSpacing()
            original_direction = current_image.GetDirection()
            
            # Reorder spacing according to the transpose axis
            new_spacing = [original_spacing[i] for i in axis_order]
            
            # Create new SimpleITK images
            transposed_image = sitk.GetImageFromArray(transposed_volume)
            transposed_image.SetSpacing(new_spacing)
            
            transposed_mask_img = sitk.GetImageFromArray(transposed_mask)
            transposed_mask_img.SetSpacing(new_spacing)
            
            # Save the final processed volume and mask with the standard naming convention
            final_volume_path = patient_output_dir / f"P{patient_id.replace('P', '')}_{pulse_name}.nii.gz"
            final_mask_path = patient_output_dir / f"P{patient_id.replace('P', '')}_{pulse_name}_seg.nii.gz"
            
            sitk.WriteImage(transposed_image, str(final_volume_path))
            sitk.WriteImage(transposed_mask_img, str(final_mask_path))
            
            # Update the processed data with the final file paths
            processed_pulse["final_volume_path"] = str(final_volume_path)
            processed_pulse["final_mask_path"] = str(final_mask_path)
            
            if verbose:
                print(f"[RM / TRANSPOSE] Final processed volume saved to {final_volume_path}")
                print(f"[RM / TRANSPOSE] Final processed mask saved to {final_mask_path}")
            
        except Exception as e:
            print(f"[RM / TRANSPOSE] Error: {str(e)}")
            # Continue even if transpose fails, but use the current images as final outputs
            
            # Save the current images as final outputs with the standard naming convention
            final_volume_path = patient_output_dir / f"P{patient_id.replace('P', '')}_{pulse_name}.nii.gz"
            final_mask_path = patient_output_dir / f"P{patient_id.replace('P', '')}_{pulse_name}_seg.nii.gz"
            
            sitk.WriteImage(current_image, str(final_volume_path))
            sitk.WriteImage(current_mask, str(final_mask_path))
            
            processed_pulse["final_volume_path"] = str(final_volume_path)
            processed_pulse["final_mask_path"] = str(final_mask_path)
            
            if verbose:
                print(f"[RM / TRANSPOSE] Error during transposing, saving untransposed images instead")
                print(f"[RM / TRANSPOSE] Final processed volume saved to {final_volume_path}")
                print(f"[RM / TRANSPOSE] Final processed mask saved to {final_mask_path}")
    elif current_image is not None and current_mask is not None:
        # Even if no transpose step is specified, save the final outputs with standard naming
        final_volume_path = patient_output_dir / f"P{patient_id.replace('P', '')}_{pulse_name}.nii.gz"
        final_mask_path = patient_output_dir / f"P{patient_id.replace('P', '')}_{pulse_name}_seg.nii.gz"
        
        sitk.WriteImage(current_image, str(final_volume_path))
        sitk.WriteImage(current_mask, str(final_mask_path))
        
        processed_pulse["final_volume_path"] = str(final_volume_path)
        processed_pulse["final_mask_path"] = str(final_mask_path)
        
        if verbose:
            print(f"[RM / FINAL] Final processed volume saved to {final_volume_path}")
            print(f"[RM / FINAL] Final processed mask saved to {final_mask_path}")


def apply_preprocessing_steps(
    plan: Dict[str, Any], 
    patient_data: Dict[str, Dict[str, Dict[str, Any]]], 
    output_dir: Path,
    verbose: bool = True
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
    
    Returns:
        Dictionary containing the updated patient data with processed file paths
    """
    # Get preprocessing steps for RM and TC
    rm_preprocessing = plan.get("preprocessing_plan", {}).get("RM", {})
    tc_preprocessing = plan.get("preprocessing_plan", {}).get("TC", {})
    
    
    # Iterate through patients and their pulses
    for patient_id, pulses in patient_data.items():
        if verbose:
            print(f"\n[PATIENT] Processing {patient_id}")
        
        patient_output_dir = output_dir / patient_id
        
        for pulse_name, pulse_data in pulses.items():
            modality = pulse_data.get("modality", "")
            
            if verbose:
                print(f"\n[{modality} / {pulse_name}] Starting preprocessing")
            
            # Process based on modality
            if modality == "RM":
                # Apply RM preprocessing pipeline
                rm_pipeline(
                    pulse_data,
                    rm_preprocessing,
                    patient_output_dir,
                    patient_id,
                    pulse_name,
                    verbose
                )
                
                
            elif modality == "TC":
                # In the future, we would call a tc_pipeline function here
                # For now, just add the original pulse data
                
                if verbose:
                    print(f"[TC / {pulse_name}] TC processing not yet implemented")
            
            else:
                if verbose:
                    print(f"[UNKNOWN / {pulse_name}] Unknown modality: {modality}")


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
    args = parser.parse_args()
    
    # Convert patient IDs to list of integers
    patient_ids = [int(p.strip()) for p in args.patients.split(',')]
    
    # Load preprocessing plan
    plan = load_plan(args.plan)
    
    # Extract data for specified patients
    patient_data = extract_patient_data(plan, patient_ids)
    
    # Create output directories
    output_base_dir = create_output_directories(args.output, patient_ids)
    print(f"\nCreated output directories in: {output_base_dir}")
    
    # Apply preprocessing steps
    apply_preprocessing_steps(plan, patient_data, output_base_dir)


if __name__ == "__main__":
    main()