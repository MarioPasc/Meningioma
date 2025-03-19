from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import SimpleITK as sitk
import os
import json
import argparse
import collections

from Meningioma.preprocessing.tools.remove_extra_channels import remove_first_channel
from Meningioma.preprocessing.tools.nrrd_to_nifti import nifti_write_3d
from Meningioma.preprocessing.tools.casting import cast_volume_and_mask
from Meningioma.preprocessing.tools.reorient import reorient_images



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
) -> Tuple[Dict[str, Any], bool]:
    """
    Apply RM-specific preprocessing steps to a pulse sequence.
    
    Implements the full RM preprocessing pipeline with steps including:
    - Channel removal
    - NIfTI export
    - Volume casting
    - Image reorientation
    
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
            return processed_pulse, False  # Return with failure
    
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
                volume=input_data,
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
    
    # Return the processed pulse data and success status
    return processed_pulse, True


def apply_preprocessing_steps(
    plan: Dict[str, Any], 
    patient_data: Dict[str, Dict[str, Dict[str, Any]]], 
    output_dir: Path,
    verbose: bool = True
) -> Dict[str, Dict[str, Dict[str, Any]]]:
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
    
    # Dictionary to store processed data
    processed_data = {}
    
    # Iterate through patients and their pulses
    for patient_id, pulses in patient_data.items():
        if verbose:
            print(f"\n[PATIENT] Processing {patient_id}")
        
        patient_output_dir = output_dir / patient_id
        processed_pulses = {}
        
        for pulse_name, pulse_data in pulses.items():
            modality = pulse_data.get("modality", "")
            
            if verbose:
                print(f"\n[{modality} / {pulse_name}] Starting preprocessing")
            
            # Process based on modality
            if modality == "RM":
                # Apply RM preprocessing pipeline
                processed_pulse, success = rm_pipeline(
                    pulse_data,
                    rm_preprocessing,
                    patient_output_dir,
                    patient_id,
                    pulse_name,
                    verbose
                )
                
                # Only add to processed pulses if successful
                if success:
                    processed_pulses[pulse_name] = processed_pulse
                
            elif modality == "TC":
                # In the future, we would call a tc_pipeline function here
                # For now, just add the original pulse data
                processed_pulses[pulse_name] = pulse_data.copy()
                
                if verbose:
                    print(f"[TC / {pulse_name}] TC processing not yet implemented")
            
            else:
                if verbose:
                    print(f"[UNKNOWN / {pulse_name}] Unknown modality: {modality}")
        
        # Add the patient's processed pulses to the result
        if processed_pulses:
            processed_data[patient_id] = processed_pulses
    
    return processed_data


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