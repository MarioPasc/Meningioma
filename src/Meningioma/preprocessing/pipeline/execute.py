from typing import Dict, Any, List, Union
from pathlib import Path
import os
import json
import argparse
import logging

from Meningioma.preprocessing.pipeline.rm_pipeline import rm_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger('preprocessing')

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
            logger.warning(f"Patient {patient_id} not found in the plan")
            continue
        
        patient_data = plan["data"][patient_key]
        pulses_data = {}
        
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
                pulses_data[pulse_name] = {
                    "volume_path": volume_path,
                    "mask_path": mask_path,
                    "modality": pulse_info.get("modality", ""),
                }
                
                # Verify files exist
                if not os.path.exists(volume_path):
                    logger.warning(f"Volume file not found: {volume_path}")
                if not os.path.exists(mask_path):
                    logger.warning(f"Mask file not found: {mask_path}")
        
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


def preprocess(
    plan: Dict[str, Any], 
    output_dir: Union[str, Path],
    patient_data: Dict[str, Dict[str, Dict[str, Any]]] = None, 
    patient_ids: str = None,
    verbose: bool = True,
    save_intermediate: bool = False
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Apply preprocessing steps to patient data according to the preprocessing plan.
    
    The function processes each patient's pulse sequences, applying the appropriate
    preprocessing steps based on the modality (RM or TC). It delegates to specific
    pipeline functions for each modality. T1 pulse is always processed first.
    
    Args:
        plan: The preprocessing plan dictionary containing steps for each modality
        patient_data: Dictionary of patient data containing pulse sequences and their metadata
        output_dir: Base output directory for storing preprocessed files
        verbose: Whether to print detailed processing information
        save_intermediate: If True, save intermediate processing files to 'others' directory
    
    Returns:
        Dictionary containing the updated patient data with processed file paths
    """

    if patient_data is None and patient_ids is None:
        raise ValueError("Either patient_data or patient_ids must be provided.")

    # Configure logging based on verbosity
    if verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    if type(plan) is str:
        plan = load_plan(plan)

    if patient_ids is not None:
        patient_ids = [int(p.strip(" P")) for p in patient_ids.split(',')]
        patient_data = extract_patient_data(plan, patient_ids)

    if type(output_dir) is str:
        output_base_dir = create_output_directories(output_dir, patient_ids)
        output_dir = Path(output_dir)
        logger.info(f"\nCreated output directories in: {output_base_dir}")

    # Get preprocessing steps for RM and TC
    rm_preprocessing = plan.get("preprocessing_plan", {}).get("RM", {})
    tc_preprocessing = plan.get("preprocessing_plan", {}).get("TC", {})
    
    # Dictionary to store updated patient data
    updated_patient_data = {}
    
    # Iterate through patients and their pulses
    for patient_id, pulses in patient_data.items():
        if verbose:
            logger.info(f"\n[PATIENT] Processing {patient_id}")
        
        patient_output_dir = output_dir / patient_id
        
        # Check if this patient has a T1 pulse
        has_t1 = False
        for pulse_name in pulses.keys():
            if pulse_name == "T1":
                has_t1 = True
                break
        
        # Skip the entire patient if no T1 pulse is found
        if not has_t1:
            logger.warning(f"[PATIENT] {patient_id} - No T1 pulse found. Skipping this patient.")
            continue
        
        # Store pulses for this patient in updated data
        updated_patient_data[patient_id] = {}
        
        # Create an ordered dictionary of pulses with T1 first, then all other pulses
        ordered_pulses = {}
        if "T1" in pulses:
            ordered_pulses["T1"] = pulses["T1"]
            
        # Add all other pulses in their original order
        for pulse_name, pulse_data in pulses.items():
            if pulse_name != "T1":
                ordered_pulses[pulse_name] = pulse_data
        
        # Track T1 brain mask path to use for other pulses
        t1_brain_mask_path = None
        
        # Process all pulses in the new order (T1 will be first)
        for pulse_name, pulse_data in ordered_pulses.items():
            modality = pulse_data.get("modality", "")
            
            if pulse_name == "T1" and verbose:
                logger.info(f"\n[{modality} / {pulse_name}] Starting preprocessing (T1 priority)")
            elif verbose:
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
                    verbose,
                    t1_brain_mask_path
                )
                updated_patient_data[patient_id][pulse_name] = processed_pulse
                
                # If this was T1, store the brain mask path for subsequent pulses
                if pulse_name == "T1":
                    # Get brain extraction mask path from T1 processing
                    t1_brain_mask_path = processed_pulse.get("brain_extraction_mask_path")
                    if verbose and t1_brain_mask_path:
                        logger.info(f"[PATIENT] {patient_id} - T1 brain mask will be used for other pulses: {t1_brain_mask_path}")
                
            elif modality == "TC":
                # In the future, we would call a tc_pipeline function here
                if verbose:
                    logger.info(f"[TC / {pulse_name}] TC processing not yet implemented")
                updated_patient_data[patient_id][pulse_name] = pulse_data
            
            else:
                if verbose:
                    logger.info(f"[UNKNOWN / {pulse_name}] Unknown modality: {modality}")
                updated_patient_data[patient_id][pulse_name] = pulse_data
    
    return updated_patient_data

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
    
    # Extract data for specified patients
    patient_data = extract_patient_data(plan, patient_ids)
    
    # Create output directories
    output_base_dir = create_output_directories(args.output, patient_ids)
    logger.info(f"\nCreated output directories in: {output_base_dir}")
    
    # Apply preprocessing steps
    preprocess(
        plan, 
        patient_data, 
        output_base_dir, 
        args.verbose, 
        args.save_intermediate
    )
    
    logger.info("Preprocessing completed successfully")


if __name__ == "__main__":
    main()