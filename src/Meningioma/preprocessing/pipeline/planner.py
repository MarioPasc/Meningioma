#!/usr/bin/env python3
import os
import json
import numpy as np
import logging
import sys
import argparse
import yaml
import nrrd  # type: ignore
from tqdm import tqdm  # type: ignore
from natsort import natsorted
from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import Optional, Dict, Any

from Meningioma.preprocessing.pipeline.metadata import (
    create_json_from_csv,
    apply_hardcoded_codification,
)
from Meningioma.utils.parse_nrrd_header import numpy_converter

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Redirect uncaught exceptions to logger
def log_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught Exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = log_exceptions



def process_patient(pulse, patient_dir, output_folder, modality):
    """
    Process a single patient folder for a given pulse.

    Opens the volume file (and its header) and computes a few simple statistics.
    Also attempts to open the segmentation file to compute the total segmented volume.

    Returns a dictionary with the following keys for the given pulse:
      - error: True if the volume file failed to open (e.g. invalid NRRD magic line)
      - volume: { route, dtype, min, max, transversal_axis }
      - header: the full header dictionary from the volume file (or {} on error)
      - segmentation: { route, total_volume } (total_volume is None if segmentation is missing)
      - modality: RM or TC
      - outputs: paths for preprocessed volume and log
    """
    patient_name = os.path.basename(patient_dir)  # e.g., "P1"
    volume_filename = f"{pulse}_{patient_name}.nrrd"
    seg_filename = f"{pulse}_{patient_name}_seg.nrrd"
    volume_path = os.path.join(patient_dir, volume_filename)
    seg_path = os.path.join(patient_dir, seg_filename)

    error = False
    try:
        volume, header = nrrd.read(volume_path)
        dtype = str(volume.dtype)
        min_intensity = np.min(volume)
        max_intensity = np.max(volume)
        volume_info = {
            "route": volume_path,
            "dtype": dtype,
            "min": min_intensity,
            "max": max_intensity,
        }
    except Exception as e:
        if "Invalid NRRD magic line" in str(e):
            error = True
        else:
            error = True
        logger.error(f"Error reading volume file {volume_path}: {e}")
        header = {}
        volume_info = {
            "route": volume_path,
            "dtype": None,
            "min": None,
            "max": None,
            "transversal_axis": None,
        }

    # Process segmentation: if file does not exist, mark as control for this pulse.
    if not os.path.isfile(seg_path):
        segmentation_info = {"route": seg_path, "total_volume": None}
    else:
        try:
            seg_img, _ = nrrd.read(seg_path)
            total_volume = int(np.sum(seg_img))
        except Exception as e:
            logger.error(f"Error processing segmentation file {seg_path}: {e}")
            total_volume = None
        segmentation_info = {"route": seg_path, "total_volume": total_volume}

    outputs = {
        "preprocessed_volume": os.path.join(
            output_folder, f"{pulse}_{patient_name}_preprocessed.nrrd"
        ),
        "logs": os.path.join(output_folder, f"{pulse}_{patient_name}.log"),
    }

    entry = {
        "error": error,
        "volume": volume_info,
        "header": header,
        "segmentation": segmentation_info,
        "modality": modality,
        "outputs": outputs,
    }
    return entry


def process_pulse(pulse, pulse_folder, output_folder, modality):
    """
    Process all patients in a given pulse folder.

    Iterates over the patient folders in the pulse folder and returns a tuple:
    (pulse, { patient_id: pulse_entry, ... })
    """
    pulse_results = {}
    for patient in tqdm(
        natsorted(os.listdir(pulse_folder)),
        desc=f"Processing patients in {pulse}",
        leave=True,
    ):
        patient_path = os.path.join(pulse_folder, patient)
        if os.path.isdir(patient_path) and patient.startswith("P"):
            entry = process_patient(
                pulse, patient_path, output_folder, modality
            )
            if entry is not None:
                pulse_results[patient] = entry
    return (pulse, pulse_results)


def plan(yaml_path: str) -> Optional[str]:
    """
    Execute the preprocessing pipeline.
    Loads configuration from a YAML file, processes patient data, and saves the results to a JSON file.
    Args:
        yaml_path (str): Path to the YAML configuration file.
    Returns:
        Optional[str]: Path to the output JSON file.
    """
    # Load configuration from YAML file
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error reading config file: {e}")
        return ""

    # Extract configuration parameters
    ROOT = config.get("paths", {}).get("root", "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition")
    OUTPUT_FOLDER = config.get("paths", {}).get("output", "/home/mariopasc/Python/Datasets/Meningiomas/jsons")
    threads = config.get("processing", {}).get("threads", 1)
    
    # Define preprocessing plans from config
    preprocessing_plan = {
        "RM": config.get("preprocessing", {}).get("RM", {}),
        "TC": config.get("preprocessing", {}).get("TC", {})
    }

    # Ensure output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Define log file path
    log_file = os.path.join(OUTPUT_FOLDER, "preprocessing.log")

    # Avoid duplicate handlers
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Validate threads
    if threads < 1 or threads > 4:
        logger.error("Number of threads must be between 1 and 4.")
        return ""

    xlsx_path = os.path.join(ROOT, "metadata.xlsx")
    csv_path = os.path.join(OUTPUT_FOLDER, "metadata_recodified.csv")
    # Apply the hardcoded codification
    apply_hardcoded_codification(
        xlsx_path=xlsx_path,
        output_csv_path=csv_path,
    )

    # Load patient metadata using your helper function.
    try:
        metadata_dict = create_json_from_csv(csv_path)
    except Exception as e:
        logger.error(f"Error reading metadata CSV: {e}")
        metadata_dict = {}

    # Global dictionary to collect patient data
    patients: Dict[Any, Any] = {}

    # Define modalities and their respective pulses from config
    modalities = config.get("modalities", {
        "RM": ["T1", "T1SIN", "T2", "SUSC"],
        "TC": ["TC"]  
    })

    # Prepare tasks for pulses that exist.
    tasks = []
    with ProcessPoolExecutor(max_workers=threads) as executor:
        for modality, pulses in modalities.items():
            modality_folder = os.path.join(ROOT, modality)
            if not os.path.exists(modality_folder):
                logger.warning(f"Warning: Modality folder '{modality_folder}' does not exist. Skipping {modality}.")
                continue
                
            for pulse in pulses:
                logger.info(f"Processing {modality}/{pulse}")
                if modality != "TC": 
                    pulse_folder = os.path.join(modality_folder, pulse)
                else:
                    pulse_folder = modality_folder
                if not os.path.exists(pulse_folder):
                    logger.warning(
                        f"Warning: Pulse folder '{pulse_folder}' does not exist. Skipping pulse {pulse}."
                    )
                    continue
                # Submit one task per pulse folder.
                tasks.append(
                    executor.submit(
                        process_pulse,
                        pulse,
                        pulse_folder,
                        OUTPUT_FOLDER,
                        modality,
                    )
                )

        # Collect results as they complete.
        for future in as_completed(tasks):
            try:
                pulse, pulse_data = future.result()
                # Merge pulse_data into the global patients dictionary.
                for patient_id, pulse_entry in pulse_data.items():
                    if patient_id not in patients:
                        patients[patient_id] = {"pulses": {}}
                    patients[patient_id]["pulses"][pulse] = pulse_entry
            except Exception as e:
                logger.error(f"Error processing a pulse: {e}")

    # Compute patient-level "control": if all available pulses lack segmentation (total_volume is None), then control = True.
    for patient_id, patient_data in patients.items():
        pulse_entries = patient_data.get("pulses", {})
        is_control = True
        for pulse_data in pulse_entries.values():
            if pulse_data["segmentation"]["total_volume"] is not None:
                is_control = False
                break
        patient_data["control"] = is_control
        # Add metadata for the patient (if available).
        patient_data["metadata"] = metadata_dict.get(patient_id, {})

    # Create the new structure with preprocessing_plan at the top level
    result = {
        "preprocessing_plan": preprocessing_plan,
        "data": patients
    }

    output_json = os.path.join(OUTPUT_FOLDER, "plan_meningioma.json")
    with open(output_json, "w") as outfile:
        json.dump(result, outfile, indent=2, default=numpy_converter)
    logger.info(f"Preprocessing plan JSON saved to '{output_json}'.")
    return output_json