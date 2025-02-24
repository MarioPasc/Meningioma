#!/usr/bin/env python3
import os
import json
import numpy as np
import logging 
from tqdm import tqdm # type: ignore
from natsort import natsorted
import sys 

from Meningioma.image_processing.nrrd_processing import open_nrrd, transversal_axis

# --------------------------------------------------------------------------------
# Logging configuration
# --------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def numpy_converter(o):
    """Convert NumPy objects into JSON-serializable objects."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.float64, np.float32)):
        return float(o)
    # For any other type, return its string representation.
    return str(o)

def process_patient(pulse, patient_dir, output_folder, preprocessing_steps):
    """
    Process a single patient folder for a given pulse.
    
    Opens the volume file and extracts its header, storing it entirely.
    Also opens the segmentation file and computes the total volume.
    
    Additionally, adds:
      - error_opening: True if the volume file fails to open due to an invalid NRRD magic line.
      - control: True if the segmentation (_seg) file does not exist.
      
    Returns a dictionary with keys: unique_id, volume, header, segmentation,
    preprocessing_plan, outputs, error_opening, and control.
    """
    patient_name = os.path.basename(patient_dir)  # e.g., "P1"
    volume_filename = f"{pulse}_{patient_name}.nrrd"
    seg_filename = f"{pulse}_{patient_name}_seg.nrrd"
    volume_path = os.path.join(patient_dir, volume_filename)
    seg_path = os.path.join(patient_dir, seg_filename)
    
    error_opening = False
    try:
        volume, header = open_nrrd(volume_path, return_header=True)
        transversal_axis_number = transversal_axis(nrrd_path=volume_path)
        
        dtype = volume.dtype
        min_intensity = np.min(volume)
        max_intensity = np.max(volume)
        
        # For volume, we only store the route.
        volume_info = {
                    "route": volume_path,
                    "dtype": dtype,
                    "min": min_intensity,
                    "max": max_intensity,
                    "transversal_axis": transversal_axis_number}
        
    except Exception as e:
        # Check for the specific error message regarding the NRRD magic line.
        if "Invalid NRRD magic line" in str(e):
            error_opening = True
        else:
            error_opening = True
        logger.error(f"Error reading volume file {volume_path}: {e}")
        header = {}  # Empty header on error.
        volume_info = {
            "route": volume_path,
            "dtype": np.NaN,
            "min": np.NaN,
            "max": np.NaN,
            "transversal_axis": np.NaN
            }


    
    # Check if the segmentation file exists; if not, mark as control patient.
    if not os.path.isfile(seg_path):
        control = True
        segmentation_info = {"route": seg_path, "total_volume": None}
    else:
        control = False
        try:
            seg_img = open_nrrd(seg_path, return_header=False)
            total_volume = int(np.sum(seg_img))
        except Exception as e:
            logger.error(f"Error processing segmentation file {seg_path}: {e}")
            total_volume = None
        segmentation_info = {"route": seg_path, "total_volume": total_volume}
    
    unique_id = f"{pulse}_{patient_name}"
    
    outputs = {
        "preprocessed_volume": os.path.join(output_folder, f"{unique_id}_preprocessed.nrrd"),
        "logs": os.path.join(output_folder, f"{unique_id}.log")
    }
    
    # Build the final entry for the patient.
    entry = {
        "unique_id": unique_id,
        "error_opening": error_opening,
        "control": control,
        "volume": volume_info,
        "header": header,
        "segmentation": segmentation_info,
        "preprocessing_plan": preprocessing_steps, 
        "outputs": outputs,
    }
    
    return entry

def main():
    # Define dataset root and output folder.
    root = "/home/mario/Python/Datasets/Meningiomas/Meningioma_Adquisition"
    output_folder = "/home/mario/Python/Results/Meningioma/preprocessing"

    
    # In this example, we assume an empty preprocessing plan.
    preprocessing_steps = {}
    
    data = {}
    pulses = ["T1", "T1SIN", "T2", "SUSC"]
    
    # If an "RM" subfolder exists under the root, use that.
    rm_folder = os.path.join(root, "RM")
    dataset_folder = rm_folder if os.path.exists(rm_folder) else root
    
    # Iterate over each pulse folder.
    for pulse in pulses:
        logger.info(f"Processing {pulse} pulse")
        pulse_folder = os.path.join(dataset_folder, pulse)
        if not os.path.exists(pulse_folder):
            logger.warning(f"Warning: Pulse folder '{pulse_folder}' does not exist. Skipping pulse {pulse}.")
            continue
        data[pulse] = {}
        # Iterate over patient folders (folders starting with "P").
        for patient in tqdm(iterable=natsorted(seq=os.listdir(pulse_folder)), desc="Processing patients"):
            patient_path = os.path.join(pulse_folder, patient)
            if os.path.isdir(patient_path) and patient.startswith("P"):
                entry = process_patient(pulse, patient_path, output_folder, preprocessing_steps)
                if entry is not None:
                    data[pulse][patient] = entry
    
    output_json = os.path.join(output_folder, "plan_meningioma.json")
    # Write the final JSON structure to file using the numpy_converter to handle NumPy objects.
    with open(output_json, "w") as outfile:
        json.dump(data, outfile, indent=2, default=numpy_converter)
    logger.info(f"Preprocessing plan JSON saved to '{output_json}'.")

if __name__ == "__main__":
    main()
