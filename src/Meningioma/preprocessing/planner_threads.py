#!/usr/bin/env python3
import os
import json
import numpy as np
import logging
import sys
import argparse
from tqdm import tqdm  # type: ignore
from natsort import natsorted
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock

from Meningioma.image_processing.nrrd_processing import open_nrrd, transversal_axis
from Meningioma.preprocessing.metadata import (
    create_json_from_csv,
    apply_hardcoded_codification,
)

# Set up a global lock for tqdm
pbar_lock = Lock()
tqdm.set_lock(pbar_lock)

# User-defined variables


# Define dataset root, output folder, and metadata CSV file.
ROOT = "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition/RM"
OUTPUT_FOLDER = "/home/mariopasc/Python/Results/Meningioma/preprocessing"


# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ensure log directory exists
output_folder = "/home/mariopasc/Python/Results/Meningioma/preprocessing"
os.makedirs(output_folder, exist_ok=True)

# Define log file path
log_file = os.path.join(output_folder, "preprocessing.log")

# Avoid duplicate handlers
if not logger.hasHandlers():
    file_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# Redirect uncaught exceptions to logger
def log_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.error("Uncaught Exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = log_exceptions


def numpy_converter(o):
    """Convert NumPy objects into JSON-serializable objects."""
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.int64, np.int32)):
        return int(o)
    if isinstance(o, (np.float64, np.float32)):
        return float(o)
    return str(o)


def process_patient(pulse, patient_dir, output_folder, preprocessing_steps):
    """
    Process a single patient folder for a given pulse.

    Opens the volume file (and its header) and computes a few simple statistics.
    Also attempts to open the segmentation file to compute the total segmented volume.

    Returns a dictionary with the following keys for the given pulse:
      - error: True if the volume file failed to open (e.g. invalid NRRD magic line)
      - volume: { route, dtype, min, max, transversal_axis }
      - header: the full header dictionary from the volume file (or {} on error)
      - segmentation: { route, total_volume } (total_volume is None if segmentation is missing)
      - preprocessing_plan: (empty or loaded externally)
      - outputs: paths for preprocessed volume and log
    """
    patient_name = os.path.basename(patient_dir)  # e.g., "P1"
    volume_filename = f"{pulse}_{patient_name}.nrrd"
    seg_filename = f"{pulse}_{patient_name}_seg.nrrd"
    volume_path = os.path.join(patient_dir, volume_filename)
    seg_path = os.path.join(patient_dir, seg_filename)

    error = False
    try:
        volume, header = open_nrrd(volume_path, return_header=True)
        trans_axis = transversal_axis(nrrd_path=volume_path)
        dtype = str(volume.dtype)
        min_intensity = np.min(volume)
        max_intensity = np.max(volume)
        volume_info = {
            "route": volume_path,
            "dtype": dtype,
            "min": min_intensity,
            "max": max_intensity,
            "transversal_axis": trans_axis,
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
            "dtype": np.NaN,
            "min": np.NaN,
            "max": np.NaN,
            "transversal_axis": np.NaN,
        }

    # Process segmentation: if file does not exist, mark as control for this pulse.
    if not os.path.isfile(seg_path):
        segmentation_info = {"route": seg_path, "total_volume": None}
    else:
        try:
            seg_img = open_nrrd(seg_path, return_header=False)
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
        "preprocessing_plan": preprocessing_steps,
        "outputs": outputs,
    }
    return entry


def process_pulse(pulse, pulse_folder, output_folder, preprocessing_steps):
    """
    Process all patients in a given pulse folder.

    Iterates over the patient folders in the pulse folder and returns a tuple:
    (pulse, { patient_id: pulse_entry, ... })
    """
    pulses = ["T1", "T1SIN", "T2", "SUSC"]
    pulse_results = {}
    for patient in tqdm(
        natsorted(os.listdir(pulse_folder)),
        desc=f"Processing patients in {pulse}",
        leave=True,
        position=pulses.index(pulse),
    ):
        patient_path = os.path.join(pulse_folder, patient)
        if os.path.isdir(patient_path) and patient.startswith("P"):
            entry = process_patient(
                pulse, patient_path, output_folder, preprocessing_steps
            )
            if entry is not None:
                pulse_results[patient] = entry
    return (pulse, pulse_results)


def main():
    parser = argparse.ArgumentParser(
        description="Generate patient-based preprocessing plan JSON with parallel pulse processing."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads to use (between 1 and 4)",
    )
    args = parser.parse_args()
    threads = args.threads
    if threads < 1 or threads > 4:
        logger.error("Number of threads must be between 1 and 4.")
        return

    xlsx_path = "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition/metadata.xlsx"

    csv_path = os.path.join(OUTPUT_FOLDER, "metadata_recodified.csv")
    # 1. Apply the hardcoded codification
    apply_hardcoded_codification(
        xlsx_path=xlsx_path,
        output_csv_path=csv_path,
    )
    preprocessing_steps = {}  # Currently empty

    # Load patient metadata using your helper function.
    try:
        metadata_dict = create_json_from_csv(csv_path)
    except Exception as e:
        logger.error(f"Error reading metadata CSV: {e}")
        metadata_dict = {}

    # Global dictionary to collect patient data
    patients = {}

    pulses = ["T1", "T1SIN", "T2", "SUSC"]

    # Use the "RM" subfolder if it exists.
    rm_folder = os.path.join(ROOT, "RM")
    dataset_folder = rm_folder if os.path.exists(rm_folder) else ROOT

    # Prepare tasks for pulses that exist.
    tasks = []
    with tqdm(total=threads) as pbar:
        with ProcessPoolExecutor(max_workers=threads) as executor:
            for pulse in pulses:
                pulse_folder = os.path.join(dataset_folder, pulse)
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
                        output_folder,
                        preprocessing_steps,
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

    output_json = os.path.join(output_folder, "plan_meningioma.json")
    with open(output_json, "w") as outfile:
        json.dump(patients, outfile, indent=2, default=numpy_converter)
    logger.info(f"Preprocessing plan JSON saved to '{output_json}'.")


if __name__ == "__main__":
    main()
