import os
import json
import logging
from typing import Dict, Any, List

import numpy as np
from tqdm import tqdm  # type: ignore

from Meningioma import ImageProcessing, BlindNoiseEstimation  # type: ignore

# =============================================================================
# User-defined variables
# =============================================================================
PATIENT: str = "P50"
PULSES: List[str] = ["T1", "T1SIN", "T2", "SUSC"]
SEEDS: List[str] = ["42", "123", "55", "23102003", "1122002"]

# Base paths (update these paths as needed)
BASE_NPZ_PATH: str = "/home/mariopasc/Python/Datasets/Meningiomas/npz"
OUTPUT_JSON_FOLDER: str = "results/variogram_models"
os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)
OUTPUT_JSON_FILE: str = os.path.join(
    OUTPUT_JSON_FOLDER, f"{PATIENT}_variogram_models.json"
)

# Variogram estimation parameters
VARIOGRAM_BINS = np.linspace(0, 100, 100)
VARIOGRAM_SAMPLING_SIZE = 3000
ESTIMATOR = "cressie"
LEN_SCALE_GUESS = 20  # initial guess for correlation length
# Note: var_guess is computed per volume using background voxels

# Default directions and labels for anisotropic variogram estimation.
DIRECTIONS = [
    np.array([1, 0, 0]),
    np.array([-1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 1]),
    np.array([1, 1, 0]),
    np.array([1, 0, 1]),
    np.array([0, 1, 1]),
    np.array([1, 1, 1]),
]
DIRECTION_LABELS = [
    r"X-axis $[1,0,0]$",
    r"Opposite X-axis $[-1,0,0]$",
    r"Y-axis $[0,1,0]$",
    r"Z-axis $[0,0,1]$",
    r"Diagonal_XY $[1,1,0]$",
    r"Diagonal_XZ $[1,0,1]$",
    r"Diagonal_YZ $[0,1,1]$",
    r"Diagonal_XYZ $[1,1,1]$",
]

# =============================================================================
# Logging configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


# =============================================================================
# Main Routine: Process all pulses and seeds for a given patient
# =============================================================================
def run_variogram_fitting_pipeline(
    patient: str,
    base_npz_path: str,
    pulses: List[str],
    seeds: List[str],
    variogram_bins: np.ndarray,
    sampling_size: int,
    estimator: str,
    len_scale_guess: float,
    directions: List[np.ndarray],
    direction_labels: List[str],
) -> Dict[str, Any]:
    """
    Process the MRI npz files for all pulses and seeds for a given patient,
    estimate variograms, fit covariance models, and record the best-fitting
    model (by r²) for each variogram.

    Parameters:
        patient: Patient identifier (e.g. "P50").
        base_npz_path: Base directory path to the npz files.
        pulses: List of pulse types (e.g. ["T1", "T1SIN", "T2", "SUSC"]).
        seeds: List of random seeds (as strings) to use for variogram estimation.
        variogram_bins: 1D array defining the bin edges for variogram estimation.
        sampling_size: Number of voxel pairs to sample.
        estimator: Variogram estimator name (e.g., "cressie").
        len_scale_guess: Initial guess for the correlation length scale.
        directions: List of 3D direction vectors for anisotropic variogram estimation.
        direction_labels: List of corresponding labels for the directions.

    Returns:
        A nested dictionary with the structure:
        {patient: {seed: {pulse: {"variograms": {variogram_label: { "model": str, "params": dict }}}}}}
    """
    results: Dict[str, Any] = {patient: {}}
    logging.info(f"Starting variogram fitting pipeline for patient: {patient}")

    # Loop over seeds (using tqdm for progress indication)
    for seed in tqdm(seeds, desc="Processing seeds", unit="seed"):
        seed_int = int(seed)  # Convert seed to int for function calls
        results[patient][seed] = {}

        # Loop over pulses for each seed
        for pulse in tqdm(pulses, desc="Processing pulses", unit="pulse", leave=False):
            logging.info(f"Processing patient {patient}, pulse {pulse}, seed {seed}")
            # Construct file path for the current pulse
            npz_filepath = os.path.join(
                base_npz_path, patient, f"{patient}_{pulse}.npz"
            )
            if not os.path.exists(npz_filepath):
                logging.warning(
                    f"File not found: {npz_filepath}. Skipping pulse {pulse}."
                )
                continue

            # Load volume and mask using the segmentation function
            try:
                volume, mask = ImageProcessing.segment_3d_volume(
                    npz_filepath, threshold_method="li"
                )
            except Exception as e:
                logging.error(
                    f"Error loading volume for pulse {pulse} with seed {seed}: {e}"
                )
                continue

            # Compute statistics from background (outside mask) to set variance guess.
            outside_pixels = volume[~mask]
            var_guess = np.var(outside_pixels) if outside_pixels.size > 1 else 0.0

            # --- Estimate Isotropic Variogram and Fit Covariance Models ---
            try:
                iso_bin_center, iso_gamma = (
                    BlindNoiseEstimation.estimate_variogram_isotropic_3d(
                        data=volume,
                        bins=variogram_bins,
                        mask=mask,
                        estimator=estimator,
                        sampling_size=sampling_size,
                        sampling_seed=seed_int,
                    )
                )
                iso_models = BlindNoiseEstimation.fit_model_3d(
                    bin_center=iso_bin_center,
                    gamma=iso_gamma,
                    var=var_guess,
                    len_scale=len_scale_guess,
                )
                # Select best isotropic model based on highest r^2
                if iso_models:
                    best_iso_model_name = max(
                        iso_models, key=lambda k: iso_models[k][1]["r2"]
                    )
                    best_iso_params = iso_models[best_iso_model_name][1]["params"]
                else:
                    best_iso_model_name = None
                    best_iso_params = {}
            except Exception as e:
                logging.error(
                    f"Error processing isotropic variogram for pulse {pulse} with seed {seed}: {e}"
                )
                best_iso_model_name = None
                best_iso_params = {}

            # --- Estimate Anisotropic Variograms and Fit Covariance Models ---
            anisotropic_results: Dict[str, Any] = {}
            try:
                anisotropic_variograms = (
                    BlindNoiseEstimation.estimate_variogram_anisotropic_3d(
                        data=volume,
                        bins=variogram_bins,
                        mask=mask,
                        directions=directions,
                        direction_labels=direction_labels,
                        estimator=estimator,
                        sampling_size=sampling_size,
                        sampling_seed=seed_int,
                    )
                )
                # For each anisotropic direction, fit models and select the best
                for label, (bin_centers, gamma) in anisotropic_variograms.items():
                    models_dict = BlindNoiseEstimation.fit_model_3d(
                        bin_center=bin_centers,
                        gamma=gamma,
                        var=var_guess,
                        len_scale=len_scale_guess,
                    )
                    if models_dict:
                        best_model_name = max(
                            models_dict, key=lambda k: models_dict[k][1]["r2"]
                        )
                        best_params = models_dict[best_model_name][1]["params"]
                    else:
                        best_model_name = None
                        best_params = {}
                    anisotropic_results[label] = {
                        "model": best_model_name,
                        "params": best_params,
                    }
            except Exception as e:
                logging.error(
                    f"Error processing anisotropic variograms for pulse {pulse} with seed {seed}: {e}"
                )

            # Build the nested results for this pulse
            pulse_results = {
                "variograms": {
                    "Isotropic": {
                        "model": best_iso_model_name,
                        "params": best_iso_params,
                    },
                }
            }
            # Merge the anisotropic results
            pulse_results["variograms"].update(anisotropic_results)

            # Save the pulse results under the current seed.
            results[patient][seed][pulse] = pulse_results

    logging.info("Completed processing all seeds and pulses.")
    return results


def main() -> None:
    """
    Main entry point for the variogram fitting pipeline.
    Processes the data for the given patient and writes the results as a JSON file.
    """
    results = run_variogram_fitting_pipeline(
        patient=PATIENT,
        base_npz_path=BASE_NPZ_PATH,
        pulses=PULSES,
        seeds=SEEDS,
        variogram_bins=VARIOGRAM_BINS,
        sampling_size=VARIOGRAM_SAMPLING_SIZE,
        estimator=ESTIMATOR,
        len_scale_guess=LEN_SCALE_GUESS,
        directions=DIRECTIONS,
        direction_labels=DIRECTION_LABELS,
    )

    # Write results to JSON file
    try:
        with open(OUTPUT_JSON_FILE, "w") as json_file:
            json.dump(results, json_file, indent=4)
        logging.info(f"Variogram model metadata saved to {OUTPUT_JSON_FILE}")
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")


if __name__ == "__main__":
    main()
