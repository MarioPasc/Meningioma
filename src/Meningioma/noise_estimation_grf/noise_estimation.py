import os
import json
import logging
import time
from typing import Dict, Any, List, Tuple

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
OUTPUT_JSON_FOLDER: str = (
    "/home/mariopasc/Python/Results/Meningioma/noise_modelling/variogram_models_fitting"
)
os.makedirs(OUTPUT_JSON_FOLDER, exist_ok=True)
OUTPUT_JSON_FILE: str = os.path.join(
    OUTPUT_JSON_FOLDER, f"{PATIENT}_variogram_models.json"
)
EXECUTION_TIME_FILE: str = os.path.join(
    OUTPUT_JSON_FOLDER, f"{PATIENT}_execution_times.json"
)

# Variogram estimation parameters
VARIOGRAM_BINS = np.linspace(0, 100, 100)
VARIOGRAM_SAMPLING_SIZE = 3000
ESTIMATOR = "cressie"
LEN_SCALE_GUESS = 20  # initial guess for correlation length

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

# Flag to ignore anisotropic variogram computation if set to True.
IGNORE_ANISOTROPIC: bool = True

# =============================================================================
# Logging configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


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
    ignore_anisotropic: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Process the MRI npz files for all pulses and seeds for a given patient,
    estimate variograms, fit covariance models, and record the best-fitting
    model (by r²) for each variogram. Also records the execution time for
    each major step.

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
        ignore_anisotropic: If True, only the isotropic variogram is computed.

    Returns:
        A tuple containing:
         - A nested dictionary with the structure:
           {patient: {seed: {pulse: {"variograms": {variogram_label: { "model": str, "r2": float, "params": dict }}}}}}
         - A nested dictionary with timing information for each pulse.
    """
    results: Dict[str, Any] = {patient: {}}
    timings: Dict[str, Any] = {patient: {}}
    logging.info(f"Starting variogram fitting pipeline for patient: {patient}")

    # Verify that all pulse files exist; remove missing pulses with a warning.
    valid_pulses: List[str] = []
    for pulse in pulses:
        npz_filepath = os.path.join(base_npz_path, patient, f"{patient}_{pulse}.npz")
        if not os.path.exists(npz_filepath):
            logging.warning(
                f"Pulse {pulse} not found at {npz_filepath}. It will be removed from the processing list."
            )
        else:
            valid_pulses.append(pulse)

    if not valid_pulses:
        logging.error("No valid pulses found for patient. Exiting pipeline.")
        return results, timings

    # Loop over seeds (with tqdm for progress indication)
    for seed in tqdm(seeds, desc="Processing seeds", unit="seed"):
        try:
            seed_int = int(seed)
        except Exception as e:
            logging.error(f"Error converting seed {seed} to integer: {e}")
            continue

        results[patient][seed] = {}
        timings[patient][seed] = {}

        # Loop over valid pulses for each seed
        for pulse in tqdm(
            valid_pulses, desc="Processing pulses", unit="pulse", leave=False
        ):
            pulse_timing: Dict[str, float] = {}
            t_pulse_start = time.time()
            logging.info(f"Processing patient {patient}, pulse {pulse}, seed {seed}")

            npz_filepath = os.path.join(
                base_npz_path, patient, f"{patient}_{pulse}.npz"
            )
            # Step 1: Load volume and mask
            try:
                t0 = time.time()
                volume, mask = ImageProcessing.segment_3d_volume(
                    npz_filepath, threshold_method="li"
                )
                pulse_timing["load_volume"] = time.time() - t0
            except Exception as e:
                logging.error(
                    f"Error loading volume for pulse {pulse} with seed {seed}: {e}"
                )
                continue

            # Step 2: Compute background variance
            try:
                t0 = time.time()
                outside_pixels = volume[~mask]
                var_guess = np.var(outside_pixels) if outside_pixels.size > 1 else 0.0
                pulse_timing["background_variance"] = time.time() - t0
            except Exception as e:
                logging.error(
                    f"Error computing background variance for pulse {pulse} with seed {seed}: {e}"
                )
                var_guess = 0.0

            # Step 3: Isotropic variogram estimation and model fitting
            try:
                t0 = time.time()
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
                pulse_timing["isotropic_estimation"] = time.time() - t0

                t0 = time.time()
                iso_models = BlindNoiseEstimation.fit_model_3d(
                    bin_center=iso_bin_center,
                    gamma=iso_gamma,
                    var=var_guess,
                    len_scale=len_scale_guess,
                )
                pulse_timing["isotropic_model_fitting"] = time.time() - t0

                if iso_models:
                    best_iso_model_name = max(
                        iso_models, key=lambda k: iso_models[k][1]["r2"]
                    )
                    best_iso_r2 = iso_models[best_iso_model_name][1]["r2"]
                    best_iso_params = iso_models[best_iso_model_name][1]["params"]
                else:
                    best_iso_model_name = None
                    best_iso_r2 = None
                    best_iso_params = {}
            except Exception as e:
                logging.error(
                    f"Error processing isotropic variogram for pulse {pulse} with seed {seed}: {e}"
                )
                best_iso_model_name = None
                best_iso_r2 = None
                best_iso_params = {}

            # Step 4: Anisotropic variogram estimation and model fitting (if not ignored)
            anisotropic_results: Dict[str, Any] = {}
            if not ignore_anisotropic:
                try:
                    t0 = time.time()
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
                    pulse_timing["anisotropic_estimation"] = time.time() - t0
                except Exception as e:
                    logging.error(
                        f"Error processing anisotropic variograms for pulse {pulse} with seed {seed}: {e}"
                    )
                    anisotropic_variograms = {}

                total_aniso_fit_time = 0.0
                for label, (bin_centers, gamma) in anisotropic_variograms.items():
                    try:
                        t0 = time.time()
                        models_dict = BlindNoiseEstimation.fit_model_3d(
                            bin_center=bin_centers,
                            gamma=gamma,
                            var=var_guess,
                            len_scale=len_scale_guess,
                        )
                        total_aniso_fit_time += time.time() - t0
                        if models_dict:
                            best_model_name = max(
                                models_dict, key=lambda k: models_dict[k][1]["r2"]
                            )
                            best_model_r2 = models_dict[best_model_name][1]["r2"]
                            best_params = models_dict[best_model_name][1]["params"]
                        else:
                            best_model_name = None
                            best_model_r2 = None
                            best_params = {}
                    except Exception as e:
                        logging.error(
                            f"Error fitting anisotropic model for {label} in pulse {pulse} with seed {seed}: {e}"
                        )
                        best_model_name = None
                        best_model_r2 = None
                        best_params = {}
                    anisotropic_results[label] = {
                        "model": best_model_name,
                        "r2": best_model_r2,
                        "params": best_params,
                    }
                pulse_timing["anisotropic_model_fitting"] = total_aniso_fit_time

            # Record total time for the pulse
            pulse_timing["total_pulse_time"] = time.time() - t_pulse_start

            # Build results for this pulse
            pulse_results = {
                "variograms": {
                    "Isotropic": {
                        "model": best_iso_model_name,
                        "r2": best_iso_r2,
                        "params": best_iso_params,
                    }
                }
            }
            if not ignore_anisotropic:
                pulse_results["variograms"].update(anisotropic_results)

            results[patient][seed][pulse] = pulse_results
            timings[patient][seed][pulse] = pulse_timing

    logging.info("Completed processing all seeds and pulses.")
    return results, timings


def main() -> None:
    """
    Main entry point for the variogram fitting pipeline.
    Processes the data for the given patient, writes the model metadata
    to a JSON file, and writes the execution times of each step to a separate JSON file.
    """
    try:
        results, exec_timings = run_variogram_fitting_pipeline(
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
            ignore_anisotropic=IGNORE_ANISOTROPIC,
        )
    except Exception as e:
        logging.error(f"Critical error during the variogram fitting pipeline: {e}")
        return

    # Save the variogram model metadata
    try:
        with open(OUTPUT_JSON_FILE, "w") as json_file:
            json.dump(results, json_file, indent=4)
        logging.info(f"Variogram model metadata saved to {OUTPUT_JSON_FILE}")
    except Exception as e:
        logging.error(f"Error saving JSON file: {e}")

    # Save the execution timings
    try:
        with open(EXECUTION_TIME_FILE, "w") as json_file:
            json.dump(exec_timings, json_file, indent=4)
        logging.info(f"Execution time metadata saved to {EXECUTION_TIME_FILE}")
    except Exception as e:
        logging.error(f"Error saving execution time JSON file: {e}")


if __name__ == "__main__":
    main()
