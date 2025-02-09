import os
import json
import logging
import time
from collections import Counter
from typing import Dict, Any, List, Tuple

import numpy as np
from tqdm import tqdm  # type: ignore

from Meningioma import (  # type: ignore
    ImageProcessing,
    BlindNoiseEstimation,
    Stats,
    Metrics,
)

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

# We will create a patient folder inside OUTPUT_JSON_FOLDER:
patient_folder: str = os.path.join(OUTPUT_JSON_FOLDER, PATIENT)
os.makedirs(patient_folder, exist_ok=True)

# In the patient folder we create two folders: "Variograms" and "Noise_Comparison"
variograms_folder: str = os.path.join(patient_folder, "Variograms")
noise_comp_folder: str = os.path.join(patient_folder, "Noise_Comparison")
os.makedirs(variograms_folder, exist_ok=True)
os.makedirs(noise_comp_folder, exist_ok=True)

# Files to save overall summaries (if needed)
OUTPUT_JSON_FILE: str = os.path.join(
    variograms_folder, f"{PATIENT}_variogram_models.json"
)
EXECUTION_TIME_FILE: str = os.path.join(
    patient_folder, f"{PATIENT}_execution_times.json"
)
SUMMARY_JSON_FILE: str = os.path.join(
    patient_folder, f"{PATIENT}_best_model_summary.json"
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

# Parzen-Rosenblatt bandwidth (used for PDF estimation)
H: float = 0.5

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
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    For each pulse of the given patient, perform segmentation (once per pulse)
    and compute the variogram and covariance model fitting across seeds.
    The full results (variogram info) are saved in a JSON file under the "Variograms"
    folder, containing only the best model info (keys: "model", "r2", "params").

    In addition, for each seed and each slice of the volume, noise metrics are computed
    (background and generated noise PDFs, Rayleigh parameters, JS divergence) and saved
    as separate JSON files in the folder structure:

      patient/Noise_Comparison/<pulse>/<slice_index>/<seed>.json

    Returns:
        A tuple of three dictionaries:
         - results: Nested results per patient > pulse > seed with full details.
         - timings: Timing information per patient > pulse > seed.
         - summary: A summary per pulse with averaged metrics (and standard deviations)
    """
    results: Dict[str, Any] = {patient: {}}
    timings: Dict[str, Any] = {patient: {}}
    summary: Dict[str, Any] = {patient: {}}
    logging.info(f"Starting variogram fitting pipeline for patient: {patient}")

    # Check pulses exist (remove missing ones)
    valid_pulses: List[str] = []
    for pulse in pulses:
        npz_filepath = os.path.join(base_npz_path, patient, f"{patient}_{pulse}.npz")
        if not os.path.exists(npz_filepath):
            logging.warning(
                f"Pulse {pulse} not found at {npz_filepath}. It will be removed from processing."
            )
        else:
            valid_pulses.append(pulse)

    if not valid_pulses:
        logging.error("No valid pulses found for patient. Exiting pipeline.")
        return results, timings, summary

    # Process each pulse once (segmentation and background extraction are independent of seed)
    for pulse in tqdm(valid_pulses, desc="Processing pulses", unit="pulse"):
        pulse_start_time = time.time()
        logging.info(f"Processing pulse {pulse} for patient {patient}")

        npz_filepath = os.path.join(base_npz_path, patient, f"{patient}_{pulse}.npz")
        try:
            t0 = time.time()
            volume, mask = ImageProcessing.segment_3d_volume(
                npz_filepath, threshold_method="li"
            )
            pulse_seg_time = time.time() - t0
            logging.info("Volume segmentation completed.")
            assert (
                volume.shape == mask.shape
            ), f"Volume and mask shapes differ for pulse {pulse}"
        except Exception as e:
            logging.error(f"Error during segmentation for pulse {pulse}: {e}")
            continue

        # Extract background (outside mask) from full volume (for variogram fitting)
        outside_pixels = volume[~mask]
        if outside_pixels.size == 0:
            logging.error(f"No background pixels found for pulse {pulse}. Skipping.")
            continue

        # Initialize dictionaries for this pulse
        results[patient][pulse] = {}
        timings[patient][pulse] = {
            "segmentation": pulse_seg_time,
        }
        seed_summaries: List[Dict[str, Any]] = []

        # Process each seed for this pulse
        for seed in tqdm(
            seeds, desc=f"Pulse {pulse}: processing seeds", unit="seed", leave=False
        ):
            seed_timing: Dict[str, float] = {}
            t_seed_start = time.time()
            try:
                seed_int = int(seed)
            except Exception as e:
                logging.error(
                    f"Error converting seed {seed} to integer for pulse {pulse}: {e}"
                )
                continue

            # --- Variogram Estimation & Covariance Model Fitting (full volume) ---
            # Step 1: Isotropic variogram estimation and fitting
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
                seed_timing["isotropic_estimation"] = time.time() - t0

                t0 = time.time()
                iso_models = BlindNoiseEstimation.fit_model_3d(
                    bin_center=iso_bin_center,
                    gamma=iso_gamma,
                    var=np.var(outside_pixels),
                    len_scale=len_scale_guess,
                )
                seed_timing["isotropic_model_fitting"] = time.time() - t0

                if iso_models:
                    best_iso_key = max(iso_models, key=lambda k: iso_models[k][1]["r2"])
                    best_iso_r2 = iso_models[best_iso_key][1]["r2"]
                    best_iso_params = iso_models[best_iso_key][1]["params"]
                    best_iso_instance = iso_models[best_iso_key][0]
                else:
                    best_iso_key = None
                    best_iso_r2 = None
                    best_iso_params = {}
                    best_iso_instance = None
            except Exception as e:
                logging.error(
                    f"Error in isotropic variogram processing for pulse {pulse}, seed {seed}: {e}"
                )
                best_iso_key = None
                best_iso_r2 = None
                best_iso_params = {}
                best_iso_instance = None

            isotropic_result = {
                "model": best_iso_key,
                "r2": best_iso_r2,
                "params": best_iso_params,
                "cov_model_instance": best_iso_instance,
            }

            # Step 2: (Optional) Anisotropic variogram estimation and fitting
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
                    seed_timing["anisotropic_estimation"] = time.time() - t0
                except Exception as e:
                    logging.error(
                        f"Error in anisotropic variogram processing for pulse {pulse}, seed {seed}: {e}"
                    )
                    anisotropic_variograms = {}

                total_aniso_fit_time = 0.0
                for label, (bin_centers, gamma) in anisotropic_variograms.items():
                    try:
                        t0 = time.time()
                        models_dict = BlindNoiseEstimation.fit_model_3d(
                            bin_center=bin_centers,
                            gamma=gamma,
                            var=np.var(outside_pixels),
                            len_scale=len_scale_guess,
                        )
                        total_aniso_fit_time += time.time() - t0
                        if models_dict:
                            best_model_key = max(
                                models_dict, key=lambda k: models_dict[k][1]["r2"]
                            )
                            best_model_r2 = models_dict[best_model_key][1]["r2"]
                            best_params = models_dict[best_model_key][1]["params"]
                            best_model_instance = models_dict[best_model_key][0]
                        else:
                            best_model_key = None
                            best_model_r2 = None
                            best_params = {}
                            best_model_instance = None
                    except Exception as e:
                        logging.error(
                            f"Error fitting anisotropic model for {label} in pulse {pulse}, seed {seed}: {e}"
                        )
                        best_model_key = None
                        best_model_r2 = None
                        best_params = {}
                        best_model_instance = None
                    anisotropic_results[label] = {
                        "model": best_model_key,
                        "r2": best_model_r2,
                        "params": best_params,
                        "cov_model_instance": best_model_instance,
                    }
                seed_timing["anisotropic_model_fitting"] = total_aniso_fit_time

            # Build per-seed results (retain cov_model_instance temporarily for noise generation)
            pulse_results = {"variograms": {"Isotropic": isotropic_result}}
            if not ignore_anisotropic:
                pulse_results["variograms"].update(anisotropic_results)

            logging.info(
                "Phase 2: Selecting best overall model and generating noise volume"
            )
            # --- Phase 2: Select best overall model (by highest r2) across variogram types ---
            best_overall_r2 = -np.inf
            best_overall_key = None
            best_overall_instance = None
            for key, value in pulse_results["variograms"].items():
                r2_val = value.get("r2")
                if r2_val is not None and r2_val > best_overall_r2:
                    best_overall_r2 = r2_val
                    best_overall_key = key
                    best_overall_instance = value.get("cov_model_instance")
            if best_overall_instance is not None:
                try:
                    logging.info("Generating noise volume using best overall model")
                    t0 = time.time()
                    real_noise, imag_noise, combined_noise = (
                        BlindNoiseEstimation.gaussian_random_fields_noise_3d(
                            model=best_overall_instance, shape=volume.shape
                        )
                    )
                    noise_generation_time = time.time() - t0
                    seed_timing["noise_generation"] = noise_generation_time
                    logging.info(
                        f"Generated a noise volume of shape {volume.shape} in {noise_generation_time:.2f} seconds"
                    )
                except Exception as e:
                    logging.error(
                        f"Error generating noise volume for pulse {pulse}, seed {seed}: {e}"
                    )
                    combined_noise = None
            else:
                combined_noise = None

            # --- Slice-by-slice Noise Comparison ---
            # For each slice in the volume, compute the background and generated noise PDFs,
            # the Rayleigh parameters, and the JS divergence, and save the data to a JSON file.
            if combined_noise is not None:
                num_slices = volume.shape[2]
                for slice_idx in range(num_slices):
                    # Define folder for this pulse and slice
                    slice_folder = os.path.join(
                        noise_comp_folder, pulse, f"slice_{slice_idx}"
                    )
                    os.makedirs(slice_folder, exist_ok=True)

                    # Extract slice data for background noise (from original volume & mask)
                    bg_slice_pixels = volume[:, :, slice_idx][~mask[:, :, slice_idx]]
                    # Extract generated noise slice
                    gen_slice = combined_noise[:, :, slice_idx]

                    # Validate that there are background pixels in this slice
                    if bg_slice_pixels.size == 0:
                        logging.warning(
                            f"Slice {slice_idx} of pulse {pulse} has no background pixels. Skipping noise comparison for this slice."
                        )
                        continue

                    try:
                        t0 = time.time()
                        # Compute PDF for background slice
                        (
                            bg_x_slice,
                            bg_kde_slice,
                            bg_pdf_slice,
                            bg_param_str_slice,
                            bg_param_series_slice,
                        ) = Stats.compute_pdf(
                            bg_slice_pixels, h=H, dist="rayleigh", num_points=1000
                        )
                        # Compute PDF for generated noise slice
                        # (Note: we flatten the generated slice)
                        (
                            gen_x_slice,
                            gen_kde_slice,
                            gen_pdf_slice,
                            gen_param_str_slice,
                            gen_param_series_slice,
                        ) = Stats.compute_pdf(
                            gen_slice.flatten(), h=H, dist="rayleigh", num_points=1000
                        )
                        slice_pdf_time = time.time() - t0
                    except Exception as e:
                        logging.error(
                            f"Error computing PDFs for pulse {pulse}, slice {slice_idx}, seed {seed}: {e}"
                        )
                        continue

                    try:
                        t0 = time.time()
                        js_div_slice = Metrics.compute_jensen_shannon_divergence_pdfs(
                            pdf1=bg_pdf_slice,
                            pdf2=gen_pdf_slice,
                            x_values=bg_x_slice,
                            epsilon=1e-10,
                        )
                        slice_js_time = time.time() - t0
                    except Exception as e:
                        logging.error(
                            f"Error computing JS divergence for pulse {pulse}, slice {slice_idx}, seed {seed}: {e}"
                        )
                        js_div_slice = None

                    # Build dictionary with the data needed to recreate the PDFs
                    noise_comp_data = {
                        "pdf_background": {
                            "x": bg_x_slice.tolist(),
                            "kde_est": bg_kde_slice.tolist(),
                            "pdf_fit": bg_pdf_slice.tolist(),
                            "bandwidth": H,
                        },
                        "pdf_generated": {
                            "x": gen_x_slice.tolist(),
                            "kde_est": gen_kde_slice.tolist(),
                            "pdf_fit": gen_pdf_slice.tolist(),
                            "bandwidth": H,
                        },
                        "Rayleigh_coefficients_background_noise": (
                            bg_param_series_slice.to_dict()
                            if bg_param_series_slice is not None
                            else None
                        ),
                        "Rayleigh_coefficients_generated_noise": (
                            gen_param_series_slice.to_dict()
                            if gen_param_series_slice is not None
                            else None
                        ),
                        "JS_Divergence_score": js_div_slice,
                        "slice_pdf_time": slice_pdf_time,
                        "slice_js_time": slice_js_time,
                    }
                    # Save the noise comparison data to a JSON file under folder:
                    # patient/Noise_Comparison/<pulse>/slice_<slice_idx>/<seed>.json
                    noise_comp_filepath = os.path.join(slice_folder, f"{seed}.json")
                    try:
                        with open(noise_comp_filepath, "w") as nf:
                            json.dump(noise_comp_data, nf, indent=4)
                        logging.info(
                            f"Saved noise comparison for pulse {pulse}, slice {slice_idx}, seed {seed}"
                        )
                    except Exception as e:
                        logging.error(
                            f"Error saving noise comparison JSON for pulse {pulse}, slice {slice_idx}, seed {seed}: {e}"
                        )

            # Finalize per-seed results: remove temporary model instances
            for key, value in pulse_results["variograms"].items():
                value.pop("cov_model_instance", None)

            seed_timing["total_seed_time"] = time.time() - t_seed_start
            results[patient][pulse][seed] = pulse_results
            timings[patient][pulse][seed] = seed_timing

            # Store best overall info for summary (only keep the three keys)
            best_overall_summary = {
                "model": best_overall_key,
                "r2": best_overall_r2,
                "params": (
                    pulse_results["variograms"][best_overall_key]["params"]
                    if best_overall_key
                    else None
                ),
            }
            seed_summaries.append(best_overall_summary)

        # End seed loop: aggregate summary for this pulse (averaging over seeds)
        valid_seed_summaries = [s for s in seed_summaries if s.get("r2") is not None]
        if valid_seed_summaries:
            r2_values = [s["r2"] for s in valid_seed_summaries]
            model_names = [
                s["model"] for s in valid_seed_summaries if s["model"] is not None
            ]
            most_common_model = (
                Counter(model_names).most_common(1)[0][0] if model_names else None
            )
            best_seed_summary = max(valid_seed_summaries, key=lambda s: s["r2"])
            summary[patient][pulse] = {
                "model": most_common_model,
                "r2": {"mean": np.mean(r2_values), "std": np.std(r2_values)},
                "params": best_seed_summary.get("params"),
            }
            logging.info(f"Pulse {pulse} summary: {summary[patient][pulse]}")
        else:
            summary[patient][pulse] = {}
            logging.warning(
                f"No valid seed results for pulse {pulse} to aggregate a summary."
            )

        timings[patient][pulse]["total_pulse_time"] = time.time() - pulse_start_time

    logging.info("Completed processing all pulses.")
    return results, timings, summary


def main() -> None:
    """
    Main entry point for the extended variogram fitting pipeline.
    Processes the data for the given patient, writes the full model metadata
    (Variograms), execution timings, and a summary (averaged per pulse) to separate JSON files.
    The noise comparison metrics (slice-by-slice per seed) are saved to separate JSON files under
    the Noise_Comparison folder.
    """
    try:
        results, exec_timings, summary = run_variogram_fitting_pipeline(
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
        logging.error(f"Critical error during the pipeline: {e}")
        return

    # Save the Variograms results (only model, r2, params per pulse, aggregated across seeds)
    try:
        with open(OUTPUT_JSON_FILE, "w") as json_file:
            json.dump(summary, json_file, indent=4)
        logging.info(f"Variogram model summary saved to {OUTPUT_JSON_FILE}")
    except Exception as e:
        logging.error(f"Error saving variogram summary JSON file: {e}")

    # Save the execution timings
    try:
        with open(EXECUTION_TIME_FILE, "w") as json_file:
            json.dump(exec_timings, json_file, indent=4)
        logging.info(f"Execution time metadata saved to {EXECUTION_TIME_FILE}")
    except Exception as e:
        logging.error(f"Error saving execution time JSON file: {e}")

    # Optionally, also save the summary dictionary (here identical to the Variograms summary)
    try:
        with open(SUMMARY_JSON_FILE, "w") as json_file:
            json.dump(summary, json_file, indent=4)
        logging.info(f"Summary of best models per pulse saved to {SUMMARY_JSON_FILE}")
    except Exception as e:
        logging.error(f"Error saving summary JSON file: {e}")


if __name__ == "__main__":
    main()
