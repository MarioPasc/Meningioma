#!/usr/bin/env python
import os
import time
from datetime import datetime
import logging
import yaml  # type: ignore
import json
from typing import List, Optional

import numpy as np

# Meningioma imports (assuming local modules)
from Meningioma import ImageProcessing, BlindNoiseEstimation  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

BASE_OUTPUT_FOLDER = "/home/mariopascual/Projects/MENINGIOMA/Results"

# -----------------------------------------------------------------------------
# User-Defined Parameters (Adjust as needed)
# -----------------------------------------------------------------------------
PATIENT = "P50"
PULSE_TYPES = ["T1SIN", "T2", "SUSC", "T1"]
SEEDS = [123, 456, 789]
USE_VOXELS: bool = False
BASE_NRRD_PATH = "/media/hddb/mario/data/Meningioma_Adquisition"

# Percentage (fraction) of slices to remove *in total*.
# Example: 0.1 = remove 10% of slices. 5% at the beginning, 5% at the end.
IGNORE_FRACTION = 0.1

variogram_bins = np.linspace(0, 100, 100)
variogram_sampling_size = 3000
variogram_sampling_seed = 42
estimator = "cressie"
len_scale_guess = 10


def make_serializable(obj):
    """Recursively convert numpy types to pure Python types for JSON/YAML."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj


def compute_voxel_sizes(nrrd_file_path: str) -> Optional[List[Optional[float]]]:
    """Extract voxel sizes if 'space directions' is in the NRRD header."""
    _, header = ImageProcessing.open_nrrd_file(
        nrrd_path=nrrd_file_path, return_header=True
    )
    if "space directions" in header:
        space_directions = header["space directions"]
        voxel_sizes = []
        for direction in space_directions:
            if direction is not None:
                voxel_size = float(np.linalg.norm(direction))
            else:
                voxel_size = None
            voxel_sizes.append(voxel_size)
        return voxel_sizes
    else:
        logging.warning("No 'space directions' found in header.")
        return None


def main():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(BASE_OUTPUT_FOLDER, f"NoiseEstimation_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    logging.info(f"Output folder created at: {run_folder}")

    # We will record all results into this list, then write them into one JSON:
    results_for_json = []

    # Also record user input parameters in a YAML:
    user_input_params = {
        "patient": PATIENT,
        "pulses": PULSE_TYPES,
        "seeds": SEEDS,
        "ignore_fraction": IGNORE_FRACTION,
        "variogram_bins": variogram_bins.tolist(),
        "variogram_sampling_size": variogram_sampling_size,
        "variogram_sampling_seed": variogram_sampling_seed,
        "estimator": estimator,
        "len_scale_guess": len_scale_guess,
    }

    # -------------------------------------------------------------------------
    # Outer loop over pulses
    # -------------------------------------------------------------------------
    for pulse in PULSE_TYPES:
        # Build the path to the patient+pulse .nrrd
        nrrd_filepath = os.path.join(
            BASE_NRRD_PATH, "RM", pulse, PATIENT, f"{pulse}_{PATIENT}.nrrd"
        )
        logging.info(f"Loading NRRD file for pulse '{pulse}': {nrrd_filepath}")
        try:
            volume_nrrd, _ = ImageProcessing.open_nrrd_file(
                nrrd_path=nrrd_filepath, return_header=True
            )
        except Exception as e:
            logging.error(f"Failed to open file {nrrd_filepath}: {e}")
            continue

        # Compute voxel sizes if available
        voxel_sizes_list = compute_voxel_sizes(nrrd_filepath)
        if voxel_sizes_list and all(v is not None for v in voxel_sizes_list):
            voxel_sizes = tuple(voxel_sizes_list)  # type: ignore
        else:
            voxel_sizes = None

        # Segment volume (threshold) to get mask
        try:
            volume_data, mask_data = ImageProcessing.segment_3d_volume(
                volume_nrrd, threshold_method="li"
            )
        except Exception as e:
            logging.error(f"Segmentation failed for {nrrd_filepath}: {e}")
            continue

        if volume_data.shape != mask_data.shape:
            logging.error(
                f"Volume and mask shapes do not match. Skipping {nrrd_filepath}"
            )
            continue

        # ---------------------------------------------------------------------
        # Remove slices by percentage (ignore_fraction)
        # ---------------------------------------------------------------------
        nz = volume_data.shape[2]
        # total number of slices to ignore:
        total_ignore = int(round(IGNORE_FRACTION * nz))
        # split half to the start, half to the end
        ignore_start = total_ignore // 2  # integer division
        ignore_end = total_ignore - ignore_start

        # ensure a minimum of 1 slice at each end if possible
        if ignore_start < 1 and nz > 2:
            ignore_start = 1
        if ignore_end < 1 and nz > 2:
            ignore_end = 1

        # clamp if ignoring more slices than we have
        if ignore_start + ignore_end >= nz:
            logging.warning(
                f"Requested ignoring {ignore_start + ignore_end} slices, "
                f"but volume has only {nz} in z-dim. Skipping slice removal."
            )
        else:
            logging.info(
                f"Ignoring {ignore_start} slices from start, "
                f"{ignore_end} from end (total {ignore_start + ignore_end})."
            )
            volume_data = volume_data[:, :, ignore_start : nz - ignore_end]
            mask_data = mask_data[:, :, ignore_start : nz - ignore_end]

        # ---------------------------------------------------------------------
        # Stats inside vs. outside
        # ---------------------------------------------------------------------
        inside_pixels = volume_data[mask_data]
        outside_pixels = volume_data[~mask_data]
        inside_mean = inside_pixels.mean()
        inside_std = inside_pixels.std()
        outside_mean = outside_pixels.mean()
        outside_std = outside_pixels.std()
        var_guess = float(np.var(outside_pixels)) if outside_pixels.size > 0 else 0.0

        logging.info(
            f"[{pulse}] inside_mask mean={inside_mean:.3f}, std={inside_std:.3f}"
        )
        logging.info(
            f"[{pulse}] outside_mask mean={outside_mean:.3f}, std={outside_std:.3f}"
        )

        # ---------------------------------------------------------------------
        # Isotropic Variogram Estimation
        # ---------------------------------------------------------------------
        iso_bin_center, iso_gamma = (
            BlindNoiseEstimation.estimate_variogram_isotropic_3d(
                data=volume_data,
                bins=variogram_bins,
                mask=mask_data,
                estimator=estimator,
                sampling_size=variogram_sampling_size,
                sampling_seed=variogram_sampling_seed,
            )
        )

        # Fit covariance models
        iso_models = BlindNoiseEstimation.fit_model_3d(
            bin_center=iso_bin_center,
            gamma=iso_gamma,
            var=var_guess,
            len_scale=len_scale_guess,
        )

        # Pick the best model by R²
        best_r2 = -np.inf
        best_model_name = None
        best_model = None
        best_model_info = None

        for model_name, (model_obj, info) in iso_models.items():
            if info["r2"] > best_r2:
                best_r2 = info["r2"]
                best_model_name = model_name
                best_model = model_obj
                best_model_info = info

        if not best_model:
            logging.warning(f"No best model found for pulse {pulse}, skipping.")
            continue

        logging.info(f"[{pulse}] Best model: {best_model_name} with R²={best_r2:.4f}")
        logging.info(f"[{pulse}] Model params: {best_model_info['params']}")

        # ---------------------------------------------------------------------
        # Inner loop: Different seeds for noise generation
        # ---------------------------------------------------------------------
        for seed in SEEDS:
            # Generate a 3D noise volume using best_model
            logging.info(f"[{pulse}, seed={seed}] Generating 3D noise volume...")
            logging.info(f"Volume shape: {volume_data.shape}")
            if USE_VOXELS:
                real_vol, imag_vol, final_vol = (
                    BlindNoiseEstimation.gaussian_random_fields_noise_3d(
                        model=best_model,
                        shape=volume_data.shape,
                        voxel_size=voxel_sizes,
                        seed_real=seed,
                        seed_imag=seed + 200,
                    )
                )

            else:
                real_vol, imag_vol, final_vol = (
                    BlindNoiseEstimation.gaussian_random_fields_noise_3d(
                        model=best_model,
                        shape=volume_data.shape,
                        voxel_size=None,
                        seed_real=seed,
                        seed_imag=seed + 200,
                    )
                )

            # Some stats on the generated final noise
            gen_mean = float(final_vol.mean())
            gen_std = float(final_vol.std())
            logging.info(f"Generated shape: {final_vol.shape}")
            logging.info(f"Generated mean: {gen_mean}")
            logging.info(f"Generated std: {gen_std}")

            # -------------------------------------------------------------
            # Save results to NPZ: we store x,y,z arrays + a 4D data array
            # -------------------------------------------------------------
            nx, ny, nz = volume_data.shape
            x_coords = np.arange(nx)
            y_coords = np.arange(ny)
            z_coords = np.arange(nz)

            # 4D array: data[:,:,:,0] = volume, data[:,:,:,1] = mask, ...
            combined_array = np.zeros((nx, ny, nz, 5), dtype=np.float32)
            combined_array[..., 0] = volume_data
            combined_array[..., 1] = mask_data.astype(np.float32)
            combined_array[..., 2] = real_vol
            combined_array[..., 3] = imag_vol
            combined_array[..., 4] = final_vol

            npz_filename = f"noise_volume_{pulse}_seed{seed}.npz"
            npz_filepath = os.path.join(run_folder, npz_filename)
            np.savez_compressed(
                npz_filepath, x=x_coords, y=y_coords, z=z_coords, data=combined_array
            )
            logging.info(f"Saved NPZ: {npz_filepath}")

            # -------------------------------------------------------------
            # Accumulate info for JSON
            # -------------------------------------------------------------
            result_entry = {
                "patient": PATIENT,
                "pulse": pulse,
                "seed": seed,
                "covariance_model": [
                    {
                        "model_name": best_model_name,
                        "params": best_model_info["params"],
                        "r2": best_model_info["r2"],
                    }
                ],
                "original_volume_stats": [
                    {
                        "mean_outside_mask": float(outside_mean),
                        "std_outside_mask": float(outside_std),
                        "mean_inside_mask": float(inside_mean),
                        "std_inside_mask": float(inside_std),
                    }
                ],
                "generated_volume_stats": [{"mean": gen_mean, "std": gen_std}],
            }
            results_for_json.append(result_entry)

    # -------------------------------------------------------------------------
    # Write JSON with all results
    # -------------------------------------------------------------------------
    json_path = os.path.join(run_folder, "results.json")
    with open(json_path, "w") as jf:
        safe_results = make_serializable(results_for_json)
        json.dump(safe_results, jf, indent=2)
    logging.info(f"Saved JSON with results: {json_path}")

    # -------------------------------------------------------------------------
    # Write YAML with user input parameters
    # -------------------------------------------------------------------------
    yaml_path = os.path.join(run_folder, "run_parameters.yaml")
    with open(yaml_path, "w") as yf:
        yaml.dump(make_serializable(user_input_params), yf)
    logging.info(f"Saved YAML with user input parameters: {yaml_path}")

    elapsed = time.time() - start_time
    logging.info(f"Pipeline completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
