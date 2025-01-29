import os
import json
import logging
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

import gstools as gs

from Meningioma import ImageProcessing

# Example dictionary of models, if not already provided inside ImageProcessing
MODEL_CLASSES: Dict[str, Any] = {
    "Gaussian": gs.Gaussian,
    "Exponential": gs.Exponential,
    "Matern": gs.Matern,
    "Stable": gs.Stable,
    "Rational": gs.Rational,
    "Circular": gs.Circular,
    "Spherical": gs.Spherical,
    "SuperSpherical": gs.SuperSpherical,
    "JBessel": gs.JBessel,
    "TLPGaussian": gs.TPLGaussian,
    "TLPSTable": gs.TPLStable,
    "TLPSimple": gs.TPLSimple,
}


def discover_nrrd_files(root_folder: str) -> List[Tuple[str, str, str, str]]:
    """
    Recursively search `root_folder` for .nrrd files with structure:
        root_folder/RM/{SUSC|T1|T1SIN|T2}/P{patient_id}/{pulse}_P{patient_id}.nrrd

    Returns (pulse_name, patient_id, full_file_path, subdir).
    """
    found_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for f in filenames:
            if f.endswith(".nrrd"):
                rel_path = os.path.relpath(os.path.join(dirpath, f), root_folder)
                parts = rel_path.split(os.sep)
                # Expecting something like [ 'RM', 'T1', 'P5', 'T1_P5.nrrd' ]
                if len(parts) >= 4 and parts[0] == "RM":
                    subdir = parts[1]  # e.g. "SUSC", "T1", ...
                    patient_folder = parts[2]  # e.g. "P5"
                    patient_id = patient_folder.replace("P", "")
                    # We call pulse = subdir for simplicity
                    pulse = subdir
                    full_path = os.path.join(dirpath, f)
                    found_files.append((pulse, patient_id, full_path, subdir))
    return found_files


def run_experiment_to_json(
    root_folder: str,
    angles_deg: List[int],
    bins: np.ndarray,
    output_json: str,
    sampling_size: int = 3000,
    sampling_seed: int = 19920516,
    angles_tol: float = np.pi / 8,
) -> None:
    """
    Main experiment driver. Builds a nested dictionary and dumps it to JSON.
    Added verbose logging so the user knows what's happening at every stage.
    """
    # --- Start-of-run logs for clarity ---
    logging.info("=== Starting run_experiment_to_json ===")
    logging.info(f"Root folder: {root_folder}")
    logging.info(f"Output JSON: {output_json}")
    logging.info(f"Angles (deg): {angles_deg}")
    logging.info(f"Distance bins: {bins}")
    logging.info(f"Sampling size: {sampling_size}")
    logging.info(f"Sampling seed: {sampling_seed}")
    logging.info(f"Angles tolerance: {angles_tol}")

    logging.info("Discovering .nrrd files in the specified root folder...")
    file_tuples = discover_nrrd_files(root_folder)

    # Prepopulate main structure
    experiment_dict: Dict[Any, Any] = {
        "Variogram_Experiment": {
            "SUSC": {},
            "T1": {},
            "T1SIN": {},
            "T2": {},
        }
    }

    if not file_tuples:
        logging.info("No files found. Exiting. Writing empty JSON structure.")
        with open(output_json, "w") as jf:
            json.dump(experiment_dict, jf, indent=2)
        return

    # We'll gather a quick summary to log how many files/patients we have per pulse
    pulse_to_patients: Dict[Any, Any] = {}
    for pulse, patient_id, file_path, subdir in file_tuples:
        if pulse not in pulse_to_patients:
            pulse_to_patients[pulse] = set()
        pulse_to_patients[pulse].add(patient_id)

    # Log the summary
    for pulse, patients_set in pulse_to_patients.items():
        logging.info(f"Pulse '{pulse}' => # of patients: {len(patients_set)}")

    # Process each file
    for pulse, patient_id, file_path, subdir in tqdm(
        file_tuples, desc="Processing NRRD files"
    ):
        logging.info(
            f"--- Processing: Pulse={pulse}, Patient={patient_id}, File={file_path}"
        )
        # 1) Load 3D data
        logging.info("Loading 3D NRRD data...")
        img_3d = ImageProcessing.open_nrrd_file(file_path, return_header=False)

        # 2) Extract transversal slice
        logging.info("Extracting transversal slice...")
        t_axis = ImageProcessing.get_transversal_axis(file_path)
        slice_2d = ImageProcessing.extract_transversal_slice(
            img_3d, t_axis, slice_index=-1
        )

        # 3) Create a mask
        logging.info("Creating convex hull mask of the slice...")
        hull = ImageProcessing.convex_hull_mask(image=slice_2d, threshold_method="li")
        mask = hull > 0

        # 4) Heuristic: variance guess
        logging.info("Computing variance guess from masked region...")
        masked_values = slice_2d[mask]
        var_guess = float(np.var(masked_values)) if len(masked_values) > 1 else 1.0
        len_scale_guess = 50.0

        # Ensure the subdir key is in the dictionary
        # If subdir is not recognized (not in {SUSC, T1, T1SIN, T2}), we create it
        if subdir not in experiment_dict["Variogram_Experiment"]:
            experiment_dict["Variogram_Experiment"][subdir] = {}
        if patient_id not in experiment_dict["Variogram_Experiment"][subdir]:
            experiment_dict["Variogram_Experiment"][subdir][patient_id] = {}

        patient_dict = experiment_dict["Variogram_Experiment"][subdir][patient_id]

        # --- 5) Isotropic
        logging.info("Estimating isotropic variogram...")
        iso_bin, iso_gamma = ImageProcessing.estimate_isotropic_variogram(
            data=slice_2d,
            bins=bins,
            mask=mask,
            sampling_size=sampling_size,
            sampling_seed=sampling_seed,
        )
        logging.info("Fitting covariance models to isotropic variogram...")
        iso_fits = ImageProcessing.fit_covariance_models(
            bin_center=iso_bin,
            gamma=iso_gamma,
            var=var_guess,
            len_scale=len_scale_guess,
            model_classes=MODEL_CLASSES,
        )

        iso_key = "Isotropic"
        patient_dict[iso_key] = {
            "bin_centers": iso_bin.tolist(),
            "gamma": iso_gamma.tolist(),
            "models": {},
        }
        logging.info(f"Saving isotropic results under '{iso_key}'...")

        for model_name, (model_obj, fit_data) in iso_fits.items():
            pcov_array = fit_data["pcov"]
            pcov_list = pcov_array.tolist() if pcov_array is not None else None
            params_dict = dict(fit_data["params"])  # ensure plain dict
            r2_val = float(fit_data["r2"])
            patient_dict[iso_key]["models"][model_name] = {
                "params": params_dict,
                "pcov": pcov_list,
                "r2": r2_val,
            }

        # --- 6) Anisotropic for each angle
        logging.info(f"Estimating anisotropic variograms for angles = {angles_deg}...")
        for angle in angles_deg:
            logging.info(f"Computing anisotropic variogram at {angle} degrees...")
            angle_rad = np.deg2rad(angle)
            direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            bin_a, gamma_a = ImageProcessing.estimate_anisotropic_variogram(
                data=slice_2d,
                bins=bins,
                direction=direction,
                mask=mask,
                angles_tol=angles_tol,
                sampling_size=sampling_size,
                sampling_seed=sampling_seed,
            )
            logging.info(
                f"Fitting covariance models to anisotropic variogram @ {angle} deg..."
            )
            aniso_fits = ImageProcessing.fit_covariance_models(
                bin_center=bin_a,
                gamma=gamma_a,
                var=var_guess,
                len_scale=len_scale_guess,
                model_classes=MODEL_CLASSES,
            )

            vario_key = f"Anisotropic_{angle}_degree"
            patient_dict[vario_key] = {
                "bin_centers": bin_a.tolist(),
                "gamma": gamma_a.tolist(),
                "models": {},
            }
            logging.info(f"Saving anisotropic results under '{vario_key}'...")

            for model_name, (model_obj, fit_data) in aniso_fits.items():
                pcov_array = fit_data["pcov"]
                pcov_list = pcov_array.tolist() if pcov_array is not None else None
                params_dict = dict(fit_data["params"])
                r2_val = float(fit_data["r2"])
                patient_dict[vario_key]["models"][model_name] = {
                    "params": params_dict,
                    "pcov": pcov_list,
                    "r2": r2_val,
                }

    # Final write-out
    logging.info(f"Writing experiment results to JSON => {output_json}")
    with open(output_json, "w") as jf:
        json.dump(experiment_dict, jf, indent=2)

    logging.info("=== Experiment completed successfully ===")


def main():
    """
    Example main function that sets up logging, defines angles, bins, etc.,
    then runs the experiment and dumps to a JSON file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    root_folder: str = "/data" # in-contained path for the dataset
    output_json: str = "/out/variogram_experiment_results.json" # in-contained path for the results file
    angles_deg: List[int] = [0, 45, 90, 135]
    bins: np.array = np.linspace(0, 150, 151)  # e.g. 150 bins up to distance 150
    sampling_size: int =3000
    sampling_seed: int =19920516
    angles_tol: float =np.pi / 8
    
    run_experiment_to_json(
        root_folder=root_folder,
        angles_deg=angles_deg,
        bins=bins,
        output_json=output_json,
        sampling_size=sampling_size,
        sampling_seed=sampling_seed,
        angles_tol=angles_tol,
    )


if __name__ == "__main__":
    main()
