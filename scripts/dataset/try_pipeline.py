from Meningioma.preprocessing.tools.remove_extra_channels import remove_first_channel
from Meningioma.preprocessing.tools.nrrd_to_nifti import nifti_write_3d
from Meningioma.preprocessing.tools.reorient import reorient_images
from Meningioma.preprocessing.tools.resample import resample_images
from Meningioma.preprocessing.tools.casting import cast_volume_and_mask
from Meningioma.preprocessing.tools.denoise_susan import denoise_susan
from Meningioma.preprocessing.tools.bias_field_corr_n4 import (
    n4_bias_field_correction,
    n4_bias_field_correction_monitored,
    generate_brain_mask_sitk,
)
from Meningioma.preprocessing.tools.skull_stripping.ants_bet import (
    ants_brain_extraction,
)
from Meningioma.preprocessing.tools.skull_stripping.fsl_bet import (
    fsl_bet_brain_extraction,
)
from Meningioma.preprocessing.tools.padding import pad_in_plane
from Meningioma.preprocessing.tools.registration.ants_sri24_reg import (
    register_to_sri24_with_mask,
)

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import time

from typing import List, Tuple, Dict, Any

PULSES: List[str] = ["T1", "T1SIN", "T2", "SUSC"]
PATIENT: str = "P1"
ROOT: str = "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition"
PROCESSED_DATASET_FOLDER: str = (
    "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Processed"
)
ATLAS_ROOT: str = (
    "/home/mariopasc/Python/Datasets/Meningiomas/ATLAS/sri24_spm8/templates"
)


def pipeline(
    pulses: List[str] = PULSES,
    traspose_tuple: Tuple[int, int, int] = (2, 1, 0),
    denoise: bool = False,
) -> Dict[str, Any]:

    results = {}

    for pulse in pulses:
        print(f"========== {pulse} ==========")

        volume_nrrd_path = os.path.join(
            ROOT, "RM", pulse, PATIENT, f"{pulse}_{PATIENT}.nrrd"
        )
        mask_path = os.path.join(
            ROOT, "RM", pulse, PATIENT, f"{pulse}_{PATIENT}_seg.nrrd"
        )

        patient_new_path = os.path.join(PROCESSED_DATASET_FOLDER, "RM", pulse, PATIENT)
        patient_new_volume = os.path.join(patient_new_path, f"{pulse}_{PATIENT}.nii.gz")
        os.makedirs(patient_new_path, exist_ok=True)

        t0 = time.time()
        # (Optional) Remove channel if multi-component
        volume_original, header_nrrd = remove_first_channel(
            nrrd_path=volume_nrrd_path, channel=0, verbose=True
        )

        # Pre. Load original volume and mask
        mask = sitk.ReadImage(mask_path)

        # Pre. Save the original volume as niigz
        nifti_write_3d(
            volume=(volume_original, header_nrrd),
            out_file=patient_new_volume,
            verbose=True,
        )

        print(f"Wrote nifti file in {time.time() - t0} seconds")
        t0 = time.time()

        # Read volume & mask
        volume_img = sitk.ReadImage(patient_new_volume)

        # Cast
        cast_vol, cast_mask = cast_volume_and_mask(volume_img, mask)

        # (Optional) Reorient
        reor_vol, reor_mask = reorient_images(cast_vol, cast_mask, orientation="LPS")

        # Resample to 1 mm isotropic
        resampled_volume, resampled_mask = resample_images(
            reor_vol, reor_mask, new_spacing=(1, 1, 1)
        )

        print(f"Cast, reorient, and resample in {time.time() - t0} seconds")

        if denoise:
            t0 = time.time()
            susan_vol = denoise_susan(
                image_sitk=resampled_volume,
                brightness_threshold=0.001,
                fwhm=0.5,
                dimension=3,
                verbose=True,
            )
            print(f"Denoising in {time.time() - t0} seconds")

        t0 = time.time()
        _, mask_brain = generate_brain_mask_sitk(
            volume_sitk=resampled_volume,
            threshold_method="li",
            structure_size_2d=7,
            iterations_2d=3,
            structure_size_3d=3,
            iterations_3d=1,
        )

        corrected_vol, final_field, conv_log = n4_bias_field_correction_monitored(
            image_sitk=resampled_volume,
            shrink_factor=4,
            max_iterations=100,
            control_points=6,
            bias_field_fwhm=0.1,
            mask_sitk=mask_brain,
            verbose=True,
        )
        print(f"Anatomical parts mask and BFCN4 in {time.time() - t0} seconds")

        t0 = time.time()

        extracted_brain_fsl, extracted_mask_fsl = fsl_bet_brain_extraction(
            input_image_sitk=corrected_vol,
            frac=0.5,
            robust=True,
            vertical_gradient=0,
            skull=False,
            verbose=True,
        )
        print(f"Brain mask in {time.time() - t0} seconds")

        # For anatomical pulses, register to SRI24
        if pulse in ["T1", "T2"]:
            t0 = time.time()
            atlas_path = f"{ATLAS_ROOT}/{pulse}_brain.nii"
            if not os.path.exists(atlas_path):
                raise FileNotFoundError(f"Atlas not found: {atlas_path}")
            atlas_img = sitk.ReadImage(atlas_path)
            # Call the registration function (assumed to be defined/imported)
            reg_vol, reg_mask, reg_params = register_to_sri24_with_mask(
                moving_image_sitk=extracted_brain_fsl,
                moving_mask_sitk=resampled_mask,
                fixed_image_sitk=atlas_img,
                output_dir=patient_new_path,
                output_transform_prefix=f"{pulse}_{PATIENT}_transform",
                output_image_prefix=f"{pulse}_{PATIENT}_reg_volume",
                output_mask_prefix=f"{pulse}_{PATIENT}_reg_mask",
                num_threads=2,
                verbose=True,
            )
            print(f"Registration to SRI24 completed in {time.time() - t0} seconds")
        else:
            reg_vol, reg_mask, reg_params = None, None, {}
        #### STORE THE DATA ####

        # Transpose accordingly to return [X, Y, Z]:
        volume_original_array = sitk.GetArrayFromImage(volume_original).transpose(
            traspose_tuple
        )

        ## processed ##
        # Transpose accordingly to return [X, Y, Z]:
        stg1_volume_processed_array = sitk.GetArrayFromImage(
            resampled_volume
        ).transpose(traspose_tuple)
        mask_processed_array = sitk.GetArrayFromImage(resampled_mask).transpose(
            traspose_tuple
        )

        print(
            f"Volume processed shape: {stg1_volume_processed_array.shape}\nMask processed shape: {mask_processed_array.shape}"
        )
        indexes_segmentation = [
            idx
            for idx in range(mask_processed_array.shape[2])
            if np.max(mask_processed_array[:, :, idx]) > 0
        ]
        print(f"{pulse} indexes with segmentation:\n{indexes_segmentation}")
        print("===========================")
        ## Bias field corrected ##
        # Transpose accordingly to return [X, Y, Z]:
        volume_bfcn4_array = sitk.GetArrayFromImage(corrected_vol).transpose(
            traspose_tuple
        )
        field_bfcn4_array = sitk.GetArrayFromImage(final_field).transpose(
            traspose_tuple
        )
        brain_mask_array = sitk.GetArrayFromImage(mask_brain).transpose(traspose_tuple)
        ## Skull stripping ##
        extracted_brain_array = sitk.GetArrayFromImage(extracted_brain_fsl).transpose(
            traspose_tuple
        )
        extracted_mask_fsl_brain_array = sitk.GetArrayFromImage(
            extracted_mask_fsl
        ).transpose(traspose_tuple)
        ## Registration ##
        if reg_vol and reg_mask:
            registered_volume_array = sitk.GetArrayFromImage(reg_vol).transpose(
                traspose_tuple
            )
            registered_mask_array = sitk.GetArrayFromImage(reg_mask).transpose(
                traspose_tuple
            )
        else:
            registered_volume_array = []
            registered_mask_array = []

        df = pd.DataFrame(
            data={
                "Volume (Original)": [
                    volume_original_array.shape,
                    volume_original_array.dtype,
                    np.max(volume_original_array),
                    np.min(volume_original_array),
                ],
                "Volume (Processed)": [
                    stg1_volume_processed_array.shape,
                    stg1_volume_processed_array.dtype,
                    np.max(stg1_volume_processed_array),
                    np.min(stg1_volume_processed_array),
                ],
                "Mask (Processed)": [
                    mask_processed_array.shape,
                    mask_processed_array.dtype,
                    np.max(mask_processed_array),
                    np.min(mask_processed_array),
                ],
            },
            index=["Shape", "dtype", "max", "min"],
        )

        key = f"{PATIENT}_{pulse}"

        entry = {
            "original_volume": volume_original_array,
            "stg1_volume": stg1_volume_processed_array,
            "bfcn4": {
                "vol": volume_bfcn4_array,
                "final_field": field_bfcn4_array,
                "conv_log": conv_log,
            },
            "skull_stripping": {"fsl": extracted_brain_array},
            "sri24_registration": {
                "vol": registered_volume_array,
                "params": reg_params,
            },
            "mask": {
                "vol": mask_processed_array,
                "indexes_segmentation": indexes_segmentation,
                "brain_mask": brain_mask_array,
                "skull_stripping_fsl": extracted_mask_fsl_brain_array,
                "registered_mask": registered_mask_array,
            },
        }

        results[key] = entry

    return results


results = pipeline()

import matplotlib.pyplot as plt

# Example pulses and patient (these should match what you used in the pipeline)
PULSES = ["T1", "T1SIN", "T2", "SUSC"]
PATIENT = "P1"


def visualize_planes_with_same_segmentation_slice(
    results, pulses, patient, data_key, seg_key=("mask", "vol")
):
    """
    Visualizes the transversal (axial), coronal, and sagittal planes for a given volume
    in the pipeline results, overlaying the segmentation mask on each view.

    Rather than taking the middle slice of each dimension independently, this function
    computes the center-of-mass of the segmentation mask (i.e., the lesion) and uses that
    coordinate for all three planes. This ensures the lesion is visible in all views.

    Parameters:
      results (dict): The pipeline results dictionary.
      pulses (list): List of pulse types (e.g., ["T1", "T1SIN", "T2", "SUSC"]).
      patient (str): Patient identifier.
      data_key (str or tuple): Key to access the desired volume.
          - If a tuple is provided (e.g., ("bfcn4", "vol")), then it is accessed via:
              results[f"{patient}_{pulse}"][data_key[0]][data_key[1]]
          - If a string is provided, then via results[f"{patient}_{pulse}"][data_key].
      seg_key (str or tuple): Key to access the segmentation mask. Defaults to ("mask", "vol").

    The function creates a grid with:
      - 4 rows (one per pulse, with the pulse type as the row label)
      - 3 columns: Transversal (Axial), Coronal, and Sagittal.
    """
    num_pulses = len(pulses)
    fig, axes = plt.subplots(nrows=num_pulses, ncols=3, figsize=(12, 3 * num_pulses))

    # Set column titles on the top row.
    col_titles = ["Transversal (Axial)", "Coronal", "Sagittal"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12)

    for i, pulse in enumerate(pulses):
        key = f"{patient}_{pulse}"

        # Retrieve volume data
        if isinstance(data_key, tuple):
            volume = results[key][data_key[0]][data_key[1]]
        else:
            volume = results[key][data_key]

        # Retrieve segmentation mask data
        if isinstance(seg_key, tuple):
            seg_mask = results[key][seg_key[0]][seg_key[1]]
        else:
            seg_mask = results[key][seg_key]

        # Compute the center-of-mass of the lesion using the segmentation mask.
        lesion_coords = np.where(seg_mask > 0)
        if lesion_coords[0].size > 0:
            center_x = int(np.mean(lesion_coords[0]))
            center_y = int(np.mean(lesion_coords[1]))
            center_z = int(np.mean(lesion_coords[2]))
        else:
            # Fallback to the volume center if no segmentation is present.
            center_x = volume.shape[0] // 2
            center_y = volume.shape[1] // 2
            center_z = volume.shape[2] // 2

        # --- Transversal (Axial) plane: use center_z ---
        ax_trans = axes[i, 0]
        vol_slice = volume[:, :, center_z]
        seg_slice = seg_mask[:, :, center_z]
        ax_trans.imshow(vol_slice, cmap="gray", interpolation="none")
        ax_trans.imshow(
            np.ma.masked_where(seg_slice == 0, seg_slice),
            cmap="Reds_r",
            alpha=0.5,
            interpolation="none",
        )
        ax_trans.set_ylabel(pulse, fontsize=10)  # Label with pulse type.
        # ax_trans.axis("off")

        # --- Coronal plane: use center_y ---
        ax_cor = axes[i, 1]
        vol_slice = volume[:, center_y, :]
        seg_slice = seg_mask[:, center_y, :]
        # Rotate for natural orientation.
        ax_cor.imshow(np.rot90(vol_slice), cmap="gray", interpolation="none")
        ax_cor.imshow(
            np.rot90(np.ma.masked_where(seg_slice == 0, seg_slice)),
            cmap="Reds_r",
            alpha=0.5,
            interpolation="none",
        )
        # ax_cor.axis("off")

        # --- Sagittal plane: use center_x ---
        ax_sag = axes[i, 2]
        vol_slice = volume[center_x, :, :]
        seg_slice = seg_mask[center_x, :, :]
        ax_sag.imshow(np.rot90(vol_slice), cmap="gray", interpolation="none")
        ax_sag.imshow(
            np.rot90(np.ma.masked_where(seg_slice == 0, seg_slice)),
            cmap="Reds_r",
            alpha=0.5,
            interpolation="none",
        )
        # ax_sag.axis("off")

    plt.tight_layout()
    plt.show()


#############################
# 1) STEP-BY-STEP VISUALIZATION
#############################
def visualize_pipeline_steps(results_dict, pulses, patient, axis: int = 2):
    """
    Displays a step-by-step view of the pipeline in 4 columns:
    1) Original volume
    2) Processed volume (or your first stage after reorient/resample if you store it)
    3) stg1_volume_processed_array
    4) volume_bfcn4_array

    Rows = different pulses (T1, T1SIN, T2, SUSC).
    """
    # Adapt this list if you add more steps
    steps = [
        ("Original", "original_volume"),
        # If you have a separate entry for an intermediate volume, replace "processed_volume" below
        ("Processed", "stg1_volume"),
        ("STAGE 1 Preproc", "stg1_volume"),
        ("BFCN4", ("bfcn4", "vol")),
        ("SkullStripping FSL", ("skull_stripping", "fsl")),
        ("Registration", ("sri24_registration", "vol")),
    ]

    # Create subplots
    fig, axes = plt.subplots(nrows=len(pulses), ncols=len(steps), figsize=(16, 10))

    for row, pulse in enumerate(pulses):
        key = f"{patient}_{pulse}"

        # Retrieve the middle slice from indexes_segmentation
        mask_lesion = results_dict[key]["mask"]["vol"]
        if axis == 2:
            seg_indexes = results_dict[key]["mask"]["indexes_segmentation"]
        else:
            seg_indexes = [
                idx
                for idx in range(mask_lesion.shape[axis])
                if np.max(np.take(mask_lesion, idx, axis=axis)) > 0
            ]

        print(f"{axis}: {seg_indexes}")

        if seg_indexes:
            mid_idx = seg_indexes[len(seg_indexes) // 2]
        else:
            # Fallback if no segmentation slices exist
            mid_idx = results_dict[key]["original_volume"].shape[2] // 2

        for col, (step_name, dict_path) in enumerate(steps):
            # Some steps might be two-level dictionary references (e.g. bfcn4 -> vol)
            if isinstance(dict_path, tuple):
                array_data = results_dict[key][dict_path[0]][dict_path[1]]
            else:
                array_data = results_dict[key][dict_path]

            ax = axes[row, col]
            ax.imshow(
                np.take(array_data, indices=mid_idx, axis=axis),
                cmap="gray",
                interpolation="none",
            )
            mask_slice = np.take(mask_lesion, mid_idx, axis=axis)
            ax.imshow(
                np.ma.masked_where(mask_slice == 0, mask_slice),
                cmap="Reds_r",
                alpha=0.5,
                interpolation="none",
            )
            ax.set_title(f"{pulse} - {step_name}", fontsize=10)
            # ax.axis("off")

    plt.tight_layout()
    plt.show()


#############################
# 2) BFCN4 VISUALIZATION
#############################
def visualize_bfcn4(results_dict, pulses, patient, axis: int = 2):
    """
    Shows, for each pulse (row):
    1) BFCN4-corrected volume slice (with optional overlay of the brain mask).
    2) The final field slice.
    3) The convergence log plot.
    """
    # Create subplots: 4 rows (one per pulse), 3 columns
    fig, axes = plt.subplots(nrows=len(pulses), ncols=3, figsize=(15, 10))

    for row, pulse in enumerate(pulses):
        key = f"{patient}_{pulse}"

        # Retrieve the middle slice from indexes_segmentation
        mask_lesion = results_dict[key]["mask"]["vol"]
        if axis == 2:
            seg_indexes = results_dict[key]["mask"]["indexes_segmentation"]
        else:
            seg_indexes = [
                idx
                for idx in range(mask_lesion.shape[axis])
                if np.max(np.take(mask_lesion, idx, axis=axis)) > 0
            ]
        if seg_indexes:
            mid_idx = seg_indexes[len(seg_indexes) // 2]
        else:
            mid_idx = results_dict[key]["original_volume"].shape[2] // 2

        # BFCN4-corrected volume and final field
        bfcn4_vol = results_dict[key]["bfcn4"]["vol"]
        bfcn4_field = results_dict[key]["bfcn4"]["final_field"]
        conv_log = results_dict[key]["bfcn4"]["conv_log"]

        # If you want to overlay the brain mask, retrieve it
        # (If you saved the actual brain mask in your dictionary; adjust accordingly)
        mask_brain = results_dict[key]["mask"]["brain_mask"]

        # --- Column 1: BFCN4 volume (with mask overlay, if desired) ---
        axes[row, 0].imshow(
            np.take(bfcn4_vol, mid_idx, axis=axis), cmap="gray", interpolation="none"
        )
        # Overlay mask in red, for example:
        mask_slice = np.take(mask_brain, mid_idx, axis=axis)
        axes[row, 0].imshow(
            np.ma.masked_where(mask_slice == 0, mask_slice),
            cmap="Reds_r",
            alpha=0.5,
            interpolation="none",
        )
        axes[row, 0].set_title(f"{pulse} - BFCN4", fontsize=10)
        # axes[row, 0].axis("off")

        # --- Column 2: Final N4 Field ---
        axes[row, 1].imshow(
            np.take(bfcn4_field, mid_idx, axis=axis), cmap="jet", interpolation="none"
        )
        axes[row, 1].set_title("N4 Field", fontsize=10)
        # axes[row, 1].axis("off")

        # --- Column 3: Convergence plot ---
        for level, data_list in conv_log.items():
            # Sort by iteration (if not already)
            data_list = sorted(data_list, key=lambda x: x[0])
            iters = [x[0] for x in data_list]
            costs = [x[1] for x in data_list]
            axes[row, 2].plot(
                iters, costs, marker="o", linestyle="-", label=f"Level {level}"
            )

        axes[row, 2].set_xlabel("Iteration", fontsize=10)
        axes[row, 2].set_ylabel("Cost", fontsize=10)
        axes[row, 2].set_title("N4 Convergence", fontsize=10)
        axes[row, 2].grid(True)
        axes[row, 2].legend(fontsize=8)

    plt.tight_layout()
    plt.show()


##############################################
# Example usage (assuming 'results' is ready):
##############################################

visualize_planes_with_same_segmentation_slice(
    results, PULSES, PATIENT, data_key=("skull_stripping", "fsl")
)

# 1) Pipeline steps visualization
visualize_pipeline_steps(results, PULSES, PATIENT, axis=2)

# 2) BFCN4 visualization
visualize_bfcn4(results, PULSES, PATIENT)

import numpy as np
import matplotlib.pyplot as plt


def visualize_pre_post_planes(entry, pulse):
    """
    Visualizes the Transversal (Axial), Coronal, and Sagittal planes for both pre‐registration
    and post‐registration volumes, overlaying the segmentation mask on each view.

    The first row corresponds to pre‐registration data, and the second row to post‐registration data.

    Parameters:
      entry (dict): Dictionary containing the image arrays.
         Expected keys:
           - Pre-registration volume: entry['skull_stripping']['fsl']
           - Pre-registration mask: entry['mask']['skull_stripping_fsl']
           - Post-registration volume: entry['sri24_registration']['vol']
           - Post-registration mask: entry['mask']['registered_mask']
      pulse (str): Pulse type (e.g., "T1" or "T2"), used for the output filename.
    """
    # Retrieve pre-registration data
    pre_volume = entry["skull_stripping"]["fsl"]
    pre_mask = entry["mask"]["vol"]

    # Retrieve post-registration data
    post_volume = entry["sri24_registration"]["vol"]
    post_mask = entry["mask"]["registered_mask"]

    # Set up figure grid: 2 rows (pre, post) x 3 columns (axial, coronal, sagittal)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))

    # Set axes background to black and tick colors to white
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            ax.set_facecolor("black")
            ax.tick_params(colors="white")

    # Column titles (set on the top row)
    col_titles = ["Transversal (Axial)", "Coronal", "Sagittal"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, color="white")

    # Function to compute the center-of-mass of the segmentation mask
    def compute_center(seg):
        coords = np.where(seg > 0)
        if coords[0].size > 0:
            center_x = int(np.mean(coords[0]))
            center_y = int(np.mean(coords[1]))
            center_z = int(np.mean(coords[2]))
        else:
            center_x = seg.shape[0] // 2
            center_y = seg.shape[1] // 2
            center_z = seg.shape[2] // 2
        return center_x, center_y, center_z

    # Compute centers for pre- and post-registration masks
    pre_center = compute_center(pre_mask)
    post_center = compute_center(post_mask)

    # Plot function for a given volume and mask using a specified center coordinate
    def plot_planes(ax_row, volume, mask, center):
        cx, cy, cz = center

        # --- Transversal (Axial) plane: slice along the third dimension ---
        ax = axes[ax_row, 0]
        vol_slice = volume[:, :, cz]
        mask_slice = mask[:, :, cz]
        ax.imshow(vol_slice, cmap="gray", interpolation="none")
        ax.imshow(
            np.ma.masked_where(mask_slice == 0, mask_slice),
            cmap="Reds_r",
            alpha=0.5,
            interpolation="none",
        )

        # --- Coronal plane: slice along the second dimension ---
        ax = axes[ax_row, 1]
        vol_slice = volume[:, cy, :]
        mask_slice = mask[:, cy, :]
        ax.imshow(np.rot90(vol_slice), cmap="gray", interpolation="none")
        ax.imshow(
            np.rot90(np.ma.masked_where(mask_slice == 0, mask_slice)),
            cmap="Reds_r",
            alpha=0.5,
            interpolation="none",
        )

        # --- Sagittal plane: slice along the first dimension ---
        ax = axes[ax_row, 2]
        vol_slice = volume[cx, :, :]
        mask_slice = mask[cx, :, :]
        ax.imshow(np.rot90(vol_slice), cmap="gray", interpolation="none")
        ax.imshow(
            np.rot90(np.ma.masked_where(mask_slice == 0, mask_slice)),
            cmap="Reds_r",
            alpha=0.5,
            interpolation="none",
        )

    # Plot pre-registration (row 0)
    plot_planes(0, pre_volume, pre_mask, pre_center)
    # Plot post-registration (row 1)
    plot_planes(1, post_volume, post_mask, post_center)

    # Set row labels
    axes[0, 0].set_ylabel("Pre-registration", fontsize=12, color="white")
    axes[1, 0].set_ylabel("Post-registration", fontsize=12, color="white")

    plt.tight_layout()

    # Save the figure as an SVG file with a black background
    output_filename = f"./scripts/dataset/registration_planes_{pulse}.svg"
    plt.savefig(output_filename, format="svg", facecolor="black")
    plt.show()


# Example usage:
visualize_pre_post_planes(results["P1_T1"], "T1")
