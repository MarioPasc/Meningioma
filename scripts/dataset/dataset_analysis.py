#!/usr/bin/env python3
"""
This script reads a JSON file containing meningioma patient data,
performs three main analyses:
    1) Patient metadata analysis (age/sex distribution, plus new groundtruth plots)
    2) Per-pulse volume analysis
    3) Per-pulse segmentation analysis

The script saves summary results to three separate subdirectories.
It uses type hints for clarity, but doesn't rely on any strict-typing or
Pydantic data models.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import os

from mgmGrowth.image_processing.nrrd_processing import open_nrrd  # type: ignore

PULSE_LABELS = ["T1", "T1SIN", "T2", "SUSC"]

import matplotlib.pyplot as plt
import scienceplots  # type: ignore

plt.style.use(["science", "ieee"])

# Define the set of timepoint labels for volume plotting
TIMEPOINT_LABELS = ["first_study", "c1", "c2", "c3", "c4", "c5"]


def load_data(json_path: str) -> Dict[str, Any]:
    """
    Loads the JSON file and returns a Python dictionary.

    :param json_path: The path to the meningioma JSON file.
    :return: A dictionary representing the entire meningioma dataset.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def parse_binary_label(value: Any) -> Optional[int]:
    """
    Attempts to parse a JSON groundtruth field (e.g. 'growth' or 'progr_calcif') into a binary 0 or 1.
    Returns:
      1 if value is recognized as '1', 1.0, etc.
      0 if value is recognized as '0', 0.0, etc.
      None if missing, NaN, or unrecognized.
    """
    if value in [1, 1.0, "1", "1.0"]:
        return 1
    elif value in [0, 0.0, "0", "0.0"]:
        return 0
    return None


def analyze_patient_metadata(data: Dict[str, Any], output_dir: Path) -> None:
    """
    Analyzes patient-level metadata and saves:
      - Age/sex distribution (separate figure)
      - A single 1-row x 3-column figure that contains:
          [0] a bar plot of groundtruth availability,
          [1] a confusion matrix (progr_calcif vs. growth),
          [2] a volume-over-time plot (colored by groundtruth/growth),
        plus a legend for the color scheme (red=1, blue=0, grey=NaN).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Basic Age / Sex analysis (same as before)
    # ------------------------------------------------------------------
    ages: List[float] = []
    sexes: List[str] = []

    for patient_id, patient_data in data.items():
        metadata = patient_data.get("metadata", {})
        general = metadata.get("general", {})
        age_str = general.get("age")
        sex_str = general.get("sex")

        # Try to convert age to float
        if age_str is not None:
            try:
                ages.append(float(age_str))
            except ValueError:
                pass

        # Collect sex as a raw string
        if sex_str is not None:
            sexes.append(str(sex_str))

    # Basic numeric summary for ages
    if ages:
        avg_age = float(np.mean(ages))
        min_age = float(np.min(ages))
        max_age = float(np.max(ages))
    else:
        avg_age = min_age = max_age = 0.0

    # Write a text summary
    summary_file = output_dir / "patient_metadata_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("=== Patient Metadata Summary ===\n")
        f.write(f"Number of patients: {len(data)}\n")
        f.write(f"Average age: {avg_age}\n")
        f.write(f"Minimum age: {min_age}\n")
        f.write(f"Maximum age: {max_age}\n")
        f.write(f"Sex distribution raw: {sexes}\n")

    # Plot a simple histogram of ages if we have any
    if ages:
        plt.figure(figsize=(6, 4))
        plt.hist(
            ages, bins=range(int(min_age), int(max_age) + 2, 2), alpha=0.7, color="blue"
        )
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.savefig(output_dir / "age_distribution.png", dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # 2) Prepare data for groundtruth availability and confusion matrix
    # ------------------------------------------------------------------
    count_nan = 0
    count_avail = 0

    # 2x2 confusion matrix: row => progr_calcif in {0,1}, col => growth in {0,1}
    conf_matrix = np.zeros((2, 2), dtype=int)

    # We'll also gather volume-over-time data here
    # (Though we will plot it in the third subplot below.)
    # We'll store a list of (volumes_list, color, alpha) for each patient
    volume_lines = []

    for patient_id, patient_data in data.items():
        metadata = patient_data.get("metadata", {})
        groundtruth = metadata.get("groundtruth", {})

        # parse 'growth' and 'progr_calcif'
        calcif_label = parse_binary_label(groundtruth.get("progr_calcif"))
        growth_label = parse_binary_label(groundtruth.get("growth"))

        # GT label availability
        if calcif_label is None or growth_label is None:
            count_nan += 1
        else:
            count_avail += 1
            conf_matrix[calcif_label, growth_label] += 1

        # Determine color for volume lines
        if growth_label == 1:
            color = "red"
            alpha = 0.9
        elif growth_label == 0:
            color = "blue"
            alpha = 0.9
        else:
            color = "grey"
            alpha = 0.3

        # Collect volumes from each timepoint
        volumes_for_patient = []
        for label in TIMEPOINT_LABELS:
            label_info = metadata.get(label, {})
            if label == "first_study":
                # Typically stored in label_info["measurements"]["vol"]
                measurements = label_info.get("measurements", {})
                vol_str = measurements.get("vol")
            else:
                # c1, c2, c3, c4, c5
                vol_str = label_info.get("vol")

            if vol_str is not None:
                try:
                    vol_float = float(vol_str)
                except ValueError:
                    vol_float = np.nan
            else:
                vol_float = np.nan

            volumes_for_patient.append(vol_float)

        volume_lines.append((volumes_for_patient, color, alpha))

    volume_lines.sort(key=lambda x: x[1] != "grey")
    # ------------------------------------------------------------------
    # 3) Single figure: 1 row, 3 columns
    #    - col0 => bar plot of GT availability
    #    - col1 => confusion matrix
    #    - col2 => volume lines (colored by growth)
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    # --- (A) Left subplot: bar plot of GT label availability ---
    labels_bar = ["Available GT Label", "NaN GT Label"]
    counts_bar = [count_avail, count_nan]
    axs[0].bar(labels_bar, counts_bar, color=["green", "gray"])
    axs[0].set_ylabel("Count")
    axs[0].set_title("Groundtruth Availability")

    # --- (B) Middle subplot: confusion matrix ---
    im = axs[1].imshow(conf_matrix, cmap="Blues", origin="upper")
    axs[1].set_xticks([0, 1])
    axs[1].set_yticks([0, 1])
    axs[1].set_xticklabels(["growth=0", "growth=1"])
    axs[1].set_yticklabels(["calcif=0", "calcif=1"])
    axs[1].set_xlabel("Groundtruth/Growth")
    axs[1].set_ylabel("Groundtruth/Progr_Calcif")
    axs[1].set_title("Confusion Matrix")

    # numeric annotations
    for i in range(2):
        for j in range(2):
            axs[1].text(
                j,
                i,
                conf_matrix[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    # --- (C) Right subplot: Volume-over-time lines ---
    x_indices = range(len(TIMEPOINT_LABELS))  # e.g., 0..5
    axs[2].set_xticks(list(x_indices))
    axs[2].set_xticklabels(TIMEPOINT_LABELS)
    axs[2].set_xlabel("Timepoints")
    axs[2].set_ylabel("Tumor Volume (vol)")
    axs[2].set_title("Tumor Volume Over Time")

    for volumes_for_patient, color, alpha in volume_lines:
        axs[2].plot(
            x_indices,
            volumes_for_patient,
            marker="o",
            color=color,
            alpha=alpha,
            linestyle=":",
        )

    # Add a custom legend to show color meaning
    # We can create invisible lines for the legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="red", lw=2, label="growth=1"),
        Line2D([0], [0], color="blue", lw=2, label="growth=0"),
        Line2D([0], [0], color="grey", lw=2, alpha=0.3, label="growth=NaN"),
    ]
    axs[2].legend(handles=legend_elements, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "metadata_1x3_plots.png", dpi=150)
    plt.close(fig)


def analyze_per_pulse_volumes(data: Dict[str, Any], output_dir: Path) -> None:
    """
    Analyzes per-pulse volume info (like min, max intensities) across patients.

    :param data: Dictionary of meningioma data loaded from JSON.
    :param output_dir: Directory to save analysis results (text, plots).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    min_values: List[float] = []
    max_values: List[float] = []
    volume_routes: List[str] = []

    for patient_id, patient_data in data.items():
        pulses = patient_data.get("pulses", {})
        for pulse_name, pulse_info in pulses.items():
            if pulse_info.get("error") is True:
                # Skip pulses that have "error": true
                continue
            volume_info = pulse_info.get("volume", {})
            vol_min = volume_info.get("min")
            vol_max = volume_info.get("max")
            route = volume_info.get("route")

            # Attempt to convert to float
            if vol_min is not None and vol_max is not None:
                try:
                    min_values.append(float(vol_min))
                    max_values.append(float(vol_max))
                    if route:
                        volume_routes.append(str(route))
                except ValueError:
                    # Some volumes might not be strictly numeric
                    pass

    # Summaries
    summary_lines: List[str] = []
    if min_values and max_values:
        global_min = float(np.min(min_values))
        global_max = float(np.max(max_values))
        summary_lines.append(f"Global minimum intensity: {global_min}")
        summary_lines.append(f"Global maximum intensity: {global_max}")
        summary_lines.append(f"Volume routes found: {len(volume_routes)}")

    # Write summary to a text file
    summary_file = output_dir / "per_pulse_volume_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("=== Per-Pulse Volume Analysis ===\n")
        for line in summary_lines:
            f.write(line + "\n")

    # Optionally, we can plot sorted min and max intensities
    if min_values and max_values:
        plt.figure(figsize=(6, 4))
        plt.plot(sorted(min_values), label="Min Intensities")
        plt.plot(sorted(max_values), label="Max Intensities")
        plt.title("Sorted Min/Max Intensities Across All Pulses")
        plt.xlabel("Index")
        plt.ylabel("Intensity")
        plt.legend()
        plt.savefig(output_dir / "min_max_intensity_plot.png", dpi=150)
        plt.close()


def analyze_per_pulse_segmentation(
    data: Dict[str, Any], output_dir: Path, data_file_path: Optional[Path] = None
) -> None:
    """
    Analyzes per-pulse segmentation volumes across patients, plus a 1x3 subplot:
      1) Bar chart (horizontal) showing total patients vs. non-empty segmentation
         for each pulse.
      2) Boxplot of mask volume (# of voxels in segmentation) for each pulse.
      3) Boxplot of the volume intensities (from the underlying volume) in
         the masked region, for each pulse.

    This function also supports saving/loading intermediate data to/from a JSON file:
      - If 'data_file_path' is not provided or doesn't exist, we compute from the NRRD files,
        save the JSON if a file path is given, then plot.
      - If 'data_file_path' is provided and exists, we skip the NRRD reading and load
        the data from JSON directly, then create the plots.

    :param data: Dictionary of meningioma data loaded from JSON.
    :param output_dir: Directory to save analysis results (text, plots).
    :param data_file_path: If provided, path to a JSON file to load/store
        the intermediate data. (Default: None)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    seg_volumes: List[float] = []

    # Will we load from a JSON file, or compute now?
    if data_file_path and data_file_path.is_file():
        # ------------------------------
        # 1) Load data from JSON
        # ------------------------------
        with open(data_file_path, "r", encoding="utf-8") as f:
            stored = json.load(f)

        seg_volumes = stored["seg_volumes"]
        total_count = stored["total_count"]
        non_empty_seg_count = stored["non_empty_seg_count"]
        mask_volumes = stored["mask_volumes"]
        mask_intensities = stored["mask_intensities"]

    else:
        # ------------------------------
        # 1) Compute data from NRRDs
        # ------------------------------
        # For the bar chart
        total_count = {pulse: 0 for pulse in PULSE_LABELS}
        non_empty_seg_count = {pulse: 0 for pulse in PULSE_LABELS}
        # For boxplot #1
        mask_volumes = {pulse: [] for pulse in PULSE_LABELS}
        # For boxplot #2
        mask_intensities = {pulse: [] for pulse in PULSE_LABELS}

        # First, gather the original text summary logic
        for patient_id, patient_data in data.items():
            pulses = patient_data.get("pulses", {})
            for pulse_name, pulse_info in pulses.items():
                seg_info = pulse_info.get("segmentation", {})
                total_vol = seg_info.get("total_volume")
                if total_vol is not None:
                    try:
                        seg_volumes.append(float(total_vol))
                    except ValueError:
                        pass

        # Then, gather the pulse-level data for the 1×3 subplot
        for patient_id, patient_data in data.items():
            pulses = patient_data.get("pulses", {})

            for pulse in PULSE_LABELS:
                pulse_info = pulses.get(pulse)
                if not pulse_info or pulse_info.get("error") is True:
                    # This patient doesn't have this pulse or it's flagged as error
                    continue

                # If we get here, the patient has the pulse
                total_count[pulse] += 1

                # Try opening the segmentation file
                seg_info = pulse_info.get("segmentation", {})
                seg_path = seg_info.get("route")
                if not seg_path or not os.path.isfile(seg_path):
                    continue

                try:
                    seg_data = open_nrrd(seg_path)  # load segmentation
                except:
                    # If there's an error reading the file, skip
                    continue

                unique_vals = np.unique(seg_data)
                if len(unique_vals) < 2:
                    # Means single intensity => probably empty or invalid
                    continue
                # Otherwise, we consider it a "non-empty" segmentation
                non_empty_seg_count[pulse] += 1

                max_val = np.max(unique_vals)
                mask = seg_data == max_val
                voxel_count = np.count_nonzero(mask)
                mask_volumes[pulse].append(voxel_count)

                # Now gather underlying volume intensities in mask
                vol_path = pulse_info.get("volume", {}).get("route")
                if not vol_path or not os.path.isfile(vol_path):
                    continue
                try:
                    vol_data = open_nrrd(vol_path)
                except:
                    continue
                # (Optional) check for shape mismatch
                if seg_data.shape != vol_data.shape:
                    print(
                        f"Warning: shape mismatch for patient {patient_id}, pulse {pulse}"
                    )
                    continue

                region_intensity = vol_data[mask]
                mask_intensities[pulse].extend(region_intensity.tolist())

        # If user gave us a path to store this data, store it
        if data_file_path:
            to_store = {
                "seg_volumes": seg_volumes,
                "total_count": total_count,
                "non_empty_seg_count": non_empty_seg_count,
                "mask_volumes": mask_volumes,
                "mask_intensities": mask_intensities,
            }
            with open(data_file_path, "w", encoding="utf-8") as f:
                json.dump(to_store, f, indent=2)

    # -----------------------------------------
    # 2) Summaries & text file
    # -----------------------------------------
    if seg_volumes:
        mean_vol = float(np.mean(seg_volumes))
        median_vol = float(np.median(seg_volumes))
        min_vol = float(np.min(seg_volumes))
        max_vol = float(np.max(seg_volumes))
    else:
        mean_vol = median_vol = min_vol = max_vol = 0.0

    summary_file = output_dir / "segmentation_summary.txt"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("=== Per-Pulse Segmentation Analysis ===\n")
        f.write(f"Count of segmentations: {len(seg_volumes)}\n")
        f.write(f"Mean segmentation volume: {mean_vol}\n")
        f.write(f"Median segmentation volume: {median_vol}\n")
        f.write(f"Min segmentation volume: {min_vol}\n")
        f.write(f"Max segmentation volume: {max_vol}\n")

    # -----------------------------------------
    # 3) Create the 1-row, 3-column figure
    # -----------------------------------------
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    # A) subplot 0: counts barplot => total vs. non-empty segmentation
    y_positions = np.arange(len(PULSE_LABELS))
    total_vals = [total_count[p] for p in PULSE_LABELS]
    seg_vals = [non_empty_seg_count[p] for p in PULSE_LABELS]

    bar_height = 0.4
    axs[0].barh(
        y_positions - bar_height / 2,
        total_vals,
        bar_height,
        color="lightblue",
        label="Total Patients",
    )
    axs[0].barh(
        y_positions + bar_height / 2,
        seg_vals,
        bar_height,
        color="salmon",
        label="Non-empty Seg",
    )
    axs[0].set_yticks(y_positions)
    axs[0].set_yticklabels(PULSE_LABELS)
    axs[0].set_xlabel("Count")
    axs[0].set_title("Total vs. Non-empty Seg. Count")
    axs[0].legend()

    # B) subplot 1: horizontal boxplot of mask volumes
    box_data_vol = [
        mask_volumes[p] if len(mask_volumes[p]) > 0 else [0] for p in PULSE_LABELS
    ]
    axs[1].boxplot(box_data_vol, vert=False, labels=PULSE_LABELS)
    axs[1].set_xlabel("Mask Volume (voxel count)")
    axs[1].set_title("Distribution of Mask Volume")

    # C) subplot 2: horizontal boxplot of intensities from underlying volume
    box_data_intensities = []
    for p in PULSE_LABELS:
        if len(mask_intensities[p]) == 0:
            box_data_intensities.append([0])
        else:
            box_data_intensities.append(mask_intensities[p])

    axs[2].boxplot(box_data_intensities, vert=False, labels=PULSE_LABELS)
    axs[2].set_xlabel("Volume Intensities in Mask")
    axs[2].set_title("Distribution of Underlying Intensities")

    # Force the same y-positions so that T1 is top row, T1SIN second, etc.
    for i, ax in enumerate(axs):
        if i == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels(PULSE_LABELS)
        else:
            yticks = ax.get_yticks()
            ax.set_yticks(yticks - 0.6)
            ax.set_yticklabels([])
        ax.invert_yaxis()

    fig.tight_layout()
    save_path = output_dir / "segmentation_1x3_summary.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """
    Main function to orchestrate loading, analysis, and output structure.
    """

    # Default paths (as requested)
    json_path = (
        "/home/mariopasc/Python/Results/Meningioma/preprocessing/plan_meningioma.json"
    )
    output_dir = "/home/mariopasc/Python/Results/Meningioma/preprocessing"

    # 1. Load data as a dictionary
    meningioma_data: Dict[str, Any] = load_data(json_path)

    # 2. Prepare output directories
    output_base = Path(output_dir)
    metadata_dir = output_base / "patient_metadata_analysis"
    volume_dir = output_base / "per_pulse_volume_analysis"
    segmentation_dir = output_base / "per_pulse_segmentation_analysis"

    # 3. Run analyses
    analyze_patient_metadata(meningioma_data, metadata_dir)
    analyze_per_pulse_volumes(meningioma_data, volume_dir)
    analyze_per_pulse_segmentation(
        data=meningioma_data,
        output_dir=segmentation_dir,
        data_file_path=Path(
            os.path.join(segmentation_dir, "segmentation_3x1_data.json")
        ),
    )

    print(f"Analysis complete. Results available in {output_base.absolute()}")


if __name__ == "__main__":
    main()
