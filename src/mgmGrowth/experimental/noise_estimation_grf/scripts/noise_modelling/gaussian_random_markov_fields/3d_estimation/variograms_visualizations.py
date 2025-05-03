#!/usr/bin/env python
import os
import sys
import time
from datetime import datetime
import logging
import yaml  # type: ignore
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import gstools as gs  # type: ignore
from mgmGrowth import ImageProcessing, BlindNoiseEstimation, Metrics, Stats  # type: ignore

"""
This script aims to explore:
    1. The visualization of variograms and covariance models fitted to a volume of noise .
    2. The comparsion of noise generation methods:
        2.1. Generate one slice of noise, and compare it to its corresponding slice of the volume.
        2.2. Generate slice by slice all th study, varying the seed, until you have the volume depth.
        2.3. Generate a volume of noise directly.
        2.4. Generate a volume of noise incorporating information about the voxel size.
"""

# -----------------------------------------------------------------------------
# Set up plotting style and logging
# -----------------------------------------------------------------------------
import scienceplots  # type: ignore

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

# -----------------------------------------------------------------------------
# Create run folder with date/time stamp for all outputs
# -----------------------------------------------------------------------------
BASE_OUTPUT_FOLDER = "/home/mariopasc/Python/Results/Meningioma/noise_modelling/NoiseEstimation_20250210_231309/Figures"
run_folder = os.path.join(BASE_OUTPUT_FOLDER, datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(run_folder, exist_ok=True)
logging.info(f"Created run folder: {run_folder}")

# -----------------------------------------------------------------------------
# Output file paths (inside run folder)
# -----------------------------------------------------------------------------
OUTPUT_FIGURE_VOLUME = os.path.join(run_folder, "volume_noise.svg")
OUTPUT_FIGURE_VOLUME_VOXELS = os.path.join(run_folder, "volume_noise_voxels.svg")
OUTPUT_FIGURE_SLICE = os.path.join(run_folder, "slice_noise.svg")
OUTPUT_FIGURE_COMPARISON_VOXELS = os.path.join(
    run_folder, "noise_comparison_3x4_voxels.svg"
)
OUTPUT_FIGURE_COMPARISON = os.path.join(run_folder, "noise_comparison_3x4.svg")
OUTPUT_FIGURE_RAYLEIGH = os.path.join(run_folder, "rayleigh_comparison.svg")
NOISE_VOLUME_NPZ = os.path.join(run_folder, "noise_volume.npz")
NOISE_VOLUME_VOXELS_NPZ = os.path.join(run_folder, "noise_volume_voxels.npz")
PER_SLICE_NOISE_NPZ = os.path.join(run_folder, "per_slice_noise.npz")
SEGMENTATION_NPZ = os.path.join(run_folder, "segmentation.npz")
RUN_PARAMETERS_YAML = os.path.join(run_folder, "run_parameters.yaml")

# -----------------------------------------------------------------------------
# User-defined parameters
# -----------------------------------------------------------------------------
PATIENT = "P50"
PULSE = "T1"
SEED = "42"  # a single seed for this test
SLICE_INDICES: List[int] = [32, 72, 112, 152]  # slices to visualize

ONLY_VARIOGRAM: bool = True
GENERATE_PER_SLICE_NOISE: bool = False


# Input NPZ folder and file (adjust as needed)
BASE_NRRD_PATH: str = (
    "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition"
)
# In our pipeline, the segmentation file is located as follows:
FILEPATH_NRRD = os.path.join(
    BASE_NRRD_PATH, "RM", PULSE, PATIENT, f"{PULSE}_{PATIENT}.nrrd"
)

# Slices to ignore at beginning and end.
X = 20

# Variogram parameters
variogram_bins = np.linspace(0, 100, 100)
variogram_sampling_size = 3000
variogram_sampling_seed = 42
estimator = "cressie"
# initial length scale guess
len_scale_guess = 10

# Seeds for noise generation.
seed_real = 12012002
seed_imag = 23102003
seed_3d = 11011969

# Parzen–Rosenblatt bandwidth.
H: float = 0.5


def compute_voxel_sizes(nrrd_file_path: str) -> Optional[List[Optional[float]]]:
    """
    Read an NRRD file and extract the voxel sizes from its 'space directions'.
    Returns a list of 3 floats (or None if missing/invalid).
    """
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
        print("Warning: 'space directions' not found in the header.")
        return None


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


# -----------------------------------------------------------------------------
# VISUALIZATION FUNCTIONS (with logging added)
# -----------------------------------------------------------------------------


def generate_visualizations(
    volume: np.ndarray,
    mask: np.ndarray,
    combined_noise: np.ndarray,
    slice_indices: List[int],
    output_figure: str,
    title_suffix: str,
) -> None:
    """
    For each requested slice index, generate a row with three subplots:
      Column 1: Original image with mask overlay.
      Column 2: Theoretical PDFs and KDE estimates (here using Rayleigh fits) for the generated noise.
      Column 3: Empirical (normalized histogram) distributions with JS divergence between background and generated noise.
    """
    n_slices = len(slice_indices)
    fig, axs = plt.subplots(n_slices, 3, figsize=(20, 6 * n_slices))
    if n_slices == 1:
        axs = [axs]

    for i, slice_idx in enumerate(slice_indices):
        logging.info(f"Visualizing slice {slice_idx}")
        image_slice = volume[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]
        noise_slice = combined_noise[:, :, slice_idx]

        # Column 1: Original image with mask overlay.
        ax0 = axs[i][0]
        ax0.imshow(image_slice, cmap="gray", origin="lower")
        mask_overlay = np.where(mask_slice, 1, np.nan)
        ax0.imshow(mask_overlay, cmap="Reds_r", alpha=0.6, origin="lower")
        ax0.set_title(f"Slice {slice_idx}: Original + Mask {title_suffix}")
        ax0.set_xlabel("X")
        ax0.set_ylabel("Y")

        # Column 2: Theoretical PDFs and KDE estimates (Rayleigh).
        original_bg = image_slice[~mask_slice].flatten()
        generated_noise = noise_slice.flatten()
        try:
            # Compute PDFs and KDE using Stats.compute_pdf.
            x_orig, kde_est_orig, pdf_rayleigh_orig, str_rayleigh_orig, _ = (
                Stats.compute_pdf(original_bg, h=H, dist="rayleigh")
            )
            x_gen, kde_est_gen, pdf_rayleigh_gen, str_rayleigh_gen, _ = (
                Stats.compute_pdf(generated_noise, h=H, dist="rayleigh")
            )
        except Exception as e:
            logging.error(f"Error computing KDE for slice {slice_idx}: {e}")
            continue

        ax1 = axs[i][1]
        # Use the x-values returned by Stats.compute_pdf (fixed length, e.g. 1000 points)
        ax1.plot(
            x_orig,
            pdf_rayleigh_orig,
            linestyle="--",
            color="red",
            label=rf"Rayleigh (Orig): {str_rayleigh_orig}",
        )
        ax1.plot(
            x_gen,
            pdf_rayleigh_gen,
            linestyle="--",
            color="blue",
            label=rf"Rayleigh (Gen): {str_rayleigh_gen}",
        )
        ax1.plot(x_orig, kde_est_orig, color="black", linewidth=2, label="KDE (Orig)")
        ax1.plot(x_gen, kde_est_gen, color="purple", linewidth=2, label="KDE (Gen)")
        ax1.set_title(f"Slice {slice_idx}: PDFs + KDE {title_suffix}")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Probability Density")
        ax1.legend(fontsize=8, loc="best")

        # Column 3: Empirical PDF comparisons.
        try:
            min_val = np.floor(min(np.min(original_bg), np.min(generated_noise)))
            max_val = np.ceil(max(np.max(original_bg), np.max(generated_noise)))
        except Exception as e:
            logging.error(f"Error computing min/max for slice {slice_idx}: {e}")
            continue
        bins_common = np.arange(min_val, max_val + 2) - 0.5
        bin_centers = (bins_common[:-1] + bins_common[1:]) / 2
        hist_orig, _ = np.histogram(original_bg, bins=bins_common, density=False)
        emp_pdf_orig = hist_orig / hist_orig.sum()
        hist_gen, _ = np.histogram(generated_noise, bins=bins_common, density=False)
        emp_pdf_gen = hist_gen / hist_gen.sum()
        ax2 = axs[i][2]
        width = (bins_common[1] - bins_common[0]) * 0.9
        ax2.bar(
            bin_centers,
            emp_pdf_orig,
            width=width,
            alpha=0.4,
            color="red",
            label="Empirical (Orig)",
        )
        ax2.bar(
            bin_centers,
            emp_pdf_gen,
            width=width,
            alpha=0.4,
            color="blue",
            label="Empirical (Gen)",
        )
        ax2.set_title(f"Slice {slice_idx}: Empirical PDFs {title_suffix}")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Probability Density")
        ax2.legend(loc="best")
        try:
            js_empirical = Metrics.compute_jensen_shannon_divergence_pdfs(
                emp_pdf_orig, emp_pdf_gen, bin_centers
            )
        except Exception as e:
            logging.error(f"Error computing JS divergence for slice {slice_idx}: {e}")
            js_empirical = None
        textstr = (
            f"JS divergence: {js_empirical:.4f}"
            if js_empirical is not None
            else "JS divergence: N/A"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax2.text(
            0.95,
            0.05,
            textstr,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=props,
        )

        for ax in (ax0, ax1, ax2):
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.xaxis.tick_bottom()
            ax.yaxis.tick_left()

    plt.tight_layout()
    plt.savefig(output_figure, bbox_inches="tight")
    logging.info(f"Visualization saved to {output_figure}")


def visualize_noise_comparison(noise_volume, noise_slice, rep_slice: int, H=0.5):
    """
    Creates a 3x4 figure comparing noise from volume vs. per-slice generation.
      - Rows correspond to: Real, Imaginary, Final noise.
      - Column 1: Volume-generated noise (representative slice).
      - Column 2: Slice-generated noise (representative slice).
      - Column 3: Theoretical PDF fits (using Stats.compute_pdf; "norm" for real/imaginary, "rayleigh" for final).
      - Column 4: KDE estimates from Stats.compute_pdf.
    """
    volume_slice = noise_volume[:, :, rep_slice, :]
    slice_slice = noise_slice[:, :, rep_slice, :]

    fig, axs = plt.subplots(3, 4, figsize=(24, 18))
    noise_labels = ["Real", "Imaginary", "Final"]
    for i in range(3):
        axs[i, 0].imshow(volume_slice[:, :, i], cmap="gray")
        axs[i, 0].set_title(f"Volume Noise - {noise_labels[i]}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(slice_slice[:, :, i], cmap="gray")
        axs[i, 1].set_title(f"Slice Noise - {noise_labels[i]}")
        axs[i, 1].axis("off")

        data_vol = volume_slice[:, :, i].flatten()
        data_slice = slice_slice[:, :, i].flatten()

        dist_name = "norm" if i < 2 else "rayleigh"
        vol_color = "blue"
        slice_color = "red"
        alpha_fill = 0.3

        # Compute theoretical PDF and KDE in one call per dataset.
        x_vol, kde_vol, pdf_vol, param_str_vol, _ = Stats.compute_pdf(
            data_vol, h=H, dist=dist_name, num_points=1000
        )
        x_slice, kde_slice, pdf_slice, param_str_slice, _ = Stats.compute_pdf(
            data_slice, h=H, dist=dist_name, num_points=1000
        )

        ax_theo = axs[i, 2]
        ax_theo.plot(
            x_vol,
            pdf_vol,
            color=vol_color,
            linestyle="--",
            label=f"Volume: {param_str_vol}",
        )
        ax_theo.fill_between(x_vol, pdf_vol, color=vol_color, alpha=alpha_fill)
        ax_theo.plot(
            x_slice,
            pdf_slice,
            color=slice_color,
            linestyle="--",
            label=f"Slice: {param_str_slice}",
        )
        ax_theo.fill_between(x_slice, pdf_slice, color=slice_color, alpha=alpha_fill)
        ax_theo.set_title(f"{noise_labels[i]} - Theoretical PDF")
        ax_theo.set_xlabel("Value")
        ax_theo.set_ylabel("Probability Density")
        ax_theo.legend(fontsize=10)

        ax_kde = axs[i, 3]
        ax_kde.plot(x_vol, kde_vol, color=vol_color, linewidth=2, label="Volume KDE")
        ax_kde.fill_between(x_vol, kde_vol, color=vol_color, alpha=alpha_fill)
        ax_kde.plot(
            x_slice, kde_slice, color=slice_color, linewidth=2, label="Slice KDE"
        )
        ax_kde.fill_between(x_slice, kde_slice, color=slice_color, alpha=alpha_fill)
        ax_kde.set_title(f"{noise_labels[i]} - KDE")
        ax_kde.set_xlabel("Value")
        ax_kde.set_ylabel("Probability Density")
        ax_kde.legend(fontsize=10)

    plt.tight_layout()
    return fig


import os
import logging
import matplotlib.pyplot as plt
import gstools as gs
from typing import Dict, Tuple, Any

# Assuming we have:
#  iso_bin_center, iso_gamma, iso_models: isotropic variogram results & models
#  anisotropic_variograms: dict with keys like "X-axis $[1,0,0]$", etc.
#  variogram_bins, var_guess, len_scale_guess
#  BlindNoiseEstimation.fit_model_3d(...) is available


def plot_variograms_1x4(
    iso_bin_center: np.ndarray,
    iso_gamma: np.ndarray,
    iso_models: Dict[str, Tuple[gs.CovModel, Dict[str, Any]]],
    anisotropic_variograms: Dict[str, Tuple[np.ndarray, np.ndarray]],
    variogram_bins: np.ndarray,
    var_guess: float,
    len_scale_guess: float,
    save_path: str,
    filename: str = "variograms_1x4.svg",
) -> None:
    """
    Creates a 1x4 figure showing variograms for:
      - X-axis [1,0,0]
      - Y-axis [0,1,0]
      - Z-axis [0,0,1]
      - Isotropic [0,0,0]

    Each column plots:
      - The estimated variogram (scatter)
      - The best-fitted covariance model (dashed line)
      - R^2 in the legend.

    Parameters
    ----------
    iso_bin_center : np.ndarray
        Distances (bin centers) for the isotropic variogram.
    iso_gamma : np.ndarray
        Semivariance (gamma) for the isotropic variogram.
    iso_models : Dict[str, Tuple[gs.CovModel, Dict[str, Any]]]
        Dictionary of model name → (CovModel, fit info) for isotropic.
    anisotropic_variograms : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Keys are direction labels (e.g. "X-axis $[1,0,0]$", etc.);
        Values are (bin_center, gamma) for that direction.
    variogram_bins : np.ndarray
        The array of distance bins originally used (for x_max in plotting).
    var_guess : float
        Initial variance guess for model fitting.
    len_scale_guess : float
        Initial length scale guess for model fitting.
    save_path : str
        Directory where the figure will be saved.
    filename : str
        Name of the output file (SVG, PNG, etc.).

    Returns
    -------
    None. Saves the figure to disk and closes the figure.
    """

    logging.info("Plotting X, Y, Z, and Isotropic variograms in 1x4 layout...")

    # We only care about these three directions in the anisotropic variograms
    directions_of_interest = [
        r"X-axis $[1,0,0]$",
        r"Y-axis $[0,1,0]$",
        r"Z-axis $[0,0,1]$",
    ]

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))  # 1x4
    color_cycle = plt.cm.viridis(np.linspace(0, 1, 10))  # type: ignore

    # -------------------------------------------------------------------------
    # 1) X, Y, Z in columns [0..2]
    # -------------------------------------------------------------------------
    for i, direction in enumerate(directions_of_interest):
        ax = axs[i]
        ax.set_xlabel("Distance")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(direction)

        # Plot the estimated variogram if it exists in anisotropic_variograms
        if direction in anisotropic_variograms:
            bin_center, gamma = anisotropic_variograms[direction]
            ax.plot(
                bin_center, gamma, "o", markersize=4, color="black", label="Estimated"
            )
            # Fit a model for this direction
            logging.info(f"Fitting model for {direction} direction variogram...")
            models_dir = BlindNoiseEstimation.fit_model_3d(
                bin_center, gamma, var=var_guess, len_scale=len_scale_guess
            )
            if models_dir:
                best_key = max(models_dir, key=lambda k: models_dir[k][1]["r2"])
                best_model, best_fit_params = models_dir[best_key]
                label_text = f"{best_key}\n$r^2$={best_fit_params['r2']:.2f}"
                best_model.plot(
                    x_max=variogram_bins[-1],
                    ax=ax,
                    color=color_cycle[0],
                    linestyle="--",
                    label=label_text,
                )
        else:
            # If that direction is missing, just show "No data"
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )

        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
    # -------------------------------------------------------------------------
    # 2) Isotropic in column [3]
    # -------------------------------------------------------------------------
    ax_iso = axs[3]
    ax_iso.set_xlabel("Distance")
    ax_iso.set_ylabel(r"$\gamma$")
    ax_iso.set_title("Isotropic [0,0,0]")

    ax_iso.plot(
        iso_bin_center, iso_gamma, "o", markersize=4, color="black", label="Estimated"
    )
    if iso_models:
        best_iso_key = max(iso_models, key=lambda k: iso_models[k][1]["r2"])
        best_iso_model, best_iso_params = iso_models[best_iso_key]
        label_text = f"{best_iso_key}\n$r^2$={best_iso_params['r2']:.2f}"
        best_iso_model.plot(
            x_max=variogram_bins[-1],
            ax=ax_iso,
            color=color_cycle[0],
            linestyle="--",
            label=label_text,
        )
    else:
        ax_iso.text(
            0.5,
            0.5,
            "No isotropic model fitted",
            ha="center",
            va="center",
            transform=ax_iso.transAxes,
        )
    ax_iso.legend()
    ax_iso.spines["top"].set_visible(False)
    ax_iso.spines["right"].set_visible(False)
    ax_iso.xaxis.tick_bottom()
    ax_iso.yaxis.tick_left()
    plt.tight_layout()
    fpath = os.path.join(save_path, filename)
    plt.savefig(fpath, bbox_inches="tight")
    logging.info(f"Saved 1x4 variogram plot to {fpath}")
    plt.close(fig)


def plot_fitted_variograms_3x3(
    iso_bin_center: np.ndarray,
    iso_gamma: np.ndarray,
    iso_models: Dict[str, Tuple[gs.CovModel, Dict[str, Any]]],
    anisotropic_variograms: Dict[str, Tuple[np.ndarray, np.ndarray]],
    variogram_bins: np.ndarray,
    var_guess: float,
    len_scale_guess: float,
    save_path: str,
) -> None:
    logging.info("Plotting fitted variograms in 3x3 grid...")
    anisotropic_order = [
        r"X-axis $[1,0,0]$",
        r"Opposite X-axis $[-1,0,0]$",
        r"Y-axis $[0,1,0]$",
        r"Z-axis $[0,0,1]$",
        r"Diagonal\_XY $[1,1,0]$",
        r"Diagonal\_XZ $[1,0,1]$",
        r"Diagonal\_YZ $[0,1,1]$",
        r"Diagonal\_XYZ $[1,1,1]$",
    ]
    anisotropic_positions = [0, 1, 2, 3, 5, 6, 7, 8]

    fig, axs = plt.subplots(3, 3, figsize=(14, 14))
    axs = axs.flatten()
    color_cycle = plt.cm.viridis(np.linspace(0, 1, 10))  # type: ignore

    for pos, direction in zip(anisotropic_positions, anisotropic_order):
        ax = axs[pos]
        if direction in anisotropic_variograms:
            bin_center, gamma = anisotropic_variograms[direction]
            ax.plot(
                bin_center, gamma, "o", markersize=4, color="black", label="Estimated"
            )
            logging.info(f"Fitting model for anisotropic variogram: {direction}")
            models_dir = BlindNoiseEstimation.fit_model_3d(
                bin_center, gamma, var=var_guess, len_scale=len_scale_guess
            )
            if models_dir:
                best_key = max(models_dir, key=lambda k: models_dir[k][1]["r2"])
                best_model, best_fit_params = models_dir[best_key]
                label_text = f"{best_key}\n$r^2$ = {best_fit_params['r2']:.2f}"
                best_model.plot(
                    x_max=variogram_bins[-1],
                    ax=ax,
                    color=color_cycle[0],
                    linestyle="--",
                    label=label_text,
                )
                ax.set_title(direction)
            else:
                ax.set_title(f"{direction}\nNo model fitted")
        else:
            ax.set_visible(False)
        ax.set_xlabel("Distance")
        ax.set_ylabel(r"$\gamma$")
    ax_center = axs[4]
    ax_center.plot(
        iso_bin_center, iso_gamma, "o", markersize=4, color="black", label="Estimated"
    )
    if iso_models:
        best_iso_key = max(iso_models, key=lambda k: iso_models[k][1]["r2"])
        best_iso_model, best_iso_params = iso_models[best_iso_key]
        label_text = f"{best_iso_key}\n$r^2$ = {best_iso_params['r2']:.2f}"
        best_iso_model.plot(
            x_max=variogram_bins[-1],
            ax=ax_center,
            color=color_cycle[0],
            linestyle="--",
            label=label_text,
        )
        ax_center.set_title(r"Isotropic $[0,0,0]$")
    else:
        ax_center.set_title(r"Isotropic $[0,0,0]$" + "\nNo model fitted")
    ax_center.set_xlabel("Distance")
    ax_center.set_ylabel(r"$\gamma$")
    plt.tight_layout()
    fpath = os.path.join(save_path, "combined_figure_variograms_3d.svg")
    plt.savefig(fpath, bbox_inches="tight")
    logging.info(f"Saved 3x3 variogram plot to {fpath}")
    plt.close()


def plot_variograms_individually(
    all_variograms: Dict[str, Tuple[np.ndarray, np.ndarray]],
    variogram_bins: np.ndarray,
    var_guess: float,
    len_scale_guess: float,
    output_folder: str,
) -> None:
    logging.info("Plotting individual variograms...")
    color_cycle = plt.cm.viridis(np.linspace(0, 1, 10))  # type: ignore
    for name, (bin_center, gamma) in all_variograms.items():
        logging.info(f"Plotting variogram for {name}...")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(bin_center, gamma, "o", markersize=4, color="black", label="Estimated")
        models = BlindNoiseEstimation.fit_model_3d(
            bin_center, gamma, var=var_guess, len_scale=len_scale_guess
        )
        if models:
            best_key = max(models, key=lambda k: models[k][1]["r2"])
            best_model, best_fit_params = models[best_key]
            label_text = f"{best_key}\n$r^2$ = {best_fit_params['r2']:.2f}"
            best_model.plot(
                x_max=variogram_bins[-1],
                ax=ax,
                color=color_cycle[0],
                linestyle="--",
                label=label_text,
            )
            ax.legend(fontsize="small")
        ax.set_xlabel("Distance")
        ax.set_ylabel(r"$\gamma$")
        ax.set_title(f"Variogram: {name}")
        plt.tight_layout()
        fname = f"variogram_{name.replace(' ','').replace('$','').replace('[','').replace(']','')}.svg"
        fpath = os.path.join(output_folder, fname)
        plt.savefig(fpath, bbox_inches="tight")
        logging.info(f"Saved individual variogram plot for {name} to {fpath}")
        plt.close(fig)


def plot_noise_distributions(
    noise_real: np.ndarray,
    noise_imag: np.ndarray,
    noise_final: np.ndarray,
    output_path: str,
    h: float = 1.0,
) -> None:
    logging.info("Plotting noise distributions...")
    # Real noise.
    real_vals = noise_real.flatten()
    bins_real = np.arange(np.min(real_vals), np.max(real_vals) + 2) - 0.5
    hist_real, bin_edges_real = np.histogram(real_vals, bins=bins_real, density=False)
    emp_hist_real = hist_real / hist_real.sum()
    bin_centers_real = (bin_edges_real[:-1] + bin_edges_real[1:]) / 2

    # Imaginary noise.
    imag_vals = noise_imag.flatten()
    bins_imag = np.arange(np.min(imag_vals), np.max(imag_vals) + 2) - 0.5
    hist_imag, bin_edges_imag = np.histogram(imag_vals, bins=bins_imag, density=False)
    emp_hist_imag = hist_imag / hist_imag.sum()
    bin_centers_imag = (bins_imag[:-1] + bins_imag[1:]) / 2

    # Final noise.
    final_vals = noise_final.flatten()
    bins_final = np.arange(np.min(final_vals), np.max(final_vals) + 2) - 0.5
    hist_final, bin_edges_final = np.histogram(
        final_vals, bins=bins_final, density=False
    )
    emp_hist_final = hist_final / hist_final.sum()
    bin_centers_final = (bins_final[:-1] + bins_final[1:]) / 2

    x_real, _, theo_pdf_gauss_real, str_real, _ = Stats.compute_pdf(
        real_vals, h=h, dist="norm"
    )
    x_imag, _, theo_pdf_gauss_imag, str_imag, _ = Stats.compute_pdf(
        imag_vals, h=h, dist="norm"
    )
    x_rayleigh, _, theo_pdf_rayleigh, str_rayleigh, _ = Stats.compute_pdf(
        imag_vals, h=h, dist="rayleigh"
    )

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs[0, 0].imshow(noise_real, cmap="gray", aspect="auto")
    axs[0, 0].set_title("Real Noise Image")
    axs[0, 0].set_xlabel("X")
    axs[0, 0].set_ylabel("Y")

    axs[0, 1].imshow(noise_imag, cmap="gray", aspect="auto")
    axs[0, 1].set_title("Imaginary Noise Image")
    axs[0, 1].set_xlabel("X")
    axs[0, 1].set_ylabel("Y")

    axs[0, 2].imshow(noise_final, cmap="gray", aspect="auto")
    axs[0, 2].set_title("Final Noise Image")
    axs[0, 2].set_xlabel("X")
    axs[0, 2].set_ylabel("Y")

    ax = axs[1, 0]
    ax.bar(
        bin_centers_real,
        emp_hist_real,
        width=1,
        alpha=0.3,
        color="#DDAA33",
        label="Empirical Histogram",
    )
    ax.plot(
        x_real,
        theo_pdf_gauss_real,
        linestyle="--",
        color="blue",
        label=rf"Gaussian PDF: {str_real}",
    )
    ax.set_title("Real Noise Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="best")

    ax = axs[1, 1]
    ax.bar(
        bin_centers_imag,
        emp_hist_imag,
        width=1,
        alpha=0.3,
        color="#DDAA33",
        label="Empirical Histogram",
    )
    ax.plot(
        x_imag,
        theo_pdf_gauss_imag,
        linestyle="--",
        color="blue",
        label=rf"Gaussian PDF: {str_imag}",
    )
    ax.set_title("Imaginary Noise Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="best")

    ax = axs[1, 2]
    ax.bar(
        bin_centers_final,
        emp_hist_final,
        width=1,
        alpha=0.3,
        color="#DDAA33",
        label="Empirical Histogram",
    )
    ax.plot(
        x_rayleigh,
        theo_pdf_rayleigh,
        linestyle="--",
        color="red",
        label=rf"Rayleigh PDF: {str_rayleigh}",
    )
    ax.set_title("Final Noise Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="best")

    for row in axs:
        for axis in row:
            axis.spines["right"].set_visible(False)
            axis.spines["top"].set_visible(False)
            axis.xaxis.tick_bottom()
            axis.yaxis.tick_left()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved noise distribution figure to {output_path}")


def plot_mask_and_pdf_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    noise_final: np.ndarray,
    output_path: str,
    h: float = 1.0,
) -> None:
    logging.info("Plotting mask and PDF comparison...")
    original_bg = image[~mask].flatten()
    generated_noise = noise_final.flatten()
    min_val = np.floor(min(np.min(original_bg), np.min(generated_noise)))
    max_val = np.ceil(max(np.max(original_bg), np.max(generated_noise)))
    bins_common = np.arange(min_val, max_val + 2) - 0.5
    bin_centers = (bins_common[:-1] + bins_common[1:]) / 2

    x_orig, kde_est_orig, pdf_rayleigh_orig, str_rayleigh_orig, _ = Stats.compute_pdf(
        original_bg, h=h, dist="rayleigh"
    )
    x_gen, kde_est_gen, pdf_rayleigh_gen, str_rayleigh_gen, _ = Stats.compute_pdf(
        generated_noise, h=h, dist="rayleigh"
    )

    hist_orig, _ = np.histogram(original_bg, bins=bins_common, density=False)
    emp_pdf_orig = hist_orig / hist_orig.sum()
    hist_gen, _ = np.histogram(generated_noise, bins=bins_common, density=False)
    emp_pdf_gen = hist_gen / hist_gen.sum()

    js_empirical = Metrics.compute_jensen_shannon_divergence_pdfs(
        emp_pdf_orig, emp_pdf_gen, bin_centers
    )
    js_rayleigh = Metrics.compute_jensen_shannon_divergence_pdfs(
        pdf_rayleigh_orig, pdf_rayleigh_gen, bin_centers
    )

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs[0].imshow(image, cmap="gray", origin="lower")
    mask_overlay = np.where(mask, 1, np.nan)
    axs[0].imshow(mask_overlay, cmap="Reds_r", alpha=0.6, origin="lower")
    axs[0].set_title("Original Image with Mask")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")

    ax2 = axs[1]
    ax2.plot(
        x_orig,
        pdf_rayleigh_orig,
        linestyle="--",
        color="red",
        label=rf"Rayleigh PDF (Original): {str_rayleigh_orig}",
    )
    ax2.plot(
        x_gen,
        pdf_rayleigh_gen,
        linestyle="--",
        color="blue",
        label=rf"Rayleigh PDF (Generated): {str_rayleigh_gen}",
    )
    ax2.plot(x_orig, kde_est_orig, color="black", linewidth=2, label="KDE (Original)")
    ax2.plot(x_gen, kde_est_gen, color="purple", linewidth=2, label="KDE (Generated)")
    ax2.set_title("Theoretical PDFs (Rayleigh) + KDE")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Probability Density")
    ax2.legend(loc="best", fontsize=8)

    ax3 = axs[2]
    width = (bins_common[1] - bins_common[0]) * 0.9
    ax3.bar(
        bin_centers,
        emp_pdf_orig,
        width=width,
        alpha=0.4,
        color="red",
        label="Empirical (Original)",
    )
    ax3.bar(
        bin_centers,
        emp_pdf_gen,
        width=width,
        alpha=0.4,
        color="blue",
        label="Empirical (Generated)",
    )
    ax3.set_title("Empirical PDFs (Normalized Histograms)")
    ax3.set_xlabel("Value")
    ax3.set_ylabel("Probability Density")
    ax3.legend(loc="best")
    textstr = f"JS divergence:\nRayleigh PDFs: {js_rayleigh:.4f}\nEmpirical: {js_empirical:.4f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    ax3.text(
        0.95,
        0.05,
        textstr,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=props,
    )

    for ax in axs:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved mask and PDF comparison figure to {output_path}")
    print(f"JS divergence (Rayleigh PDFs): {js_rayleigh:.4f}")
    print(f"JS divergence (Empirical PDFs): {js_empirical:.4f}")
    print(f"Mask and PDF comparison saved to {output_path}")


# -----------------------------------------------------------------------------
# Noise Generation Functions (with logging)
# -----------------------------------------------------------------------------


def get_volume_noise(model, volume_shape, npz_filepath, voxel_size=None):
    if os.path.exists(npz_filepath):
        logging.info(f"Loading noise volume from {npz_filepath}")
        data = np.load(npz_filepath)
        real_vol = data["real"]
        imag_vol = data["imaginary"]
        final_vol = data["final"]
        noise_volume = np.stack((real_vol, imag_vol, final_vol), axis=-1)
    else:
        logging.info("Generating full-volume noise using 3D generator...")
        if voxel_size is not None:
            real_vol, imag_vol, final_vol = (
                BlindNoiseEstimation.gaussian_random_fields_noise_3d(
                    model=model, shape=volume_shape, voxel_size=voxel_size
                )
            )
        else:
            real_vol, imag_vol, final_vol = (
                BlindNoiseEstimation.gaussian_random_fields_noise_3d(
                    model=model, shape=volume_shape
                )
            )
        noise_volume = np.stack((real_vol, imag_vol, final_vol), axis=-1)
        np.savez(npz_filepath, real=real_vol, imaginary=imag_vol, final=final_vol)
        logging.info(f"Saved noise volume to {npz_filepath}")
    return noise_volume


def generate_slice_noise(model, volume_shape, npz_filepath):

    logging.info("Getting full-volume noise...")
    if os.path.exists(npz_filepath):
        logging.info(f"Loading noise volume from {npz_filepath}")
        data = np.load(npz_filepath)
        real_vol = data["real"]
        imag_vol = data["imaginary"]
        final_vol = data["final"]
        noise_slices = np.stack((real_vol, imag_vol, final_vol), axis=-1)
    else:
        logging.info("Generating per-slice noise using 2D generator...")
        nx, ny, nz = volume_shape
        noise_slices = np.zeros((nx, ny, nz, 3), dtype=np.float32)
        for z in range(nz):
            logging.info(f"Generating noise for slice {z}...")
            real_field, imag_field, combined_field = (
                BlindNoiseEstimation.gaussian_random_fields_noise_2d(
                    model=model,
                    shape=(nx, ny),
                    independent=True,
                    seed_real=100 + z,
                    seed_imag=500 + z,
                    seed_3d=11011969,
                )
            )
            noise_slices[:, :, z, 0] = np.squeeze(real_field)
            noise_slices[:, :, z, 1] = np.squeeze(imag_field)
            noise_slices[:, :, z, 2] = np.squeeze(combined_field)
        np.savez(
            PER_SLICE_NOISE_NPZ,
            real=noise_slices[:, :, :, 0],
            imaginary=noise_slices[:, :, :, 1],
            final=noise_slices[:, :, :, 2],
        )
        logging.info(f"Saved per-slice noise volume to {PER_SLICE_NOISE_NPZ}")
    return noise_slices


# -----------------------------------------------------------------------------
# Main Pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    start_time = time.time()
    logging.info("Starting main pipeline...")

    logging.info(f"Loading volume from NRRD: {FILEPATH_NRRD}")
    try:
        volume_nrrd, _ = ImageProcessing.open_nrrd_file(
            nrrd_path=FILEPATH_NRRD, return_header=True
        )
    except Exception as e:
        logging.error(f"Error reading NRRD file: {e}")
        sys.exit(1)

    voxel_sizes_list = compute_voxel_sizes(FILEPATH_NRRD)
    if voxel_sizes_list and all(v is not None for v in voxel_sizes_list):
        # Convert to tuple[float,float,float]
        voxel_sizes = tuple(float(v) for v in voxel_sizes_list)  # type: ignore
        logging.info(f"Extracted voxel sizes (mm): {voxel_sizes}")
    else:
        voxel_sizes = None
        logging.warning(
            "Voxel sizes not found or invalid. Will generate noise without coarse-graining."
        )

    # --- Segmentation and saving volume & mask ---
    logging.info(f"Loading volume from: {FILEPATH_NRRD}")
    try:
        volume, mask = ImageProcessing.segment_3d_volume(
            volume_nrrd, threshold_method="li"
        )
    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        sys.exit(1)
    if volume.shape != mask.shape:
        logging.error("Volume and mask shapes do not match.")
        sys.exit(1)
    logging.info(f"Original volume shape: {volume.shape}; mask shape: {mask.shape}")

    # Ignore X slices at beginning and end.
    volume = volume[:, :, X:-X]
    mask = mask[:, :, X:-X]
    logging.info(f"Volume shape after ignoring {2*X} slices: {volume.shape}")

    # Save the (post-ignore) volume and mask.
    np.savez_compressed(SEGMENTATION_NPZ, volume=volume, mask=mask)
    logging.info(f"Saved segmentation (volume and mask) to {SEGMENTATION_NPZ}")

    # --- Compute statistics ---
    inside_pixels = volume[mask]
    outside_pixels = volume[~mask]
    inside_mean = inside_pixels.mean()
    inside_std = inside_pixels.std()
    outside_mean = outside_pixels.mean()
    outside_std = outside_pixels.std()
    logging.info(f"Inside mask: mean = {inside_mean:.2f}, std = {inside_std:.2f}")
    logging.info(f"Outside mask: mean = {outside_mean:.2f}, std = {outside_std:.2f}")

    var_guess = np.var(outside_pixels) if outside_pixels.size > 0 else 0.0
    logging.info(f"Initial variance guess (outside mask): {var_guess:.4f}")

    # --- Variogram estimation and model fitting ---
    logging.info("Estimating isotropic variogram...")
    iso_bin_center, iso_gamma = BlindNoiseEstimation.estimate_variogram_isotropic_3d(
        data=volume,
        bins=variogram_bins,
        mask=mask,
        estimator=estimator,
        sampling_size=variogram_sampling_size,
        sampling_seed=variogram_sampling_seed,
    )
    logging.info("Fitting covariance model for isotropic variogram...")
    iso_models = BlindNoiseEstimation.fit_model_3d(
        bin_center=iso_bin_center,
        gamma=iso_gamma,
        var=var_guess,
        len_scale=len_scale_guess,
    )

    # --- Anisotropic variograms ---
    directions = [
        np.array([1, 0, 0]),
        np.array([-1, 0, 0]),
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        np.array([1, 1, 0]),
        np.array([1, 0, 1]),
        np.array([0, 1, 1]),
        np.array([1, 1, 1]),
    ]
    direction_labels = [
        r"X-axis $[1,0,0]$",
        r"Opposite X-axis $[-1,0,0]$",
        r"Y-axis $[0,1,0]$",
        r"Z-axis $[0,0,1]$",
        r"Diagonal\_XY $[1,1,0]$",
        r"Diagonal\_XZ $[1,0,1]$",
        r"Diagonal\_YZ $[0,1,1]$",
        r"Diagonal\_XYZ $[1,1,1]$",
    ]
    logging.info("Estimating anisotropic variograms...")
    anisotropic_variograms = BlindNoiseEstimation.estimate_variogram_anisotropic_3d(
        data=volume,
        bins=variogram_bins,
        mask=mask,
        directions=directions,
        direction_labels=direction_labels,
        estimator=estimator,
        sampling_size=variogram_sampling_size,
        sampling_seed=variogram_sampling_seed,
    )

    logging.info("Plotting fitted variograms (3x3 grid)...")
    plot_fitted_variograms_3x3(
        iso_bin_center=iso_bin_center,
        iso_gamma=iso_gamma,
        iso_models=iso_models,
        anisotropic_variograms=anisotropic_variograms,
        variogram_bins=variogram_bins,
        var_guess=var_guess,
        len_scale_guess=len_scale_guess,
        save_path=run_folder,
    )

    logging.info("Plotting fitted variograms (1x4 grid)...")
    plot_variograms_1x4(
        iso_bin_center=iso_bin_center,
        iso_gamma=iso_gamma,
        iso_models=iso_models,
        anisotropic_variograms=anisotropic_variograms,
        variogram_bins=variogram_bins,
        var_guess=var_guess,
        len_scale_guess=len_scale_guess,
        save_path=run_folder,
    )

    all_variograms = {"Isotropic": (iso_bin_center, iso_gamma)}
    for key, val in anisotropic_variograms.items():
        all_variograms[key] = val
    logging.info("Plotting individual variograms...")
    plot_variograms_individually(
        all_variograms=all_variograms,
        variogram_bins=variogram_bins,
        var_guess=var_guess,
        len_scale_guess=len_scale_guess,
        output_folder=run_folder,
    )

    # --- Determine best model ---
    best_r2 = -np.inf
    best_model = None
    best_model_info = None
    best_variogram_type = None
    for model_name, (model, info) in iso_models.items():
        if info["r2"] > best_r2:
            best_r2 = info["r2"]
            best_model = model
            best_model_info = info
            best_variogram_type = "Isotropic"
    for variogram_type, _ in anisotropic_variograms.items():
        models_dict = BlindNoiseEstimation.fit_model_3d(
            bin_center=anisotropic_variograms.get(variogram_type, (None, None))[0],
            gamma=anisotropic_variograms.get(variogram_type, (None, None))[1],
            len_scale=len_scale_guess,
            var=var_guess,
        )
        for model_name, (model, info) in models_dict.items():
            if info["r2"] > best_r2:
                best_r2 = info["r2"]
                best_model = model
                best_model_info = info
                best_variogram_type = variogram_type

    logging.info("Best model found:")
    logging.info(f"Variogram type: {best_variogram_type}")
    logging.info(f"Best r^2: {best_r2}")
    logging.info(f"Fitted parameters: {best_model_info['params']}")  # type: ignore

    if not ONLY_VARIOGRAM:
        # --- Generate noise fields ---
        logging.info("Generating independent noise slices (2D, independent GRF)...")
        real_field_independent, imaginary_field_independent, final_noise_independent = (
            BlindNoiseEstimation.gaussian_random_fields_noise_2d(
                model=best_model,
                shape=(volume.shape[0], volume.shape[1]),
                independent=True,
                seed_real=seed_real,
                seed_imag=seed_imag,
                seed_3d=seed_3d,
            )
        )

        logging.info("Plotting noise distributions for independent noise...")
        plot_noise_distributions(
            real_field_independent,
            imaginary_field_independent,
            final_noise_independent,
            os.path.join(
                run_folder,
                f"noise_distributions_independent_real{seed_real}_imag{seed_imag}.svg",
            ),
            h=0.5,
        )

        logging.info("Plotting mask and PDF comparison for independent noise...")
        slice_idx = SLICE_INDICES[2]
        image_slice = volume[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]
        plot_mask_and_pdf_comparison(
            image=image_slice,
            mask=mask_slice,
            noise_final=final_noise_independent,
            output_path=os.path.join(
                run_folder,
                f"final_noise_comparison_independent_real{seed_real}_imag{seed_imag}.svg",
            ),
        )

        # --- Generate full-volume and per-slice noise ---
        logging.info("Generating full-volume noise...")
        noise_volume = get_volume_noise(
            model=best_model,
            volume_shape=volume.shape,
            npz_filepath=NOISE_VOLUME_NPZ,
            voxel_size=None,
        )
        logging.info("Generating full-volume noise using voxel sizes...")
        noise_volume_voxels = get_volume_noise(
            model=best_model,
            volume_shape=volume.shape,
            npz_filepath=NOISE_VOLUME_VOXELS_NPZ,
            voxel_size=voxel_sizes,  # pass the voxel sizes for coarse-graining
        )

        logging.info("Creating visualization for volume-generated noise...")
        generate_visualizations(
            volume,
            mask,
            noise_volume,
            SLICE_INDICES,
            OUTPUT_FIGURE_VOLUME,
            "(Volume Noise)",
        )

        logging.info("Creating visualization for volume-generated noise with voxels...")
        generate_visualizations(
            volume,
            mask,
            noise_volume_voxels,
            SLICE_INDICES,
            OUTPUT_FIGURE_VOLUME_VOXELS,
            "(Volume Noise using Voxel Size)",
        )

        logging.info(
            "Creating 3x4 comparison visualization of volume with and without voxels..."
        )
        rep_slice = SLICE_INDICES[2]
        fig_comp = visualize_noise_comparison(
            noise_volume, noise_volume_voxels, rep_slice, H=H
        )
        plt.savefig(OUTPUT_FIGURE_COMPARISON_VOXELS, bbox_inches="tight")
        logging.info(
            f"Saved 3x4 comparison visualization to {OUTPUT_FIGURE_COMPARISON_VOXELS}"
        )

        if GENERATE_PER_SLICE_NOISE:
            logging.info("Generating per-slice noise...")
            noise_slice = generate_slice_noise(
                best_model, volume.shape, PER_SLICE_NOISE_NPZ
            )
            logging.info("Creating visualization for per-slice generated noise...")
            noise_array_2d = np.zeros_like(volume, dtype=np.float32)
            for slice_idx in SLICE_INDICES:
                noise_array_2d[:, :, slice_idx] = noise_slice[:, :, slice_idx, 0]
            generate_visualizations(
                volume,
                mask,
                noise_array_2d,
                SLICE_INDICES,
                OUTPUT_FIGURE_SLICE,
                "(Slice Noise)",
            )
            logging.info(
                "Creating 3x4 comparison visualization of volume against slice-generated..."
            )
            rep_slice = SLICE_INDICES[2]
            fig_comp = visualize_noise_comparison(
                noise_volume, noise_slice, rep_slice, H=H
            )
            plt.savefig(OUTPUT_FIGURE_COMPARISON, bbox_inches="tight")
            logging.info(
                f"Saved 3x4 comparison visualization to {OUTPUT_FIGURE_COMPARISON}"
            )

        total_time = time.time() - start_time
        logging.info(f"Pipeline execution time: {total_time:.2f} seconds")

        # --- Write run parameters and results to YAML ---
        run_parameters = {
            "patient": PATIENT,
            "pulse": PULSE,
            "seed": SEED,
            "slice_indices": SLICE_INDICES,
            "input_filepath": FILEPATH_NRRD,
            "volume_shape_after_ignore": volume.shape,
            "mask_shape_after_ignore": mask.shape,
            "inside_mask": {"mean": float(inside_mean), "std": float(inside_std)},
            "outside_mask": {"mean": float(outside_mean), "std": float(outside_std)},
            "initial_variance_guess": float(var_guess),
            "variogram": {
                "bins": variogram_bins.tolist(),
                "sampling_size": variogram_sampling_size,
                "estimator": estimator,
                "len_scale_guess": len_scale_guess,
            },
            "best_covariance_model": best_model_info["params"],  # type: ignore
            "seeds": {
                "seed_real": seed_real,
                "seed_imag": seed_imag,
                "seed_3d": seed_3d,
            },
            "output_files": {
                "noise_volume_npz": NOISE_VOLUME_NPZ,
                "per_slice_noise_npz": PER_SLICE_NOISE_NPZ,
                "volume_noise_figure": OUTPUT_FIGURE_VOLUME,
                "slice_noise_figure": OUTPUT_FIGURE_SLICE,
                "comparison_figure": OUTPUT_FIGURE_COMPARISON,
                "rayleigh_comparison_figure": OUTPUT_FIGURE_RAYLEIGH,
                "segmentation_npz": SEGMENTATION_NPZ,
            },
            "execution_time_sec": total_time,
        }

        # Convert numpy types to native Python types
        run_parameters["best_covariance_model"] = make_serializable(
            run_parameters["best_covariance_model"]
        )

        with open(RUN_PARAMETERS_YAML, "w") as f:
            yaml.dump(run_parameters, f)
        logging.info(f"Saved run parameters to {RUN_PARAMETERS_YAML}")


if __name__ == "__main__":
    main()
