import os
import logging
import time
from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh  # type: ignore

from Meningioma import (  # type: ignore
    ImageProcessing,
    BlindNoiseEstimation,
    Stats,
    Metrics,
)

import scienceplots  # type: ignore

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# =============================================================================
# User-defined variables for testing
# =============================================================================
PATIENT: str = "P50"
PULSE: str = "T1"
SEED: str = "42"  # single seed for this test
# List of slice indices to visualize
SLICE_INDICES: List[int] = [32, 72, 112, 152]

# Base paths (update these paths as needed)
BASE_NPZ_PATH: str = "/home/mariopasc/Python/Datasets/Meningiomas/npz"
OUTPUT_FOLDER: str = (
    "/home/mariopasc/Python/Results/Meningioma/noise_modelling/debug_visualizations"
)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Output figures for each test
OUTPUT_FIGURE_VOLUME: str = os.path.join(
    OUTPUT_FOLDER, f"{PATIENT}_{PULSE}_volume_noise.svg"
)
OUTPUT_FIGURE_SLICE: str = os.path.join(
    OUTPUT_FOLDER, f"{PATIENT}_{PULSE}_slice_noise.svg"
)
OUTPUT_FIGURE_COMPARISON: str = os.path.join(
    OUTPUT_FOLDER, f"{PATIENT}_{PULSE}_noise_comparison.svg"
)

# Variogram estimation parameters (full-volume processing)
VARIOGRAM_BINS = np.linspace(0, 100, 100)
VARIOGRAM_SAMPLING_SIZE = 3000
ESTIMATOR = "cressie"
LEN_SCALE_GUESS = 20
# Use only the isotropic variogram; ignore anisotropic computations
IGNORE_ANISOTROPIC: bool = True

# Parzen–Rosenblatt bandwidth for PDF estimation
H: float = 0.5

# =============================================================================
# Logging configuration
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


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
      Column 2: Theoretical Rayleigh PDFs and KDE estimates (from the volume-generated noise).
      Column 3: Empirical (normalized histogram) distributions with JS divergence computed between
                background and generated noise.
    The title_suffix is used to indicate whether this visualization uses volume-based or slice-based noise.
    """
    n_slices = len(slice_indices)
    fig, axs = plt.subplots(n_slices, 3, figsize=(20, 6 * n_slices))
    if n_slices == 1:
        axs = [axs]

    for i, slice_idx in enumerate(slice_indices):
        logging.info(f"Visualizing slice {slice_idx}")
        # Extract the slices
        image_slice = volume[:, :, slice_idx]
        mask_slice = mask[:, :, slice_idx]
        noise_slice = combined_noise[
            :, :, slice_idx
        ]  # could be from volume or slice generator

        # Column 1: Original image with mask overlay
        ax0 = axs[i][0]
        ax0.imshow(image_slice, cmap="gray", origin="lower")
        mask_overlay = np.where(mask_slice, 1, np.nan)
        ax0.imshow(mask_overlay, cmap="Reds_r", alpha=0.6, origin="lower")
        ax0.set_title(f"Slice {slice_idx}: Original + Mask {title_suffix}")
        ax0.set_xlabel("X")
        ax0.set_ylabel("Y")

        # Column 2: Theoretical PDFs and KDE estimates (volume-based noise)
        # Extract background pixels from original slice (mask false)
        original_bg = image_slice[~mask_slice].flatten()
        generated_noise = noise_slice.flatten()

        # Determine common bins
        try:
            min_val = np.floor(min(np.min(original_bg), np.min(generated_noise)))
            max_val = np.ceil(max(np.max(original_bg), np.max(generated_noise)))
        except Exception as e:
            logging.error(f"Error computing min/max for slice {slice_idx}: {e}")
            continue
        bins_common = np.arange(min_val, max_val + 2) - 0.5
        bin_centers = (bins_common[:-1] + bins_common[1:]) / 2

        # Fit Rayleigh distributions
        loc_orig, scale_orig = rayleigh.fit(original_bg)
        loc_gen, scale_gen = rayleigh.fit(generated_noise)
        pdf_rayleigh_orig = rayleigh.pdf(bin_centers, loc=loc_orig, scale=scale_orig)
        pdf_rayleigh_gen = rayleigh.pdf(bin_centers, loc=loc_gen, scale=scale_gen)

        # Compute KDE estimates via Stats.compute_pdf (with dist="norm")
        try:
            x_orig, kde_est_orig, _, _, _ = Stats.compute_pdf(
                original_bg, h=H, dist="norm"
            )
            x_gen, kde_est_gen, _, _, _ = Stats.compute_pdf(
                generated_noise, h=H, dist="norm"
            )
        except Exception as e:
            logging.error(f"Error computing KDE for slice {slice_idx}: {e}")
            continue

        ax1 = axs[i][1]
        ax1.plot(
            bin_centers,
            pdf_rayleigh_orig,
            linestyle="--",
            color="red",
            label=rf"Rayleigh (Orig): loc={loc_orig:.2f}, $\sigma$={scale_orig:.2f}",
        )
        ax1.plot(
            bin_centers,
            pdf_rayleigh_gen,
            linestyle="--",
            color="blue",
            label=rf"Rayleigh (Gen): loc={loc_gen:.2f}, $\sigma$={scale_gen:.2f}",
        )
        ax1.plot(x_orig, kde_est_orig, color="black", linewidth=2, label="KDE (Orig)")
        ax1.plot(x_gen, kde_est_gen, color="purple", linewidth=2, label="KDE (Gen)")
        ax1.set_title(f"Slice {slice_idx}: PDFs + KDE {title_suffix}")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Probability Density")
        ax1.legend(fontsize=8, loc="best")

        # Column 3: Empirical PDF comparisons
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

        # Compute JS divergence between empirical PDFs
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
    plt.show()
    logging.info(f"Visualization saved to {output_figure}")


def main() -> None:
    """
    Run the pipeline with a single seed and pulse, then generate:
      1. A visualization using volume-generated noise (as before).
      2. A visualization using noise generated slice-by-slice via the 2D generator.
      3. A comparison plot (for a representative slice) comparing the Rayleigh PDFs
         and KDE PDFs of volume-generated noise vs per-slice-generated noise.
    """
    npz_filepath = os.path.join(BASE_NPZ_PATH, PATIENT, f"{PATIENT}_{PULSE}.npz")
    logging.info(f"Loading volume from: {npz_filepath}")
    try:
        volume, mask = ImageProcessing.segment_3d_volume(
            npz_filepath, threshold_method="li"
        )
    except Exception as e:
        logging.error(f"Error during segmentation: {e}")
        return

    assert volume.shape == mask.shape, "Volume and mask shapes do not match."
    logging.info(f"Volume shape: {volume.shape}")

    try:
        seed_int = int(SEED)
    except Exception as e:
        logging.error(f"Error converting SEED to integer: {e}")
        return

    # --- Step 1: Variogram estimation and noise generation (full-volume) ---
    logging.info(
        "Estimating isotropic variogram and fitting covariance model (full-volume)..."
    )
    try:
        iso_bin_center, iso_gamma = (
            BlindNoiseEstimation.estimate_variogram_isotropic_3d(
                data=volume,
                bins=VARIOGRAM_BINS,
                mask=mask,
                estimator=ESTIMATOR,
                sampling_size=VARIOGRAM_SAMPLING_SIZE,
                sampling_seed=seed_int,
            )
        )
        iso_models = BlindNoiseEstimation.fit_model_3d(
            bin_center=iso_bin_center,
            gamma=iso_gamma,
            var=np.var(volume[~mask]),
            len_scale=LEN_SCALE_GUESS,
        )
        if iso_models:
            best_model_key = max(iso_models, key=lambda k: iso_models[k][1]["r2"])
            best_model_r2 = iso_models[best_model_key][1]["r2"]
            best_model_instance = iso_models[best_model_key][0]
            logging.info(f"Best model: {best_model_key} (r2 = {best_model_r2:.3f})")
        else:
            logging.error("No valid model was fitted.")
            return
    except Exception as e:
        logging.error(f"Error during variogram estimation/model fitting: {e}")
        return

    # Generate full-volume noise using the best model
    try:
        logging.info("Generating full-volume noise...")
        _, _, combined_noise_volume = (
            BlindNoiseEstimation.gaussian_random_fields_noise_3d(
                model=best_model_instance, shape=volume.shape
            )
        )
    except Exception as e:
        logging.error(f"Error generating full-volume noise: {e}")
        return

    # --- Step 2: Generate slice-by-slice noise using the 2D function ---
    # The 2D noise will be generated only for the requested slices.
    noise_slices_2d = {}
    slice_shape = volume.shape[:2]
    try:
        logging.info("Generating noise slices using 2D noise generator...")
        # For each requested slice, generate a 2D noise field
        # (using independent=True for independent real and imaginary parts)
        for slice_idx in SLICE_INDICES:
            _, _, noise_slice = BlindNoiseEstimation.gaussian_random_fields_noise_2d(
                model=best_model_instance,
                shape=slice_shape,
                independent=True,
                seed_real=1122022,
                seed_imag=23102003,
                seed_3d=11021969,
            )
            noise_slices_2d[slice_idx] = noise_slice
    except Exception as e:
        logging.error(f"Error generating 2D noise slices: {e}")
        return

    # --- Step 3: Create visualizations ---
    # Visualization 1: Using volume-generated noise
    logging.info("Creating visualization using volume-generated noise...")
    generate_visualizations(
        volume=volume,
        mask=mask,
        combined_noise=combined_noise_volume,
        slice_indices=SLICE_INDICES,
        output_figure=OUTPUT_FIGURE_VOLUME,
        title_suffix="(Volume Noise)",
    )

    # Visualization 2: Using per-slice 2D noise generation
    # For this, we build a "combined noise" array from the per-slice noise fields.
    noise_array_2d = np.zeros_like(volume, dtype=np.float32)
    for slice_idx in SLICE_INDICES:
        noise_array_2d[:, :, slice_idx] = noise_slices_2d[slice_idx]
    logging.info("Creating visualization using per-slice generated noise...")
    generate_visualizations(
        volume=volume,
        mask=mask,
        combined_noise=noise_array_2d,
        slice_indices=SLICE_INDICES,
        output_figure=OUTPUT_FIGURE_SLICE,
        title_suffix="(Slice Noise)",
    )

    # Visualization 3: Comparison plot for a representative slice (use the first slice in the list)
    rep_slice = SLICE_INDICES[0]
    logging.info(f"Creating comparison plot for representative slice {rep_slice}")
    # Extract data for the representative slice
    image_slice = volume[:, :, rep_slice]
    mask_slice = mask[:, :, rep_slice]
    noise_vol_slice = combined_noise_volume[:, :, rep_slice]
    noise_2d_slice = noise_slices_2d[rep_slice]

    # For both methods, compute the Rayleigh PDFs and KDE estimates.
    original_bg = image_slice[~mask_slice].flatten()
    gen_vol = noise_vol_slice.flatten()
    gen_2d = noise_2d_slice.flatten()

    bins_common = np.linspace(
        np.floor(min(np.min(original_bg), np.min(gen_vol), np.min(gen_2d))),
        np.ceil(max(np.max(original_bg), np.max(gen_vol), np.max(gen_2d))),
        100,
    )
    bin_centers = (bins_common[:-1] + bins_common[1:]) / 2

    # Fit Rayleigh distributions
    loc_vol, scale_vol = rayleigh.fit(gen_vol)
    loc_2d, scale_2d = rayleigh.fit(gen_2d)
    pdf_vol = rayleigh.pdf(bin_centers, loc=loc_vol, scale=scale_vol)
    pdf_2d = rayleigh.pdf(bin_centers, loc=loc_2d, scale=scale_2d)

    # Compute KDE estimates (using Stats.compute_pdf with dist="norm")
    try:
        x_vol, kde_vol, _, _, _ = Stats.compute_pdf(gen_vol, h=H, dist="norm")
        x_2d, kde_2d, _, _, _ = Stats.compute_pdf(gen_2d, h=H, dist="norm")
    except Exception as e:
        logging.error(f"Error computing KDE for comparison plot: {e}")
        return

    # Create a comparison figure with two subplots.
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    # Subplot 0: Compare Rayleigh PDFs
    axs[0].plot(
        bin_centers,
        pdf_vol,
        linestyle="--",
        color="blue",
        label=rf"Volume Noise: loc={loc_vol:.2f}, $\sigma$={scale_vol:.2f}",
    )
    axs[0].plot(
        bin_centers,
        pdf_2d,
        linestyle="--",
        color="green",
        label=rf"Slice Noise: loc={loc_2d:.2f}, $\sigma$={scale_2d:.2f}",
    )
    axs[0].set_title("Comparison of Theoretical Rayleigh PDFs")
    axs[0].set_xlabel("Value")
    axs[0].set_ylabel("Probability Density")
    axs[0].legend(loc="best", fontsize=10)

    # Subplot 1: Compare KDE estimates
    axs[1].plot(x_vol, kde_vol, color="blue", linewidth=2, label="KDE Volume Noise")
    axs[1].plot(x_2d, kde_2d, color="green", linewidth=2, label="KDE Slice Noise")
    axs[1].set_title("Comparison of KDE PDFs")
    axs[1].set_xlabel("Value")
    axs[1].set_ylabel("Probability Density")
    axs[1].legend(loc="best", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE_COMPARISON, bbox_inches="tight")
    plt.show()
    logging.info(f"Comparison plot saved to {OUTPUT_FIGURE_COMPARISON}")


if __name__ == "__main__":
    main()
