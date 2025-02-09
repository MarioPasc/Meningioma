import os
import logging
from typing import List
import numpy as np
import matplotlib.pyplot as plt

from Meningioma import ImageProcessing, BlindNoiseEstimation, Stats, Metrics  # type: ignore
import scienceplots  # type: ignore

# Set up plotting style.
plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# =============================================================================
# User-defined parameters
# =============================================================================
PATIENT = "P50"
PULSE = "T1"
SEED = "42"  # a single seed for this test
SLICE_INDICES: List[int] = [32, 72, 112, 152]  # slices to visualize

# Base paths (update these paths as needed)
BASE_NPZ_PATH: str = "/home/mariopasc/Python/Datasets/Meningiomas/npz"
OUTPUT_FOLDER: str = (
    "/home/mariopasc/Python/Results/Meningioma/noise_modelling/debug_visualizations"
)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Output figure filepaths
OUTPUT_FIGURE_VOLUME = os.path.join(
    OUTPUT_FOLDER, f"{PATIENT}_{PULSE}_volume_noise.svg"
)
OUTPUT_FIGURE_SLICE = os.path.join(OUTPUT_FOLDER, f"{PATIENT}_{PULSE}_slice_noise.svg")
OUTPUT_FIGURE_COMPARISON = os.path.join(
    OUTPUT_FOLDER, f"{PATIENT}_{PULSE}_noise_comparison_3x4.svg"
)
OUTPUT_FIGURE_RAYLEIGH = os.path.join(
    OUTPUT_FOLDER, f"{PATIENT}_{PULSE}_rayleigh_comparison.svg"
)

# NPZ filepaths for noise volumes
NOISE_VOLUME_NPZ = os.path.join(OUTPUT_FOLDER, f"{PATIENT}_{PULSE}_noise_volume.npz")
PER_SLICE_NOISE_NPZ = os.path.join(
    OUTPUT_FOLDER, f"{PATIENT}_{PULSE}_per_slice_noise.npz"
)

# Variogram estimation parameters
VARIOGRAM_BINS = np.linspace(0, 100, 100)
VARIOGRAM_SAMPLING_SIZE = 3000
ESTIMATOR = "cressie"
LEN_SCALE_GUESS = 20
IGNORE_ANISOTROPIC = True

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

# =============================================================================
# Utility Functions
# =============================================================================


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


def get_volume_noise(model, volume_shape, npz_filepath):
    """
    If a noise volume NPZ file exists, load it; otherwise, generate full-volume noise
    using the 3D generator, pack the real, imaginary, and final channels into a 4D array,
    and save the result.
    """
    if os.path.exists(npz_filepath):
        logging.info(f"Loading noise volume from {npz_filepath}")
        data = np.load(npz_filepath)
        real_vol = data["real"]
        imag_vol = data["imaginary"]
        final_vol = data["final"]
        noise_volume = np.stack((real_vol, imag_vol, final_vol), axis=-1)
    else:
        logging.info("Generating full-volume noise using 3D generator...")
        real_vol, imag_vol, final_vol = (
            BlindNoiseEstimation.gaussian_random_fields_noise_3d(
                model=model, shape=volume_shape
            )
        )
        noise_volume = np.stack((real_vol, imag_vol, final_vol), axis=-1)
        np.savez(npz_filepath, real=real_vol, imaginary=imag_vol, final=final_vol)
        logging.info(f"Saved noise volume to {npz_filepath}")
    return noise_volume


def generate_slice_noise(model, volume_shape):
    """
    Generate a noise volume by generating 2D noise slices independently.
    Returns a 4D array with shape (x, y, z, 3) where the last axis channels are:
       0: real, 1: imaginary, 2: final (combined) noise.
    Also saves the per-slice noise to the file specified by PER_SLICE_NOISE_NPZ.
    """
    nx, ny, nz = volume_shape
    noise_slices = np.zeros((nx, ny, nz, 3), dtype=np.float32)
    for z in range(nz):
        seed_offset = z  # vary seed for each slice
        real_field, imag_field, combined_field = (
            BlindNoiseEstimation.gaussian_random_fields_noise_2d(
                model=model,
                shape=(nx, ny),
                independent=True,
                seed_real=1122022 + seed_offset,
                seed_imag=23102003 + seed_offset,
                seed_3d=11021969,
            )
        )
        noise_slices[:, :, z, 0] = np.squeeze(real_field)
        noise_slices[:, :, z, 1] = np.squeeze(imag_field)
        noise_slices[:, :, z, 2] = np.squeeze(combined_field)
    # Save per-slice noise.
    np.savez(
        PER_SLICE_NOISE_NPZ,
        real=noise_slices[:, :, :, 0],
        imaginary=noise_slices[:, :, :, 1],
        final=noise_slices[:, :, :, 2],
    )
    logging.info(f"Saved per-slice noise volume to {PER_SLICE_NOISE_NPZ}")
    return noise_slices


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


# =============================================================================
# Main Pipeline
# =============================================================================


def main() -> None:
    # Load the volume and mask (segmentation)
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

    # Step 1: Variogram estimation and covariance model fitting.
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

    # Step 2: Generate (or load) full-volume noise.
    noise_volume = get_volume_noise(best_model_instance, volume.shape, NOISE_VOLUME_NPZ)

    # Step 3: Generate per-slice noise (and save it).
    noise_slice = generate_slice_noise(best_model_instance, volume.shape)

    # Step 4: Create visualizations.
    # (A) Visualization using volume-generated noise.
    logging.info("Creating visualization using volume-generated noise...")
    generate_visualizations(
        volume,
        mask,
        noise_volume,
        SLICE_INDICES,
        OUTPUT_FIGURE_VOLUME,
        "(Volume Noise)",
    )

    # (B) Visualization using slice-generated noise.
    logging.info("Creating visualization using slice-generated noise...")
    # Build a 3D array by extracting the first channel from the 4D per-slice noise.
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

    # (C) Comparison visualization: 3x4 figure comparing volume vs. per-slice noise.
    rep_slice = 112  # choose a representative slice
    fig_comp = visualize_noise_comparison(noise_volume, noise_slice, rep_slice, H=H)
    plt.savefig(OUTPUT_FIGURE_COMPARISON, bbox_inches="tight")
    logging.info(f"3x4 Comparison visualization saved to {OUTPUT_FIGURE_COMPARISON}")


if __name__ == "__main__":
    main()
