import os
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import gstools as gs  # type: ignore
import pandas as pd

from scipy.stats import rice, rayleigh, ncx2, norm  # type: ignore
from scipy.interpolate import interp1d  # type: ignore

# Import the convex hull function from your package.
from Meningioma import ImageProcessing, BlindNoiseEstimation, Metrics, Stats  # type: ignore

# ========================================================== VISUALIZATIONS ==================================================================================


# =============================================================================
# VARIOGRAM VISUALIZATIONS
# =============================================================================
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
    """
    Create a 3x3 grid of plots showing:
      - The isotropic variogram (labeled as "Isotropic [0,0,0]") in the center (cell [1,1]).
      - Eight anisotropic variograms in the surrounding cells.

    For each variogram, only the best-fitting covariance model is plotted.
    The legend (label on the model curve) includes the model name and its fitted r^2.

    The anisotropic directions (and the corresponding keys in anisotropic_variograms)
    are assumed to be:
      - r"X-axis $[1,0,0]$"
      - r"Y-axis $[0,1,0]$"
      - r"Z-axis $[0,0,1]$"
      - r"Diagonal\_XY $[1,1,0]$"
      - r"Diagonal\_XZ $[1,0,1]$"
      - r"Diagonal\_YZ $[0,1,1]$"
      - r"Diagonal\_XYZ $[1,1,1]$"
      - r"Opposite X-axis $[-1,0,0]$"
    """
    import matplotlib.pyplot as plt

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
    # In a 3x3 grid (indices 0 to 8), reserve cell index 4 for isotropic.
    anisotropic_positions = [0, 1, 2, 3, 5, 6, 7, 8]

    fig, axs = plt.subplots(3, 3, figsize=(14, 14))
    axs = axs.flatten()

    color_cycle = plt.cm.viridis(np.linspace(0, 1, 10))  # type: ignore

    # Plot anisotropic variograms.
    for pos, direction in zip(anisotropic_positions, anisotropic_order):
        ax = axs[pos]
        if direction in anisotropic_variograms:
            bin_center, gamma = anisotropic_variograms[direction]
            ax.plot(
                bin_center, gamma, "o", markersize=4, color="black", label="Estimated"
            )
            # For this variogram, fit the covariance models and select the best.
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
                ax.set_title(f"{direction}")
            else:
                ax.set_title(f"{direction}\nNo model fitted")
        else:
            ax.set_visible(False)
        ax.set_xlabel("Distance")
        ax.set_ylabel(r"$\gamma$")

    # Plot the isotropic variogram in the center cell (index 4).
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
    print(f"Saved individual variogram plot to {fpath}")
    plt.close()


def plot_variograms_individually(
    all_variograms: Dict[str, Tuple[np.ndarray, np.ndarray]],
    variogram_bins: np.ndarray,
    var_guess: float,
    len_scale_guess: float,
    output_folder: str,
) -> None:
    """
    Plot each variogram (both isotropic and anisotropic) in a separate figure.
    For each variogram, the best-fitting covariance model (as determined by the highest r²)
    is computed and overlaid using the model's built-in plot method. The label on the curve
    includes the model name and its fitted r² value.

    Parameters
    ----------
    all_variograms : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary mapping variogram names (e.g., "Isotropic", "X-axis $[1,0,0]$", etc.)
        to a tuple (bin_centers, gamma) containing the estimated variogram.
    variogram_bins : np.ndarray
        The bin edges used for variogram estimation (used to set the x_max for plotting).
    var_guess : float
        The initial variance guess used for model fitting.
    len_scale_guess : float
        The initial length scale guess used for model fitting.
    output_folder : str
        The folder in which to save the individual variogram image files.
    """
    import matplotlib.pyplot as plt

    color_cycle = plt.cm.viridis(np.linspace(0, 1, 10))  # type: ignore

    for name, (bin_center, gamma) in all_variograms.items():
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(bin_center, gamma, "o", markersize=4, color="black", label="Estimated")
        # Compute the best-fitting model for this variogram.
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
        # Prepare a filename that avoids spaces and special characters.
        fname = f"variogram_{name.replace(' ','').replace('$','').replace('[','').replace(']','')}.svg"
        fpath = os.path.join(output_folder, fname)
        plt.savefig(fpath, bbox_inches="tight")
        print(f"Saved individual variogram plot for {name} to {fpath}")
        plt.close(fig)


# =============================================================================
# GENERATED NOISE VISUALIZATIONS
# =============================================================================


from scipy.stats import norm, rayleigh


def plot_noise_distributions(
    noise_real: np.ndarray,
    noise_imag: np.ndarray,
    noise_final: np.ndarray,
    output_path: str,
    h: float = 1.0,
) -> None:
    """
    Create a 2x3 subplot figure displaying the noise images and their corresponding distributions.

    Top row:
      - [0,0]: Real noise image.
      - [0,1]: Imaginary noise image.
      - [0,2]: Final combined noise image.

    Bottom row (distribution plots):
      - [1,0]: Real noise: Empirical histogram (bars in yellow, color "#DDAA33") and theoretical
               Gaussian PDF (dashed blue) with parameters (μ, σ).
      - [1,1]: Imaginary noise: Empirical histogram and theoretical Gaussian PDF.
      - [1,2]: Final noise: Empirical histogram and theoretical Rayleigh PDF (dashed red;
               with parameter σ = √(scale) from the fit).

    The theoretical PDFs are computed directly using the corresponding PDF functions.

    Parameters:
        noise_real: 2D array for the generated real noise component.
        noise_imag: 2D array for the generated imaginary noise component.
        noise_final: 2D array for the generated magnitude (final) noise.
        output_path: Path where to save the figure.
        h: Bandwidth for the KDE estimation (not used here, retained for interface compatibility).
    """
    # --- Prepare empirical histograms for each noise image ---
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

    # --- Fit theoretical distributions and compute theoretical PDFs ---
    # For real and imaginary noise: Gaussian fit.
    mu_real, sigma_real = norm.fit(real_vals)
    mu_imag, sigma_imag = norm.fit(imag_vals)
    theo_pdf_gauss_real = norm.pdf(bin_centers_real, loc=mu_real, scale=sigma_real)
    theo_pdf_gauss_imag = norm.pdf(bin_centers_imag, loc=mu_imag, scale=sigma_imag)

    # For final noise: Rayleigh fit.
    loc_r, scale_r = rayleigh.fit(final_vals)
    theo_pdf_rayleigh = rayleigh.pdf(bin_centers_final, loc=loc_r, scale=scale_r)

    # --- Create 2x3 subplot figure ---
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: Noise images.
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

    # Bottom row: Distribution plots.
    # [1,0] Real noise.
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
        bin_centers_real,
        theo_pdf_gauss_real,
        linestyle="--",
        color="blue",
        label=rf"Gaussian PDF: $\mu={mu_real:.2f},\ \sigma={sigma_real:.2f}$",
    )
    ax.set_title("Real Noise Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="best")

    # [1,1] Imaginary noise.
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
        bin_centers_imag,
        theo_pdf_gauss_imag,
        linestyle="--",
        color="blue",
        label=rf"Gaussian PDF: $\mu={mu_imag:.2f},\ \sigma={sigma_imag:.2f}$",
    )
    ax.set_title("Imaginary Noise Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="best")

    # [1,2] Final noise.
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
        bin_centers_final,
        theo_pdf_rayleigh,
        linestyle="--",
        color="red",
        label=rf"Rayleigh PDF: $\sigma={np.sqrt(scale_r):.2f}$",
    )
    ax.set_title("Final Noise Distribution")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Density")
    ax.legend(loc="best")

    # Improve appearance by removing top/right spines and adjusting ticks.
    for row in axs:
        for axis in row:
            axis.spines["right"].set_visible(False)
            axis.spines["top"].set_visible(False)
            axis.xaxis.tick_bottom()
            axis.yaxis.tick_left()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Noise distributions saved to {output_path}")


# =============================================================================
# GENERATED NOISE COMPARISON WITH ACTUAL NOISE DISTRIBUTION ON ORIGINAL SLICE
# =============================================================================


def plot_mask_and_pdf_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    noise_final: np.ndarray,
    output_path: str,
    h: float = 1.0,
) -> None:
    """
    Create a three-panel figure to compare the generated noise to the original background noise.

    Panel details:
      - Subplot 1: Original image with the mask overlay (mask in red, α=0.6).
      - Subplot 2: Comparison of theoretical Rayleigh PDFs and the PDF approximated via
                   Parzen–Rosenblatt KDE.
                   Two theoretical curves are plotted:
                       • Rayleigh PDF for original background (red, dashed)
                       • Rayleigh PDF for generated noise (blue, dashed)
                   Additionally, the KDE-estimated PDFs are overlaid.
      - Subplot 3: Empirical PDFs (normalized histograms) for the original background (red bars)
                   and the generated noise (blue bars). Also, the Jensen–Shannon divergence (JS divergence)
                   between the distributions is annotated.

    Parameters:
        image: Original image (2D array).
        mask: Boolean mask (2D array); pixels outside the mask are considered background.
        noise_final: Generated noise image (magnitude; 2D array).
        output_path: File path to save the resulting figure.
        h: Bandwidth for KDE estimation.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import rayleigh

    # Assume compute_pdf and compute_js_divergence are available.
    # from your_module import compute_pdf, compute_js_divergence

    # --- Extract data: original background and generated noise ---
    original_bg = image[~mask].flatten()
    generated_noise = noise_final.flatten()

    # Determine common bins (assuming quantized values)
    min_val = np.floor(min(np.min(original_bg), np.min(generated_noise)))
    max_val = np.ceil(max(np.max(original_bg), np.max(generated_noise)))
    bins_common = np.arange(min_val, max_val + 2) - 0.5
    bin_centers = (bins_common[:-1] + bins_common[1:]) / 2

    # --- Fit theoretical distributions using Rayleigh ---
    loc_orig, scale_orig = rayleigh.fit(original_bg)
    loc_gen, scale_gen = rayleigh.fit(generated_noise)

    # --- Compute theoretical PDFs directly ---
    pdf_rayleigh_orig = rayleigh.pdf(bin_centers, loc=loc_orig, scale=scale_orig)
    pdf_rayleigh_gen = rayleigh.pdf(bin_centers, loc=loc_gen, scale=scale_gen)

    # --- Compute empirical PDFs ---
    hist_orig, _ = np.histogram(original_bg, bins=bins_common, density=False)
    emp_pdf_orig = hist_orig / hist_orig.sum()
    hist_gen, _ = np.histogram(generated_noise, bins=bins_common, density=False)
    emp_pdf_gen = hist_gen / hist_gen.sum()

    # --- Compute Jensen–Shannon Divergence between empirical PDFs ---
    js_empirical = Metrics.compute_jensen_shannon_divergence_pdfs(
        emp_pdf_orig, emp_pdf_gen, bin_centers
    )
    js_rayleigh = Metrics.compute_jensen_shannon_divergence_pdfs(
        pdf_rayleigh_orig, pdf_rayleigh_gen, bin_centers
    )

    # --- Compute KDE estimates via Parzen–Rosenblatt method ---
    x_orig, kde_est_orig, _, _, _ = Stats.compute_pdf(original_bg, h=h, dist="norm")
    x_gen, kde_est_gen, _, _, _ = Stats.compute_pdf(generated_noise, h=h, dist="norm")

    # --- Create 3-panel figure ---
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # Subplot 1: Original image with mask overlay.
    axs[0].imshow(image, cmap="gray", origin="lower")
    mask_overlay = np.where(mask, 1, np.nan)
    axs[0].imshow(mask_overlay, cmap="Reds_r", alpha=0.6, origin="lower")
    axs[0].set_title("Original Image with Mask")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")

    # Subplot 2: Theoretical PDF and KDE comparisons.
    ax2 = axs[1]
    ax2.plot(
        bin_centers,
        pdf_rayleigh_orig,
        linestyle="--",
        color="red",
        label=rf"Rayleigh PDF (Original): $\mathrm{{loc}}={loc_orig:.2f},\ \sigma={scale_orig:.2f}$",
    )
    ax2.plot(
        bin_centers,
        pdf_rayleigh_gen,
        linestyle="--",
        color="blue",
        label=rf"Rayleigh PDF (Generated): $\mathrm{{loc}}={loc_gen:.2f},\ \sigma={scale_gen:.2f}$",
    )
    ax2.plot(
        x_orig,
        kde_est_orig,
        color="black",
        linewidth=2,
        label="KDE (Original)",
    )
    ax2.plot(
        x_gen,
        kde_est_gen,
        color="purple",
        linewidth=2,
        label="KDE (Generated)",
    )
    ax2.set_title("Theoretical PDFs (Rayleigh) + KDE")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Probability Density")
    ax2.legend(loc="best", fontsize=8)

    # Subplot 3: Empirical PDF comparisons.
    ax3 = axs[2]
    width = (bins_common[1] - bins_common[0]) * 0.9  # bar width
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

    # Annotate Jensen–Shannon divergence values in subplot 3.
    textstr = (
        "JS divergence:\n"
        f"Rayleigh PDFs: {js_rayleigh:.4f}\n"
        f"Empirical: {js_empirical:.4f}"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    # Place text box in bottom-right corner of ax3.
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

    # Remove top/right spines for all axes.
    for ax in axs:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"JS divergence (Rayleigh PDFs): {js_rayleigh:.4f}")
    print(f"JS divergence (Empirical PDFs): {js_empirical:.4f}")
    print(f"Mask and PDF comparison saved to {output_path}")


# ========================================================== END OF VISUALIZATIONS =======================================================================


# =============================================================================
# Call in the Main Pipeline
# =============================================================================
if __name__ == "__main__":
    output_folder = (
        "scripts/noise_modelling/gaussian_random_markov_fields/images/experiment_images"
    )
    os.makedirs(output_folder, exist_ok=True)

    base_path = "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition"
    output_npz_path = "/home/mariopasc/Python/Datasets/Meningiomas/npz"

    patient = "P50"
    pulse = "T1"
    filepath = os.path.join(output_npz_path, patient, f"{patient}_{pulse}.npz")

    # Load volume and compute the exclusion mask.
    volume, mask = ImageProcessing.segment_3d_volume(filepath, threshold_method="li")
    print("Data shape:", volume.shape)
    print("Original Mask field shape:", mask.shape)

    # Print statistics inside and outside the mask.
    inside_pixels = volume[mask]
    outside_pixels = volume[np.logical_not(mask)]
    print(
        "Inside mask: mean = {:.2f}, std = {:.2f}".format(
            inside_pixels.mean(), inside_pixels.std()
        )
    )
    print(
        "Outside mask: mean = {:.2f}, std = {:.2f}".format(
            outside_pixels.mean(), outside_pixels.std()
        )
    )

    # Compute initial variance guess from the background (outside the convex hull).
    var_guess = np.var(outside_pixels) if outside_pixels.size > 1 else 0.0
    len_scale_guess = 20

    # Define variogram bins and sampling parameters.
    variogram_bins = np.linspace(0, 100, 100)
    variogram_sampling_size = 3000
    variogram_sampling_seed = 19920516
    estimator = "cressie"
    shape = (512, 512)

    # Isotropic variogram estimation and model fitting.
    iso_bin_center, iso_gamma = BlindNoiseEstimation.estimate_variogram_isotropic_3d(
        data=volume,
        bins=variogram_bins,
        mask=mask,
        estimator=estimator,
        sampling_size=variogram_sampling_size,
        sampling_seed=variogram_sampling_seed,
    )
    iso_models = BlindNoiseEstimation.fit_model_3d(
        bin_center=iso_bin_center,
        gamma=iso_gamma,
        len_scale=len_scale_guess,
        var=var_guess,
    )

    # Define eight anisotropic directions and labels.
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

    plot_fitted_variograms_3x3(
        iso_bin_center=iso_bin_center,
        iso_gamma=iso_gamma,
        iso_models=iso_models,
        anisotropic_variograms=anisotropic_variograms,
        variogram_bins=variogram_bins,
        var_guess=var_guess,
        len_scale_guess=len_scale_guess,
        save_path=output_folder,
    )

    # Combine the isotropic variogram with the anisotropic ones.
    all_variograms = {}
    all_variograms["Isotropic"] = (iso_bin_center, iso_gamma)
    for key, val in anisotropic_variograms.items():
        all_variograms[key] = val

    # Plot every variogram separately, saving each as a single image.
    plot_variograms_individually(
        all_variograms=all_variograms,
        variogram_bins=variogram_bins,
        var_guess=var_guess,
        len_scale_guess=len_scale_guess,
        output_folder=output_folder,
    )

    # Find the best model

    # Initialize variables to hold the best model info.
    best_r2 = -np.inf
    best_model = None  # This will be the covariance model instance.
    best_model_info = (
        None  # Dictionary with fitted parameters, including 'r2' and 'params'.
    )
    best_variogram_type = None  # A string indicating which variogram (e.g., "Isotropic" or a given anisotropic label).

    # Compute anisotropic covariance models from the anisotropic variograms.
    anisotropic_models: Dict[str, Dict[str, Tuple[gs.CovModel, Dict[str, Any]]]] = {}
    for direction_label, (bin_centers, gamma) in anisotropic_variograms.items():
        anisotropic_models[direction_label] = BlindNoiseEstimation.fit_model_3d(
            bin_center=bin_centers,
            gamma=gamma,
            len_scale=len_scale_guess,
            var=var_guess,
        )

    # First, check the isotropic models.
    for model_name, (model, info) in iso_models.items():
        r2 = info["r2"]
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_info = info
            best_variogram_type = "Isotropic"

    # Next, check the anisotropic models.
    for variogram_type, models_dict in anisotropic_models.items():
        for model_name, (model, info) in models_dict.items():
            r2 = info["r2"]
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_model_info = info
                best_variogram_type = variogram_type

    # Print the results.
    print("Best model found:")
    print("Variogram type:", best_variogram_type)
    print("Best r^2:", best_r2)
    print("Fitted parameters:", best_model_info["params"])  # type: ignore

    # Now, let's generate random fields using the obtained covariance model.
    # Independently-generated GRF
    print("Generating independent noise slices ...")
    real_field_independent, imaginary_field_independent, final_noise_independent = (
        BlindNoiseEstimation.gaussian_random_fields_noise_2d(
            model=best_model,
            shape=shape,
            independent=True,
            seed_real=1122022,
            seed_imag=23102003,
            seed_3d=11021969,
        )
    )
    # Dependent GRF (coming from the same noise volume)
    print("Generating same-volume noise slices ...")
    real_field_dependent, imaginary_field_dependent, final_noise_dependent = (
        BlindNoiseEstimation.gaussian_random_fields_noise_2d(
            model=best_model,
            shape=shape,
            independent=False,
            seed_real=1122022,
            seed_imag=23102003,
            seed_3d=11021969,
        )
    )

    # Plot the noise comparison and distributions for both generated fields.
    print("Plotting noise distributions ...")
    # Independently-generated GRF
    plot_noise_distributions(
        real_field_independent,
        imaginary_field_independent,
        final_noise_independent,
        os.path.join(output_folder, "noise_distributions_independent.svg"),
        h=0.5,
    )
    # Dependent GRF (coming from the same noise volume)
    plot_noise_distributions(
        real_field_dependent,
        imaginary_field_dependent,
        final_noise_dependent,
        os.path.join(output_folder, "noise_distributions_dependent.svg"),
        h=0.5,
    )

    # Plot noise comparison between generated and actual noise

    # Let's pick a single slice for this visualization

    slice: int = 112

    image = volume[:, :, slice]
    mask = mask[:, :, slice]

    plot_mask_and_pdf_comparison(
        image=image,
        mask=mask,
        noise_final=final_noise_dependent,
        output_path=os.path.join(output_folder, "final_noise_comparison_dependent.svg"),
    )

    plot_mask_and_pdf_comparison(
        image=image,
        mask=mask,
        noise_final=final_noise_independent,
        output_path=os.path.join(
            output_folder, "final_noise_comparison_independent.svg"
        ),
    )

    # 3D Case

    generate_3d_noise = False
    if generate_3d_noise:
        print("Generating 3D Noise")
        shape_3d = (volume.shape[0], volume.shape[1], volume.shape[2])
        real_vol, imag_vol, combined_vol = (
            BlindNoiseEstimation.gaussian_random_fields_noise_3d(
                model=best_model,
                shape=shape_3d,
                seed_real=1122022,
                seed_imag=23102003,
            )
        )
