import os
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import gstools as gs  # type: ignore
import pandas as pd

from scipy.stats import rice, rayleigh, ncx2, norm  # type: ignore
from scipy.interpolate import interp1d  # type: ignore

# Import the convex hull function from your package.
from Meningioma import ImageProcessing  # type: ignore

# ========================================================= HELPER FUNCTIONS =================================================================================


def compute_js_divergence(
    pdf1: np.ndarray, pdf2: np.ndarray, x_vals: np.ndarray, epsilon: float = 1e-10
) -> float:
    """
    Compute the Jensen–Shannon divergence between two PMFs.
    Assumes pdf1 and pdf2 are already discrete probability mass functions defined at x_vals.
    """
    # Add epsilon to avoid division by zero and log(0).
    p1 = pdf1 + epsilon
    p2 = pdf2 + epsilon
    p1 /= p1.sum()
    p2 /= p2.sum()
    m = 0.5 * (p1 + p2)
    kl1 = np.sum(p1 * np.log(p1 / m))
    kl2 = np.sum(p2 * np.log(p2 / m))
    return 0.5 * (kl1 + kl2)


def compute_pdf(data: np.ndarray, h: float, dist: str = "norm") -> tuple:
    """
    Estimate the probability density function (PDF) for a 1D data array by computing:
      - A Parzen–Rosenblatt KDE (using ImageProcessing.kde)
      - A theoretical PDF fitted to the data using one of several distributions.

    Parameters:
        data: 1D numpy array of noise values.
        h: Bandwidth for the KDE estimation.
        dist: A string specifying the distribution to fit. One of:
              "norm"     → Gaussian (normal) distribution.
              "rayleigh" → Rayleigh distribution.
              "rice"     → Rice distribution.
              "ncx2"     → Non-central chi-squared distribution.

    Returns:
        A tuple with:
          - x_common: Common x-axis values (numpy array).
          - kde_est: The KDE-estimated PDF (numpy array).
          - pdf_fit: The theoretical PDF evaluated on x_common (numpy array).
          - param_str: A string summarizing the fitted parameters.
          - param_series: A pandas Series with the fitted parameter values.
    """
    data = data.flatten()
    x_min = data.min()
    x_max = data.max()
    x_common = np.linspace(x_min, x_max, 1000)

    # Compute KDE using the custom function.
    kde_vals, x_kde = ImageProcessing.kde(
        data, h=h, num_points=1000, return_x_values=True
    )
    f_interp = interp1d(x_kde, kde_vals, bounds_error=False, fill_value=0)
    kde_est = f_interp(x_common)

    # Fit the theoretical distribution and compute the PDF.
    if dist.lower() == "norm":
        mu, sigma = norm.fit(data)
        pdf_fit = norm.pdf(x_common, loc=mu, scale=sigma)
        param_str = f"μ={mu:.2f}, σ={sigma:.2f}"
        param_series = pd.Series({"mu": mu, "sigma": sigma})
    elif dist.lower() == "rayleigh":
        loc, scale = rayleigh.fit(data)
        pdf_fit = rayleigh.pdf(x_common, loc=loc, scale=scale)
        # Report sigma as sqrt(scale) if desired.
        sigma_est = np.sqrt(scale)
        param_str = f"loc={loc:.2f}, σ̂={sigma_est:.2f}"
        param_series = pd.Series({"loc": loc, "scale": scale, "sigma": sigma_est})
    elif dist.lower() == "rice":
        # rice.fit returns: shape parameter b, location, and scale.
        b, loc, scale = rice.fit(data)
        pdf_fit = rice.pdf(x_common, b, loc=loc, scale=scale)
        param_str = f"b={b:.5f}, loc={loc:.2f}, scale={scale:.2f}"
        param_series = pd.Series({"b": b, "loc": loc, "scale": scale})
    elif dist.lower() == "ncx2":
        # ncx2.fit returns: degrees of freedom, noncentrality, location, and scale.
        df, nc, loc, scale = ncx2.fit(data)
        pdf_fit = ncx2.pdf(x_common, df, nc, loc=loc, scale=scale)
        sigma_est = np.sqrt(scale)
        param_str = f"L={df:.2f}, NC={nc:.2f}, σ={sigma_est:.2f}"
        param_series = pd.Series(
            {"df": df, "nc": nc, "loc": loc, "scale": scale, "sigma": sigma_est}
        )
    else:
        raise ValueError(
            f"Distribution '{dist}' not recognized. Choose among 'norm', 'rayleigh', 'rice', 'ncx2'."
        )

    return x_common, kde_est, pdf_fit, param_str, param_series


# ========================================================= HELPER FUNCTIONS =================================================================================


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
            models_dir = fit_model_3d(
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
        models = fit_model_3d(
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
    js_empirical = compute_js_divergence(emp_pdf_orig, emp_pdf_gen, bin_centers)
    js_rayleigh = compute_js_divergence(
        pdf_rayleigh_orig, pdf_rayleigh_gen, bin_centers
    )

    # --- Compute KDE estimates via Parzen–Rosenblatt method ---
    x_orig, kde_est_orig, _, _, _ = compute_pdf(original_bg, h=h, dist="norm")
    x_gen, kde_est_gen, _, _, _ = compute_pdf(generated_noise, h=h, dist="norm")

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
# 1. Variogram Estimation Functions (Using Only Background Voxels)
# =============================================================================
def estimate_variogram_isotropic_3d(
    data: np.ndarray,
    bins: np.ndarray,
    mask: Optional[np.ndarray],
    estimator: str = "matheron",
    sampling_size: int = 2000,
    sampling_seed: int = 19920516,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the isotropic variogram from 3D data using gstools.vario_estimate.
    Only voxels where the mask is False (i.e. background) are used.

    Parameters
    ----------
    data : np.ndarray
        3D volume data.
    bins : np.ndarray
        1D array defining the bin edges.
    mask : Optional[np.ndarray]
        3D boolean exclusion mask.
    sampling_size : int, optional
        Number of voxel pairs to sample.
    sampling_seed : int, optional
        Seed for random sampling.

    Returns
    -------
    bin_centers : np.ndarray
        Centers of the distance bins.
    gamma : np.ndarray
        Estimated variogram values.
    """
    # Only use pixels outside the mask (background).
    if mask is not None:
        valid_indices = np.flatnonzero(np.logical_not(mask.flatten()))
    else:
        valid_indices = np.arange(data.size)
    valid_data = data.flatten()[valid_indices]

    # Create a grid of coordinates.
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])
    pos_x, pos_y, pos_z = np.meshgrid(x, y, z, indexing="ij")
    pos_all = np.vstack((pos_x.flatten(), pos_y.flatten(), pos_z.flatten()))
    pos_valid = pos_all[:, valid_indices]

    if valid_data.size < sampling_size:
        sampling_size = valid_data.size

    print(f"Valid voxel positions: {valid_data.size}")

    bin_centers, gamma = gs.vario_estimate(
        pos=pos_valid,
        field=valid_data,
        bin_edges=bins,
        mesh_type="unstructured",
        estimator=estimator,
        sampling_size=sampling_size,
        sampling_seed=sampling_seed,
    )
    return bin_centers, gamma


def estimate_variogram_anisotropic_3d(
    data: np.ndarray,
    bins: np.ndarray,
    mask: Optional[np.ndarray] = None,
    directions: Optional[List[np.ndarray]] = None,
    direction_labels: Optional[List[str]] = None,
    estimator: str = "matheron",
    angles_tol: float = np.pi / 8,
    sampling_size: int = 2000,
    sampling_seed: int = 19920516,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate directional variograms from 3D data using gstools.vario_estimate.
    Only background voxels (mask False) are used.

    Parameters
    ----------
    data : np.ndarray
        3D volume data.
    bins : np.ndarray
        1D array defining the bin edges.
    mask : Optional[np.ndarray]
        3D boolean exclusion mask.
    directions : Optional[List[np.ndarray]], optional
        List of 3D direction vectors. If None, a default set of 7 directions is used.
        (Note: To obtain eight directions, provide a custom list.)
    direction_labels : Optional[List[str]], optional
        List of labels corresponding to each direction. If None and directions is provided,
        the default labels are generated as "Direction 1", "Direction 2", etc.
    angles_tol : float, optional
        Tolerance for directional variogram (in radians).
    sampling_size : int, optional
        Number of voxel pairs to sample.
    sampling_seed : int, optional
        Seed for random sampling.

    Returns
    -------
    variograms : Dict[str, Tuple[np.ndarray, np.ndarray]]
        Mapping of direction labels to (bin_centers, variogram values).
    """
    # Only use pixels outside the mask.
    if mask is not None:
        valid_indices = np.flatnonzero(np.logical_not(mask.flatten()))
    else:
        valid_indices = np.arange(data.size)
    valid_data = data.flatten()[valid_indices]

    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])
    pos_x, pos_y, pos_z = np.meshgrid(x, y, z, indexing="ij")
    pos_all = np.vstack((pos_x.flatten(), pos_y.flatten(), pos_z.flatten()))
    pos_valid = pos_all[:, valid_indices]

    print(f"Valid voxel positions: {valid_data.size}")
    if valid_data.size == 0:
        raise ValueError("No valid voxel positions remain after applying mask.")
    sampling_size = min(sampling_size, valid_data.size)

    # Use provided directions if given; otherwise, use a default list of 8 directions.
    if directions is None:
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
    else:
        if direction_labels is None:
            direction_labels = [f"Direction {i+1}" for i in range(len(directions))]

    variograms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for direction, label in zip(directions, direction_labels):
        print(f"Estimating anisotropic variogram for direction: {label}")
        bin_centers, gamma = gs.vario_estimate(
            pos=pos_valid,
            field=valid_data,
            bin_edges=bins,
            mesh_type="unstructured",
            estimator=estimator,
            direction=[direction],
            angles_tol=angles_tol,
            sampling_size=sampling_size,
            sampling_seed=sampling_seed,
        )
        variograms[label] = (bin_centers, gamma)
    return variograms


# =============================================================================
# 2. Covariance Model Fitting Function (3D)
# =============================================================================
def fit_model_3d(
    bin_center: np.ndarray,
    gamma: np.ndarray,
    var: float = 1.0,
    len_scale: float = 10.0,
) -> Dict[str, Tuple[gs.CovModel, Dict[str, Any]]]:
    """
    Fit several theoretical variogram models to the estimated 3D variogram.
    (Some models may not converge; this is reported in the console.)

    Parameters
    ----------
    bin_center : np.ndarray
        Centers of the distance bins.
    gamma : np.ndarray
        Estimated variogram values.
    var : float, optional
        Initial variance guess.
    len_scale : float, optional
        Initial guess for the correlation length scale.

    Returns
    -------
    results : Dict[str, Tuple[gs.CovModel, Dict[str, Any]]]
        Mapping of model names to (fitted model, fit parameters including r^2).
    """
    models = {
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
    print("Fitting 3D covariance models")
    results: Dict[str, Tuple[gs.CovModel, Dict[str, Any]]] = {}
    for model_name, model_class in models.items():
        try:
            model = model_class(dim=3, var=var, len_scale=len_scale)
            params, pcov, r2 = model.fit_variogram(bin_center, gamma, return_r2=True)
            results[model_name] = (model, {"params": params, "pcov": pcov, "r2": r2})
            print(f"Model {model_name} fitted with r^2 = {r2:.3f}")
        except Exception as e:
            print(f"Model {model_name} failed to fit: {e}")
    return results


# =============================================================================
# 3. Gaussian Random Fields generation
# =============================================================================


def gaussian_random_fields_noise_2d(
    model: gs.CovModel,
    shape: Tuple[int, int],
    independent: bool = True,
    seed_real: int = 19770928,
    seed_imag: int = 19773022,
    seed_3d: int = 19770928,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic Gaussian random field noise for a 2D image.

    Two modes are available:

      1. **independent=True** (default):
         Two independent 2D Gaussian random fields (representing the real and imaginary parts)
         are generated using gs.SRF with a "structured" mesh. The final noise is computed as:

             noise = sqrt(real_field^2 + imag_field^2)

         The use of two independent seeds ensures that the two fields are generated independently.

      2. **independent=False**:
         A single 3D Gaussian random field is generated over a volume of shape (n, m, 2)
         using gs.SRF with a "structured" mesh. The two slices along the third dimension are then
         extracted as the real and imaginary parts and combined via the modulus operation:

             noise = sqrt(slice_0^2 + slice_1^2)

         In this case the two channels are correlated as they come from one 3D realization.

    **Mesh Type Considerations:**
    When using gs.SRF, the `mesh_type` parameter determines how the input coordinate tuple is interpreted.
    With `"structured"`, the provided arrays (or ranges) define the grid along each axis, which is ideal
    for regularly spaced images or volumes. For irregular grids one might use `"unstructured"`.

    Parameters
    ----------
    model : gs.CovModel
        The best-fit covariance model to be used for generating the noise.
    shape : Tuple[int, int]
        A tuple (n, m) defining the size of the 2D image.
    independent : bool, optional
        If True (default), generate two independent 2D fields (using separate seeds).
        If False, generate one 3D field of shape (n, m, 2) and extract two slices.
    seed_real : int, optional
        Random seed for the real part (used if independent is True).
    seed_imag : int, optional
        Random seed for the imaginary part (used if independent is True).
    seed_3d : int, optional
        Random seed for the 3D field (used if independent is False).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple (real_field, imag_field, combined_field) where each array has shape (n, m).
        The combined field is computed as sqrt(real_field**2 + imag_field**2).
    """
    n, m = shape

    if independent:
        # Generate independent fields for real and imaginary parts using a structured mesh.
        x = np.arange(n)
        y = np.arange(m)
        z = np.arange(1)
        # Generate real part.
        srf_real = gs.SRF(model, seed=seed_real)
        real_field = srf_real((x, y, z), mesh_type="structured")
        # Generate imaginary part.
        srf_imag = gs.SRF(model, seed=seed_imag)
        imag_field = srf_imag((x, y, z), mesh_type="structured")

    else:
        # Generate a single 3D volume of noise with two slices along the third dimension.
        x = np.arange(n)
        y = np.arange(m)
        z = np.arange(2)  # Two slices.
        srf_3d = gs.SRF(model, seed=seed_3d)
        # Use a structured mesh since the grid is regularly spaced.
        volume_3d = srf_3d((x, y, z), mesh_type="structured")
        # Extract the two slices.
        real_field = volume_3d[:, :, 0]
        imag_field = volume_3d[:, :, 1]

    # Combine the two fields via the modulus operation.
    combined_field = (np.sqrt(real_field**2 + imag_field**2)) / (np.sqrt(2))
    return real_field, imag_field, combined_field


def gaussian_random_fields_noise_3d(
    model: gs.CovModel,
    shape: Tuple[int, int, int],
    seed_real: int = 19770928,
    seed_imag: int = 19773022,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate independent 3D Gaussian random field volumes representing the real and
    imaginary parts of noise and combine them using the modulus operation.

    In this function, two independent 3D fields are simulated over a structured grid
    defined by `shape` = (nx, ny, nz) using two separate seeds. The final noise volume
    is computed as:

        combined = sqrt(real_field**2 + imag_field**2) / sqrt(2)

    The division by sqrt(2) normalizes the noise (consistent with the 2D version).

    Parameters
    ----------
    model : gs.CovModel
        The best-fit covariance model to use for noise generation.
    shape : Tuple[int, int, int]
        Desired shape of the noise volume as (nx, ny, nz).
    seed_real : int, optional
        Random seed for generating the real part of the noise (default 19770928).
    seed_imag : int, optional
        Random seed for generating the imaginary part of the noise (default 19773022).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple (real_volume, imag_volume, combined_volume), where each array has shape `shape`.
        The combined_volume is computed as the modulus (Euclidean norm) of the two volumes.
    """
    nx, ny, nz = shape
    # Define coordinate arrays for a structured grid.
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)

    # Generate the real noise volume.
    srf_real = gs.SRF(model, seed=seed_real)
    real_volume = srf_real((x, y, z), mesh_type="structured")

    # Generate the imaginary noise volume.
    srf_imag = gs.SRF(model, seed=seed_imag)
    imag_volume = srf_imag((x, y, z), mesh_type="structured")

    # Normalize signal amplitude
    real_volume = real_volume / np.sqrt(2)
    imag_volume = imag_volume / np.sqrt(2)

    # Combine the two volumes by taking the Euclidean norm and normalize by √2.
    combined_volume = np.sqrt(real_volume**2 + imag_volume**2)

    return real_volume, imag_volume, combined_volume


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
    iso_bin_center, iso_gamma = estimate_variogram_isotropic_3d(
        data=volume,
        bins=variogram_bins,
        mask=mask,
        estimator=estimator,
        sampling_size=variogram_sampling_size,
        sampling_seed=variogram_sampling_seed,
    )
    iso_models = fit_model_3d(
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
    anisotropic_variograms = estimate_variogram_anisotropic_3d(
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
        anisotropic_models[direction_label] = fit_model_3d(
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
        gaussian_random_fields_noise_2d(
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
        gaussian_random_fields_noise_2d(
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
        real_vol, imag_vol, combined_vol = gaussian_random_fields_noise_3d(
            model=best_model,
            shape=shape_3d,
            seed_real=1122022,
            seed_imag=23102003,
        )
