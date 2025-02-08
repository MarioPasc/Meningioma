from typing import Any, Tuple, Dict, List
from numpy.typing import NDArray
import os

import numpy as np
import pandas as pd
import csv
import gstools as gs  # type: ignore

from Meningioma.image_processing import ImageProcessing  # type: ignore
from Meningioma.utils import Stats  # type: ignore

from scipy.stats import rice, rayleigh, ncx2, norm  # type: ignore
from scipy.interpolate import interp1d  # type: ignore

import matplotlib.pyplot as plt
from cycler import cycler
import scienceplots  # type: ignore
import pyvista as pv  # type: ignore
from threedim_pipeline import fit_model_3d  # type: ignore

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


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
    plt.show()


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


def plot_noise_distributions(
    noise_real: np.ndarray,
    noise_imag: np.ndarray,
    noise_final: np.ndarray,
    output_path: str,
    h: float = 1.0,
) -> None:
    """
    Create a 2x3 subplot figure displaying the noise images and their corresponding PMFs.

    Top row:
      - [0,0]: Real noise image.
      - [0,1]: Imaginary noise image.
      - [0,2]: Final combined noise image.

    Bottom row (PMF plots):
      - [1,0]: Real noise: Empirical PMF (bars in yellow, color "#DDAA33") and theoretical
               Gaussian PMF (dashed blue) with parameters (\mu, \sigma).
      - [1,1]: Imaginary noise: Empirical PMF and theoretical Gaussian PMF.
      - [1,2]: Final noise: Empirical PMF, theoretical Rayleigh PMF (dashed red; \(\sigma=\sqrt{scale}\))
               and NC-\(\chi^2\) PMF (dashed green, "#117733"; legend shows degrees of freedom \(L\), noncentrality,
               and \(\sigma=\sqrt{scale}\)).

    In all cases, the theoretical PMFs are computed by differencing the CDF at the histogram bin edges.

    Parameters:
        noise_real: 2D array for the generated real noise component.
        noise_imag: 2D array for the generated imaginary noise component.
        noise_final: 2D array for the generated magnitude (final) noise.
        output_path: Path where to save the figure.
        h: Bandwidth for the KDE estimation (not used here, retained for interface compatibility).
    """
    # --- Prepare empirical PMFs for each noise image ---
    # For consistency with the attached example we define bins as if values were quantized.
    # Real noise.
    real_vals = noise_real.flatten()
    bins_real = np.arange(np.min(real_vals), np.max(real_vals) + 2) - 0.5
    hist_real, bin_edges_real = np.histogram(real_vals, bins=bins_real, density=False)
    pmf_real = hist_real / hist_real.sum()
    bin_centers_real = (bin_edges_real[:-1] + bin_edges_real[1:]) / 2

    # Imaginary noise.
    imag_vals = noise_imag.flatten()
    bins_imag = np.arange(np.min(imag_vals), np.max(imag_vals) + 2) - 0.5
    hist_imag, bin_edges_imag = np.histogram(imag_vals, bins=bins_imag, density=False)
    pmf_imag = hist_imag / hist_imag.sum()
    bin_centers_imag = (bin_edges_imag[:-1] + bin_edges_imag[1:]) / 2

    # Final noise.
    final_vals = noise_final.flatten()
    bins_final = np.arange(np.min(final_vals), np.max(final_vals) + 2) - 0.5
    hist_final, bin_edges_final = np.histogram(
        final_vals, bins=bins_final, density=False
    )
    pmf_final = hist_final / hist_final.sum()
    bin_centers_final = (bin_edges_final[:-1] + bin_edges_final[1:]) / 2

    # --- Fit distributions and compute theoretical PMFs via CDF differencing ---
    # For real and imaginary noise: Gaussian fit.
    mu_real, sigma_real = norm.fit(real_vals)
    mu_imag, sigma_imag = norm.fit(imag_vals)
    theo_pmf_gauss_real = norm.cdf(
        bin_edges_real[1:], loc=mu_real, scale=sigma_real
    ) - norm.cdf(bin_edges_real[:-1], loc=mu_real, scale=sigma_real)
    theo_pmf_gauss_imag = norm.cdf(
        bin_edges_imag[1:], loc=mu_imag, scale=sigma_imag
    ) - norm.cdf(bin_edges_imag[:-1], loc=mu_imag, scale=sigma_imag)

    # For final noise: Rayleigh and NC-chi2 fits.
    loc_r, scale_r = rayleigh.fit(final_vals)
    df_ncx2, nc_ncx2, loc_ncx2, scale_ncx2 = ncx2.fit(final_vals)
    theo_pmf_rayleigh = rayleigh.cdf(
        bin_edges_final[1:], loc=loc_r, scale=scale_r
    ) - rayleigh.cdf(bin_edges_final[:-1], loc=loc_r, scale=scale_r)
    theo_pmf_ncx2 = ncx2.cdf(
        bin_edges_final[1:], df_ncx2, nc_ncx2, loc_ncx2, scale_ncx2
    ) - ncx2.cdf(bin_edges_final[:-1], df_ncx2, nc_ncx2, loc_ncx2, scale_ncx2)

    # --- Create 2x3 subplot figure ---
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Top row: Noise images.
    im0 = axs[0, 0].imshow(noise_real, cmap="gray", aspect="auto")
    axs[0, 0].set_title("Real Noise Image")
    axs[0, 0].set_xlabel("X")
    axs[0, 0].set_ylabel("Y")

    im1 = axs[0, 1].imshow(noise_imag, cmap="gray", aspect="auto")
    axs[0, 1].set_title("Imaginary Noise Image")
    axs[0, 1].set_xlabel("X")
    axs[0, 1].set_ylabel("Y")

    im2 = axs[0, 2].imshow(noise_final, cmap="gray", aspect="auto")
    axs[0, 2].set_title("Final Noise Image")
    axs[0, 2].set_xlabel("X")
    axs[0, 2].set_ylabel("Y")

    # Bottom row: PMF plots.
    # [1,0] Real noise PMF.
    ax = axs[1, 0]
    # Plot the empirical PMF as bars (color yellow).
    ax.bar(
        bin_centers_real,
        pmf_real,
        width=1,
        alpha=0.3,
        color="#DDAA33",
        label="Empirical PMF",
    )
    # Overlay the theoretical Gaussian PMF (dashed blue).
    ax.plot(
        bin_centers_real,
        theo_pmf_gauss_real,
        linestyle="--",
        color="blue",
        label=rf"Gaussian: $\mu={mu_real:.2f},\ \sigma={sigma_real:.2f}$",
    )
    ax.set_title("Real Noise PMF")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Mass")
    ax.legend(loc="best")

    # [1,1] Imaginary noise PMF.
    ax = axs[1, 1]
    ax.bar(
        bin_centers_imag,
        pmf_imag,
        width=1,
        alpha=0.3,
        color="#DDAA33",
        label="Empirical PMF",
    )
    ax.plot(
        bin_centers_imag,
        theo_pmf_gauss_imag,
        linestyle="--",
        color="blue",
        label=rf"Gaussian: $\mu={mu_imag:.2f},\ \sigma={sigma_imag:.2f}$",
    )
    ax.set_title("Imaginary Noise PMF")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Mass")
    ax.legend(loc="best")

    # [1,2] Final noise PMF.
    ax = axs[1, 2]
    ax.bar(
        bin_centers_final,
        pmf_final,
        width=1,
        alpha=0.3,
        color="#DDAA33",
        label="Empirical PMF",
    )
    # Overlay the theoretical Rayleigh PMF (dashed red).
    ax.plot(
        bin_centers_final,
        theo_pmf_rayleigh,
        linestyle="--",
        color="red",
        label=rf"Rayleigh: $\sigma={np.sqrt(scale_r):.2f}$",
    )
    # Overlay the theoretical NC-chi2 PMF (dashed green, "#117733").
    ax.plot(
        bin_centers_final,
        theo_pmf_ncx2,
        linestyle="--",
        color="#117733",
        label=rf"NC-$\chi^2$: $L={df_ncx2:.2f},\ NC={nc_ncx2:.2f},\ \sigma={np.sqrt(scale_ncx2):.2f}$",
    )
    ax.set_title("Final Noise PMF")
    ax.set_xlabel("Value")
    ax.set_ylabel("Probability Mass")
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
