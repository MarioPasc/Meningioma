from typing import Any, Tuple, Dict, List
from numpy.typing import NDArray
import os

import numpy as np
import pandas as pd
import csv
import gstools as gs  # type: ignore
from gstools.covmodel.plot import plot_variogram  # type: ignore

from Meningioma.image_processing import ImageProcessing  # type: ignore
from Meningioma.utils import Stats, npz_converter  # type: ignore

import matplotlib.pyplot as plt
from cycler import cycler
import scienceplots  # type: ignore
import pyvista as pv  # type: ignore

# Import the required distributions from scipy.stats
from scipy.stats import rice, rayleigh

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Global flag to disable colormaps and axis details (labels, ticks, numeration)
DISABLE_DETAILS = False


def disable_details(ax: plt.Axes) -> None:
    """
    If DISABLE_DETAILS is True, disable axis labels, ticks, and numeration.
    """
    if DISABLE_DETAILS:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(bottom=False, top=False, left=False, right=False)


def load_data(file_path: str, slice_index: int) -> np.ndarray:
    """
    Load the MRI data from a .npz file and extract a single slice.
    """
    data = np.load(file_path)
    slice_data = np.squeeze(data["data"][0, :, :, slice_index])
    return slice_data


def extract_phase_from_kspace(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate k-space from an image and extract the phase data.
    """
    k_space = np.fft.fftshift(np.fft.fft2(image))
    phase = np.angle(k_space)
    return phase, k_space


def to_real_imag(
    magnitude: np.ndarray, phase: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert magnitude data into synthetic complex data by applying a phase.
    """
    real_part = magnitude * np.cos(phase)
    imag_part = magnitude * np.sin(phase)
    return real_part, imag_part


def estimate_variogram_isotropic(
    data: np.ndarray,
    bins: np.ndarray,
    mask: NDArray[np.bool_],
    sampling_size: int = 2000,
    sampling_seed: int = 19920516,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the variogram from 2D data using the gstools vario_estimate method.
    """
    if mask is not None:
        valid_indices = np.argwhere(mask.flatten()).flatten()
        valid_data = data.flatten()[valid_indices]
        pos_x, pos_y = np.meshgrid(
            np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij"
        )
        pos_flat = np.vstack((pos_x.flatten(), pos_y.flatten()))[:, valid_indices]
    else:
        valid_data = data.flatten()
        pos_x, pos_y = np.meshgrid(
            np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij"
        )
        pos_flat = np.vstack((pos_x.flatten(), pos_y.flatten()))
    print(f"Valid positions: {len(valid_data)}")
    assert len(valid_data) > sampling_size, "Sampling size exceeds valid positions."
    sampling_size = min(sampling_size, len(valid_data))
    bin_centers, gamma = gs.vario_estimate(
        pos=pos_flat,
        field=valid_data,
        bin_edges=bins,
        mesh_type="unstructured",
        sampling_size=sampling_size,
        sampling_seed=sampling_seed,
    )
    return bin_centers, gamma


def estimate_variogram_anisotropic(
    data: np.ndarray,
    bins: np.ndarray,
    mask: NDArray[np.bool_] = None,
    directions: List[np.ndarray] = None,
    angles_tol: float = np.pi / 8,
    sampling_size: int = 2000,
    sampling_seed: int = 19920516,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate the directional variogram from 2D data using gstools vario_estimate.
    """
    if mask is not None:
        valid_indices = np.argwhere(mask.flatten()).flatten()
        valid_data = data.flatten()[valid_indices]
        pos_x, pos_y = np.meshgrid(
            np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij"
        )
        pos_flat = np.vstack((pos_x.flatten(), pos_y.flatten()))[:, valid_indices]
    else:
        valid_data = data.flatten()
        pos_x, pos_y = np.meshgrid(
            np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij"
        )
        pos_flat = np.vstack((pos_x.flatten(), pos_y.flatten()))
    print(f"Valid positions: {len(valid_data)}")
    assert len(valid_data) > 0, "No valid positions remain after applying mask."
    sampling_size = min(sampling_size, len(valid_data))
    if directions is None:
        directions = [
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
            np.array([-1, 1]),
        ]
        direction_labels = ["Horizontal", "Vertical", "Diagonal_45", "Diagonal_135"]
    else:
        direction_labels = [f"Direction {i+1}" for i in range(len(directions))]
    variograms = {}
    for direction, label in zip(directions, direction_labels):
        print(f"Fitting anisotropic variogram for direction: {direction}")
        bin_centers, gamma = gs.vario_estimate(
            pos=pos_flat,
            field=valid_data,
            bin_edges=bins,
            mesh_type="unstructured",
            direction=[direction],
            angles_tol=angles_tol,
            sampling_size=sampling_size,
            sampling_seed=sampling_seed,
        )
        variograms[label] = (bin_centers, gamma)
    return variograms


def fit_model(
    bin_center: np.ndarray,
    gamma: np.ndarray,
    var: float = 1.0,
    len_scale: float = 10.0,
    nugget: bool = True,
) -> Dict[str, Tuple[gs.CovModel, Dict[str, Any]]]:
    """
    Fit multiple theoretical variogram models to the estimated variogram.
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
    print("Fitting isotropic model")
    results = {}
    for model_name, model_class in models.items():
        try:
            model = model_class(dim=2, var=var, len_scale=len_scale)
            params, pcov, r2 = model.fit_variogram(bin_center, gamma, return_r2=True)
            results[model_name] = (model, {"params": params, "pcov": pcov, "r2": r2})
        except Exception as e:
            print(f"Model {model_name} failed to fit: {e}")
    return results


def generate_random_fields(
    model: gs.CovModel,
    shape: Tuple[int, int],
    seed_real: int = 19770928,
    seed_imag: int = 19773022,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic random fields for the real and imaginary parts and combine them.
    The final noise field is computed as:
        I = sqrt(real^2 + imag^2)
    Note that even if the real and imaginary parts are zero-mean, I follows a Rayleigh
    distribution with expected mean σ√(π/2) (with σ the scale parameter). We will subtract
    this bias (estimated via scipy.stats) so that the final noise has zero mean.
    """
    n, m = shape
    x_s = np.arange(n)
    y_s = np.arange(m)
    srf_real = gs.SRF(model, seed=seed_real)
    new_data_real = srf_real((x_s, y_s), mesh_type="structured")
    srf_imag = gs.SRF(model, seed=seed_imag)
    new_data_imag = srf_imag((x_s, y_s), mesh_type="structured")
    new_data = np.sqrt(new_data_real**2 + new_data_imag**2)
    return new_data_real, new_data_imag, new_data


def plot_noise_distributions(
    noise_real: np.ndarray,
    noise_imag: np.ndarray,
    noise_final: np.ndarray,
    output_path: str,
    h: float = 1.0,
) -> None:
    """
    Plot the probability density functions (PDFs) of the GRF-generated noise.
    The top subplot shows the Gaussian PDFs (via KDE) of the real and imaginary parts,
    with legends indicating their means and standard deviations.
    The bottom subplot shows the KDE of the Rayleigh (magnitude) noise along with
    the theoretical Rayleigh PDF computed from the estimated scale parameter (using scipy.stats).
    The legend includes the estimated parameters using LaTeX notation.

    Parameters:
        noise_real: 2D array for the generated real part.
        noise_imag: 2D array for the generated imaginary part.
        noise_final: 2D array for the magnitude (Rayleigh) noise.
        output_path: Path where to save the figure.
        h: Bandwidth for the KDE.
    """
    # Use the ImageProcessing.kde function for density estimation.
    from Meningioma.image_processing import ImageProcessing

    # Flatten the arrays.
    real_vals = noise_real.flatten()
    imag_vals = noise_imag.flatten()
    rayleigh_vals = noise_final.flatten()

    # Get KDE estimates.
    kde_real, x_real = ImageProcessing.kde(
        real_vals, h=h, num_points=1000, return_x_values=True
    )
    kde_imag, x_imag = ImageProcessing.kde(
        imag_vals, h=h, num_points=1000, return_x_values=True
    )
    kde_rayleigh, x_rayleigh = ImageProcessing.kde(
        rayleigh_vals, h=h, num_points=1000, return_x_values=True
    )

    # Compute sample statistics.
    mean_real, std_real = np.mean(real_vals), np.std(real_vals)
    mean_imag, std_imag = np.mean(imag_vals), np.std(imag_vals)
    mean_rayleigh, std_rayleigh = np.mean(rayleigh_vals), np.std(rayleigh_vals)

    # Estimate parameters for the Rayleigh distribution using scipy.stats.
    # The fit method returns (loc, scale)
    loc_r, scale_r = rayleigh.fit(rayleigh_vals)
    scipy_bias = scale_r * np.sqrt(np.pi / 2)

    # Theoretical Rayleigh PDF.
    def rayleigh_pdf(r, loc, scale):
        return (r - loc) / scale**2 * np.exp(-((r - loc) ** 2) / (2 * scale**2))

    x_theoretical = np.linspace(0, np.max(rayleigh_vals), 1000)
    y_theoretical = rayleigh_pdf(x_theoretical, loc_r, scale_r)

    # Create figure with two subplots.
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(10, 6))

    # Subplot 1: Gaussian PDFs for real and imaginary parts.
    axs[0].plot(
        x_real,
        kde_real,
        label=rf"Real: $\mu={mean_real:.2f},\ \sigma={std_real:.2f}$",
        color="blue",
    )
    axs[0].plot(
        x_imag,
        kde_imag,
        label=rf"Imaginary: $\mu={mean_imag:.2f},\ \sigma={std_imag:.2f}$",
        color="orange",
    )
    axs[0].set_title("Gaussian Distributions of Real and Imaginary Generated Noise")
    axs[0].set_xlabel("Value")
    axs[0].set_ylabel("Density")
    axs[0].legend()

    # Subplot 2: KDE of the Rayleigh (magnitude) noise and theoretical Rayleigh PDF.
    # Use a LaTeX line break (\\) for a multi-line legend.
    axs[1].plot(
        x_rayleigh,
        kde_rayleigh,
        label=rf"Rayleigh: $\mu={mean_rayleigh:.2f},\ \sigma={std_rayleigh:.2f}$\\"
        rf"$\hat{{\sigma}}={scale_r:.2f}$, Bias={scipy_bias:.2f}",
        color="green",
    )
    axs[1].plot(
        x_theoretical,
        y_theoretical,
        label="Theoretical Rayleigh",
        color="red",
        linestyle="--",
    )
    axs[1].set_title("Rayleigh Distribution from Generated Noise")
    axs[1].set_xlabel("Value")
    axs[1].set_ylabel("Density")
    axs[1].legend()

    for ax in axs:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Noise distributions saved to {output_path}")


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rice, rayleigh, ncx2

def plot_mask_and_distribution(image: np.ndarray, mask: np.ndarray, output_path: str) -> None:
    """
    Create a two-panel figure:
      - Left panel: The original image with the overlayed mask (displayed in red with alpha=0.6).
      - Right panel: Empirical PMFs (from normalized histograms) and the corresponding fitted discrete PMFs.
          * Rice distribution fitted to the entire (non-masked) image.
          * Rayleigh and non-central chi-square (NC χ²) distributions fitted to the background (pixels outside the mask).
    
    The fitted parameters are displayed in the legends.
    
    Parameters:
        image: The original image (2D array).
        mask: Boolean mask (2D array).
        output_path: Path where to save the figure.
    """
    # --- Left Panel: Display image with mask overlay ---
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # Show the image in grayscale.
    axs[0].imshow(image, cmap="gray", origin="lower")
    axs[0].set_title("Original Image with Mask")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    # Create an overlay from the mask (red color).
    mask_overlay = np.where(mask, 1, np.nan)
    axs[0].imshow(mask_overlay, cmap="Reds_r", alpha=0.6, origin="lower")
    
    # --- Right Panel: Empirical and Fitted PMFs ---
    # Extract values.
    rice_vals = image.flatten()  # All pixels for the Rice distribution.
    background_vals = image[~mask].flatten()  # Background pixels for Rayleigh and NC χ².
    
    # Fit distributions.
    # Rice distribution fit (returns: shape parameter 'b', location, and scale).
    b, loc_rice, scale_rice = rice.fit(rice_vals)
    # Rayleigh distribution fit (returns: location and scale).
    loc_rayleigh, scale_rayleigh = rayleigh.fit(background_vals)
    # NC χ² distribution fit (returns: degrees of freedom, noncentrality, location, and scale).
    df_ncx2, nc_ncx2, loc_ncx2, scale_ncx2 = ncx2.fit(background_vals)
    
    # Create histogram bins assuming the image intensities are quantized.
    # For the Rice distribution (all pixel values).
    bins_rice = np.arange(np.min(rice_vals), np.max(rice_vals) + 2) - 0.5  # bins for integer values
    hist_rice, bin_edges_rice = np.histogram(rice_vals, bins=bins_rice, density=False)
    pmf_rice = hist_rice / hist_rice.sum()  # Normalize to get a PMF.
    bin_centers_rice = (bin_edges_rice[:-1] + bin_edges_rice[1:]) / 2
    
    # For background pixels (Rayleigh and NC χ²).
    bins_bkg = np.arange(np.min(background_vals), np.max(background_vals) + 2) - 0.5
    hist_bkg, bin_edges_bkg = np.histogram(background_vals, bins=bins_bkg, density=False)
    pmf_bkg = hist_bkg / hist_bkg.sum()
    bin_centers_bkg = (bin_edges_bkg[:-1] + bin_edges_bkg[1:]) / 2
    
    # Compute theoretical PMFs by differencing the CDF at the bin edges.
    theo_pmf_rice = rice.cdf(bin_edges_rice[1:], b, loc_rice, scale_rice) - \
                    rice.cdf(bin_edges_rice[:-1], b, loc_rice, scale_rice)
    theo_pmf_rayleigh = rayleigh.cdf(bin_edges_bkg[1:], loc_rayleigh, scale_rayleigh) - \
                        rayleigh.cdf(bin_edges_bkg[:-1], loc_rayleigh, scale_rayleigh)
    theo_pmf_ncx2 = ncx2.cdf(bin_edges_bkg[1:], df_ncx2, nc_ncx2, loc_ncx2, scale_ncx2) - \
                    ncx2.cdf(bin_edges_bkg[:-1], df_ncx2, nc_ncx2, loc_ncx2, scale_ncx2)
    
    # Plot empirical PMF and theoretical PMF for each distribution.
    ax = axs[1]
    # Rice distribution.
    ax.plot(bin_centers_rice, theo_pmf_rice, color="blue", marker="o",
            label=rf"Rice fit: $b={b:.5f},\ loc={loc_rice:.2f}$\\$\mathrm{{scale}}={scale_rice:.2f}$")
    # Rayleigh distribution.
    ax.plot(bin_centers_bkg, theo_pmf_rayleigh, color="red", marker="o",
            label=rf"Rayleigh fit: $loc={loc_rayleigh:.2f}$\\$\mathrm{{scale}}={scale_rayleigh:.2f}$")
    # NC χ² distribution.
    ax.plot(bin_centers_bkg, theo_pmf_ncx2, color="green", marker="o",
            label=rf"NC-$\chi^2$ fit: $loc={loc_ncx2:.2f}$, $\mathrm{{scale}}={scale_ncx2:.2f}$\\$\lambda={df_ncx2:.2f}$, $NC={nc_ncx2:.2f}$")
    
    # PMF
    ax.bar(bin_centers_rice, pmf_rice, width=1, alpha=0.3, color="blue", label="Empirical Image PMF")
    ax.bar(bin_centers_bkg, pmf_bkg, width=1, alpha=0.3, color="red", label="Empirical Background Pixels PMF")
    
    ax.set_title("Empirical and Fitted PMFs")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Probability Mass")
    ax.legend()
    
    # Improve appearance by removing top and right spines.
    for axis in axs:
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)
        axis.xaxis.tick_bottom()
        axis.yaxis.tick_left()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_mask_and_pdf_comparison(
    image: np.ndarray,
    mask: np.ndarray,
    noise_final: np.ndarray,
    output_path: str,
    h: float = 1.0,
) -> None:
    """
    Create a three-panel figure:
      - Subplot 1: The original image with the overlayed mask (mask shown in red with α=0.6).
      - Subplot 2: Theoretical Rayleigh PDFs computed via scipy.stats.rayleigh.fit for:
            (i) original background noise (pixels outside mask) in red (dashed),
           (ii) generated noise (noise_final) in blue (dashed).
      - Subplot 3: Empirical PDFs (via KDE) for the same two distributions, using h=1.0.
           The original is shown in red (solid) and the generated in blue (solid).
      Both subplots 2 and 3 share the same y-axis limits.
    Additionally, the function computes and prints the Jensen–Shannon divergence between:
      - The theoretical PDFs (original vs. generated),
      - The empirical PDFs (original vs. generated).
    """
    # Extract original background noise (pixels outside the mask).
    original_bg = image[~mask].flatten()
    # Use the final generated noise; assume noise_final is the (bias-corrected) generated slice.
    generated_noise = noise_final.flatten()

    # --- Theoretical PDFs via scipy.stats.rayleigh.fit ---
    # Fit the Rayleigh distribution to the original background.
    loc_orig, scale_orig = rayleigh.fit(original_bg)
    # Fit to the generated noise.
    loc_gen, scale_gen = rayleigh.fit(generated_noise)
    # Define a common x-axis range.
    x = np.linspace(0, max(np.max(original_bg), np.max(generated_noise)), 1000)
    pdf_orig_theo = rayleigh.pdf(x, loc_orig, scale_orig)
    pdf_gen_theo = rayleigh.pdf(x, loc_gen, scale_gen)

    # --- Empirical PDFs via KDE ---
    # Use your ImageProcessing.kde function.
    from Meningioma.image_processing import ImageProcessing

    # Here we force a common range for the empirical estimation.
    x_emp = np.linspace(0, max(np.min(original_bg), np.min(generated_noise)), 1000)
    x_emp = np.linspace(0, max(np.max(original_bg), np.max(generated_noise)), 1000)
    kde_orig, x_orig = ImageProcessing.kde(
        original_bg, h=h, num_points=1000, return_x_values=True
    )
    kde_gen, x_gen = ImageProcessing.kde(
        generated_noise, h=h, num_points=1000, return_x_values=True
    )
    # Re-interpolate both onto x_emp.
    from scipy.interpolate import interp1d

    f_orig = interp1d(x_orig, kde_orig, bounds_error=False, fill_value="extrapolate")
    f_gen = interp1d(x_gen, kde_gen, bounds_error=False, fill_value="extrapolate")
    kde_orig_emp = f_orig(x_emp)
    kde_gen_emp = f_gen(x_emp)

    # Create the figure with 3 subplots.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Subplot 1: Original image with mask overlay.
    axs[0].imshow(image, cmap="gray", origin="lower")
    # Overlay the mask in red with α = 0.6.
    mask_overlay = np.where(mask, 1, np.nan)
    axs[0].imshow(mask_overlay, cmap="Reds_r", alpha=0.6, origin="lower")
    axs[0].set_title("Original Image with Mask")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")

    # Subplot 2: Theoretical PDFs.
    axs[1].plot(
        x,
        pdf_orig_theo,
        color="red",
        linestyle="--",
        label=rf"Original: $\mathrm{{loc}}={loc_orig:.2f}$, $\hat{{\sigma}}={scale_orig:.2f}$",
    )
    axs[1].plot(
        x,
        pdf_gen_theo,
        color="blue",
        linestyle="--",
        label=rf"Generated: $\mathrm{{loc}}={loc_gen:.2f}$, $\hat{{\sigma}}={scale_gen:.2f}$",
    )
    axs[1].set_title("Theoretical Rayleigh PDFs")
    axs[1].set_xlabel("Value")
    axs[1].set_ylabel("Density")
    axs[1].legend()

    # Subplot 3: Empirical PDFs (KDE).
    axs[2].plot(
        x_emp, kde_orig_emp, color="red", linestyle="-", label="Original (Empirical)"
    )
    axs[2].plot(
        x_emp, kde_gen_emp, color="blue", linestyle="-", label="Generated (Empirical)"
    )
    axs[2].set_title("Empirical PDFs (KDE)")
    axs[2].set_xlabel("Value")
    axs[2].set_ylabel("Density")
    axs[2].legend()

    # Force subplots 2 and 3 to share the same y-axis limits.
    common_ylim = (
        min(
            np.min(pdf_orig_theo),
            np.min(pdf_gen_theo),
            np.min(kde_orig_emp),
            np.min(kde_gen_emp),
        ),
        max(
            np.max(pdf_orig_theo),
            np.max(pdf_gen_theo),
            np.max(kde_orig_emp),
            np.max(kde_gen_emp),
        ),
    )
    axs[1].set_ylim(common_ylim)
    axs[2].set_ylim(common_ylim)

    for ax in axs:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Mask and PDF comparison saved to {output_path}")

    # --- Compute Jensen–Shannon Divergence ---
    js_theoretical = compute_js_divergence(pdf_orig_theo, pdf_gen_theo, x)
    js_empirical = compute_js_divergence(kde_orig_emp, kde_gen_emp, x_emp)
    print(f"Jensen–Shannon Divergence (Theoretical PDFs): {js_theoretical:.4f}")
    print(f"Jensen–Shannon Divergence (Empirical PDFs): {js_empirical:.4f}")


def compute_js_divergence(
    pdf1: np.ndarray, pdf2: np.ndarray, x_vals: np.ndarray, epsilon: float = 1e-10
) -> float:
    """
    Compute the Jensen–Shannon divergence between two PDFs by first discretizing them.
    """
    from Meningioma.utils import Stats

    p1 = Stats.approximate_pmf_from_pdf(pdf1, x_vals, epsilon)
    p2 = Stats.approximate_pmf_from_pdf(pdf2, x_vals, epsilon)
    m = 0.5 * (p1 + p2)
    kl1 = np.sum(p1 * np.log(p1 / (m + epsilon)))
    kl2 = np.sum(p2 * np.log(p2 / (m + epsilon)))
    return 0.5 * (kl1 + kl2)


def main():
    output_folder = (
        "scripts/noise_modelling/gaussian_random_markov_fields/images/experiment_images"
    )
    os.makedirs(output_folder, exist_ok=True)

    base_path = "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition"
    output_npz_path = "/home/mariopasc/Python/Datasets/Meningiomas/npz"

    slice_index = 112
    patient = "P50"
    pulse = "T1"

    filepath = os.path.join(output_npz_path, patient, f"{patient}_{pulse}.npz")
    slice_data = load_data(filepath, slice_index=slice_index)

    # Save the magnitude MRI.
    fig_slice, ax_slice = plt.subplots(figsize=(8, 8))
    im = ax_slice.imshow(slice_data, cmap="gray", origin="lower")
    ax_slice.set_xlabel("X")
    ax_slice.set_ylabel("Y")
    # plt.colorbar(im, ax=ax_slice)
    disable_details(ax_slice)
    plt.savefig(os.path.join(output_folder, "magnitude_slice.svg"), bbox_inches="tight")
    plt.close(fig_slice)

    # Extract phase and save approximated k-space.
    phase_data, k_space = extract_phase_from_kspace(slice_data)
    fig_k, ax_k = plt.subplots(figsize=(8, 8))
    im_k = ax_k.imshow(np.log1p(np.abs(k_space)), cmap="gray", origin="lower")
    ax_k.set_xlabel("Frequency X")
    ax_k.set_ylabel("Frequency Y")
    plt.colorbar(im_k, ax=ax_k)
    disable_details(ax_k)
    plt.savefig(
        os.path.join(output_folder, "approximated_kspace.svg"), bbox_inches="tight"
    )
    plt.close(fig_k)

    # Convert to complex data.
    slice_data_real, slice_data_imag = to_real_imag(slice_data, phase_data)

    # Save the approximated real part.
    fig_real, ax_real = plt.subplots(figsize=(8, 8))
    im_real = ax_real.imshow(slice_data_real, cmap="seismic", origin="lower")
    ax_real.set_xlabel("X")
    ax_real.set_ylabel("Y")
    plt.colorbar(im_real, ax=ax_real)
    disable_details(ax_real)
    plt.savefig(
        os.path.join(output_folder, "approximated_real.svg"), bbox_inches="tight"
    )
    plt.close(fig_real)

    # Save the approximated imaginary part.
    fig_imag, ax_imag = plt.subplots(figsize=(8, 8))
    im_imag = ax_imag.imshow(slice_data_imag, cmap="seismic", origin="lower")
    ax_imag.set_xlabel("X")
    ax_imag.set_ylabel("Y")
    plt.colorbar(im_imag, ax=ax_imag)
    disable_details(ax_imag)
    plt.savefig(
        os.path.join(output_folder, "approximated_imag.svg"), bbox_inches="tight"
    )
    plt.close(fig_imag)

    # Compute and save the mask overlay.
    hull = ImageProcessing.convex_hull_mask(image=slice_data, threshold_method="li")
    mask = hull > 0
    fig_mask, ax_mask = plt.subplots(figsize=(8, 8))
    im_mask = ax_mask.imshow(slice_data, cmap="gray", origin="lower")

    mask_overlay = np.where(mask, 1, np.nan)
    ax_mask.imshow(mask_overlay, cmap="Reds_r", alpha=0.6, origin="lower")

    ax_mask.set_xlabel("X")
    ax_mask.set_ylabel("Y")
    plt.colorbar(im_mask, ax=ax_mask)
    disable_details(ax_mask)
    plt.savefig(os.path.join(output_folder, "mask_overlay.svg"), bbox_inches="tight")
    plt.close(fig_mask)

    # Plot original image with mask and distributions.
    plot_mask_and_distribution(
        slice_data, mask, os.path.join(output_folder, "mask_and_distribution.svg")
    )

    variogram_bins = np.linspace(0, 20, 30)
    variogram_sampling_size = 3000
    variogram_sampling_seed = 19920516

    masked_values = slice_data[mask]
    var_guess = np.var(masked_values) if len(masked_values) > 1 else 0
    len_scale_guess = 1.5

    n, m = slice_data.shape

    # Compute isotropic variogram.
    iso_bin_center, iso_gamma = estimate_variogram_isotropic(
        data=slice_data_real,
        bins=variogram_bins,
        mask=mask,
        sampling_size=variogram_sampling_size,
        sampling_seed=variogram_sampling_seed,
    )
    iso_models = fit_model(
        bin_center=iso_bin_center,
        gamma=iso_gamma,
        len_scale=len_scale_guess,
        var=var_guess,
    )

    # Compute anisotropic variograms.
    anisotropic_variograms = estimate_variogram_anisotropic(
        data=slice_data_real,
        bins=variogram_bins,
        mask=mask,
        sampling_size=variogram_sampling_size,
        sampling_seed=variogram_sampling_seed,
    )

    directions = ["Isotropic", "Horizontal", "Vertical", "Diagonal_45", "Diagonal_135"]
    all_variograms = {
        "Isotropic": (iso_bin_center, iso_gamma),
        **anisotropic_variograms,
    }
    all_models = {"Isotropic": iso_models}
    for direction, (bin_center, gamma) in anisotropic_variograms.items():
        all_models[direction] = fit_model(
            bin_center=iso_bin_center,
            gamma=iso_gamma,
            len_scale=len_scale_guess,
            var=var_guess,
        )

    # Plot variograms for each direction.
    color_cycle = plt.cm.viridis(np.linspace(0, 1, 10))
    for direction in directions:
        bin_center, gamma = all_variograms[direction]
        models = all_models[direction]
        fig_vario, ax_vario = plt.subplots(figsize=(8, 8))
        ax_vario.plot(bin_center, gamma, "o", color="black")
        col_idx = 0
        for model_name, (model, _) in models.items():
            color = color_cycle[col_idx % len(color_cycle)]
            model.plot(
                x_max=variogram_bins[-1],
                ax=ax_vario,
                color=color,
                linestyle="--",
                label=None,
            )
            col_idx += 1
        ax_vario.set_xlabel("Distance")
        ax_vario.set_ylabel("Gamma")
        disable_details(ax_vario)
        plt.savefig(
            os.path.join(output_folder, f"variogram_{direction.lower()}.svg"),
            bbox_inches="tight",
        )
        plt.close(fig_vario)

    # Choose the best-fitting model from the isotropic variogram.
    best_model_name = max(iso_models, key=lambda name: iso_models[name][1]["r2"])
    best_model = iso_models[best_model_name][0]

    # Generate GRF noise fields.
    noise_real, noise_imag, noise_final = generate_random_fields(
        best_model, shape=(n, m)
    )

    # Use scipy.stats.rayleigh to estimate the scale and compute the bias.
    loc_r, scale_r = rayleigh.fit(noise_final.flatten())
    scipy_bias = scale_r * np.sqrt(np.pi / 2)

    # Subtract the bias and clip negative values.
    noise_final_corrected = np.maximum(noise_final - scipy_bias, 0)

    # Save the generated real part.
    fig_noise_real, ax_noise_real = plt.subplots(figsize=(8, 8))
    im_noise_real = ax_noise_real.imshow(noise_real, cmap="seismic", origin="lower")
    ax_noise_real.set_xlabel("X")
    ax_noise_real.set_ylabel("Y")
    plt.colorbar(im_noise_real, ax=ax_noise_real)
    disable_details(ax_noise_real)
    plt.savefig(os.path.join(output_folder, "generated_real.svg"), bbox_inches="tight")
    plt.close(fig_noise_real)

    # Save the generated imaginary part.
    fig_noise_imag, ax_noise_imag = plt.subplots(figsize=(8, 8))
    im_noise_imag = ax_noise_imag.imshow(noise_imag, cmap="seismic", origin="lower")
    ax_noise_imag.set_xlabel("X")
    ax_noise_imag.set_ylabel("Y")
    plt.colorbar(im_noise_imag, ax=ax_noise_imag)
    disable_details(ax_noise_imag)
    plt.savefig(os.path.join(output_folder, "generated_imag.svg"), bbox_inches="tight")
    plt.close(fig_noise_imag)

    # Save the final corrected noise slice.
    fig_noise_final, ax_noise_final = plt.subplots(figsize=(8, 8))
    im_noise_final = ax_noise_final.imshow(
        noise_final_corrected, cmap="seismic", origin="lower"
    )
    ax_noise_final.set_xlabel("X")
    ax_noise_final.set_ylabel("Y")
    plt.colorbar(im_noise_final, ax=ax_noise_final)
    disable_details(ax_noise_final)
    plt.savefig(os.path.join(output_folder, "final_noise.svg"), bbox_inches="tight")
    plt.close(fig_noise_final)

    # Print statistics.
    noise_background = slice_data_real[mask]
    mean_background = np.mean(noise_background)
    std_background = np.std(noise_background)
    print(
        "Background noise (outside brain mask) - Mean: {:.4f}, Std: {:.4f}".format(
            mean_background, std_background
        )
    )

    mean_noise_real = np.mean(noise_real)
    std_noise_real = np.std(noise_real)
    print(
        "Generated noise slice (real) - Mean: {:.4f}, Std: {:.4f}".format(
            mean_noise_real, std_noise_real
        )
    )

    mean_noise_imag = np.mean(noise_imag)
    std_noise_imag = np.std(noise_imag)
    print(
        "Generated noise slice (imaginary) - Mean: {:.4f}, Std: {:.4f}".format(
            mean_noise_imag, std_noise_imag
        )
    )

    mean_noise_final = np.mean(noise_final_corrected)
    std_noise_final = np.std(noise_final_corrected)
    print(
        "Final GRF-generated noise slice (corrected) - Mean: {:.4f}, Std: {:.4f}".format(
            mean_noise_final, std_noise_final
        )
    )

    print("Estimated Rayleigh scale (from scipy): {:.4f}".format(scale_r))
    print("Computed Rayleigh bias (σ̂√(π/2)) from scipy: {:.4f}".format(scipy_bias))

    # Plot the PDFs of the generated noise components.
    plot_noise_distributions(
        noise_real,
        noise_imag,
        noise_final_corrected,
        os.path.join(output_folder, "noise_distributions.svg"),
        h=0.5,
    )

    plot_mask_and_pdf_comparison(
        slice_data,
        mask,
        noise_final_corrected,
        os.path.join(output_folder, "mask_and_pdf_comparison.svg"),
        h=0.5,
    )

    """
    # Combined figure for overview (optional).
    fig_combined, axes = plt.subplots(3, 5, figsize=(20, 12))
    for col, direction in enumerate(directions):
        bin_center, gamma = all_variograms[direction]
        axes[0, col].plot(bin_center, gamma, "o", color="black")
        color_cycle_combined = plt.cm.viridis(
            np.linspace(0, 1, len(all_models[direction]))
        )
        colors = iter(color_cycle_combined)
        for model_name, (model, _) in all_models[direction].items():
            color = next(colors)
            model.plot(
                x_max=variogram_bins[-1],
                ax=axes[0, col],
                color=color,
                linestyle="--",
                label=None,
            )
        axes[0, col].set_xlabel("Distance")
        axes[0, col].set_ylabel("Gamma")
        disable_details(axes[0, col])
        _, _, new_data = generate_random_fields(best_model, shape=(n, m))
        im2 = axes[1, col].imshow(new_data, cmap="viridis", origin="lower")
        axes[1, col].set_xlabel("X")
        axes[1, col].set_ylabel("Y")
        disable_details(axes[1, col])
        fft_noise = np.fft.fftshift(np.fft.fft2(new_data))
        power_spectrum = np.log1p(np.abs(fft_noise))
        im3 = axes[2, col].imshow(power_spectrum, cmap="viridis", origin="lower")
        axes[2, col].set_xlabel("Frequency X")
        axes[2, col].set_ylabel("Frequency Y")
        disable_details(axes[2, col])
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "combined_figure.svg"), bbox_inches="tight")
    plt.close(fig_combined)

    # Additional Step: Generate a 3D noise volume and visualize it.
    x = np.arange(10)
    y = np.arange(10)
    z = np.arange(10)
    model3d = gs.Gaussian(dim=3, len_scale=[16, 8, 4], angles=(0.8, 0.4, 0.2))
    srf3d = gs.SRF(model3d)
    field3d = srf3d((x, y, z), mesh_type="structured")
    vtk_path = os.path.join(output_folder, "3d_field.vtk")
    srf3d.vtk_export(vtk_path)
    print(f"3D field saved to {vtk_path}")
    mesh = srf3d.to_pyvista()
    mesh.contour(isosurfaces=8).plot()
    """


if __name__ == "__main__":
    main()
