from typing import Any, Tuple, Dict, List
from numpy.typing import NDArray
import os

import numpy as np
import pandas as pd
import csv
import gstools as gs
from gstools.covmodel.plot import plot_variogram # type : ignore

from Meningioma.image_processing import ImageProcessing  # type: ignore
from Meningioma.utils import Stats, npz_converter  # type: ignore

import matplotlib.pyplot as plt
import scienceplots  # type: ignore

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

def load_data(file_path: str, slice_index: int) -> np.ndarray:
    """
    Load the MRI data from a .npz file and extract a single slice.

    Parameters
    ----------
    file_path : str
        Path to the .npz file containing MRI data.
    slice_index : int
        The index of the slice to extract.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the selected MRI slice (magnitude data).
    """
    data = np.load(file_path)
    # Extract the entire spatial region (all x and y coordinates)
    slice_data = np.squeeze(data["data"][0, :, :, slice_index])
    return slice_data

def extract_phase_from_kspace(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate k-space from an image and extract the phase data.
    
    Parameters
    ----------
    image : np.ndarray
        Input MRI image data (2D numpy array).
        
    Returns
    -------
    np.ndarray
        Phase data extracted from the approximated k-space.
    """
    # Compute 2D FFT and shift to center the k-space
    k_space = np.fft.fftshift(np.fft.fft2(image))
    # Extract phase data
    phase = np.angle(k_space)
    return phase, k_space

def to_real_imag(
    magnitude: np.ndarray, phase: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert magnitude data into synthetic complex data by applying a phase.

    Parameters
    ----------
    magnitude : np.ndarray
        Magnitude image data.
    phase : np.ndarray
        Phase data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Real and imaginary parts of the complex data.
    """
    real_part = magnitude * np.cos(phase)
    imag_part = magnitude * np.sin(phase)
    return real_part, imag_part

def estimate_variogram_isotropic(
    data: np.ndarray,
    bins: np.ndarray,
    sampling_size: int = 2000,
    sampling_seed: int = 19920516,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the variogram from 2D data using the gstools vario_estimate method.

    Parameters
    ----------
    data : np.ndarray
        2D image data (e.g., real part of complex image).
    bins : np.ndarray
        Array of bin edges for distance classes.
    sampling_size : int
        Number of random pairs sampled to estimate the variogram.
    sampling_seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        bin_center, gamma values for the estimated variogram.
        
    Docs 
    -------
    https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.variogram.vario_estimate.html#gstools.variogram.vario_estimate
    """
    # Structured grid positions
    n, m = data.shape
    x = np.arange(n)  # Row indices as x positions
    y = np.arange(m)  # Column indices as y positions

    # Use gstools.vario_estimate
    bin_centers, gamma = gs.vario_estimate(
        pos=(x, y),                 # Structured grid axes
        field=data,                 # 2D field data
        bin_edges=bins,             # Bin edges for distance classes
        mesh_type="structured",     # Data is structured
        sampling_size=sampling_size,  # Random sampling size
        sampling_seed=sampling_seed,  # Seed for reproducibility
    )
    return bin_centers, gamma

def estimate_variogram_anisotropic(
    data: np.ndarray,
    bins: np.ndarray,
    directions: List[np.ndarray] = None,
    angles_tol: float = np.pi / 8,
    sampling_size: int = 2000,
    sampling_seed: int = 19920516,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate the directional variogram from 2D data using gstools vario_estimate.

    Parameters
    ----------
    data : np.ndarray
        2D image data (e.g., real part of complex image).
    bins : np.ndarray
        Array of bin edges for distance classes.
    directions : List[np.ndarray], optional
        List of direction vectors to evaluate the variogram.
        Default: horizontal, vertical, and diagonal directions.
    angles_tol : float
        Angular tolerance for directional variograms (radians).
        Default: np.pi / 8 (22.5 degrees).
    sampling_size : int
        Number of random pairs sampled to estimate the variogram.
    sampling_seed : int
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, Tuple[np.ndarray, np.ndarray]]
        Dictionary containing bin centers and gamma values for each direction.
        Keys: "Horizontal", "Vertical", "Diagonal 45°", "Diagonal 135°".
    """
    # Structured grid positions
    n, m = data.shape
    x = np.arange(n)  # Row indices as x positions
    y = np.arange(m)  # Column indices as y positions

    # Default directions: horizontal, vertical, diagonal (45° and 135°)
    if directions is None:
        directions = [
            np.array([1, 0]),   # Horizontal (x-axis)
            np.array([0, 1]),   # Vertical (y-axis)
            np.array([1, 1]),   # Diagonal 45°
            np.array([-1, 1]),  # Diagonal 135°
        ]
        direction_labels = ["Horizontal", "Vertical", "Diagonal_45", "Diagonal_135"]
    else:
        direction_labels = [f"Direction {i+1}" for i in range(len(directions))]

    # Store results for each direction
    variograms = {}

    # Estimate variogram for each direction
    for direction, label in zip(directions, direction_labels):
        print(f"Fitting anisotrophic {direction} model")
        bin_centers, gamma = gs.vario_estimate(
            pos=(x, y),                 # Structured grid axes
            field=data,                 # 2D field data
            bin_edges=bins,             # Bin edges
            mesh_type="structured",     # Structured grid
            direction=[direction],      # Directional vector
            angles_tol=angles_tol,      # Angular tolerance
            sampling_size=sampling_size,  # Sampling
            sampling_seed=sampling_seed,  # Seed for reproducibility
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

    Parameters
    ----------
    bin_center : np.ndarray
        The distances at which the variogram was estimated.
    gamma : np.ndarray
        The estimated variogram values.
    var : float
        Initial guess for variance parameter.
    len_scale : float
        Initial guess for length scale parameter.
    nugget : bool
        Whether to fit a nugget parameter.

    Returns
    -------
    Dict[str, Tuple[gs.CovModel, Dict[str, Any]]]
        Dictionary with model names as keys and tuples containing:
        - Fitted covariance model
        - Dictionary with fitting statistics: params, pcov, r2
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
        "TLPExponential": gs.TPLExponential,
        "TLPSTable": gs.TPLStable,
        "TLPSimple": gs.TPLSimple
    }
    print("Fitting isotrophic model")
    results = {}

    for model_name, model_class in models.items():
        try:
            # Instantiate and fit the model
            model = model_class(dim=2, var=var, len_scale=len_scale)
            params, pcov, r2 = model.fit_variogram(bin_center, gamma, return_r2=True)

            # Save the model and stats in the results dictionary
            results[model_name] = (model, {'params': params, 'pcov': pcov, 'r2': r2})
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

    Parameters
    ----------
    model : gs.CovModel
        The fitted covariance model.
    shape : Tuple[int, int]
        Shape of the 2D field (n, m).
    seed_real : int
        Random seed for the real part SRF.
    seed_imag : int
        Random seed for the imaginary part SRF.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        new_data_real, new_data_imag, and combined magnitude field.
    """
    n, m = shape
    x_s = np.arange(0, n)
    y_s = np.arange(0, m)

    srf_real = gs.SRF(model, seed=seed_real)
    new_data_real = srf_real((x_s, y_s), mesh_type="structured")

    srf_imag = gs.SRF(model, seed=seed_imag)
    new_data_imag = srf_imag((x_s, y_s), mesh_type="structured")

    # Combine into magnitude
    new_data = np.sqrt(new_data_real**2 + new_data_imag**2)
    return new_data_real, new_data_imag, new_data


def experiment():
    # Example Input Paths and Parameters
    base_path = "/home/mario/Python/Datasets/Meningiomas/Meningioma_Adquisition"
    output_npz_path = "/home/mario/Python/Datasets/Meningiomas/npz"

    slice_index = 18
    patient = "P15"
    pulse = "T1"

    # Convert data to NPZ format
    """
    npz_converter.convert_to_npz(
        base_path=base_path, output_path=output_npz_path, patient=patient, pulse=pulse
    )
    """
    # Load the data
    filepath = os.path.join(output_npz_path, patient, f"{patient}_{pulse}.npz")
    slice_data = npz_converter.load_mri_slice(filepath=filepath, slice_index=slice_index)



    # Tunable hyperparameters
    variogram_bins = np.linspace(0, 150, 150)  # Limit to 50 pixels distance
    variogram_sampling_size = 3000  # How many pairs to sample
    variogram_sampling_seed = 19920516
    covariance_len_scale = 100.0  # Length scale for covariance model

    n, m = slice_data.shape

    # Extract phase from approximated k-space
    phase_data, k_space = extract_phase_from_kspace(slice_data)


    # Convert to complex data
    slice_data_real, slice_data_imag = to_real_imag(slice_data, phase_data)

    # Estimate variogram on the real part
    bin_center, gamma = estimate_variogram_isotropic(
        data=slice_data_real,
        bins=variogram_bins,
        sampling_size=variogram_sampling_size,
        sampling_seed=variogram_sampling_seed,
    )

    # Fit all models
    results = fit_model(
        bin_center=bin_center, gamma=gamma, var=np.var(slice_data_real), len_scale=covariance_len_scale
    )

    # Find the best model based on R2
    best_model_name = max(results, key=lambda model: results[model][1]['r2'])
    best_model, best_stats = results[best_model_name]
    best_r2 = best_stats['r2']
    best_params = best_stats['params']
    perr = np.round(np.sqrt(np.diag(best_stats['pcov'])), 3)
    
    # Generate new random fields
    new_data_real, new_data_imag, new_data = generate_random_fields(
        best_model, shape=(n, m), seed_real=19770928, seed_imag=19773022
    )

    # Plotting 
    
    # Visualization
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))  # 3 rows, 2 columns

    # First Row: Original Image and K-space Magnitude
    axes[0, 0].set_title("Original Image")
    im0 = axes[0, 0].imshow(slice_data, cmap="gray", origin="lower")
    fig.colorbar(im0, ax=axes[0, 0], label="Intensity")

    axes[0, 1].set_title(f"K-space Magnitude\nPhase: {phase_data.mean()}")
    im_kspace = axes[0, 1].imshow(np.log1p(np.abs(k_space)), cmap="gray", origin="lower")
    fig.colorbar(im_kspace, ax=axes[0, 1], label="Log Magnitude")

    # Second Row: Real and Imaginary Parts
    axes[1, 0].set_title("Generated Real Part")
    im1 = axes[1, 0].imshow(new_data_real, cmap="seismic", origin="lower")
    fig.colorbar(im1, ax=axes[1, 0], label="Intensity")

    axes[1, 1].set_title("Generated Imaginary Part")
    im2 = axes[1, 1].imshow(new_data_imag, cmap="seismic", origin="lower")
    fig.colorbar(im2, ax=axes[1, 1], label="Intensity")

    # Third Row: Original and Generated Slices
    axes[2, 0].set_title(f"Original Slice\nMean: {slice_data.mean():.4f}, Std: {slice_data.std():.4f}")
    im3 = axes[2, 0].imshow(slice_data, cmap="seismic", origin="lower")
    fig.colorbar(im3, ax=axes[2, 0], label="Intensity")

    axes[2, 1].set_title(f"Generated Slice\nMean: {new_data.mean():.4f}, Std: {new_data.std():.4f}")
    im4 = axes[2, 1].imshow(new_data, cmap="seismic", origin="lower")
    fig.colorbar(im4, ax=axes[2, 1], label="Intensity")

    # Adjust layout and save combined plot
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, hspace=0.4)  # Adjust spacing
    plt.savefig("scripts/noise_modelling/gaussian_random_markov_fields/MRF_slices.svg", format="svg")

    # Separate Variogram and Fitted Model Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the variogram
    ax.set_title("Variogram and Fitted Models")
    ax.plot(bin_center, gamma, "o", label="Estimated Variogram", color="black")

    # Plot all models with a unique color
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    handles = []

    for (model_name, (model, stats)), color in zip(results.items(), colors):
        model.plot(x_max=variogram_bins[-1], ax=ax, color=color, linestyle="--")
        handles.append(plt.Line2D([0], [0], color=color, linestyle="--", label=f"{model_name} (R²={stats['r2']:.4f})"))

    # Highlight the best model
    handles.append(plt.Line2D([0], [0], color='red', linestyle="-", label=f"Best Model: {best_model_name}"))
    best_model.plot(x_max=variogram_bins[-1], ax=ax, color='red', linestyle="-")

    # Add text block for best model stats
    ax.text(
        0.05, 0.95,
        f"Best Model: {best_model_name}\n"
        f"R² = {best_r2:.4f}\n"
        f"Variance = {best_params['var']:.4f} ± {perr[0]}\n"
        f"Length Scale = {best_params['len_scale']:.4f} ± {perr[1]}\n"
        f"Nugget = {best_params['nugget']:.4f} ± {perr[2]}",
        fontsize=10,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round", facecolor="lightgrey", alpha=0.5),
        transform=ax.transAxes
    )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Add legend outside the plot
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4)

    # Labels and adjustments
    ax.set_xlabel("Distance")
    ax.set_ylabel("Gamma")
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Adjust space for legend
    plt.savefig("scripts/noise_modelling/gaussian_random_markov_fields/MRF_variogram_fitting.svg", format="svg")

def main():
    # Example Input Paths and Parameters
    base_path = "/home/mario/Python/Datasets/Meningiomas/Meningioma_Adquisition"
    output_npz_path = "/home/mario/Python/Datasets/Meningiomas/npz"

    slice_index = 18
    patient = "P15"
    pulse = "T1"

    # Load the data
    filepath = os.path.join(output_npz_path, patient, f"{patient}_{pulse}.npz")
    slice_data = load_data(filepath, slice_index=slice_index)

    # Tunable hyperparameters
    variogram_bins = np.linspace(0, 150, 150)
    variogram_sampling_size = 3000
    variogram_sampling_seed = 19920516
    covariance_len_scale = 100.0

    n, m = slice_data.shape

    # Extract phase from approximated k-space
    phase_data, k_space = extract_phase_from_kspace(slice_data)

    # Convert to complex data
    slice_data_real, _ = to_real_imag(slice_data, phase_data)

    # Compute isotropic variogram
    iso_bin_center, iso_gamma = estimate_variogram_isotropic(
        data=slice_data_real,
        bins=variogram_bins,
        sampling_size=variogram_sampling_size,
        sampling_seed=variogram_sampling_seed,
    )
    iso_models = fit_model(bin_center=iso_bin_center, gamma=iso_gamma)

    # Compute anisotropic variograms
    anisotropic_variograms = estimate_variogram_anisotropic(
        data=slice_data_real,
        bins=variogram_bins,
        sampling_size=variogram_sampling_size,
        sampling_seed=variogram_sampling_seed,
    )

    # Combine isotropic and anisotropic models
    directions = ["Isotropic", "Horizontal", "Vertical", "Diagonal_45", "Diagonal_135"]
    all_variograms = {"Isotropic": (iso_bin_center, iso_gamma), **anisotropic_variograms}
    all_models = {"Isotropic": iso_models}

    for direction, (bin_center, gamma) in anisotropic_variograms.items():
        all_models[direction] = fit_model(bin_center=bin_center, gamma=gamma)

    # Initialize CSV file
    results_file = "anisotropic_variogram_results.csv"
    with open(results_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Variogram", "Model", "R2", "variance", "variance_error",
            "len_scale", "len_scale_error", "nugget", "nugget_error"
        ])

        # Visualization
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        handles = []

        for col, direction in enumerate(directions):
            # Row 1: Variograms and fitted models
            bin_center, gamma = all_variograms[direction]
            models = all_models[direction]

            best_model_name = max(models, key=lambda name: models[name][1]['r2'])
            best_model, best_stats = models[best_model_name]
            best_r2 = best_stats['r2']
            best_params = best_stats['params']
            best_perr = np.round(np.sqrt(np.diag(best_stats['pcov'])), 3)

            # Write to CSV
            writer.writerow([
                f"{direction.lower()}", best_model_name, best_r2,
                best_params['var'], best_perr[0],
                best_params['len_scale'], best_perr[1],
                best_params['nugget'], best_perr[2]
            ])

            # Plot the variogram and models
            axes[0, col].plot(bin_center, gamma, "o", color="black", label="Variogram")
            for model_name, (model, _) in models.items():
                color = next(plt.gca()._get_lines.prop_cycler)['color']
                model.plot(x_max=variogram_bins[-1], ax=axes[0, col], color=color, linestyle="--")
                if col == 0:  # Collect legend handles only once
                    handles.append(plt.Line2D([0], [0], color=color, linestyle="--", label=model_name))

            axes[0, col].set_title(f"{direction} | Best Model: {best_model_name} (R²={best_r2:.4f})")
            axes[0, col].set_xlabel("Distance")
            axes[0, col].set_ylabel("Gamma")

            # Row 2: Generated noise map
            new_data_real, _, new_data = generate_random_fields(best_model, shape=(n, m))
            im2 = axes[1, col].imshow(new_data, cmap="seismic", origin="lower")
            axes[1, col].set_title(f"Noise Map: {direction}")
            fig.colorbar(im2, ax=axes[1, col])

            # Row 3: FFT Power spectrum
            fft_noise = np.fft.fftshift(np.fft.fft2(new_data))
            power_spectrum = np.log1p(np.abs(fft_noise))
            im3 = axes[2, col].imshow(power_spectrum, cmap="gray", origin="lower")
            axes[2, col].set_title(f"Power Spectrum: {direction}")
            fig.colorbar(im3, ax=axes[2, col])

        # Unified legend below all subplots
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Add space for the legend
        plt.savefig("anisotropic_variogram_analysis.svg", format="svg")


if __name__ == "__main__":
    main()
