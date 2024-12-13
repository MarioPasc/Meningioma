import numpy as np
import matplotlib.pyplot as plt
from gstools import vario_estimate_unstructured, TPLExponential, SRF # type : ignore
from gstools.covmodel.plot import plot_variogram # type : ignore
from typing import Tuple
from Meningioma.utils import npz_converter # type : ignore
import os


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


def extract_phase_from_kspace(image: np.ndarray) -> np.ndarray:
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
    return phase



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


def estimate_variogram(
    data: np.ndarray,
    bins: np.ndarray,
    sampling_size: int = 2000,
    sampling_seed: int = 19920516,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the variogram from 2D data.

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
    """
    n, m = data.shape
    x_s = np.arange(0, n)
    y_s = np.arange(0, m)
    x_u, y_u = np.meshgrid(x_s, y_s)
    x_u = np.reshape(x_u, n * m)
    y_u = np.reshape(y_u, n * m)

    bin_center, gamma = vario_estimate_unstructured(
        (x_u, y_u),
        data.flatten(),
        bins,
        sampling_size=sampling_size,
        sampling_seed=sampling_seed,
    )
    return bin_center, gamma


def fit_model(
    bin_center: np.ndarray,
    gamma: np.ndarray,
    model_type: str = "TPLExponential",
    var: float = 1.0,
    len_scale: float = 10.0,
    nugget: bool = True,
) -> TPLExponential:
    """
    Fit a theoretical variogram model to the estimated variogram.

    Parameters
    ----------
    bin_center : np.ndarray
        The distances at which the variogram was estimated.
    gamma : np.ndarray
        The estimated variogram values.
    model_type : str
        Choice of covariance model. For demonstration, we use TPLExponential.
    var : float
        Initial guess for variance parameter.
    len_scale : float
        Initial guess for length scale parameter.
    nugget : bool
        Whether to fit a nugget parameter.

    Returns
    -------
    TPLExponential
        The fitted covariance model.
    """
    # Here we fix the model to TPLExponential. Can switch to Matern, Rational, etc.
    fit_model = TPLExponential(dim=2, var=var, len_scale=len_scale)
    fit_model.fit_variogram(bin_center, gamma, nugget=nugget)
    return fit_model


def generate_random_fields(
    model: TPLExponential,
    shape: Tuple[int, int],
    seed_real: int = 19770928,
    seed_imag: int = 19773022,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic random fields for the real and imaginary parts and combine them.

    Parameters
    ----------
    model : TPLExponential
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

    srf_real = SRF(model, seed=seed_real)
    new_data_real = srf_real((x_s, y_s), mesh_type="structured")

    srf_imag = SRF(model, seed=seed_imag)
    new_data_imag = srf_imag((x_s, y_s), mesh_type="structured")

    # Combine into magnitude
    new_data = np.sqrt(new_data_real**2 + new_data_imag**2)
    return new_data_real, new_data_imag, new_data


def main():
    # Example Input Paths and Parameters
    base_path = "/home/mario/Python/Datasets/Meningiomas/Meningioma_Adquisition"
    output_npz_path = "/home/mario/Python/Datasets/Meningiomas/npz"

    slice_index = 18
    patient = "P15"
    pulse = "T1"

    # 1. Convert data to NPZ format
    """
    npz_converter.convert_to_npz(
        base_path=base_path, output_path=output_npz_path, patient=patient, pulse=pulse
    )
    """
    # 2. Define the file path
    filepath = os.path.join(output_npz_path, patient, f"{patient}_{pulse}.npz")
    # 3. Load MRI Slice
    slice_data = load_data(file_path=filepath, slice_index=slice_index)
    slice_data = np.squeeze(slice_data)
    # Tunable hyperparameters
    spatial_extent = 5.0  # Coordinate range for phase field
    variogram_bins = np.linspace(0, 100, 100)  # Limit to 50 pixels distance
    variogram_sampling_size = 2000  # How many pairs to sample
    variogram_sampling_seed = 19920516
    covariance_var = 1.0  # Initial variance for the fitted model
    covariance_len_scale = 20.0  # Length scale for covariance model

    n, m = slice_data.shape

    # Extract phase from approximated k-space
    phase_data = extract_phase_from_kspace(slice_data)


    # Convert to complex data
    slice_data_real, slice_data_imag = to_real_imag(slice_data, phase_data)

    # Estimate variogram on the real part
    bin_center, gamma = estimate_variogram(
        data=slice_data_real,
        bins=variogram_bins,
        sampling_size=variogram_sampling_size,
        sampling_seed=variogram_sampling_seed,
    )

    # Fit a covariance model
    fitted_model = fit_model(
        bin_center=bin_center,
        gamma=gamma,
        model_type="TPLExponential",
        var=np.var(slice_data_real),
        len_scale=covariance_len_scale,
        nugget=True,
    )

    # Generate new random fields
    new_data_real, new_data_imag, new_data = generate_random_fields(
        fitted_model, shape=(n, m), seed_real=19770928, seed_imag=19773022
    )

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Generated Real Part")
    plt.imshow(new_data_real, cmap="seismic", origin="lower")
    plt.colorbar(label="Intensity")

    plt.subplot(1, 2, 2)
    plt.title("Generated Imaginary Part")
    plt.imshow(new_data_imag, cmap="seismic", origin="lower")
    plt.colorbar(label="Intensity")

    plt.tight_layout()
    plt.show()

    # Compute some statistics
    original_mean = slice_data.mean()
    original_std = slice_data.std()
    generated_mean = new_data.mean()
    generated_std = new_data.std()

    # Print statistics
    print("Original Slice Statistics:")
    print(f"Mean: {original_mean:.4f}, Std: {original_std:.4f}")

    print("\nGenerated Slice Statistics:")
    print(f"Mean: {generated_mean:.4f}, Std: {generated_std:.4f}")

    # Visualization
    # 1. Variogram
    plt.figure(figsize=(8, 4))
    plt.title("Variogram and Fitted Model")
    plt.plot(bin_center, gamma, "o", label="Estimated Variogram")
    plot_variogram(fitted_model, x_max=variogram_bins[-1], label="Fitted Model")
    plt.legend()
    plt.xlabel("Distance")
    plt.ylabel("Gamma")
    plt.tight_layout()
    plt.show()

    # 2. Original and Generated Slices
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Slice")
    plt.imshow(slice_data, cmap="seismic", origin="lower")
    plt.colorbar(label="Intensity")

    plt.subplot(1, 2, 2)
    plt.title("Generated Slice")
    plt.imshow(new_data, cmap="seismic", origin="lower")
    plt.colorbar(label="Intensity")

    plt.tight_layout()
    plt.savefig("./markov.svg")
    plt.show()


if __name__ == "__main__":
    main()
