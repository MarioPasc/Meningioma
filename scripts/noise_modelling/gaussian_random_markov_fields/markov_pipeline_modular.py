from turtle import color
from typing import Any, Tuple, Dict, List, Optional
from numpy.typing import NDArray
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import gstools as gs
from gstools.covmodel.plot import plot_variogram  # type : ignore
from tqdm import tqdm

# If these are your local modules:
from Meningioma.image_processing import ImageProcessing  # type: ignore
from Meningioma.utils import Stats, npz_converter  # type: ignore

import scienceplots

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
    slice_data = np.squeeze(data["data"][0, :, :, slice_index])
    return slice_data


def extract_phase_from_kspace(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate k-space from an image and extract the phase data.

    Parameters
    ----------
    image : np.ndarray
        Input MRI image data (2D).

    Returns
    -------
    phase : np.ndarray
        Phase data from the FFT of the image.
    k_space : np.ndarray
        The 2D FFT (shifted) of the image.
    """
    k_space = np.fft.fftshift(np.fft.fft2(image))
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
    (real_part, imag_part) : Tuple[np.ndarray, np.ndarray]
        Real and imaginary parts of the complex data.
    """
    real_part = magnitude * np.cos(phase)
    imag_part = magnitude * np.sin(phase)
    return real_part, imag_part


def estimate_variogram_isotropic(
    data: np.ndarray,
    bins: np.ndarray,
    mask: Optional[NDArray[np.bool_]],
    sampling_size: int = 2000,
    sampling_seed: int = 19920516,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the isotropic variogram using gstools.vario_estimate.

    Parameters
    ----------
    data : np.ndarray
        2D data array (e.g., real part of an MRI slice).
    bins : np.ndarray
        Distance bin edges.
    mask : np.ndarray, optional
        Boolean mask. If provided, only use masked pixels.
    sampling_size : int
        Number of random pairs for vario_estimate.
    sampling_seed : int
        Random seed for reproducibility.

    Returns
    -------
    (bin_centers, gamma) : Tuple[np.ndarray, np.ndarray]
        The bin centers and semi-variances.
    """
    # Flatten the data with or without mask
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

    # Ensure sampling_size is not bigger than total valid pixels
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


def estimate_variogram_direction(
    data: np.ndarray,
    bins: np.ndarray,
    direction: np.ndarray,
    mask: Optional[NDArray[np.bool_]] = None,
    angles_tol: float = np.pi / 8,
    sampling_size: int = 2000,
    sampling_seed: int = 19920516,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a directional variogram for a single direction vector.

    Parameters
    ----------
    data : np.ndarray
        2D data array (e.g., real part of an MRI slice).
    bins : np.ndarray
        Distance bin edges.
    direction : np.ndarray
        Direction vector (e.g. [1, 0]).
    mask : np.ndarray, optional
        Boolean mask. If provided, only use masked pixels.
    angles_tol : float
        Angular tolerance in radians around 'direction'.
    sampling_size : int
        Number of random pairs for vario_estimate.
    sampling_seed : int
        Random seed for reproducibility.

    Returns
    -------
    (bin_centers, gamma) : Tuple[np.ndarray, np.ndarray]
        The bin centers and semi-variances for the given direction.
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

    sampling_size = min(sampling_size, len(valid_data))

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
    return bin_centers, gamma


def fit_model(
    bin_center: np.ndarray,
    gamma: np.ndarray,
    var: float = 1.0,
    len_scale: float = 10.0,
) -> Dict[str, Tuple[gs.CovModel, Dict[str, Any]]]:
    """
    Fit multiple theoretical variogram models to the (bin_center, gamma) data.

    Parameters
    ----------
    bin_center : np.ndarray
        The distance axis values.
    gamma : np.ndarray
        The variogram (semi-variance) values.
    var : float
        Initial guess for variance parameter.
    len_scale : float
        Initial guess for length scale parameter.

    Returns
    -------
    results : Dict[str, Tuple[gs.CovModel, Dict[str, Any]]]
        Each entry has (model_instance, {'params':..., 'pcov':..., 'r2':...}).
    """
    model_classes = {
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
        "TLPSimple": gs.TPLSimple,
    }

    results: Dict[str, Tuple[gs.CovModel, Dict[str, Any]]] = {}
    for name, ModelClass in model_classes.items():
        try:
            mod = ModelClass(dim=2, var=var, len_scale=len_scale)
            params, pcov, r2 = mod.fit_variogram(bin_center, gamma, return_r2=True)
            results[name] = (mod, {"params": params, "pcov": pcov, "r2": r2})
        except Exception as e:
            print(f"[WARNING] Model '{name}' failed to fit: {e}")
    return results


def generate_random_fields(
    model: gs.CovModel,
    shape: Tuple[int, int],
    seed_real: int = 19770928,
    seed_imag: int = 19773022,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate two random fields (real and imaginary) from a fitted covariance model
    and combine them into a magnitude field.

    Parameters
    ----------
    model : gs.CovModel
        A gstools CovModel fitted to some variogram data.
    shape : Tuple[int, int]
        (n, m) shape for the output field.
    seed_real : int
        Random seed for the real part.
    seed_imag : int
        Random seed for the imaginary part.

    Returns
    -------
    (real_part, imag_part, magnitude) : Tuple[np.ndarray, np.ndarray, np.ndarray]
        The generated real part, imaginary part, and magnitude.
    """
    n, m = shape
    x_s = np.arange(0, n)
    y_s = np.arange(0, m)

    srf_real = gs.SRF(model, seed=seed_real)
    real_part = srf_real((x_s, y_s), mesh_type="structured")

    srf_imag = gs.SRF(model, seed=seed_imag)
    imag_part = srf_imag((x_s, y_s), mesh_type="structured")

    magnitude = np.sqrt(real_part**2 + imag_part**2)
    return real_part, imag_part, magnitude


# ---------------------------------------------------------------------------
# NEW MODULAR FUNCTIONS
# ---------------------------------------------------------------------------
def compute_variograms(
    data: np.ndarray,
    mask: Optional[np.ndarray],
    bins: np.ndarray,
    angles_deg: List[float],
    sampling_size: int = 2000,
    sampling_seed: int = 19920516,
    angles_tol: float = np.pi / 8,
    var_guess: float = 1.0,
    len_scale_guess: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute isotropic and multiple anisotropic variograms, then fit models.

    Parameters
    ----------
    data : np.ndarray
        2D array, e.g. real-part of the MRI slice.
    mask : np.ndarray, optional
        Boolean mask to exclude certain pixels.
    bins : np.ndarray
        The bin edges for distance.
    angles_deg : List[float]
        List of angles in degrees for anisotropic variograms (e.g. [0, 45, 90, 135, ...]).
    sampling_size : int
        Number of random pixel-pairs for vario_estimate.
    sampling_seed : int
        Random seed for reproducibility.
    angles_tol : float
        Angular tolerance around each direction vector.
    var_guess : float
        Initial guess for model variance.
    len_scale_guess : float
        Initial guess for length scale.

    Returns
    -------
    results : Dict[str, Dict[str, Any]]
        A dictionary with keys = "Isotropic" and each angle label.
        Each value is another dict with:
        {
          "bin_centers": np.ndarray,
          "gamma": np.ndarray,
          "fits": Dict[str, (CovModel, fit_stats)],
          "best_model_name": str,
          "best_model_stats": Dict[str, Any]
        }
    """
    results: Dict[str, Dict[str, Any]] = {}

    # --- 1) Compute isotropic variogram
    iso_bin_center, iso_gamma = estimate_variogram_isotropic(
        data=data,
        bins=bins,
        mask=mask,
        sampling_size=sampling_size,
        sampling_seed=sampling_seed,
    )
    iso_fits = fit_model(
        iso_bin_center, iso_gamma, var=var_guess, len_scale=len_scale_guess
    )
    # find best model
    best_iso = max(iso_fits.keys(), key=lambda m: iso_fits[m][1]["r2"])
    results["Isotropic"] = {
        "bin_centers": iso_bin_center,
        "gamma": iso_gamma,
        "fits": iso_fits,
        "best_model_name": best_iso,
        "best_model_stats": iso_fits[best_iso][1],
    }

    # --- 2) Compute anisotropic for each angle
    for angle_deg in tqdm(iterable=angles_deg, colour="green"):
        angle_rad = np.deg2rad(angle_deg)
        direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        bin_center, gamma = estimate_variogram_direction(
            data=data,
            bins=bins,
            direction=direction_vector,
            mask=mask,
            angles_tol=angles_tol,
            sampling_size=sampling_size,
            sampling_seed=sampling_seed,
        )
        fits = fit_model(bin_center, gamma, var=var_guess, len_scale=len_scale_guess)
        best_model = max(fits.keys(), key=lambda m: fits[m][1]["r2"])

        label = f"{angle_deg}°"
        results[label] = {
            "bin_centers": bin_center,
            "gamma": gamma,
            "fits": fits,
            "best_model_name": best_model,
            "best_model_stats": fits[best_model][1],
        }

    return results


def plot_mask_images(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.3,
    title_prefix: str = "",
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Visualize the original image, the mask, and an overlay of the mask on the image.
    Titles include mean and std of all pixels (in the first plot)
    and mean and std of pixels outside the mask (in the third plot).

    Parameters
    ----------
    image : np.ndarray
        The 2D image.
    mask : np.ndarray
        Boolean mask array of the same shape.
    alpha : float
        Alpha blending value for overlay.
    title_prefix : str
        A prefix to include in each subplot title.
    figsize : Tuple[int, int]
        Size of the figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1) Original image
    mean_all = image.mean()
    std_all = image.std()
    axes[0].imshow(image, cmap="gray", origin="lower")
    axes[0].set_title(
        f"{title_prefix} Original:\nMean={mean_all:.3f}, Std={std_all:.3f}"
    )

    # 2) Mask
    axes[1].imshow(mask, cmap="gray", origin="lower")
    axes[1].set_title(f"{title_prefix} Mask")

    # 3) Overlay
    #   We'll show the image in grayscale, then overlay the mask in e.g. red with alpha
    axes[2].imshow(image, cmap="gray", origin="lower")
    axes[2].imshow(mask, cmap="Reds", origin="lower", alpha=alpha)
    # Mean/Std for outside the mask: mask==True => outside or inside depends on your definition.
    # If your mask indicates the region to EXCLUDE, then we compute stats on those pixels:
    masked_values = image[mask]
    mean_mask = masked_values.mean() if len(masked_values) > 0 else 0
    std_mask = masked_values.std() if len(masked_values) > 1 else 0
    axes[2].set_title(
        f"{title_prefix} Overlay:\nMean={mean_mask:.3f}, Std={std_mask:.3f}"
    )

    plt.tight_layout()
    plt.savefig(
        "scripts/noise_modelling/gaussian_random_markov_fields/mask_images.svg",
        format="svg",
    )
    plt.show()


def plot_variogram_results(
    variogram_dict: Dict[str, Dict[str, Any]],
    dist_max: float,
    shape: Tuple[int, int],
    output_csv: Optional[str] = None,
    colormap: str = "seismic",
) -> None:
    """
    Create one column per variogram result (isotropic + anisotropic angles).
    For each column:
      - Row1: measured variogram + fitted models (highlight best)
      - Row2: random field generated from best model
      - Row3: FFT power spectrum of that random field

    Parameters
    ----------
    variogram_dict : Dict[str, Dict[str, Any]]
        Output from 'compute_variograms'. Each key is a label (Isotropic, "45°", etc.),
        value has 'bin_centers', 'gamma', 'fits', 'best_model_name', 'best_model_stats'.
    dist_max : float
        The maximum distance to plot for the model curves.
    shape : Tuple[int, int]
        Shape (n, m) to generate new random fields.
    output_csv : str, optional
        If provided, writes best model fits to this CSV file.
    colormap : str
        Colormap for the random field.
    """
    # Sort the dictionary so that "Isotropic" is first (if you like a specific order):
    sorted_keys = sorted(variogram_dict.keys(), key=lambda k: (k != "Isotropic", k))

    # If saving CSV
    if output_csv is not None:
        with open(output_csv, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "Variogram",
                    "BestModel",
                    "R2",
                    "Variance",
                    "VarError",
                    "LenScale",
                    "LenScaleError",
                    "Nugget",
                    "NuggetError",
                ]
            )
            # We'll write a line for each best model

    # Create figure with 3 rows, #cols = # of variograms
    fig, axes = plt.subplots(3, len(sorted_keys), figsize=(5 * len(sorted_keys), 10))
    if len(sorted_keys) == 1:
        # If there's only 1 column, axes might be 1D
        axes = np.array([axes]).T  # shape (3,1)

    # Keep track of model legend handles (one-time creation)
    global_legend_handles = []

    for col, label in enumerate(sorted_keys):
        data_dict = variogram_dict[label]
        bin_centers = data_dict["bin_centers"]
        gamma = data_dict["gamma"]
        fits = data_dict["fits"]
        best_model_name = data_dict["best_model_name"]
        best_stats = data_dict["best_model_stats"]
        best_model, _ = fits[best_model_name]

        # --- 1) Plot measured variogram + all fitted models
        ax_vario = axes[0, col] if axes.ndim > 1 else axes[0]
        ax_vario.plot(bin_centers, gamma, "o", color="black", label="Data")

        # Plot each model in a different color
        color_cycle = plt.cm.tab20(np.linspace(0, 1, len(fits)))
        for (m_name, (m_instance, m_stats)), color in zip(fits.items(), color_cycle):
            m_instance.plot(x_max=dist_max, ax=ax_vario, color=color, linestyle="--")
            if col == 0:
                # Collect handles for a global legend (only once)
                global_legend_handles.append(
                    plt.Line2D([0], [0], color=color, linestyle="--", label=m_name)
                )

        # Highlight best model in a thicker red line
        best_model.plot(x_max=dist_max, ax=ax_vario, color="red", linewidth=2.0)
        ax_vario.set_title(
            f"{label} | Best: {best_model_name} (R²={best_stats['r2']:.3f})"
        )
        ax_vario.set_xlabel("Distance")
        ax_vario.set_ylabel("Gamma")

        # --- 2) Generate a new random field from best model
        ax_noise = axes[1, col] if axes.ndim > 1 else axes[1]
        _, _, new_data = generate_random_fields(best_model, shape=shape)
        im_noise = ax_noise.imshow(new_data, cmap=colormap, origin="lower")
        ax_noise.set_title(f"Noise Map: {label}")
        fig.colorbar(im_noise, ax=ax_noise, fraction=0.046, pad=0.04)

        # --- 3) FFT power spectrum of the new random field
        ax_fft = axes[2, col] if axes.ndim > 1 else axes[2]
        fft_noise = np.fft.fftshift(np.fft.fft2(new_data))
        power_spectrum = np.log1p(np.abs(fft_noise))
        im_fft = ax_fft.imshow(power_spectrum, cmap="gray", origin="lower")
        ax_fft.set_title(f"Power Spectrum: {label}")
        fig.colorbar(im_fft, ax=ax_fft, fraction=0.046, pad=0.04)

        # If saving to CSV, write best model info
        if output_csv is not None:
            # Extract param errors
            pcov = best_stats["pcov"]
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else [0, 0, 0]
            # Format row
            row = [
                label,
                best_model_name,
                f"{best_stats['r2']:.4f}",
                f"{best_stats['params']['var']:.4f}",
                f"{perr[0]:.4f}",
                f"{best_stats['params']['len_scale']:.4f}",
                f"{perr[1]:.4f}",
                f"{best_stats['params'].get('nugget', 0.0):.4f}",
                f"{perr[2]:.4f}" if len(perr) >= 3 else "0.0000",
            ]
            with open(output_csv, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row)

    # Global legend outside the subplots
    fig.legend(
        handles=global_legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=min(5, len(fits)),  # or some other layout
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(
        "scripts/noise_modelling/gaussian_random_markov_fields/anisotropic_variogram_analysis.svg",
        format="svg",
    )
    plt.show()


def main():
    """
    Example usage combining everything in one flow:
    1) Load data
    2) Compute mask
    3) Plot the slice + mask
    4) Compute isotropic + anisotropic variograms for angles
    5) Plot results
    """
    # --- Example data (replace with your actual paths)
    base_path = "/home/mario/Python/Datasets/Meningiomas/Meningioma_Adquisition"
    output_npz_path = "/home/mario/Python/Datasets/Meningiomas/npz"

    slice_index = 18
    patient = "P15"
    pulse = "T1"

    # Load the data
    filepath = os.path.join(output_npz_path, patient, f"{patient}_{pulse}.npz")
    slice_data = load_data(filepath, slice_index=slice_index)

    # Compute the mask that overlays on top of the brain and skull
    mask_inverted = ImageProcessing.convex_hull_mask(
        image=slice_data, threshold_method="li", min_object_size=100
    )
    mask = (mask_inverted == 0).astype(bool)

    # Sanity check
    rotation_angle = 90
    k = rotation_angle // 90  # Calculate k based on the angle
    slice_data = np.rot90(slice_data, k=k)
    mask = np.rot90(mask, k=k)

    # Tunable hyperparameters
    variogram_bins = np.linspace(0, 150, 150)
    variogram_sampling_size = 3000
    variogram_sampling_seed = 19920516
    covariance_len_scale = 100.0
    angles_tol = np.pi / 8

    # Extract phase from approximated k-space
    phase_data, k_space = extract_phase_from_kspace(slice_data)

    # Convert to complex data
    slice_data_real, _ = to_real_imag(slice_data, phase_data)

    masked_values = slice_data[mask]
    var_mask = np.var(masked_values) if len(masked_values) > 1 else 0

    # 3) Plot the slice + mask
    plot_mask_images(
        image=slice_data,
        mask=mask,
        alpha=0.6,
        title_prefix="MRI Slice",
        figsize=(12, 4),
    )

    # 4) Compute isotropic + anisotropic variograms
    #    E.g. angles every 45 degrees
    angles_list = [0, 45, 90, 135, 180]
    results_dict = compute_variograms(
        data=slice_data_real,
        mask=mask,
        bins=variogram_bins,
        angles_deg=angles_list,
        sampling_size=variogram_sampling_size,
        sampling_seed=variogram_sampling_seed,
        angles_tol=angles_tol,
        var_guess=var_mask,
        len_scale_guess=covariance_len_scale,
    )

    # 5) Plot the results
    plot_variogram_results(
        variogram_dict=results_dict,
        dist_max=variogram_bins[-1],
        shape=slice_data.shape,
        output_csv="scripts/noise_modelling/gaussian_random_markov_fields/anisotropic_variogram_results.csv",
        colormap="seismic",
    )


if __name__ == "__main__":
    main()
