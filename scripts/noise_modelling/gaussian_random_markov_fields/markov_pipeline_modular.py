from turtle import color
from typing import Any, Tuple, Dict, List, Optional
from numpy.typing import NDArray
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import gstools as gs  # type: ignore
from gstools.covmodel.plot import plot_variogram  # type : ignore
from tqdm import tqdm  # type: ignore

# If these are your local modules:
from Meningioma.image_processing import ImageProcessing  # type: ignore
from Meningioma.utils import Stats, npz_converter  # type: ignore

import scienceplots  # type: ignore

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


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
    # The mask had the pixels to exclude (brain and skull) set to 255 (True)
    # We want the stats of the background pixels, so we invert the mask.
    masked_values = image[~mask]
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
        color_cycle = plt.cm.tab20(np.linspace(0, 1, len(fits)))  # type: ignore
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
        _, _, new_data = ImageProcessing.generate_random_fields(best_model, shape=shape)
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
    base_path = "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition"
    output_npz_path = "/home/mariopasc/Python/Datasets/Meningiomas/npz"

    slice_index = 84
    patient = "P15"
    pulse = "T1"

    # Load the data
    filepath = os.path.join(output_npz_path, patient, f"{patient}_{pulse}.npz")
    slice_data = npz_converter.load_mri_slice(
        filepath=filepath, slice_index=slice_index
    )

    # Compute the mask that overlays on top of the brain and skull
    hull = ImageProcessing.convex_hull_mask(image=slice_data, threshold_method="li")
    mask = hull > 0

    """
    # Sanity check
    rotation_angle = 0
    k = rotation_angle // 90  # Calculate k based on the angle
    slice_data = np.rot90(slice_data, k=k)
    mask = np.rot90(mask, k=k)
    """

    # Tunable hyperparameters
    variogram_bins = np.linspace(0, 20, 30)
    variogram_sampling_size = 5000
    variogram_sampling_seed = 19920516
    covariance_len_scale = 2.5
    angles_tol = np.pi / 8

    # Extract phase from approximated k-space
    phase_data, k_space = ImageProcessing.estimate_phase_from_kspace(slice_data)

    # Convert to complex data
    slice_data_real, _ = ImageProcessing.get_real_and_complex_images(
        slice_data, phase_data
    )

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
    angles_list = [0, 45, 90, 135]
    results_dict = ImageProcessing.estimate_all_variograms(
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
