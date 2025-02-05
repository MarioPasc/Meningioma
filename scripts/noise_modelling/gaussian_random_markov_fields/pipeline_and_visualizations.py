from typing import Any, Tuple, Dict, List
from numpy.typing import NDArray
import os

import numpy as np
import pandas as pd
import csv
import gstools as gs
from gstools.covmodel.plot import plot_variogram  # type: ignore

from Meningioma.image_processing import ImageProcessing  # type: ignore
from Meningioma.utils import Stats, npz_converter  # type: ignore

import matplotlib.pyplot as plt
from cycler import cycler
import scienceplots  # type: ignore
import pyvista as pv  # type: ignore

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
    distribution with expected mean sigma*sqrt(pi/2). We will later subtract this bias.
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

    # Save the magnitude MRI slice using viridis colormap.
    fig_slice, ax_slice = plt.subplots()
    im = ax_slice.imshow(slice_data, cmap="viridis", origin="lower")
    ax_slice.set_xlabel("X")
    ax_slice.set_ylabel("Y")
    plt.colorbar(im, ax=ax_slice)
    disable_details(ax_slice)
    plt.savefig(os.path.join(output_folder, "magnitude_slice.svg"), bbox_inches="tight")
    plt.close(fig_slice)

    # Extract phase and save approximated k-space.
    phase_data, k_space = extract_phase_from_kspace(slice_data)
    fig_k, ax_k = plt.subplots()
    im_k = ax_k.imshow(np.log1p(np.abs(k_space)), cmap="viridis", origin="lower")
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
    fig_real, ax_real = plt.subplots()
    im_real = ax_real.imshow(slice_data_real, cmap="viridis", origin="lower")
    ax_real.set_xlabel("X")
    ax_real.set_ylabel("Y")
    plt.colorbar(im_real, ax=ax_real)
    disable_details(ax_real)
    plt.savefig(
        os.path.join(output_folder, "approximated_real.svg"), bbox_inches="tight"
    )
    plt.close(fig_real)

    # Save the approximated imaginary part.
    fig_imag, ax_imag = plt.subplots()
    im_imag = ax_imag.imshow(slice_data_imag, cmap="viridis", origin="lower")
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
    fig_mask, ax_mask = plt.subplots()
    im_mask = ax_mask.imshow(slice_data, cmap="viridis", origin="lower")
    ax_mask.contour(mask, colors="red", linewidths=1)
    ax_mask.set_xlabel("X")
    ax_mask.set_ylabel("Y")
    plt.colorbar(im_mask, ax=ax_mask)
    disable_details(ax_mask)
    plt.savefig(os.path.join(output_folder, "mask_overlay.svg"), bbox_inches="tight")
    plt.close(fig_mask)

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
        fig_vario, ax_vario = plt.subplots()
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

    # Subtract Rayleigh bias from the magnitude.
    # For a Rayleigh distribution with parameter sigma, bias = sigma*sqrt(pi/2).
    sigma = np.sqrt(best_model.var)
    bias = sigma * np.sqrt(np.pi / 2)
    noise_final_corrected = np.maximum(noise_final - bias, 0)

    # Save the generated real part.
    fig_noise_real, ax_noise_real = plt.subplots()
    im_noise_real = ax_noise_real.imshow(noise_real, cmap="viridis", origin="lower")
    ax_noise_real.set_xlabel("X")
    ax_noise_real.set_ylabel("Y")
    plt.colorbar(im_noise_real, ax=ax_noise_real)
    disable_details(ax_noise_real)
    plt.savefig(os.path.join(output_folder, "generated_real.svg"), bbox_inches="tight")
    plt.close(fig_noise_real)

    # Save the generated imaginary part.
    fig_noise_imag, ax_noise_imag = plt.subplots()
    im_noise_imag = ax_noise_imag.imshow(noise_imag, cmap="viridis", origin="lower")
    ax_noise_imag.set_xlabel("X")
    ax_noise_imag.set_ylabel("Y")
    plt.colorbar(im_noise_imag, ax=ax_noise_imag)
    disable_details(ax_noise_imag)
    plt.savefig(os.path.join(output_folder, "generated_imag.svg"), bbox_inches="tight")
    plt.close(fig_noise_imag)

    # Save the final corrected noise slice.
    fig_noise_final, ax_noise_final = plt.subplots()
    im_noise_final = ax_noise_final.imshow(
        noise_final_corrected, cmap="viridis", origin="lower"
    )
    ax_noise_final.set_xlabel("X")
    ax_noise_final.set_ylabel("Y")
    plt.colorbar(im_noise_final, ax=ax_noise_final)
    disable_details(ax_noise_final)
    plt.savefig(os.path.join(output_folder, "final_noise.svg"), bbox_inches="tight")
    plt.close(fig_noise_final)

    # Print statistics.
    # 1. Background noise (from the original real slice, outside the brain mask)
    noise_background = slice_data_real[mask]
    mean_background = np.mean(noise_background)
    std_background = np.std(noise_background)
    print(
        "Background noise (outside brain mask) - Mean: {:.4f}, Std: {:.4f}".format(
            mean_background, std_background
        )
    )

    # 2. Generated noise slice (real component)
    mean_noise_real = np.mean(noise_real)
    std_noise_real = np.std(noise_real)
    print(
        "Generated noise slice (real) - Mean: {:.4f}, Std: {:.4f}".format(
            mean_noise_real, std_noise_real
        )
    )

    # 3. Generated noise slice (imaginary component)
    mean_noise_imag = np.mean(noise_imag)
    std_noise_imag = np.std(noise_imag)
    print(
        "Generated noise slice (imaginary) - Mean: {:.4f}, Std: {:.4f}".format(
            mean_noise_imag, std_noise_imag
        )
    )

    # 4. Final GRF-generated noise slice (corrected)
    mean_noise_final = np.mean(noise_final_corrected)
    std_noise_final = np.std(noise_final_corrected)
    print(
        "Final GRF-generated noise slice (corrected) - Mean: {:.4f}, Std: {:.4f}".format(
            mean_noise_final, std_noise_final
        )
    )

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
    x = np.arange(50)
    y = np.arange(50)
    z = np.arange(50)
    model3d = gs.Gaussian(dim=3, len_scale=[16, 8, 4], angles=(0.8, 0.4, 0.2))
    srf3d = gs.SRF(model3d)
    field3d = srf3d((x, y, z), mesh_type="structured")
    vtk_path = os.path.join(output_folder, "3d_field.vtk")
    srf3d.vtk_export(vtk_path)
    print(f"3D field saved to {vtk_path}")
    mesh = srf3d.to_pyvista()
    mesh.contour(isosurfaces=8).plot()


if __name__ == "__main__":
    main()
