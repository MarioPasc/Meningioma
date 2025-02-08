import os
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import gstools as gs  # type: ignore

# Import the convex hull function from your package.
from Meningioma import ImageProcessing  # type: ignore


# =============================================================================
# 1. Volume and Mask Loading Function
# =============================================================================
def load_volume_and_mask(
    filepath: str, threshold_method: str = "li"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a 3D volume from an NPZ file and generate a volumetric exclusion mask by
    applying a 2D convex hull mask on each slice.

    The volume is loaded using:
        npz_data = np.load(filepath)
        volume = np.squeeze(npz_data["data"][0, :, :, :])

    For each 2D slice in the volume, the function
        ImageProcessing.convex_hull_mask(image=slice_data, threshold_method=threshold_method)
    is applied. The resulting mask is thresholded (mask > 0) so that pixels inside the
    convex hull (brain/skull) are marked as True (to be excluded) and the background as False.

    Parameters
    ----------
    filepath : str
        Path to the NPZ file containing the volume data.
    threshold_method : str, optional
        Thresholding method used for convex hull mask estimation (default is "li").

    Returns
    -------
    volume : np.ndarray
        The loaded 3D volume.
    mask : np.ndarray
        A 3D boolean mask where True indicates regions to exclude (inside the convex hull).
    """
    npz_data = np.load(filepath)
    volume = np.squeeze(npz_data["data"][0, :, :, :])

    # Initialize an empty mask with the same shape as the volume.
    mask = np.zeros(volume.shape, dtype=bool)

    # Process each slice independently.
    for i in range(volume.shape[0]):
        slice_data = volume[i, :, :]
        hull = ImageProcessing.convex_hull_mask(
            image=slice_data, threshold_method=threshold_method
        )
        # Pixels inside the convex hull (brain/skull) are marked as True (to be excluded)
        slice_mask = hull > 0
        mask[i, :, :] = slice_mask

    print("Mask computed")
    return volume, mask


# =============================================================================
# 2. Variogram Estimation Functions (Using Only Background Voxels)
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
# 3. Covariance Model Fitting Function (3D)
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
# 4. Visualization Function (Using the Provided Plotting Style)
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
# Example Call in the Main Pipeline
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
    volume, mask = load_volume_and_mask(filepath, threshold_method="li")
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
