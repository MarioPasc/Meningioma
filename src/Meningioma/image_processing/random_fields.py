from typing import Any, Tuple, Dict, List, Optional
from numpy.typing import NDArray

import numpy as np
import gstools as gs
from tqdm import tqdm


"""
This script contains all the functions that perform the estimation of isotrophic and anisotrophic MRI noise
by fitting covariance models to variograms generated following a preferred direction (anisotrophic) or with
any direction (isotrophic).

These functions include:
    1. Computing the isotrophic variogram
    2. Computing the anisotrophic variogram -given various directions to focus. 
    3. Finding the best-fitting covariance model from a list. This selection is based on the R^2 obtained by the model. 
    4. Generating the final noise map. By combining two generated noise maps in a linear way, we convert the typical
       Gaussian-like noise seen in the real-imaginary components of the MRI images to create the theoretical Rice noise
       distributions. 
"""


def get_estimate_isotropic_variogram(
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


def get_estimate_anisotropic_variogram(
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


def get_fitted_models(
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


def get_generate_random_fields(
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


def get_estimate_all_variograms(
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
    iso_bin_center, iso_gamma = get_estimate_isotropic_variogram(
        data=data,
        bins=bins,
        mask=mask,
        sampling_size=sampling_size,
        sampling_seed=sampling_seed,
    )
    iso_fits = get_fitted_models(
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
        bin_center, gamma = get_estimate_anisotropic_variogram(
            data=data,
            bins=bins,
            direction=direction_vector,
            mask=mask,
            angles_tol=angles_tol,
            sampling_size=sampling_size,
            sampling_seed=sampling_seed,
        )
        fits = get_fitted_models(
            bin_center, gamma, var=var_guess, len_scale=len_scale_guess
        )
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
