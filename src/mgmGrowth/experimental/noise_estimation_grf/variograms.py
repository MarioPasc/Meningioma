from typing import Any, Tuple, Dict, List, Optional

import numpy as np
import gstools as gs  # type: ignore


def get_estimate_variogram_isotropic_3d(
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


def get_estimate_variogram_anisotropic_3d(
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
