from typing import Any, Tuple, Dict, List, Optional
from numpy.typing import NDArray

import numpy as np
import gstools as gs  # type: ignore
from tqdm import tqdm  # type: ignore


"""
This script contains all the functions that perform the estimation of isotrophic and anisotrophic MRI noise
by fitting covariance models to variograms generated following a preferred direction (anisotrophic) or with
any direction (isotrophic).

These functions include:
    1. Computing the isotrophic variogram
    2. Computing the anisotrophic variogram -given various directions to focus. 
    3. Finding the best-fitting covariance model from a list. This selection is based on the R^2 obtained by the model. 
    4. Getting the final noise volume or noise map in 2d.
"""


def get_gaussian_random_fields_noise_2d(
    model: gs.CovModel,
    shape: Tuple[int, int],
    independent: bool = True,
    seed_real: int = 19770928,
    seed_imag: int = 19773022,
    seed_3d: int = 19770928,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic Gaussian random field noise for a 2D image.

    Two modes are available:

      1. **independent=True** (default):
         Two independent 2D Gaussian random fields (representing the real and imaginary parts)
         are generated using gs.SRF with a "structured" mesh. The final noise is computed as:

             noise = sqrt(real_field^2 + imag_field^2)

         The use of two independent seeds ensures that the two fields are generated independently.

      2. **independent=False**:
         A single 3D Gaussian random field is generated over a volume of shape (n, m, 2)
         using gs.SRF with a "structured" mesh. The two slices along the third dimension are then
         extracted as the real and imaginary parts and combined via the modulus operation:

             noise = sqrt(slice_0^2 + slice_1^2)

         In this case the two channels are correlated as they come from one 3D realization.

    **Mesh Type Considerations:**
    When using gs.SRF, the `mesh_type` parameter determines how the input coordinate tuple is interpreted.
    With `"structured"`, the provided arrays (or ranges) define the grid along each axis, which is ideal
    for regularly spaced images or volumes. For irregular grids one might use `"unstructured"`.

    Parameters
    ----------
    model : gs.CovModel
        The best-fit covariance model to be used for generating the noise.
    shape : Tuple[int, int]
        A tuple (n, m) defining the size of the 2D image.
    independent : bool, optional
        If True (default), generate two independent 2D fields (using separate seeds).
        If False, generate one 3D field of shape (n, m, 2) and extract two slices.
    seed_real : int, optional
        Random seed for the real part (used if independent is True).
    seed_imag : int, optional
        Random seed for the imaginary part (used if independent is True).
    seed_3d : int, optional
        Random seed for the 3D field (used if independent is False).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple (real_field, imag_field, combined_field) where each array has shape (n, m).
        The combined field is computed as sqrt(real_field**2 + imag_field**2).
    """
    n, m = shape

    if independent:
        # Generate independent fields for real and imaginary parts using a structured mesh.
        x = np.arange(n)
        y = np.arange(m)
        z = np.arange(1)
        # Generate real part.
        srf_real = gs.SRF(model, seed=seed_real)
        real_field = srf_real((x, y, z), mesh_type="structured")
        # Generate imaginary part.
        srf_imag = gs.SRF(model, seed=seed_imag)
        imag_field = srf_imag((x, y, z), mesh_type="structured")

    else:
        # Generate a single 3D volume of noise with two slices along the third dimension.
        x = np.arange(n)
        y = np.arange(m)
        z = np.arange(2)  # Two slices.
        srf_3d = gs.SRF(model, seed=seed_3d)
        # Use a structured mesh since the grid is regularly spaced.
        volume_3d = srf_3d((x, y, z), mesh_type="structured")
        # Extract the two slices.
        real_field = volume_3d[:, :, 0]
        imag_field = volume_3d[:, :, 1]

    # Combine the two fields via the modulus operation.
    combined_field = (np.sqrt(real_field**2 + imag_field**2)) / (np.sqrt(2))
    return real_field, imag_field, combined_field


import numpy as np
import gstools as gs
from typing import Tuple, Optional


def get_gaussian_random_fields_noise_3d(
    model: gs.CovModel,
    shape: Tuple[int, int, int],
    seed_real: int = 19770928,
    seed_imag: int = 19773022,
    voxel_size: Optional[Tuple[float, float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate independent 3D Gaussian random field volumes representing the real and
    imaginary parts of noise and combine them using the modulus operation.

    Two independent 3D fields are simulated over a structured grid defined by
    `shape` = (nx, ny, nz) using two separate seeds. The final noise volume
    is computed as:

        combined = sqrt(real_field**2 + imag_field**2) / sqrt(2)

    If `voxel_size` is provided, the function will use upscaling with
    coarse graining to take voxel volume into account.

    Parameters
    ----------
    model : gs.CovModel
        The best-fit covariance model to use for noise generation.
    shape : Tuple[int, int, int]
        Desired shape of the noise volume as (nx, ny, nz).
    seed_real : int, optional
        Random seed for generating the real part of the noise (default 19770928).
    seed_imag : int, optional
        Random seed for generating the imaginary part of the noise (default 19773022).
    voxel_size : Optional[Tuple[float, float, float]], optional
        If provided, must be (dx, dy, dz). Each SRF call will use
        upscaling='coarse_graining' and set point_volumes = dx*dy*dz.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple (real_volume, imag_volume, combined_volume), where each array
        has shape `shape`. The combined_volume is computed as the modulus
        (Euclidean norm) of the two volumes.
    """

    nx, ny, nz = shape
    # Define coordinate arrays for a structured grid.
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)

    # If voxel_size is specified, compute the volume and use coarse_graining.
    if voxel_size is not None:
        dx, dy, dz = voxel_size
        voxel_volume = dx * dy * dz
        srf_real = gs.SRF(model, seed=seed_real, upscaling="coarse_graining")
        real_volume = srf_real(
            (x, y, z), mesh_type="structured", point_volumes=voxel_volume
        )

        srf_imag = gs.SRF(model, seed=seed_imag, upscaling="coarse_graining")
        imag_volume = srf_imag(
            (x, y, z), mesh_type="structured", point_volumes=voxel_volume
        )
    else:
        # Default behavior if no voxel_size is specified
        srf_real = gs.SRF(model, seed=seed_real, upscaling="no_scaling")
        real_volume = srf_real((x, y, z), mesh_type="structured")

        srf_imag = gs.SRF(model, seed=seed_imag, upscaling="no_scaling")
        imag_volume = srf_imag((x, y, z), mesh_type="structured")

    # Combine the two volumes by taking the Euclidean norm and normalize by √2.
    combined_volume = np.sqrt(real_volume**2 + imag_volume**2) / np.sqrt(2)

    return real_volume, imag_volume, combined_volume
