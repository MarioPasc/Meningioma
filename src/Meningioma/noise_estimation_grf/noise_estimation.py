from .gaussian_random_fields import (
    get_gaussian_random_fields_noise_2d,
    get_gaussian_random_fields_noise_3d,
)

from .variograms import (
    get_estimate_variogram_isotropic_3d,
    get_estimate_variogram_anisotropic_3d,
)

from .covariance_models import get_fit_model_3d

from typing import Any, Tuple, Dict, List, Optional

import numpy as np
import gstools as gs  # type: ignore


class BlindNoiseEstimation:

    @staticmethod
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
        return get_estimate_variogram_anisotropic_3d(
            data=data,
            bins=bins,
            mask=mask,
            directions=directions,
            direction_labels=direction_labels,
            estimator=estimator,
            angles_tol=angles_tol,
            sampling_size=sampling_size,
            sampling_seed=sampling_seed,
        )

    @staticmethod
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
        return get_estimate_variogram_isotropic_3d(
            data=data,
            bins=bins,
            mask=mask,
            estimator=estimator,
            sampling_size=sampling_size,
            sampling_seed=sampling_seed,
        )

    @staticmethod
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
        return get_fit_model_3d(
            bin_center=bin_center, gamma=gamma, var=var, len_scale=len_scale
        )

    @staticmethod
    def gaussian_random_fields_noise_2d(
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
        return get_gaussian_random_fields_noise_2d(
            model=model,
            shape=shape,
            independent=independent,
            seed_real=seed_real,
            seed_imag=seed_imag,
            seed_3d=seed_3d,
        )

    @staticmethod
    def gaussian_random_fields_noise_3d(
        model: gs.CovModel,
        shape: Tuple[int, int, int],
        seed_real: int = 19770928,
        seed_imag: int = 19773022,
        voxel_size: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate independent 3D Gaussian random field volumes representing the real and
        imaginary parts of noise and combine them using the modulus operation.

        In this function, two independent 3D fields are simulated over a structured grid
        defined by `shape` = (nx, ny, nz) using two separate seeds. The final noise volume
        is computed as:

            combined = sqrt(real_field**2 + imag_field**2) / sqrt(2)

        The division by sqrt(2) normalizes the noise (consistent with the 2D version).

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

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            A tuple (real_volume, imag_volume, combined_volume), where each array has shape `shape`.
            The combined_volume is computed as the modulus (Euclidean norm) of the two volumes.
        """
        return get_gaussian_random_fields_noise_3d(
            model=model,
            shape=shape,
            seed_real=seed_real,
            seed_imag=seed_imag,
            voxel_size=voxel_size,
        )
