from .interpolation import resize, InterpolationMethod
from .nrrd_processing import open_nrrd, transversal_axis
from .noise_modelling import (
    get_kde,
    get_noise_outside_bbox,
    get_rician,
)
from .segmentation import (
    apply_active_contours,
    get_global_umbralization,
    get_local_umbralization,
    get_largest_bbox,
    get_filled_mask,
    get_convex_hull_mask,
    get_3d_volume_segmentation,
)
from .random_fields import (
    get_estimate_isotropic_variogram,
    get_estimate_anisotropic_variogram,
    get_estimate_all_variograms,
    get_fitted_models,
    get_generate_random_fields,
)
from .k_space import get_phase_from_kspace, get_real_imag
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Tuple, Optional, Union, Dict, List, Any
import gstools as gs  # type: ignore


class ImageProcessing:

    @staticmethod
    def global_histogram_segmentation(
        image: np.ndarray, method: str = "li", threshold: float = -1
    ) -> Tuple[NDArray[np.bool_], float]:
        """
        Segment an MRI image to separate the background from the cranial and intracranial regions.

        Args:
            image (np.ndarray): MRI image to segment.
            method (str, optional): Thresholding method. Options: 'otsu', 'weighted_mean','li'. Defaults to 'li'.
            threshold (float, optional): Threshold for separating background and intracranial region. Used only if method is 'manual'.
            block_size (int, optional): Block size for adaptive thresholding. Only used if method is 'adaptive'.

        Returns:
            np.ndarray: Mask where background is 0 and intracranial region is 1.
        """
        return get_global_umbralization(image=image, method=method, threshold=threshold)

    @staticmethod
    def local_histogram_segmentation(
        slice_data: NDArray[np.float64], w: int, thresholding_method: str = "li"
    ) -> Tuple[NDArray[np.uint8], NDArray[np.float64]]:
        """
        Applies mean filtering and binarizes the image using a specified thresholding method.

        Args:
            slice_data (NDArray[np.float64]): The original MRI slice.
            w (int): Neighborhood size for the mean filter.
            thresholding_method (str): The thresholding method to apply ('otsu', 'li', 'weighted_mean', 'manual').

        Returns:
            Tuple[NDArray[np.bool_], NDArray[np.float64]]: A tuple containing the binarized image
            and the mean-filtered image.
        """
        return get_local_umbralization(
            slice_data=slice_data, w=w, thresholding_method=thresholding_method
        )

    @staticmethod
    def fill_mask(
        mask: np.ndarray, structure_size: int = 7, iterations: int = 3
    ) -> np.ndarray:
        """
        Fills the mask by applying morphological closing and hole filling multiple times.

        Args:
            mask (np.ndarray): Binary mask to be processed.
            structure_size (int): Size of the structuring element for morphological operations.
            iterations (int): Number of times to apply the closing and hole-filling process.

        Returns:
            np.ndarray: Mask after applying the filter multiple times.
        """
        return get_filled_mask(
            mask=mask, structure_size=structure_size, iterations=iterations
        )

    @staticmethod
    def convex_hull_mask(
        image: NDArray[np.float64], threshold_method: str = "li"
    ) -> NDArray[np.float64]:
        """
        Generate a convex hull mask for the brain/skull region.

        Parameters
        ----------
        image : NDArray[np.float64]
            Input 2D image (grayscale, normalized between 0-255).
        threshold_method : str
            Thresholding method for segmentation (e.g., 'li', 'otsu').
        min_object_size : int
            Minimum size of objects to keep after thresholding.

        Returns
        -------
        np.ndarray
            A binary mask (2D) with the brain/skull region inside and background outside.
        """
        return get_convex_hull_mask(
            image=image,
            threshold_method=threshold_method,
        )

    @staticmethod
    def segment_3d_volume(
        filepath: str,
        threshold_method: str = "li",
        structure_size_2d: int = 7,
        iterations_2d: int = 3,
        structure_size_3d: int = 3,
        iterations_3d: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implements the same segmentation pipeline as debug_segmentation_steps, but returns
        only the final (volume, mask) without displaying intermediate stages.

        Steps:
        1) Load volume from NPZ (assumed shape = (H, W, S) after squeeze).
        2) Global min-max normalization to [0, 255].
        3) For each slice:
            - Threshold (largest contour -> convex hull) + 2D morphological closing.
        4) (Optional) 3D morphological closing for inter-slice consistency.
        5) Largest 3D connected component is kept.
        6) Return the original volume (float32) and final mask (boolean).

        Parameters
        ----------
        filepath : str
            Path to the NPZ file. The volume is assumed in npz_data["data"][0], shape (H, W, S).
        threshold_method : str, optional
            Threshold method, e.g. "otsu" or "li". By default "otsu".
        structure_size_2d : int, optional
            Size of the kernel for 2D morphological closing. Default 7.
        iterations_2d : int, optional
            Number of iterations for 2D morphological closing. Default 3.
        structure_size_3d : int, optional
            Size of the kernel for 3D morphological closing. Default 3.
        iterations_3d : int, optional
            Number of iterations for the 3D morphological closing. Default 1.

        Returns
        -------
        volume : np.ndarray
            The loaded original volume (float32), shape (H, W, S).
        mask : np.ndarray
            A 3D boolean array (same shape as volume) with the segmented region = True.
        """
        return get_3d_volume_segmentation(
            filepath=filepath,
            threshold_method=threshold_method,
            structure_size_2d=structure_size_2d,
            iterations_2d=iterations_2d,
            structure_size_3d=structure_size_3d,
            iterations_3d=iterations_3d,
        )

    @staticmethod
    def find_largest_bbox(
        mask: np.ndarray,
        extra_margin: Tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> Tuple[int, int, int, int]:
        """
        Finds the largest bounding box within a binary mask and optionally extends it by
        specified margins in each direction. Adjusts the bounding box if it exceeds image
        boundaries, showing a message when adjustments are made.

        Parameters:
        - mask (np.ndarray): A 2D binary array representing the mask where regions are defined by 1s.
        - extra_margin (tuple of int, optional): A tuple (min_row_margin, min_col_margin,
        max_row_margin, max_col_margin) specifying the number of units to adjust the bounding
        box in each direction. Default is (0, 0, 0, 0), meaning no extension.

        Returns:
        - largest_bbox (tuple): A tuple (min_row, min_col, max_row, max_col) defining the bounding
        box coordinates for the largest region in the mask, with any specified margin applied and
        adjusted to fit within the image boundaries.
        """

        return get_largest_bbox(mask=mask, extra_margin=extra_margin)

    @staticmethod
    def active_contours(
        image: NDArray[np.float64],
        bounding_box: Tuple[int, int, int, int],
        alpha: float = 0.015,
        beta: float = 10,
        gamma: float = 0.001,
        iterations: int = 250,
    ) -> NDArray[np.bool_]:
        """
        Applies the Active Contours algorithm to segment a region within the largest bounding box.

        Args:
            image (np.ndarray): Original MRI image.
            bounding_box (Tuple[int, int, int, int]): Coordinates of the bounding box (min_row, min_col, max_row, max_col).
            alpha (float): Snake smoothness parameter. Higher values mean smoother contours.
            beta (float): Snake elasticity parameter.
            gamma (float): Time step for the snake movement.
            iterations (int): Number of iterations for the Active Contours algorithm.

        Returns:
            np.ndarray: Refined binary mask of the segmented region.
        """
        return apply_active_contours(
            image=image,
            bounding_box=bounding_box,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            iterations=iterations,
        )

    @staticmethod
    def extract_noise_outside_bbox(
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Extracts noise values from an image that lie outside a specified bounding box.

        Parameters:
        - image (np.ndarray): A 2D array representing the input image from which noise values are extracted.
        - bbox (tuple[int, int, int, int]): A tuple (min_row, min_col, max_row, max_col) defining the bounding box
        coordinates. These coordinates represent the region to exclude when extracting noise.
        - mask (np.ndarray): A binary or boolean mask array with the same shape as `image` where regions of interest
        are defined. This mask is used to identify pixels outside the bounding box.

        Returns:
        - np.ndarray: An array of pixel values from `image` that are outside the specified bounding box.
        """
        return get_noise_outside_bbox(image=image, bbox=bbox, mask=mask)

    @staticmethod
    def kde(
        noise_values: Union[NDArray[np.float64], List[float]],
        h: float = 1.0,
        num_points: int = 1000,
        return_x_values: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate the probability density function (PDF) of noise values using a Kernel Density Estimation (KDE) approach.

        This function computes a KDE of the provided noise values, allowing for adjustable smoothing with `h`
        and controlling the granularity of the output PDF with `num_points`.

        Parameters:
        - noise_values (Union[NDArray[np.float64], List[float]]): A list of noise values sampled from an MRI image's background or similar data.
        - h (float, optional): The bandwidth smoothing parameter for KDE. Controls the width of the Gaussian kernel.
        Higher values result in smoother KDEs. Default is 1.0.
        - num_points (int, optional): The number of points to evaluate the KDE over the range of `noise_values`.
        A higher number increases resolution of the PDF but may require more computation time. Default is 1000.

        Returns:
        - np.ndarray: The estimated PDF values across the specified number of points within the range of `noise_values`.
        """
        return get_kde(
            noise_values=noise_values,
            h=h,
            num_points=num_points,
            return_x_values=return_x_values,
        )

    @staticmethod
    def rician(x_values: np.ndarray, sigma: float) -> np.ndarray:
        """
        Calculate the Rician probability density function (PDF) for given values and sigma.

        Parameters:
        - x_values (np.ndarray): Values over which to calculate the Rician PDF.
        - sigma (float): Scale parameter of the Rician distribution.

        Returns:
        - np.ndarray: Calculated Rician PDF values.
        """
        return get_rician(x_values=x_values, sigma=sigma)

    @staticmethod
    def open_nrrd_file(
        nrrd_path: str, return_header: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Open the given nrrd file path.

        params:
            - nrrd_path (str): Path to the nrrd file to open.
            - return_header (bool): If True, return the header along with the image. Defaults to False.

        returns:
            - Union[np.ndarray, Tuple[np.ndarray, dict]]:
                If return_header is True, returns (image, header).
                Otherwise, returns image only.
        """
        return open_nrrd(nrrd_path=nrrd_path, return_header=return_header)

    @staticmethod
    def get_transversal_axis(nrrd_path: str) -> np.intp:
        """
        Finds the transversal axis of a nrrd file

        args:
            - nrrd_path (str): Path to the nrrd file.

        returns:
            - transversal_axis: Tranversal axis of the nrrd file.
        """
        return transversal_axis(nrrd_path=nrrd_path)

    @staticmethod
    def extract_transversal_slice(
        image_data: np.ndarray,
        transversal_axis: np.intp,
        slice_index: int = -1,
    ) -> np.ndarray:
        """
        Extracts the middle slice along the transversal axis.
        """
        if slice_index == -1:
            # Calculate the middle slice index along the transversal axis
            slice_index = image_data.shape[transversal_axis] // 2

        # Use np.take to extract the middle slice from the correct axis
        slice = np.take(image_data, slice_index, axis=transversal_axis)

        return slice

    @staticmethod
    def find_center_of_mass(
        image: np.ndarray, threshold: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        """
        Find the center of mass of the bright region in the image, optionally applying a threshold.

        Args:
            image (np.ndarray): Input image (grayscale, float32, int16, etc.).
            threshold (Optional[Tuple[int, int]]): Threshold values (low, high) for segmentation.

        Returns:
            (int, int): Coordinates of the center of mass (x, y).
        """
        if threshold is not None:
            _, image = cv2.threshold(
                image, threshold[0], threshold[1], cv2.THRESH_BINARY
            )

        image = image.astype(np.float64)
        M = cv2.moments(image)

        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = image.shape[1] // 2, image.shape[0] // 2

        return center_x, center_y

    @staticmethod
    def calculate_aspect_ratio(
        image: np.ndarray, target_size: Tuple[int, int]
    ) -> float:
        """
        Calculate the scaling factor to preserve the aspect ratio.

        Args:
            image (np.ndarray): Input image.
            target_size (tuple): Target resolution (width, height).

        Returns:
            float: Scaling factor to preserve aspect ratio.
        """
        original_h, original_w = image.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / original_w, target_h / original_h)
        return scale

    @staticmethod
    def add_padding(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Pad the image to fit the target resolution.

        Args:
            image (np.ndarray): Image with preserved aspect ratio.
            target_size (tuple): Target resolution (width, height).

        Returns:
            np.ndarray: Padded image.
        """
        target_w, target_h = target_size
        padded_image = np.zeros((target_h, target_w), dtype=image.dtype)

        new_h, new_w = image.shape[:2]
        start_y = (target_h - new_h) // 2
        start_x = (target_w - new_w) // 2
        padded_image[start_y : start_y + new_h, start_x : start_x + new_w] = image

        return padded_image

    @staticmethod
    def apply_translation(
        image: NDArray[np.uint8],
        center_x: int,
        center_y: int,
        target_size: Tuple[int, int],
    ) -> NDArray[np.uint8]:
        """
        Apply translation to center the region of interest in the image.

        Args:
            image (NDArray[np.uint8]): Input image with padding.
            center_x (int): X-coordinate of the region's center.
            center_y (int): Y-coordinate of the region's center.
            target_size (Tuple[int, int]): Target resolution (width, height).

        Returns:
            NDArray[np.uint8]: Translated image with the region centered.
        """
        # Extract target width and height
        target_w, target_h = target_size

        # Calculate the center of the target image
        image_center_x, image_center_y = target_w // 2, target_h // 2

        # Compute translation offsets
        offset_x: int = image_center_x - center_x
        offset_y: int = image_center_y - center_y

        # Create a 2D translation matrix for affine transformation
        translation_matrix: NDArray[np.float32] = np.float32(
            [[1, 0, offset_x], [0, 1, offset_y]]  # type: ignore
        )

        # Apply the affine transformation (translation) to the image
        translated_image: NDArray[np.uint8] = cv2.warpAffine(
            image, translation_matrix, (target_w, target_h)  # type: ignore
        )

        return translated_image

    @staticmethod
    def resize_image_with_method(
        image: NDArray[np.float64],
        target_size: Tuple[int, int],
        method: InterpolationMethod,
    ) -> NDArray[np.float64]:
        """
        Resize the image using the specified interpolation method.

        Args:
            image (np.ndarray): Input image.
            target_size (tuple): Target resolution (width, height).
            method (InterpolationMethod): Interpolation method to use for resizing.

        Returns:
            np.ndarray: Resized image.
        """
        resized_image = resize(image, target_size, method)
        return resized_image

    @staticmethod
    def resize_with_padding_and_center(
        image: np.ndarray,
        target_size: Tuple[int, int] = (512, 512),
        method: InterpolationMethod = InterpolationMethod.INTER_LANCZOS4,
        return_centers_of_mass: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Tuple[int, int]]]]:
        """
        Resizes the image while preserving aspect ratio, then pads and centers the brain in the target resolution.

        Args:
            image (np.ndarray): Input image (brain scan).
            target_size (tuple): Target resolution (width, height) for output image.

        Returns:
            Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int]]:
            Padded image, resized image, offset (offset_x, offset_y), center of mass (cx, cy).
        """
        # Step 1: Calculate the scale factor for aspect ratio preservation
        scale = ImageProcessing.calculate_aspect_ratio(
            image=image, target_size=target_size
        )
        new_w = int(image.shape[1] * scale)
        new_h = int(image.shape[0] * scale)

        # Resize the image to the new dimensions
        resized_image = ImageProcessing.resize_image_with_method(
            image=image, target_size=(new_w, new_h), method=method
        )

        # Step 2: Pad the resized image to the target size
        padded_image = ImageProcessing.add_padding(resized_image, target_size)

        # Step 3: Find the center of mass of the resized image (brain region)
        cx_resized, cy_resized = ImageProcessing.find_center_of_mass(resized_image)

        # Step 4: Calculate the translation offsets to center the brain region
        target_center_x, target_center_y = (
            target_size[0] // 2,
            target_size[1] // 2,
        )
        offset_x, offset_y = (
            target_center_x - cx_resized,
            target_center_y - cy_resized,
        )

        # Apply translation to center the brain in the padded image
        centered_image = ImageProcessing.apply_translation(
            padded_image, target_center_x, target_center_y, target_size
        )

        if return_centers_of_mass:
            return (
                centered_image,
                {
                    "Original_CM": (cx_resized, cy_resized),
                    "Translated_CM": (offset_x, offset_y),
                },
            )
        else:
            return centered_image

    @staticmethod
    def estimate_isotropic_variogram(
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
        return get_estimate_isotropic_variogram(
            data=data,
            bins=bins,
            mask=mask,
            sampling_size=sampling_size,
            sampling_seed=sampling_seed,
        )

    @staticmethod
    def estimate_anisotropic_variogram(
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
        return get_estimate_anisotropic_variogram(
            data=data,
            bins=bins,
            direction=direction,
            mask=mask,
            angles_tol=angles_tol,
            sampling_size=sampling_size,
            sampling_seed=sampling_seed,
        )

    @staticmethod
    def fit_covariance_models(
        bin_center: np.ndarray,
        gamma: np.ndarray,
        var: float = 1.0,
        len_scale: float = 10.0,
        model_classes: Dict[str, gs.CovModel] = {
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
        },
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
        return get_fitted_models(
            bin_center=bin_center, gamma=gamma, var=var, len_scale=len_scale
        )

    @staticmethod
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
        return get_generate_random_fields(
            model=model, shape=shape, seed_real=seed_real, seed_imag=seed_imag
        )

    @staticmethod
    def estimate_all_variograms(
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
        return get_estimate_all_variograms(
            data=data,
            mask=mask,
            bins=bins,
            angles_deg=angles_deg,
            angles_tol=angles_tol,
            sampling_size=sampling_size,
            sampling_seed=sampling_seed,
            var_guess=var_guess,
            len_scale_guess=len_scale_guess,
        )

    @staticmethod
    def estimate_phase_from_kspace(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
        return get_phase_from_kspace(image=image)

    @staticmethod
    def get_real_and_complex_images(
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
        return get_real_imag(magnitude=magnitude, phase=phase)
