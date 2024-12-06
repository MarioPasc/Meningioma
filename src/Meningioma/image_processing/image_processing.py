from .interpolation import resize, InterpolationMethod
from .nrrd_processing import open_nrrd, transversal_axis
from .noise_modelling import (
    get_detected_mri_image,
    get_filled_mask,
    get_kde,
    get_largest_bbox,
    get_noise_outside_bbox,
    get_rician,
    get_preprocess_slice_with_thresholding,
)
import numpy as np
from numpy.typing import NDArray
import cv2
from typing import Tuple, Optional, Union, Dict, List


class ImageProcessing:

    @staticmethod
    def segment_by_histogram(
        image: np.ndarray, method: str = "li", threshold: float = -1
    ):
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
        return get_detected_mri_image(image=image, method=method, threshold=threshold)

    @staticmethod
    def preprocess_slice_with_thresholding(
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
        return get_preprocess_slice_with_thresholding(
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
