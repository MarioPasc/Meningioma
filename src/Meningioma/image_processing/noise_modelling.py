from scipy.stats import gaussian_kde, rice  # type: ignore
from scipy import ndimage  # type: ignore
from skimage.filters import threshold_li, threshold_otsu
from skimage import exposure, morphology, measure
import numpy as np
from typing import Tuple, List, Union
from numpy.typing import NDArray
from typing import Any

# This part of the module is made to process the images,
# general-pourpose tools like segmenting the image with histogram techniques,
# extracting a portion of the image that belongs to the largest bbox, etc...


def get_detected_mri_image(
    image: NDArray[np.float32], method: str = "li", threshold: float = 0.0
) -> NDArray[np.bool_]:
    ...

    """
    Segment an MRI image to segment the image based on the histogram

    Args:
        image (np.ndarray): MRI image to segment.
        method (str, optional): Thresholding method. Options: 'otsu',
        'weighted_mean','li'. Defaults to 'li'.

        threshold (float, optional): Threshold for separating background
        and intracranial region. Used only if method is 'manual'.

        block_size (int, optional): Block size for adaptive thresholding.
        Only used if method is 'adaptive'.

    Returns:
        np.ndarray: Mask where background is 0 and intracranial region is 1.
    """
    # Compute histogram of the image for 'weighted_mean'
    hist, hist_centers = exposure.histogram(image)

    # Select the thresholding method
    if method == "otsu":
        threshold = threshold_otsu(image)

    elif method == "weighted_mean":
        threshold = np.average(hist_centers, weights=hist)

    elif method == "li":
        threshold = threshold_li(image)

    elif method == "manual":
        if threshold is None:
            raise ValueError(
                "For 'manual' method, " "a threshold value must be provided."
            )

    else:
        raise ValueError(
            "Invalid method specified. Choose from 'otsu', 'weighted_mean', 'li'."
        )

    # Create mask where values less than or equal to the threshold are considered background
    background_mask = image <= threshold

    # Invert the mask so that 0 is background
    intracranial_mask = ~background_mask

    return intracranial_mask


def get_preprocess_slice_with_thresholding(
    slice_data: NDArray[np.float64], w: int, thresholding_method: str = "li"
) -> Tuple[NDArray[np.uint8], NDArray[np.float64]]:
    """
    Applies mean filtering and binarizes the image using a specified thresholding method.

    Args:
        slice_data (NDArray[np.float64]): The original MRI slice.
        w (int): Neighborhood size for the mean filter.
        thresholding_method (str): The thresholding method to apply ('otsu', 'li', 'weighted_mean', 'manual').

    Returns:
        Tuple[NDArray[np.uint8], NDArray[np.float64]]: A tuple containing the binarized image
        and the mean-filtered image.
    """
    try:
        # Apply a mean filter
        mean_filtered = ndimage.uniform_filter(
            slice_data, size=w
        ) + 0.0001 * np.random.normal(size=slice_data.shape)

        # Binarize using the thresholding function
        binarized_mask = get_detected_mri_image(
            mean_filtered, method=thresholding_method
        )

        # Convert boolean mask to uint8
        binarized_image = binarized_mask.astype(np.uint8)

        return binarized_image, mean_filtered
    except Exception as e:
        raise RuntimeError(f"Error during preprocessing with thresholding: {e}")


def get_filled_mask(
    mask: NDArray[np.bool_], structure_size: int = 7, iterations: int = 3
) -> NDArray[np.uint8]:
    """
    Fills the mask by applying morphological closing and hole filling multiple times.

    Args:
        mask (np.ndarray): Binary mask to be processed.
        structure_size (int): Size of the structuring element for morphological operations.
        iterations (int): Number of times to apply the closing and hole-filling process.

    Returns:
        np.ndarray: Mask after applying the filter multiple times.
    """
    structuring_element = morphology.disk(structure_size)

    # Apply the filter multiple times
    for _ in range(iterations):
        # Morphological closing
        mask = morphology.closing(mask, structuring_element)
        # Hole filling
        mask = ndimage.binary_fill_holes(
            mask, structure=np.ones((structure_size, structure_size))
        )  # type: ignore

    return mask.astype(np.uint8)


def get_largest_bbox(
    mask: NDArray[np.bool_],
    extra_margin: Tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Tuple[int, int, int, int]:
    """
    Finds the largest bounding box within a binary mask and optionally extends it by
    specified margins in each direction. Ensures the bounding box fits within image boundaries.

    Args:
        mask (NDArray[np.bool_]): Binary mask where regions are defined by True values.
        extra_margin (Tuple[int, int, int, int]): Margins to extend the bounding box in the order
            (min_row_margin, min_col_margin, max_row_margin, max_col_margin). Defaults to (0, 0, 0, 0).

    Returns:
        Tuple[int, int, int, int]: Coordinates of the largest bounding box (min_row, min_col, max_row, max_col).
    """
    # Label the connected components in the mask
    labeled_mask, _ = ndimage.label(mask)  # type: ignore
    regions = measure.regionprops(labeled_mask)  # type: ignore

    # Initialize variables to track the largest bounding box
    largest_bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    max_area: int = 0

    # Find the region with the largest area
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        area = (max_row - min_row) * (max_col - min_col)

        if area > max_area:
            max_area = area
            largest_bbox = (min_row, min_col, max_row, max_col)

    # Apply margins to the largest bounding box
    min_row, min_col, max_row, max_col = largest_bbox
    min_row_margin, min_col_margin, max_row_margin, max_col_margin = extra_margin

    # Ensure the bounding box fits within the image boundaries
    min_row = max(0, min_row - min_row_margin)
    min_col = max(0, min_col - min_col_margin)
    max_row = min(mask.shape[0], max_row + max_row_margin)
    max_col = min(mask.shape[1], max_col + max_col_margin)

    return (min_row, min_col, max_row, max_col)


def get_noise_outside_bbox(
    image: NDArray[np.float64], bbox: tuple[int, int, int, int], mask: NDArray[np.int8]
) -> NDArray[np.float64]:
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
    min_row, min_col, max_row, max_col = bbox

    # Create a mask to exclude the bounding box region
    outside_bbox_mask = np.ones_like(mask, dtype=bool)
    outside_bbox_mask[min_row:max_row, min_col:max_col] = False

    # Extract noise values from the image that are outside the bounding box
    noise_values: NDArray[np.float64] = image[outside_bbox_mask]
    return noise_values


# Here we are defining some noise models


def get_kde(
    noise_values: Union[NDArray[np.float64], List[float]],
    h: float = 1.0,
    num_points: int = 1000,
    return_x_values: bool = False,
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
    """
    Estimate the probability density function (PDF) of noise values using
    Kernel Density Estimation (KDE).

    Args:
        noise_values (Union[NDArray[np.float_], List[float]]): List or array of noise values.
        h (float): Bandwidth smoothing parameter for KDE.
        num_points (int): Number of points to evaluate the KDE over the range of `noise_values`.
        return_x_values (bool): Whether to return the x values along with the PDF.

    Returns:
        Union[NDArray[np.float_], Tuple[NDArray[np.float_], NDArray[np.float_]]]:
            The estimated PDF or (PDF, x values).
    """
    # Define the range over which to evaluate the KDE
    x_values: NDArray[np.float64] = np.linspace(
        np.min(noise_values), np.max(noise_values), num_points
    )

    # Apply Gaussian KDE with the specified bandwidth
    kde: Any = gaussian_kde(noise_values, bw_method=h)

    # Evaluate KDE over x_values and return
    if not return_x_values:
        return kde(x_values)
    else:
        return kde(x_values), x_values


def get_rician(x_values: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """
    Calculate the Rician probability density function (PDF) for given values and sigma.

    Parameters:
    - x_values (np.ndarray): Values over which to calculate the Rician PDF.
    - sigma (float): Scale parameter of the Rician distribution.

    Returns:
    - np.ndarray: Calculated Rician PDF values.
    """
    rice_pdf: Any = rice.pdf(x_values, b=0, scale=sigma)
    return rice_pdf
