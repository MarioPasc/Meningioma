from numpy.typing import NDArray
from typing import Tuple

from scipy import ndimage  # type: ignore
import numpy as np
import cv2

from skimage.filters import threshold_li, threshold_otsu, gaussian
from skimage import exposure, morphology, measure
from skimage.segmentation import active_contour
from skimage.draw import polygon


def get_global_umbralization(
    image: NDArray[np.float32], method: str = "li", threshold: float = 0.0
) -> Tuple[NDArray[np.bool_], float]:
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

    return intracranial_mask, threshold


def get_local_umbralization(
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

        if thresholding_method in ["otsu", "li", "weighted_mean", "manual"]:
            # Binarize using the thresholding function
            binarized_mask = get_global_umbralization(
                mean_filtered, method=thresholding_method
            )
        else:
            binarized_mask = slice_data > mean_filtered

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


def apply_active_contours(
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
    # Extract ROI from the image using the bounding box
    min_row, min_col, max_row, max_col = bounding_box
    roi = image[min_row:max_row, min_col:max_col]

    # Preprocess ROI: Apply Gaussian smoothing
    smoothed_roi = gaussian(roi, sigma=1)

    # Generate initial contour points (simple rectangle around the ROI)
    rows, cols = roi.shape
    s = np.linspace(0, 2 * np.pi, 400)  # 400 points around the perimeter
    r_init = (rows / 2) * (1 - 0.4 * np.cos(s))  # Centered ellipse shape
    c_init = (cols / 2) * (1 + 0.4 * np.sin(s))

    # Adjust the contour to fit the entire image ROI
    r_init = r_init + min_row
    c_init = c_init + min_col
    init_contour = np.array([r_init, c_init]).T

    # Apply Active Contours algorithm
    snake = active_contour(
        smoothed_roi,
        init_contour,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_num_iter=iterations,
    )

    # Create binary mask from the snake contour
    mask = np.zeros_like(image, dtype=bool)
    snake_rows = np.clip(np.round(snake[:, 0]).astype(int), 0, image.shape[0] - 1)
    snake_cols = np.clip(np.round(snake[:, 1]).astype(int), 0, image.shape[1] - 1)
    mask[snake_rows, snake_cols] = True

    rr, cc = polygon(snake_rows, snake_cols, shape=image.shape)
    mask[rr, cc] = True

    return mask


def get_convex_hull_mask(
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
    # Normalize image intensity
    img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply thresholding (using the global histogram method)
    mask, _ = get_global_umbralization(image=img_norm, method=threshold_method)
    thresh = (mask * 255).astype(np.uint8)

    # Find contours and compute convex hulls
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_list = [cv2.convexHull(contour) for contour in contours]

    # Create a filled hull mask
    filled_hull_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(filled_hull_mask, hull_list, -1, 255, thickness=cv2.FILLED)

    # Apply morphological closing and hole filling
    binary_mask = get_filled_mask(filled_hull_mask > 0, structure_size=7, iterations=3)

    return binary_mask * 255  # Convert to uint8 binary mask
