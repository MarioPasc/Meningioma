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
        binarized_image = binarized_mask.astype(np.uint8)  # type:ignore

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

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros_like(
            image, dtype=np.uint8
        )  # Return empty mask if no contours found

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute the convex hull of the largest contour
    largest_hull = cv2.convexHull(largest_contour)

    # Create a filled hull mask
    hull_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.drawContours(
        hull_mask, [largest_hull], -1, 255, thickness=cv2.FILLED
    )  # type:ignore

    # Apply morphological closing and hole filling
    binary_mask = get_filled_mask(hull_mask > 0, structure_size=7, iterations=3)

    return binary_mask.astype(np.uint8) * 255  # type: ignore


def _threshold_slice(slice_img: np.ndarray, method: str = "otsu") -> np.ndarray:
    """
    Threshold a single 2D slice (uint8) using the specified method.
    Supports both 'otsu' and 'li' thresholding via scikit-image.

    Parameters
    ----------
    slice_img : np.ndarray
        2D image array in uint8 format (0..255).
    method : str
        Threshold method, e.g. "otsu" or "li".

    Returns
    -------
    np.ndarray
        A binary mask (uint8) of the same shape as slice_img, where
        foreground pixels are 255, and background pixels are 0.
    """
    if method.lower() == "otsu":
        # Otsu threshold using OpenCV
        _, thr = cv2.threshold(slice_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thr

    elif method.lower() == "li":
        # Li threshold from scikit-image
        # Convert slice to float before applying threshold_li
        slice_float = slice_img.astype(np.float32)
        li_value = threshold_li(slice_float)
        # Create a binary mask: pixels >= li_value => 255, else 0
        thr = (slice_float >= li_value).astype(np.uint8) * 255
        return thr

    else:
        # Fallback to Otsu if an unknown method is provided
        _, thr = cv2.threshold(slice_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thr


def get_3d_volume_segmentation(
    filepath: str,
    threshold_method: str = "otsu",
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
    # 1) Load volume
    npz_data = np.load(filepath)
    volume = np.squeeze(
        npz_data["data"][0]
    )  # shape: (H, W, S) or (S, H, W) depending on dataset
    volume = volume.astype(np.float32, copy=False)

    # Check we have 3D data
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}.")

    # 2) Global normalization
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin < 1e-6:
        # Degenerate volume, create a trivial mask
        mask = (volume > 0).astype(bool)
        return volume, mask

    vol_norm = ((volume - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    H, W, S = volume.shape

    # Arrays to store intermediate results for each slice
    hull_3d = np.zeros((H, W, S), dtype=np.uint8)

    # 3) Per-slice threshold + largest contour + hull + 2D closing
    kernel_2d = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (structure_size_2d, structure_size_2d)
    )

    for i in range(S):
        slice_i = vol_norm[..., i]
        thr_i = _threshold_slice(slice_i, threshold_method)

        contours, _ = cv2.findContours(
            thr_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        hull_pts = cv2.convexHull(largest_contour)

        hull_mask_slice = np.zeros_like(slice_i, dtype=np.uint8)
        cv2.drawContours(
            hull_mask_slice, [hull_pts], -1, color=255, thickness=cv2.FILLED
        )  # type: ignore

        # 2D morphological closing
        hull_closed = cv2.morphologyEx(
            hull_mask_slice, cv2.MORPH_CLOSE, kernel_2d, iterations=iterations_2d
        )
        hull_3d[..., i] = hull_closed

    # Convert to boolean
    mask_3d = hull_3d > 0

    # 4) Optional 3D morphological closing
    if structure_size_3d > 1:
        struct_3d = ndimage.generate_binary_structure(3, 1)
        struct_3d = ndimage.iterate_structure(struct_3d, structure_size_3d)
        mask_3d = ndimage.binary_closing(
            mask_3d, structure=struct_3d, iterations=iterations_3d
        )

    # 5) Keep only the largest connected 3D component
    labeled, n_labels = ndimage.label(mask_3d)
    if n_labels > 1:
        label_sizes = ndimage.sum(mask_3d, labeled, range(1, n_labels + 1))
        largest_label = np.argmax(label_sizes) + 1
        mask_3d = labeled == largest_label

    return volume, mask_3d.astype(bool)
