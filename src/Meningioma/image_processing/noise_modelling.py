from scipy.stats import gaussian_kde, rice  # type: ignore
import numpy as np
from typing import Tuple, List, Union
from numpy.typing import NDArray
from typing import Any


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
