from enum import Enum
import cv2
import numpy as np
import time
from typing import Tuple
from numpy.typing import NDArray


# Interpolation methods as enum
class InterpolationMethod(Enum):
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_LANCZOS4 = cv2.INTER_LANCZOS4


def resize(
    image: NDArray[np.float64],
    size: Tuple[int, int],
    method: InterpolationMethod,
) -> NDArray[np.float64]:
    """
    Resizes an image to the specified size using the given interpolation method.

    Args:
        image (NDArray[np.float64]): The image to resize.
        size (Tuple[int, int]): The target size as (width, height).
        method (InterpolationMethod): The interpolation method to use.

    Returns:
        NDArray[np.float64]: The resized image.
    """
    # Resize the image and explicitly cast the result to NDArray[np.float64]
    resized_image: NDArray[np.float64] = np.asarray(
        cv2.resize(image, size, interpolation=method.value), dtype=np.float64
    )
    return resized_image
