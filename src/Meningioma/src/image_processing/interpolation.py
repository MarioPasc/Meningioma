from enum import Enum
import cv2
import numpy as np
import time
from typing import Union, Tuple

# Interpolation methods as enum
class InterpolationMethod(Enum):
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_LANCZOS4 = cv2.INTER_LANCZOS4

def resize(image: np.ndarray, size: Tuple[int, int], method: InterpolationMethod, measure_time: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Resizes an image to the specified size using the given interpolation method.
    
    Args:
        image (np.ndarray): The image to resize.
        size (Tuple[int, int]): The target size as (width, height).
        method (InterpolationMethod): The interpolation method to use.
        measure_time (bool): If True, returns the resized image and the time taken.
    
    Returns:
        Union[np.ndarray, Tuple[np.ndarray, float]]: 
        If measure_time is False, returns the resized image. 
        If True, returns a tuple (resized image, execution_time).
    """
    if measure_time:
        start_time = time.time()
        resized_image = cv2.resize(image, size, interpolation=method.value)
        end_time = time.time()
        execution_time = end_time - start_time
        return resized_image, execution_time
    else:
        return cv2.resize(image, size, interpolation=method.value)