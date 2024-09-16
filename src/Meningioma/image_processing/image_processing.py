from Meningioma.image_processing.interpolation import resize, InterpolationMethod
from Meningioma.image_processing.nrrd_processing import open_nrrd, transversal_axis
import numpy as np
import cv2
from typing import Tuple, Optional, Union

class ImageProcessing:

    @staticmethod
    def open_nrrd_file(nrrd_path: str, return_header: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
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
    def get_transversal_axis(nrrd_path: str) -> int:
        """
        Finds the transversal axis of a nrrd file

        args:
            - nrrd_path (str): Path to the nrrd file.
        
        returns:
            - transversal_axis: Tranversal axis of the nrrd file.
        """
        return transversal_axis(nrrd_path=nrrd_path)

    @staticmethod
    def find_center_of_mass(image: np.ndarray, threshold: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
        """
        Find the center of mass of the bright region in the image, optionally applying a threshold.
        
        Args:
            image (np.ndarray): Input image (grayscale, float32, int16, etc.).
            threshold (Optional[Tuple[int, int]]): Threshold values (low, high) for segmentation.
        
        Returns:
            (int, int): Coordinates of the center of mass (x, y).
        """
        if threshold is not None:
            _, image = cv2.threshold(image, threshold[0], threshold[1], cv2.THRESH_BINARY)
        
        image = image.astype(np.float64)
        M = cv2.moments(image)
        
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            center_x, center_y = image.shape[1] // 2, image.shape[0] // 2
        
        return center_x, center_y

    @staticmethod
    def calculate_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int]) -> float:
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
        padded_image[start_y:start_y + new_h, start_x:start_x + new_w] = image
        
        return padded_image

    @staticmethod
    def apply_translation(image: np.ndarray, center_x: int, center_y: int, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Apply translation to center the region of interest in the image.
        
        Args:
            image (np.ndarray): Input image with padding.
            center_x (int): X-coordinate of the region's center.
            center_y (int): Y-coordinate of the region's center.
            target_size (tuple): Target resolution (width, height).
        
        Returns:
            np.ndarray: Translated image with the region centered.
        """
        target_w, target_h = target_size
        image_center_x, image_center_y = target_w // 2, target_h // 2
        offset_x, offset_y = image_center_x - center_x, image_center_y - center_y
        
        translation_matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        translated_image = cv2.warpAffine(image, translation_matrix, (target_w, target_h))
        
        return translated_image

    @staticmethod
    def resize_image_with_method(image: np.ndarray, target_size: Tuple[int, int], method: InterpolationMethod) -> np.ndarray:
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
