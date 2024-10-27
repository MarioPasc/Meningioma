from image_processing.interpolation import resize, InterpolationMethod
from image_processing.nrrd_processing import open_nrrd, transversal_axis
import numpy as np
import cv2
from typing import Tuple, Optional, Union, Dict, List

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
    def extract_middle_transversal_slice(image_data: np.ndarray, transversal_axis: int) -> np.ndarray:
        """
        Extracts the middle slice along the transversal axis.
        """
        # Calculate the middle slice index along the transversal axis
        middle_slice_idx = image_data.shape[transversal_axis] // 2
        
        # Use np.take to extract the middle slice from the correct axis
        middle_slice = np.take(image_data, middle_slice_idx, axis=transversal_axis)
        
        return middle_slice

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
    
    @staticmethod
    def resize_with_padding_and_center(image: np.ndarray, 
                                       target_size: Tuple[int]=(512, 512), 
                                       method: InterpolationMethod = InterpolationMethod.INTER_LANCZOS4,
                                       return_centers_of_mass: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Tuple[int, int]]]]:
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
        scale = ImageProcessing.calculate_aspect_ratio(image=image, target_size=target_size)
        new_w = int(image.shape[1] * scale)
        new_h = int(image.shape[0] * scale)
        
        # Resize the image to the new dimensions
        resized_image = ImageProcessing.resize_image_with_method(image=image, target_size=(new_w, new_h), method=method)
        
        # Step 2: Pad the resized image to the target size
        padded_image = ImageProcessing.add_padding(resized_image, target_size)
        
        # Step 3: Find the center of mass of the resized image (brain region)
        cx_resized, cy_resized = ImageProcessing.find_center_of_mass(resized_image)

        # Step 4: Calculate the translation offsets to center the brain region
        target_center_x, target_center_y = target_size[0] // 2, target_size[1] // 2
        offset_x, offset_y = target_center_x - cx_resized, target_center_y - cy_resized

        # Apply translation to center the brain in the padded image
        centered_image = ImageProcessing.apply_translation(padded_image, target_center_x, target_center_y, target_size)
        
        if return_centers_of_mass:
            return (centered_image, {'Original_CM': (cx_resized, cy_resized), 'Translated_CM': (offset_x, offset_y)})
        else:
            return centered_image
    


def main() -> None:
    import matplotlib.pyplot as plt

    patient = 51
    pulse = 'T1'
    rm_nrrd_path=f'/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition/RM/{pulse}/P{patient}/{pulse}_P{patient}.nrrd'
    tc_nrrd_path=f'/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition/TC/P{patient}/TC_P{patient}.nrrd'

    processing = ImageProcessing()
    image = processing.open_nrrd_file(rm_nrrd_path)
    print(f'Image size: {image.shape}, max pixel value: {np.max(image)}')
    original_im = image[:, 100, :]

    fig, axes = plt.subplots(1, 2, figsize=(10,6))
    axes[0].imshow(original_im, cmap='gray')
    axes[1].imshow(processing.resize_with_padding_and_center(image=original_im, target_size=(512,512)), cmap='gray')
    plt.show()

if __name__ == "__main__":
    main()