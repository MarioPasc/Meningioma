import Meningioma.image_processing.interpolation
import numpy as np
import cv2
import nrrd
import os
from typing import Union, Tuple, Dict

class ImageProcessing:
    """
    A class to perform various image processing tasks needed throughout the project
    """

    def _open_nrrd(self, nrrd_path: str, return_header: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
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
        if not os.path.isfile(nrrd_path):
            raise FileNotFoundError(f"{nrrd_path} is not a valid file path.")
        
        image, header = nrrd.read(nrrd_path)
        return (np.ndarray(image), header) if return_header else np.ndarray(image)


