import interpolation
import numpy as np
import cv2
import nrrd
import os
from typing import Union, Tuple

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


    def dimensionality_standarization_experiment(self, dataset_folder:str, save_path:str) -> None:
        """
        Perform a dimensionality standarization experiment. This experiment aims to achieve the most suitable 
        interpolation technique for each dataset explored in this study: T1, T1SIN, T1, SUSC and CT.  

        params:
            - dataset_folder (str): Root folder containing all the datasets
            - save_path (str): Path to the folder which will contain all the output data
        """

        adquisition_types = ['TC', 'RM/T1', 'RM/T2', 'RM/T1SIN', 'RM/SUSC']

        # Explore the datasets
        for format in adquisition_types: 
            dataset_path = os.path.join(dataset_folder, format)
            for patient in os.listdir(dataset_folder):
                patient_folder = os.path.join(dataset_path, patient)

                # All nrrd images and segmentations follow the same naming format, 
                # e.g: SUSC_P1.nrrd for image ; SUSC_P1_seg.nrrd for its corresponding segmentation mask
                image_file = f'{format.strip('RM/')}_{patient}.nrrd'
                segmentation_file = f'{format.strip('RM/')}_{patient}_seg.nrrd'
