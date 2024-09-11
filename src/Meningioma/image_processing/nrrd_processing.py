import nrrd
import numpy as np
from typing import Union, Tuple
import os

def open_nrrd(nrrd_path: str, return_header: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
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
    
    try:
        image, header = nrrd.read(nrrd_path)
        return (image, header) if return_header else image
    except Exception as e:
        print(f"    Error reading file {nrrd_path}. Error: {str(e)}")  

def transversal_axis(nrrd_path: str) -> int:
    """
    Finds the transversal axis of a nrrd file

    args:
        - nrrd_path (str): Path to the nrrd file.
    
    returns:
        - transversal_axis: Tranversal axis of the nrrd file.
    """

    _, header = open_nrrd(nrrd_path=nrrd_path, return_header=True)
    space_directions = np.array(header['space directions'])

    # Identify the axis corresponding to the transversal view (superior-inferior)
    # This is typically the axis with the largest value in the third column (superior-inferior)
    # We take the absolute values to avoid sign issues
    transversal_axis = np.argmax(np.abs(space_directions[:, 2]))

    return transversal_axis