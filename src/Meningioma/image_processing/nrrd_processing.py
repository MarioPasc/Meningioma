import nrrd  # type: ignore
import numpy as np
from typing import Union, Tuple, Optional
import os
from numpy.typing import NDArray


def open_nrrd(
    nrrd_path: str, return_header: bool = False
) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], dict]]:
    """
    Open the given NRRD file path.

    Args:
        nrrd_path (str): Path to the NRRD file to open.
        return_header (bool): If True, return the header along with the image. Defaults to False.

    Returns:
        Union[NDArray[np.float64], Tuple[NDArray[np.float64], dict]]:
            If return_header is True, returns (image, header).
            Otherwise, returns image only.
    """
    if not os.path.isfile(nrrd_path):
        raise FileNotFoundError(f"{nrrd_path} is not a valid file path.")

    try:
        image: NDArray[np.float64]
        header: dict
        image, header = nrrd.read(nrrd_path)
        return (image, header) if return_header else image
    except Exception as e:
        raise FileNotFoundError(f"Error reading file {nrrd_path}. Error: {str(e)}")


def transversal_axis(nrrd_path: str) -> np.intp:
    """
    Finds the transversal axis from the NRRD file by selecting the axis where
    the z-component dominates over the other components (x, y).

    Args:
        nrrd_path (str): Path to the NRRD file.

    Returns:
        Optional[int]: The index of the transversal axis (0, 1, or 2), or None if no valid axis is found.
    """
    try:
        # Load the NRRD file and read the header
        header: dict
        _, header = nrrd.read(nrrd_path)

        # Extract space directions from the header
        space_directions: NDArray[np.float64] = np.array(header.get("space directions"))

        # Extract the z-components from the space directions (third element of each vector)
        z_components: NDArray[np.float64] = space_directions[:, 2]

        # Normalize each direction vector to compare relative dominance
        normalized_directions: NDArray[np.float64] = np.linalg.norm(
            space_directions, axis=1
        )

        # Calculate the dominance ratios for each axis
        dominance_ratios: NDArray[np.float64] = np.array(
            [abs(z) / norm for z, norm in zip(z_components, normalized_directions)]
        )

        # Find the axis with the highest dominance ratio
        transversal_axis: np.intp = np.argmax(dominance_ratios)

        return transversal_axis

    except Exception as e:
        raise Exception(f"Error processing {nrrd_path}: {e}")
