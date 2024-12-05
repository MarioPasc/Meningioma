import os
import nrrd  # type: ignore
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple
from Meningioma.image_processing.nrrd_processing import transversal_axis

# Global variable for pulse types
PULSE_TYPES: List[str] = ["SUSC", "T1", "T2", "T1SIN"]


def reorder_to_transversal(
    image_data: NDArray[np.float64], mask_data: NDArray[np.float64], nrrd_path: str
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Reorders the image and mask data to ensure the transversal axis is in the last position.

    Args:
        image_data (NDArray[np.float64]): The image data array.
        mask_data (NDArray[np.float64]): The mask data array.
        nrrd_path (str): Path to the NRRD file for determining the transversal axis.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]: Reordered image and mask data arrays.
    """
    axis: np.intp = transversal_axis(nrrd_path)
    if axis != image_data.ndim - 1:
        image_data = np.moveaxis(image_data, axis, -1)
        mask_data = np.moveaxis(mask_data, axis, -1)
    return image_data, mask_data


def convert_to_npz(base_path: str, output_path: str, pulse: str, patient: str) -> None:
    """
    Converts a patient-pulse pair from NRRD files to NPZ format.

    Args:
        base_path (str): Base path to the dataset directory.
        output_path (str): Path to save the NPZ files.
        pulse (str): Pulse type of the patient.
        patient (str): Patient identifier.

    Returns:
        None
    """
    patient_path: str = os.path.join(base_path, "RM", pulse, patient)
    img_file: str = os.path.join(patient_path, f"{pulse}_{patient}.nrrd")
    mask_file: str = os.path.join(patient_path, f"{pulse}_{patient}_seg.nrrd")

    if not os.path.exists(img_file) or not os.path.exists(mask_file):
        print(f"Skipping: Missing files for {patient} with pulse {pulse}")
        return

    try:
        img_data: NDArray[np.float64]
        mask_data: NDArray[np.float64]
        img_data, _ = nrrd.read(img_file)
        mask_data, _ = nrrd.read(mask_file)
        img_data, mask_data = reorder_to_transversal(img_data, mask_data, img_file)
        stacked_data: NDArray[np.float64] = np.stack([img_data, mask_data], axis=0)

        patient_output_dir: str = os.path.join(output_path, patient)
        os.makedirs(patient_output_dir, exist_ok=True)
        output_file: str = os.path.join(patient_output_dir, f"{patient}_{pulse}.npz")
        np.savez_compressed(output_file, data=stacked_data)
        print(f"Converted: {output_file}")
    except Exception as e:
        print(f"Error processing {patient} with pulse {pulse}: {e}")


# Example usage
if __name__ == "__main__":
    patient: str = "P4"

    convert_to_npz(
        base_path="/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition",
        output_path="/home/mariopasc/Python/Datasets/Meningiomas/outputNPZ",
        patient=patient,
        pulse="T1",
    )

    convert_to_npz(
        base_path="/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition",
        output_path="/home/mariopasc/Python/Datasets/Meningiomas/outputNPZ",
        patient=patient,
        pulse="T1SIN",
    )

    convert_to_npz(
        base_path="/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition",
        output_path="/home/mariopasc/Python/Datasets/Meningiomas/outputNPZ",
        patient=patient,
        pulse="T2",
    )

    convert_to_npz(
        base_path="/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition",
        output_path="/home/mariopasc/Python/Datasets/Meningiomas/outputNPZ",
        patient=patient,
        pulse="SUSC",
    )
