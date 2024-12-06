from tkinter import Image
from typing_extensions import LiteralString
from Meningioma.image_processing import ImageProcessing  # type: ignore
from Meningioma.utils import Stats, npz_converter  # type: ignore
import os
import matplotlib.pyplot as plt
import scienceplots  # type: ignore

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


if __name__ == "__main__":
    base_path: str = (
        "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition"
    )
    output_npz_path: str = "/home/mariopasc/Python/Datasets/Meningiomas/npz"

    patient: str = "P15"
    pulse: str = "SUSC"
    w: int = 7
    z: int = 7
    slice_index: int = -1
    thresholding_method: str = "li"

    npz_converter.convert_to_npz(
        base_path=base_path,
        output_path=output_npz_path,
        patient=patient,
        pulse=pulse,
    )

    filepath: LiteralString = os.path.join(
        output_npz_path, patient, f"{patient}_{pulse}.npz"
    )

    # Load MRI slice
    slice_data = npz_converter.load_mri_slice(
        filepath=filepath, slice_index=slice_index
    )

    # Preprocess slice with advanced thresholding
    binarized_image, mean_filtered = ImageProcessing.preprocess_slice_with_thresholding(
        slice_data=slice_data, w=w, thresholding_method=thresholding_method
    )

    # Compute joint frequencies
    joint_frequencies = Stats.compute_joint_frequencies(
        binarized_image=binarized_image, z=z
    )

    # Compute mutual information
    mutual_info = Stats.compute_mutual_information(joint_frequencies=joint_frequencies)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Slice")
    plt.imshow(slice_data, cmap="gray")

    plt.subplot(1, 3, 2)
    plt.title("Binary Image")
    plt.imshow(binarized_image, cmap="gray")

    plt.subplot(1, 3, 3)
    plt.title("Mutual Information")
    plt.imshow(mutual_info, cmap="gray")
    plt.colorbar()

    plt.show()
