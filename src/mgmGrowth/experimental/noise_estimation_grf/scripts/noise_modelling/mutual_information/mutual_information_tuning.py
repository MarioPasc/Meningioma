import numpy as np
from numpy.typing import NDArray
from typing import List
from typing_extensions import LiteralString
from mgmGrowth.image_processing import ImageProcessing  # type: ignore
from mgmGrowth.utils import Stats, nrrd2npz  # type: ignore
import os
import matplotlib.pyplot as plt
import scienceplots  # type: ignore

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def generate_visualization(
    slice_data: NDArray[np.float64],
    binarized_image: NDArray[np.uint8],
    mutual_info: NDArray[np.float64],
    save_path: str,
    show: bool = False,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].set_title("Original Slice")
    axes[0].imshow(slice_data, cmap="gray")

    axes[1].set_title("Binary Image")
    axes[1].imshow(binarized_image, cmap="gray")

    axes[2].set_title("Mutual Information")
    mi = axes[2].imshow(mutual_info, cmap="gray")

    cbar = fig.colorbar(mi, ax=axes, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label("Mutual Information (MI) value", fontsize=12)

    for ax in axes:
        ax.axis("off")

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()


def run_pipeline_for_w_z(
    filepath: str,
    slice_index: int,
    w: int,
    z: int,
    thresholding_method: str = "mean_filter",
) -> NDArray[np.float64]:
    """
    Runs the full pipeline to compute mutual information for a specific w and z.

    Args:
        filepath (str): Path to the .npz file.
        slice_index (int): Slice index to extract.
        w (int): Mean filter size for preprocessing.
        z (int): Neighborhood size for joint frequency computation.
        thresholding_method (str): Thresholding method.

    Returns:
        NDArray[np.float64]: Mutual information matrix.
    """
    try:
        # Load MRI slice
        slice_data = nrrd2npz.load_mri_slice(
            filepath=filepath, slice_index=slice_index
        )

        # Preprocess slice with advanced thresholding
        binarized_image, _ = ImageProcessing.local_histogram_segmentation(
            slice_data=slice_data, w=w, thresholding_method=thresholding_method
        )

        # Compute joint frequencies
        joint_frequencies = Stats.compute_joint_frequencies(
            binarized_image=binarized_image, z=z
        )

        # Compute mutual information
        mutual_info = Stats.compute_mutual_information(
            joint_frequencies=joint_frequencies
        )

        return mutual_info
    except Exception as e:
        print(f"Error in pipeline for w={w}, z={z}: {e}")
        return np.zeros_like(slice_data)  # type: ignore


def visualize_w_z_effects(
    filepath: str,
    slice_index: int,
    w_values: List[int],
    z_values: List[int],
    save_path: str,
    thresholding_method: str = "mean_filter",
    show: bool = False,
) -> None:
    """
    Visualizes the mutual information for all combinations of w and z values.

    Args:
        filepath (str): Path to the .npz file.
        slice_index (int): Slice index to extract.
        w_values (List[int]): List of w values to try.
        z_values (List[int]): List of z values to try.
        thresholding_method (str): Thresholding method.
    """
    num_rows = len(w_values)
    num_cols = len(z_values)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(3 * num_cols, 3 * num_rows))
    fig.suptitle(
        "Effect of Tuning w (rows) and z (columns) on Mutual Information", fontsize=14
    )

    for i, w in enumerate(w_values):
        for j, z in enumerate(z_values):
            print(f"Processing w={w}, z={z}...")

            # Run pipeline and get mutual information
            mutual_info = run_pipeline_for_w_z(
                filepath, slice_index, w, z, thresholding_method
            )

            # Plot mutual information
            ax = axes[i, j] if num_rows > 1 and num_cols > 1 else axes[max(i, j)]
            ax.imshow(mutual_info, cmap="gray")
            ax.set_title(f"w={w}, z={z}")
            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust for the suptitle

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()


if __name__ == "__main__":
    # Paths and parameters
    base_path: str = (
        "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition"
    )
    output_npz_path: str = "/home/mariopasc/Python/Datasets/Meningiomas/npz"
    save_path: str = "./scripts/noise_modelling/mutual_information/tune_wz.pdf"

    patient: str = "P15"
    pulse: str = "T1"
    slice_index: int = 30  # Middle slice
    thresholding_method: str = "mean_filter"

    # Convert data to NPZ format
    nrrd2npz.convert_to_npz(
        base_path=base_path, output_path=output_npz_path, patient=patient, pulse=pulse
    )

    # Define the file path
    filepath = os.path.join(output_npz_path, patient, f"{patient}_{pulse}.npz")

    # Define values for w and z to test
    w_values = [3, 5, 7, 9]  # Mean filter sizes
    z_values = [3, 5, 7, 9]  # Neighborhood sizes for joint frequencies

    # Visualize the effect of tuning w and z
    visualize_w_z_effects(
        filepath=filepath,
        slice_index=slice_index,
        w_values=w_values,
        z_values=z_values,
        save_path=save_path,
        thresholding_method=thresholding_method,
        show=False,
    )
