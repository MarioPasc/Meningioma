import argparse
import matplotlib.pyplot as plt
import scipy.io as sio  # type: ignore
import numpy as np
from PIL import Image
from typing import Tuple
import numpy.typing as npt


def main(image_path: str, matfile_path: str, save_path: str) -> None:
    """
    Visualize MRI image, responsibilities matrix, and their superposition.

    Parameters:
        image_path (str): Path to the input MRI image.
        matfile_path (str): Path to the .mat file containing the responsibilities matrix.
        save_path (str): Path to save the visualization.
    """
    # Load the MRI image
    image: npt.NDArray[np.uint8] = np.array(Image.open(image_path))

    # Load the responsibilities matrix from the .mat file
    mat_data = sio.loadmat(matfile_path)
    responsibilities: npt.NDArray[np.float64] = mat_data.get(
        "SaveResponsibilities", None
    )

    if responsibilities is None:
        raise ValueError("Responsibilities matrix not found in the .mat file.")

    # Ensure the responsibilities matrix matches the image shape
    if image.shape != responsibilities.shape:
        raise ValueError("Image and responsibilities matrix must have the same shape.")

    # Normalize the responsibilities for visualization
    responsibilities_visual: npt.NDArray[np.float64] = (
        responsibilities / responsibilities.max()
    )

    # Superpose responsibilities on the image (overlay)
    overlay: npt.NDArray[np.float64] = (
        0.5 * image.astype(np.float64) / 255 + 0.5 * responsibilities_visual
    )

    # Plot the MRI image, responsibilities matrix, and overlay
    plt.figure(figsize=(15, 5))

    # Original MRI image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("MRI Image")
    plt.axis("off")

    # Responsibilities matrix
    plt.subplot(1, 3, 2)
    plt.imshow(responsibilities_visual, cmap="hot")
    plt.title("Responsibilities Matrix")
    plt.axis("off")

    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(overlay, cmap="hot")
    plt.title("Overlay: MRI + Responsibilities")
    plt.axis("off")

    # Save the visualization
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved to {save_path}")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Visualize MRI and responsibilities matrix."
    )
    parser.add_argument("--image", required=True, help="Path to the MRI image.")
    parser.add_argument(
        "--matfile", required=True, help="Path to the .mat responsibilities file."
    )
    parser.add_argument(
        "--output", required=True, help="Path to save the visualization."
    )
    args = parser.parse_args()

    # Execute the main function
    main(args.image, args.matfile, args.output)
