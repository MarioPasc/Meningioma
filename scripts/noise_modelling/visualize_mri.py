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
    # Load the MRI image and ensure it is grayscale
    image: npt.NDArray[np.uint8] = np.array(Image.open(image_path).convert('L'))

    print(f"Image shape: {image.shape}")

    # Load the responsibilities matrix from the .mat file
    mat_data = sio.loadmat(matfile_path)
    responsibilities: npt.NDArray[np.float64] = mat_data.get('SaveResponsibilities', None)

    if responsibilities is None:
        raise ValueError("Responsibilities matrix not found in the .mat file.")

    print(f"Responsibilities shape: {responsibilities.shape}")

    # Ensure the responsibilities matrix matches the image shape
    if image.shape != responsibilities.shape:
        raise ValueError("Image and responsibilities matrix must have the same shape.")

    # Normalize the responsibilities for visualization
    responsibilities_visual: npt.NDArray[np.float64] = responsibilities / responsibilities.max()

    # Superpose responsibilities on the image (overlay)
    overlay: npt.NDArray[np.float64] = (
        0.5 * image.astype(np.float64) / 255 + 0.5 * responsibilities_visual
    )

    # Create a figure with constrained layout
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)

    # Original MRI image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('MRI Image')
    axes[0].axis('off')

    # Responsibilities matrix
    im = axes[1].imshow(responsibilities_visual, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Responsibilities Matrix')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(overlay, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Overlay: MRI + Responsibilities')
    axes[2].axis('off')

    # Add a horizontal colorbar across the bottom
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.05, pad=0.08)
    cbar.set_label('Responsibility Probability', fontsize=12)

    # Save the visualization
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
