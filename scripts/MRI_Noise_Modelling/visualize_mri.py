import argparse
import matplotlib.pyplot as plt
import scipy.io as sio  # type: ignore
import numpy as np
from PIL import Image


def main(image_path, matfile_path):
    # Load the MRI image
    image = np.array(Image.open(image_path))

    # Load the responsibilities matrix
    mat_data = sio.loadmat(matfile_path)
    responsibilities = mat_data.get("SaveResponsibilities", None)

    if responsibilities is None:
        raise ValueError("Responsibilities matrix not found in the .mat file.")

    # Plot the image and responsibilities
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("MRI Image")
    plt.axis("off")

    # Responsibilities matrix
    plt.subplot(1, 2, 2)
    plt.imshow(responsibilities, cmap="hot")
    plt.title("Responsibilities Matrix")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize MRI and responsibilities.")
    parser.add_argument("--image", required=True, help="Path to the MRI image.")
    parser.add_argument(
        "--matfile", required=True, help="Path to the .mat responsibilities file."
    )
    args = parser.parse_args()
    main(args.image, args.matfile)
