import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def load_npz(npz_path: str) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Loads and extracts data from a .npz file.

    Args:
        npz_path (str): Path to the .npz file.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]: Image and mask arrays.
    """
    data: NDArray[np.float64] = np.load(npz_path)["data"]
    image: NDArray[np.float64] = data[0]
    mask: NDArray[np.float64] = data[1]
    return image, mask


def visualize_npz(npz_path: str, slice_idx: Optional[int] = None) -> None:
    """
    Visualizes the image and mask from a .npz file.

    Args:
        npz_path (str): Path to the .npz file.
        slice_idx (Optional[int]): Index of the slice to display. If None, the middle slice is shown.

    Returns:
        None
    """
    image: NDArray[np.float64]
    mask: NDArray[np.float64]
    image, mask = load_npz(npz_path)
    slice_idx = slice_idx if slice_idx is not None else image.shape[-1] // 2

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image[:, :, slice_idx], cmap="gray")
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(mask[:, :, slice_idx], cmap="gray")
    axs[1].set_title("Mask")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    from typing import List
    from typing_extensions import LiteralString
    import os

    pulses: List[str] = ["T1", "T1SIN", "T2", "SUSC"]
    patient: str = "P15"
    base: LiteralString = "/home/mariopasc/Python/Datasets/Meningiomas/outputNPZ/"

    visualize_npz(os.path.join(base, patient, f"{patient}_{pulses[3]}.npz"))


if __name__ == "__main__":
    main()
