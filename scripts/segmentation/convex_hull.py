from skimage import morphology
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from Meningioma.image_processing import ImageProcessing
from Meningioma.utils import npz_converter


def process_whole_study(filepath: str, pulse: str):
    """
    Processes an entire MRI study, applies the convex hull pipeline on each slice,
    and visualizes 9 slices in a 3x3 grid.

    Args:
        filepath (str): Path to the .npz file containing the study data.
        pulse (str): Type of pulse sequence (for labeling).

    Returns:
        None
    """
    # Load all slices from the file
    data = np.load(filepath)["data"]  # Assuming the key is 'data'
    total_slices = data.shape[-1]  # Total number of slices

    # Helper to get indices with at least 2-slice separation
    def get_indices(start, end, exclude, num_slices=3):
        step = max((end - start) // (num_slices * 2), 2)
        indices = [i for i in range(start, end, step) if i not in exclude]
        return indices[:num_slices]

    # Define fixed positions: first, middle, and last slices
    first_idx = 0
    middle_idx = total_slices // 2
    last_idx = total_slices - 1

    # Collect indices, ensuring we don’t repeat fixed positions
    exclude = {first_idx, middle_idx, last_idx}
    start_indices = get_indices(0, middle_idx // 2, exclude)
    middle_indices = get_indices(middle_idx - 5, middle_idx + 5, exclude)
    end_indices = get_indices(last_idx - (middle_idx // 2), total_slices, exclude)

    # Final 9 indices (ensure positions (0,0), (1,1), (2,2))
    selected_indices = [
        first_idx,  # Top-left corner
        start_indices[0],  # Top-center
        start_indices[1],  # Top-right
        middle_indices[0],  # Middle-left
        middle_idx,  # Middle-center
        middle_indices[1],  # Middle-right
        end_indices[0],  # Bottom-left
        end_indices[1],  # Bottom-center
        last_idx,  # Bottom-right
    ]

    # Prepare 3x3 plot
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()  # Flatten for easy indexing

    # Iterate over selected slices
    for i, slice_idx in enumerate(selected_indices):
        slice_data = np.squeeze(data[0, :, :, slice_idx])

        # Normalize image
        img = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Thresholding
        mask, _ = ImageProcessing.global_histogram_segmentation(image=img, method="li")
        thresh = (mask * 255).astype(np.uint8)

        # Find contours and convex hull
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        hull_list = [cv2.convexHull(contour) for contour in contours]

        # Create a filled hull mask
        filled_hull_mask = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(filled_hull_mask, hull_list, -1, 255, thickness=cv2.FILLED)

        # Apply hole filling and smoothing
        filled_hull_mask = ImageProcessing.fill_mask(
            filled_hull_mask > 0, structure_size=5, iterations=2
        )

        # Overlay mask onto the original image
        overlay = np.dstack((img, img, img))  # Convert grayscale to RGB
        mask_rgb = np.zeros_like(overlay)
        mask_rgb[:, :, 1] = filled_hull_mask * 255  # Green channel for the mask

        # Combine with transparency (alpha=0.5)
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1, mask_rgb, alpha, 0)

        # Plot
        ax = axes[i]
        ax.imshow(overlay)
        ax.set_title(f"Slice {slice_idx} - {pulse}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def process_and_save_study(filepath: str, output_path: str, patient: str, pulse: str):
    """
    Processes all slices in an MRI study, applies the convex hull pipeline,
    and saves each processed slice as an image.

    Args:
        filepath (str): Path to the .npz file containing the study data.
        output_path (str): Base path where processed images will be saved.
        patient (str): Patient identifier.
        pulse (str): Pulse sequence type.

    Returns:
        None
    """
    # Load all slices from the file
    data = np.load(filepath)["data"]  # Assuming the key is 'data'
    total_slices = data.shape[-1]  # Total number of slices

    # Create output directory
    save_dir = os.path.join(output_path, patient, pulse)
    os.makedirs(save_dir, exist_ok=True)

    # Process each slice
    for slice_idx in range(total_slices):
        slice_data = np.squeeze(data[0, :, :, slice_idx])

        # Normalize image
        img = cv2.normalize(slice_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Thresholding
        mask, _ = ImageProcessing.global_histogram_segmentation(image=img, method="li")
        thresh = (mask * 255).astype(np.uint8)

        # Find contours and convex hull
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        hull_list = [cv2.convexHull(contour) for contour in contours]

        # Create a filled hull mask
        filled_hull_mask = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(filled_hull_mask, hull_list, -1, 255, thickness=cv2.FILLED)

        # Apply hole filling and smoothing
        filled_hull_mask = ImageProcessing.fill_mask(
            filled_hull_mask > 0, structure_size=5, iterations=2
        )

        # Overlay mask onto the original image
        overlay = np.dstack((img, img, img))  # Convert grayscale to RGB
        mask_rgb = np.zeros_like(overlay)
        mask_rgb[:, :, 1] = filled_hull_mask * 255  # Green channel for the mask

        # Combine with transparency (alpha=0.5)
        alpha = 0.5
        overlay = cv2.addWeighted(overlay, 1, mask_rgb, alpha, 0)

        # Save image
        save_path = os.path.join(
            save_dir, f"{patient}_{pulse}_{slice_idx}_convexhull.png"
        )
        plt.imsave(save_path, overlay)
        print(f"Saved: {save_path}")


# Example Input Paths
base_path = "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition"
output_npz_path = "/home/mariopasc/Python/Datasets/Meningiomas/npz"
output_path = "/home/mariopasc/Python/Datasets/Meningiomas/ConvexHull"  # Directory to store images
patient = "P9"

pulses = ["T1SIN", "T2", "SUSC", "T1"]

# pulses = ["T1"]

os.makedirs(output_path, exist_ok=True)


# Process the whole study
# process_whole_study(filepath=filepath, pulse=pulse)

for pulse in pulses:
    try:
        npz_converter.convert_to_npz(
            base_path=base_path,
            output_path=output_npz_path,
            patient=patient,
            pulse=pulse,
        )
        filepath = os.path.join(output_npz_path, patient, f"{patient}_{pulse}.npz")
        process_and_save_study(
            filepath=filepath, output_path=output_path, patient=patient, pulse=pulse
        )
    except Exception as e:
        print(f"Error: {e}")
