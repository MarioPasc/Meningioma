# We tried adding an active contour algorithm to segment the skull and intracraneal section, but the results were
# unstable and worse than the largest bbox approach to the problem, and the trade-off between the tuning of the
# algorithm for each image and the results was not worth it

import os

import numpy as np
from Meningioma.image_processing import ImageProcessing  # type: ignore
from Meningioma.utils import nrrd_to_npz  # type: ignore
from skimage.draw import rectangle
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt
import scienceplots  # type: ignore

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def visualize_pipeline_steps(
    image: np.ndarray,
    intracranial_mask: np.ndarray,
    filled_mask: np.ndarray,
    largest_bbox: tuple,
    refined_mask: np.ndarray,
    save_path: str = "./scripts/segmentation/segmentation.pdf",
):
    """
    Visualizes the steps in the segmentation pipeline:
    1. Global histogram segmentation mask
    2. Mask after filling
    3. Bounding box around brain/skull
    4. Active contours refined mask
    5. Combined visualization with original image, segmentation mask, and refined mask

    Args:
        image (np.ndarray): Original MRI slice.
        intracranial_mask (np.ndarray): Mask from global segmentation.
        filled_mask (np.ndarray): Mask after hole-filling.
        largest_bbox (tuple): Coordinates of the largest bounding box.
        refined_mask (np.ndarray): Mask from active contours.
        save_path (str, optional): Path to save the final figure. Defaults to None.
    """
    # Extract bounding box coordinates
    min_row, min_col, max_row, max_col = largest_bbox

    # Step 1: Original image with global segmentation mask
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()  # Flatten axes for easy indexing

    # Plot 1: Original image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original MRI Slice")
    axes[0].axis("off")

    # Plot 2: Global segmentation mask
    axes[1].imshow(intracranial_mask, cmap="gray")
    axes[1].set_title("Global Segmentation Mask")
    axes[1].axis("off")

    # Plot 3: Filled mask after morphological operations
    axes[2].imshow(filled_mask, cmap="gray")
    axes[2].set_title("Filled Mask")
    axes[2].axis("off")

    # Plot 4: Bounding box on the original image
    bbox_image = np.copy(image)
    rr, cc = rectangle(
        start=(min_row, min_col), end=(max_row, max_col), shape=image.shape
    )
    bbox_image[rr, cc] = np.max(image)  # Draw white rectangle
    axes[3].imshow(bbox_image, cmap="gray")
    axes[3].set_title("Bounding Box")
    axes[3].axis("off")

    # Plot 5: Active Contours Mask
    axes[4].imshow(refined_mask, cmap="gray")
    axes[4].set_title("Active Contours Mask")
    axes[4].axis("off")

    # Plot 6: Combined visualization with masks overlaid
    axes[5].imshow(image, cmap="gray")
    axes[5].imshow(intracranial_mask, cmap="Reds", alpha=0.5)  # Red overlay
    axes[5].imshow(refined_mask, cmap="Greens", alpha=0.5)  # Green overlay
    axes[5].set_title("Combined: Global Mask (Red) and Refined (Green)")
    axes[5].axis("off")

    # Adjust layout
    plt.tight_layout()

    # Save figure if save_path is specified
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")


def compare_segmentation_methods(
    image: np.ndarray,
    filled_mask: np.ndarray,
    bounding_box: tuple,
    refined_mask: np.ndarray,
    save_path: str = "./scripts/segmentation/segmentation.pdf",
):
    """
    Visualizes segmentation steps in a single row:
    1. Filled mask (blue) over original image (background intact).
    2. Bounding box (red) over original image (background intact).
    3. Active Contours mask (green) over original image (background intact).
    4. Combined visualization on black background:
        - Bounding box (red),
        - Active Contours mask (green, alpha=0.7),
        - Filled mask (blue, alpha=0.7).

    Args:
        image (np.ndarray): Original MRI image.
        filled_mask (np.ndarray): Binary mask after filling operations.
        bounding_box (tuple): Bounding box (min_row, min_col, max_row, max_col).
        refined_mask (np.ndarray): Binary mask from Active Contours.
        save_path (str, optional): Path to save the figure. Defaults to None.
    """
    # Extract bounding box coordinates
    min_row, min_col, max_row, max_col = bounding_box
    rr, cc = rectangle(
        start=(min_row, min_col), end=(max_row, max_col), shape=image.shape
    )

    # Create a blank black background for the combined plot
    combined_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)

    # Setup the figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.ravel()

    # Function to overlay a mask on the original image
    def overlay_mask(image, mask, color, alpha=0.5):
        mask = mask > 0
        overlay = np.zeros((*image.shape, 3), dtype=np.float32)
        overlay[..., 0] = image / image.max()  # Normalize original image
        overlay[..., 1] = image / image.max()
        overlay[..., 2] = image / image.max()
        overlay[mask] = color  # Apply mask color
        return (1 - alpha) * overlay + alpha * np.dstack(
            (image, image, image)
        ) / image.max()

    # Plot 1: Filled mask (blue) over original image
    filled_overlay = overlay_mask(image, filled_mask, color=(0, 0, 1))
    axes[0].imshow(filled_overlay)
    axes[0].set_title("Filled Mask (Blue)")
    axes[0].axis("off")

    # Plot 2: Bounding box (red edges) over original image
    min_row, min_col, max_row, max_col = bounding_box
    bbox_height = max_row - min_row
    bbox_width = max_col - min_col

    axes[1].imshow(image, cmap="gray")
    rect = Rectangle(
        (min_col, min_row),
        bbox_width,
        bbox_height,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axes[1].add_patch(rect)  # Add the rectangle as edges only
    axes[1].set_title("Bounding Box (Red Edges)")
    axes[1].axis("off")

    # Plot 3: Active Contours mask (green) over original image
    contours_overlay = overlay_mask(image, refined_mask, color=(0, 1, 0))
    axes[2].imshow(contours_overlay)
    axes[2].set_title("Active Contours Mask (Green)")
    axes[2].axis("off")

    # Plot 4: Combined visualization on black background
    combined_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)

    # Step 4.1: Draw bounding box in red (edges only)
    axes[3].imshow(combined_image, cmap="gray")  # Black background
    rect_combined = Rectangle(
        (min_col, min_row),
        bbox_width,
        bbox_height,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axes[3].add_patch(rect_combined)

    # Step 4.2: Overlay Active Contours mask (green, alpha=0.6)
    green_overlay = np.zeros_like(combined_image)
    green_overlay[refined_mask, 1] = 1.0  # Green channel
    axes[3].imshow(green_overlay, alpha=0.7)

    # Step 4.3: Overlay Filled mask (blue, alpha=0.7)
    blue_overlay = np.zeros_like(combined_image)
    blue_overlay[filled_mask, 2] = 1.0  # Blue channel
    axes[3].imshow(blue_overlay, alpha=0.3)

    axes[3].set_title("Combined: Red=BBox, Green=Contours, Blue=Filled")
    axes[3].axis("off")

    # Adjust layout and save if needed
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Figure saved to {save_path}")


############################################
#          EXPERIMENT FUNCTIONS            #
############################################


def apply_active_contours_on_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.015,
    beta: float = 10,
    gamma: float = 0.001,
    iterations: int = 250,
) -> np.ndarray:
    """
    Applies Active Contours using the boundary of the filled mask as the initial contour.

    Args:
        image (np.ndarray): Original MRI image.
        mask (np.ndarray): Binary mask for initialization.
        alpha (float): Snake smoothness parameter.
        beta (float): Snake elasticity parameter.
        gamma (float): Time step for the snake movement.
        iterations (int): Number of iterations.

    Returns:
        np.ndarray: Refined binary mask.
    """

    from skimage.segmentation import active_contour
    from skimage import measure
    from skimage.filters import gaussian

    # Preprocess the image (Gaussian smoothing)
    smoothed_image = gaussian(image, sigma=1)

    # Find the initial contour from the mask boundary
    contours = measure.find_contours(mask, 0.5)
    if len(contours) == 0:
        raise ValueError("No contour found in the mask.")

    # Use the largest contour
    initial_contour = max(contours, key=len)

    # Apply Active Contours algorithm
    snake = active_contour(
        smoothed_image,
        initial_contour,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        max_num_iter=iterations,
    )

    # Create a binary mask from the snake contour
    refined_mask = np.zeros_like(mask, dtype=bool)
    snake_rows = np.clip(np.round(snake[:, 0]).astype(int), 0, mask.shape[0] - 1)
    snake_cols = np.clip(np.round(snake[:, 1]).astype(int), 0, mask.shape[1] - 1)
    refined_mask[snake_rows, snake_cols] = True

    # Fill inside the contour
    from skimage.draw import polygon

    rr, cc = polygon(snake_rows, snake_cols, shape=mask.shape)
    refined_mask[rr, cc] = True

    return refined_mask


def create_experiment_folders(base_output_path: str, pulses: list) -> dict:
    """
    Create a folder structure for segmentation experiments.

    Args:
        base_output_path (str): Base path to save results.
        pulses (list): List of pulse types.

    Returns:
        dict: Dictionary with pulse type as key and output path as value.
    """
    experiment_paths = {}
    for pulse in pulses:
        pulse_folder = os.path.join(base_output_path, pulse)
        os.makedirs(pulse_folder, exist_ok=True)
        experiment_paths[pulse] = pulse_folder
    return experiment_paths


def pipeline_one_patient(
    base_path: str,
    output_npz_path: str,
    patient: str,
    pulse: str,
    slice_index: int,
    thresholding_method: str,
    save_path: str,
) -> None:
    # Load the MRI slice
    try:
        nrrd_to_npz.convert_to_npz(
            base_path=base_path,
            output_path=output_npz_path,
            patient=patient,
            pulse=pulse,
        )
        filepath = os.path.join(output_npz_path, patient, f"{patient}_{pulse}.npz")
        slice_data = nrrd_to_npz.load_mri_slice(
            filepath=filepath, slice_index=slice_index
        )

        # Step 1: Segment the image
        intracranial_mask = ImageProcessing.global_histogram_segmentation(
            image=slice_data, method=thresholding_method
        )

        # Step 2: Fill the mask
        filled_mask = ImageProcessing.fill_mask(mask=intracranial_mask)

        # Step 3: Find the largest bounding box
        largest_bbox = ImageProcessing.find_largest_bbox(filled_mask)

        # Step 4: Apply Active Contours
        refined_mask = apply_active_contours_on_mask(image=slice_data, mask=filled_mask)

        # Step 5: Visualize the pipeline steps
        """
        visualize_pipeline_steps(
            image=slice_data,
            intracranial_mask=intracranial_mask,
            filled_mask=filled_mask,
            largest_bbox=largest_bbox,
            refined_mask=refined_mask,
            save_path=os.path.join(save_path, "pipeline_steps.pdf"),
        )
        """

        compare_segmentation_methods(
            image=slice_data,
            filled_mask=filled_mask,
            bounding_box=largest_bbox,
            refined_mask=refined_mask,
            save_path=save_path,
        )
    except FileNotFoundError:
        print(f"File not found: {filepath}")  # type: ignore
    except IOError as e:
        print(f"I/O error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        pass


if __name__ == "__main__":
    # Paths and parameters
    base_path: str = (
        "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition"
    )
    output_npz_path: str = "/home/mariopasc/Python/Datasets/Meningiomas/npz"
    save_base_path: str = (
        "/home/mariopasc/Python/Datasets/Meningiomas/segmentation_experiment"
    )

    pulses = ["SUSC", "T1", "T1SIN", "T2"]
    slice_index: int = -1  # Middle slice
    thresholding_method: str = "li"

    # Step 1: Create folder structure for experiments
    experiment_folders = create_experiment_folders(save_base_path, pulses)

    # Step 2: Iterate through pulses and patients
    for pulse in pulses:
        print(f"Processing pulse: {pulse}")
        pulse_output_path = experiment_folders[pulse]

        # List patients for the current pulse
        pulse_path = os.path.join(base_path, f"RM/{pulse}")
        patients = [
            folder
            for folder in os.listdir(pulse_path)
            if os.path.isdir(os.path.join(pulse_path, folder))
        ]
        print(f"Found {len(patients)} patients for pulse {pulse}")
        for folder in patients:
            print(os.path.basename(folder))

        for patient in patients:
            print(f"Processing patient: {patient}")
            patient_save_path = os.path.join(
                pulse_output_path, f"{patient}_results.pdf"
            )

            # Run the pipeline for the current patient
            pipeline_one_patient(
                base_path=base_path,
                output_npz_path=output_npz_path,
                patient=patient,
                pulse=pulse,
                slice_index=slice_index,
                thresholding_method=thresholding_method,
                save_path=patient_save_path,
            )

    print("Segmentation experiment completed.")
