import os
from typing import Any, Tuple, List
from numpy.typing import NDArray

import numpy as np
from sklearn.cluster import DBSCAN  # type: ignore
from scipy.stats import gaussian_kde  # type: ignore

from Meningioma.image_processing import ImageProcessing  # type: ignore
from Meningioma.utils import Stats, npz_converter  # type: ignore

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes
import scienceplots  # type: ignore

plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def extract_bounding_box(image: NDArray[np.float64]) -> Tuple[int, int, int, int]:
    """
    Extracts the bounding box of the brain and skull using segmentation and masking.

    Args:
        image (NDArray[np.float64]): The input MRI slice.

    Returns:
        Tuple[int, int, int, int]: Bounding box coordinates (min_row, min_col, max_row, max_col).
    """
    mask = ImageProcessing.global_histogram_segmentation(image)
    filled_mask = ImageProcessing.fill_mask(mask)
    bbox = ImageProcessing.find_largest_bbox(filled_mask)
    return bbox


def mask_bounding_box(
    image: NDArray[np.float64], bbox: Tuple[int, int, int, int]
) -> NDArray[np.float64]:
    """
    Masks out pixels inside the bounding box (sets them to 0).

    Args:
        image (NDArray[np.float64]): The input image.
        bbox (Tuple[int, int, int, int]): Bounding box coordinates (min_row, min_col, max_row, max_col).

    Returns:
        NDArray[np.float64]: Image with bounding box region masked out.
    """
    min_row, min_col, max_row, max_col = bbox
    masked_image = image.copy()
    masked_image[min_row:max_row, min_col:max_col] = 0  # Mask bounding box
    return masked_image


def extract_noise_candidates(
    mutual_info: NDArray[np.float64], bbox: Tuple[int, int, int, int], threshold: float
) -> List[Tuple[int, int]]:
    """
    Extracts pixel coordinates with MI above a threshold and outside the bounding box.

    Args:
        mutual_info (NDArray[np.float64]): The mutual information map.
        bbox (Tuple[int, int, int, int]): Bounding box coordinates.
        threshold (float): Threshold for selecting pixels.

    Returns:
        List[Tuple[int, int]]: List of candidate pixel coordinates.
    """
    # Mask the bounding box
    masked_mi = mask_bounding_box(mutual_info, bbox)

    # Find pixels above threshold
    candidate_coords = np.argwhere(masked_mi > threshold)
    return [(x, y) for x, y in candidate_coords]


def cluster_noise_pixels(
    pixel_coords: List[Tuple[int, int]],
    eps: float = 3,
    min_samples: int = 5,
    seed: int = 42,
) -> NDArray[np.int32]:
    """
    Applies DBSCAN clustering to pixel coordinates.

    Args:
        pixel_coords (List[Tuple[int, int]]): List of pixel coordinates to cluster.
        eps (float): The maximum distance between two samples for one to be in the same cluster.
        min_samples (int): Minimum number of points to form a cluster.
        seed (int): Seed for reproducibility. Although DBSCAN is deterministic,
                    setting a seed can ensure stable sorting or other minor variations remain consistent.

    Returns:
        NDArray[np.int32]: Cluster labels for each pixel coordinate.
    """
    if not pixel_coords:
        return np.array([], dtype=np.int32)  # No candidates to cluster

    # For reproducibility, in case any ordering steps are involved
    np.random.seed(seed)

    # Convert list of tuples into a NumPy array of shape (n_samples, 2)
    pixel_array = np.array(pixel_coords, dtype=np.float64)

    # Perform DBSCAN clustering
    # https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.DBSCAN.html
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(pixel_array)

    return labels


def visualize_clusters(
    original_image: NDArray[np.float64],
    pixel_coords: List[Tuple[int, int]],
    cluster_labels: NDArray[np.int32],
    show: bool = False,
) -> None:
    """
    Visualizes the original image with clusters overlaid.

    Args:
        original_image (NDArray[np.float64]): The original MRI slice.
        pixel_coords (List[Tuple[int, int]]): Pixel coordinates used in clustering.
        cluster_labels (NDArray[np.int32]): Labels assigned by the clustering algorithm.

    Returns:
        None
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image, cmap="gray")
    plt.title("Clusters of Background Noise")

    # Overlay clusters
    for (x, y), label in zip(pixel_coords, cluster_labels):
        if label != -1:  # Ignore outliers
            plt.scatter(y, x, c=f"C{label % 10}", s=5)

    for format in ["png", "pdf"]:
        plt.savefig(f"./scripts/noise_modelling/mutual_information/clusters.{format}")
    if show:
        plt.show()


def visualize_pipeline_results(
    original_image: NDArray[np.float64],
    mutual_info: NDArray[np.float64],
    bbox: Tuple[int, int, int, int],
    noise_coords: List[Tuple[int, int]],
    cluster_labels: NDArray[np.int32],
    w: int,
    z: int,
    eps: float,
    min_samples: int,
    show: bool = False,
) -> None:
    """
    Visualizes the pipeline results in three subplots:
    1. Original image with the bounding box.
    2. Mutual information map with a colorbar.
    3. Original image with MI map and clustering results overlaid.

    Args:
        original_image (NDArray[np.float64]): The original MRI slice.
        mutual_info (NDArray[np.float64]): The mutual information map.
        bbox (Tuple[int, int, int, int]): Bounding box coordinates (min_row, min_col, max_row, max_col).
        noise_coords (List[Tuple[int, int]]): Coordinates of noise pixels used for clustering.
        cluster_labels (NDArray[np.int32]): Cluster labels assigned to noise pixels.
        w (int): Mean filter window size for MI computation.
        z (int): Neighborhood size for joint frequency computation.
        eps (float): DBSCAN clustering distance threshold.
        min_samples (int): Minimum points to form a cluster.

    Returns:
        None
    """
    fig, axes = plt.subplots(
        1, 3, figsize=(18, 6), gridspec_kw={"height_ratios": [1], "hspace": 0.3}
    )
    fig.suptitle(
        "Background Noise Analysis with Mutual Information and Clustering", fontsize=16
    )

    # --- Subplot 1: Original Image with Bounding Box ---
    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original Image with Bounding Box", fontsize=12)
    min_row, min_col, max_row, max_col = bbox
    rect = patches.Rectangle(
        (min_col, min_row),  # Bottom-left corner
        max_col - min_col,  # Width
        max_row - min_row,  # Height
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axes[0].add_patch(rect)

    # --- Subplot 2: Mutual Information Map ---
    mi_plot = axes[1].imshow(mutual_info, cmap="hot")
    axes[1].set_title(f"Mutual Information Map (w={w}, z={z})", fontsize=12)
    axes[1].axis("off")

    # --- Subplot 3: MI Map and Clustering Results Overlaid on Original Image ---
    axes[2].imshow(original_image, cmap="gray")  # Original image without alpha
    masked_mi = np.ma.masked_where(
        mutual_info <= 0.01, mutual_info
    )  # Mask near-zero MI values
    axes[2].imshow(masked_mi, cmap="hot", alpha=0.9)  # MI map with reduced alpha

    # Overlay cluster points
    num_clusters = len(set(cluster_labels)) - (
        1 if -1 in cluster_labels else 0
    )  # Exclude noise
    for (x, y), label in zip(noise_coords, cluster_labels):
        if label != -1:  # Exclude DBSCAN noise
            axes[2].scatter(y, x, c=f"C{label % 10}", s=25)

    axes[2].set_title(
        f"Clusters Overlaid (Clusters: {num_clusters}, eps={eps}, min_samples={min_samples})",
        fontsize=12,
    )

    for ax in axes:
        ax.axis("off")

    # --- Add a Shared Colorbar for the Mutual Information Map ---
    cbar_ax: Axes = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(mi_plot, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Mutual Information (MI) Value", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    for format in ["png", "pdf"]:
        plt.savefig(f"./scripts/noise_modelling/mutual_information/clusters.{format}")
    if show:
        plt.show()


def generate_cluster_pdfs(
    noise_coords: List[Tuple[int, int]],
    cluster_labels: NDArray[np.int32],
    original_image: NDArray[np.float64],
    bbox: Tuple[int, int, int, int],
    bandwidths: List[float] = [0.01, 0.1, 0.5],
) -> None:
    """
    Generates a 3x1 subplot of Parzen-Rosenblatt KDE models for each cluster
    using different bandwidths, with pixel intensities from the original image,
    using an evaluation range determined by pixel values outside of the bounding box.

    Additionally, plots a KDE model fitted on all pixels outside the bounding box
    (i.e., background region), displayed last on top of all cluster KDE lines.

    Args:
        noise_coords (List[Tuple[int, int]]): Coordinates of noise pixels.
        cluster_labels (NDArray[np.int32]): Labels assigned to each noise pixel.
        original_image (NDArray[np.float64]): The original MRI image.
        bbox (Tuple[int, int, int, int]): Bounding box coordinates (min_row, min_col, max_row, max_col).
        bandwidths (List[float]): List of bandwidth values for KDE.

    Returns:
        None
    """
    min_row, min_col, max_row, max_col = bbox
    # Create a mask for all pixels outside the bounding box
    outside_mask = np.ones_like(original_image, dtype=bool)
    outside_mask[min_row:max_row, min_col:max_col] = False

    # Extract all pixel values outside the bounding box
    outside_values = original_image[outside_mask]

    # Determine global_min and global_max based on outside (background) values only
    global_min = np.min(outside_values)
    global_max = np.max(outside_values)

    # Extract unique clusters and their sizes (excluding noise points, label -1)
    unique_labels = [label for label in set(cluster_labels) if label != -1]
    cluster_sizes = [
        (label, np.sum(cluster_labels == label)) for label in unique_labels
    ]
    cluster_sizes.sort(
        key=lambda x: x[1], reverse=True
    )  # Sort clusters by size (descending)

    # Prepare pixel intensity values for each cluster (from the original image)
    cluster_pixel_values = {
        label: [
            original_image[x, y]
            for (x, y), l in zip(noise_coords, cluster_labels)
            if l == label
        ]
        for label, _ in cluster_sizes
    }

    # Generate 3x1 subplots for different bandwidths
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(
        "Parzen-Rosenblatt KDE for Noise Clusters (Pixel Intensities)", fontsize=16
    )

    cluster_colors = {}
    color_map = plt.cm.tab10

    # Loop through bandwidths and plot PDFs
    for idx, h in enumerate(bandwidths):
        ax = axes[idx]
        ax.set_title(f"Bandwidth ($h$) = {h}", fontsize=12)
        ax.set_xlabel("Pixel Intensity Value", fontsize=10)
        ax.set_ylabel("Probability Density" if idx == 0 else "")
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        # Set log scale and predefined y-limits as requested
        ax.set_xscale("log")
        ax.set_ylim([0.0, 0.3])

        x_values = np.linspace(global_min, global_max, 1000)

        # Plot each cluster's KDE
        for i, (label, size) in enumerate(cluster_sizes):
            pixel_values = np.array(cluster_pixel_values[label])
            kde = gaussian_kde(pixel_values, bw_method=h)
            pdf_values = kde(x_values)

            color = color_map(i % 10)
            ax.plot(
                x_values,
                pdf_values,
                color=color,
                linewidth=2,
                label=f"Cluster {label} ({size} points)",
            )
            cluster_colors[label] = color

        # Finally, plot the KDE for all outside-of-bbox points on top
        all_kde = gaussian_kde(outside_values, bw_method=h)
        all_pdf_values = all_kde(x_values)

        # Plot with a distinct style (e.g., black, thicker line, dashed)
        ax.plot(
            x_values,
            all_pdf_values,
            color="black",
            linewidth=2,
            linestyle="--",
            label="All Outside BBox",
        )

    # Add shared legend below all subplots
    # Include clusters plus the "All Outside BBox" model
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=cluster_colors[label],
            linewidth=2,
            label=f"Cluster {label} ({size} points)",
        )
        for label, size in cluster_sizes
    ]
    # Add handle for all outside bbox model at the end
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color="black",
            linewidth=2,
            linestyle="--",
            label="All Outside BBox",
        )
    )

    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=10)
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    for format in ["png", "pdf"]:
        plt.savefig(
            f"./scripts/noise_modelling/mutual_information/parzen_rosenblatt_clusters.{format}"
        )


if __name__ == "__main__":
    # Example Input Paths and Parameters
    base_path: str = (
        "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition"
    )
    output_npz_path: str = "/home/mariopasc/Python/Datasets/Meningiomas/npz"

    slice_index: int = 30  # Middle slice
    mi_threshold: float = 0.1  # Mutual information threshold
    eps: float = 10  # DBSCAN clustering distance threshold
    min_samples: int = 100  # Minimum points to form a cluster

    w: int = 9
    z: int = 5

    patient: str = "P15"
    pulse: str = "T1"
    thresholding_method: str = "mean_filter"

    # 1. Convert data to NPZ format
    """
    npz_converter.convert_to_npz(
        base_path=base_path, output_path=output_npz_path, patient=patient, pulse=pulse
    )
    """
    # 2. Define the file path
    filepath = os.path.join(output_npz_path, patient, f"{patient}_{pulse}.npz")

    # 3. Load MRI Slice
    slice_data = npz_converter.load_mri_slice(
        filepath=filepath, slice_index=slice_index
    )

    # 4. Compute Mutual Information Map
    binarized_image, _ = ImageProcessing.local_histogram_segmentation(
        slice_data, w=w, thresholding_method=thresholding_method
    )
    joint_frequencies = Stats.compute_joint_frequencies(binarized_image, z=z)
    mutual_info = Stats.compute_mutual_information(joint_frequencies)

    # 5. Extract Bounding Box
    bbox = extract_bounding_box(slice_data)

    # 6. Extract Noise Candidates
    noise_candidates = extract_noise_candidates(
        mutual_info, bbox, threshold=mi_threshold
    )

    # 7. Cluster Noise Pixels
    cluster_labels = cluster_noise_pixels(
        noise_candidates, eps=eps, min_samples=min_samples
    )

    # 8. Generate KDE Visualization for Each Cluster
    generate_cluster_pdfs(
        noise_coords=noise_candidates,
        cluster_labels=cluster_labels,
        original_image=slice_data,  # Pass the original MRI slice
        bandwidths=[0.5, 1.0, 1.5],
        bbox=bbox,
    )
    # Visualize Results
    visualize_pipeline_results(
        original_image=slice_data,
        mutual_info=mutual_info,
        bbox=bbox,
        noise_coords=noise_candidates,
        cluster_labels=cluster_labels,
        w=w,
        z=z,
        eps=eps,
        min_samples=min_samples,
    )
