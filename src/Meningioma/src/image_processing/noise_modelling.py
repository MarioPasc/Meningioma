from scipy.stats import gaussian_kde, rice
from scipy import ndimage
from skimage.filters import threshold_li, threshold_otsu
from skimage import exposure, morphology, measure
import numpy as np
from typing import Tuple, List, Union

def get_detected_mri_image(image: np.ndarray, method: str = 'li', threshold: float = None):
    """
    Segment an MRI image to separate the background from the cranial and intracranial regions.
    
    Args:
        image (np.ndarray): MRI image to segment.
        method (str, optional): Thresholding method. Options: 'otsu', 'weighted_mean','li'. Defaults to 'li'.
        threshold (float, optional): Threshold for separating background and intracranial region. Used only if method is 'manual'.
        block_size (int, optional): Block size for adaptive thresholding. Only used if method is 'adaptive'.
    
    Returns:
        np.ndarray: Mask where background is 0 and intracranial region is 1.
    """
    # Compute histogram of the image for 'weighted_mean'
    hist, hist_centers = exposure.histogram(image)
    
    # Select the thresholding method
    if method == 'otsu':
        threshold = threshold_otsu(image)
    
    elif method == 'weighted_mean':
        threshold = np.average(hist_centers, weights=hist)
    
    elif method == 'li':
        threshold = threshold_li(image)
    
    elif method == 'manual':
        if threshold is None:
            raise ValueError("For 'manual' method, a threshold value must be provided.")
    
    else:
        raise ValueError("Invalid method specified. Choose from 'otsu', 'weighted_mean', 'li'.")
    
    # Create mask where values less than or equal to the threshold are considered background
    background_mask = image <= threshold
    
    # Invert the mask so that 0 is background and 1 is intracranial region
    intracranial_mask = ~background_mask
    
    return intracranial_mask

def get_filled_mask(mask: np.ndarray, structure_size: int = 7, iterations: int = 3) -> np.ndarray:
    """
    Fills the mask by applying morphological closing and hole filling multiple times.
    
    Args:
        mask (np.ndarray): Binary mask to be processed.
        structure_size (int): Size of the structuring element for morphological operations.
        iterations (int): Number of times to apply the closing and hole-filling process.
    
    Returns:
        np.ndarray: Mask after applying the filter multiple times.
    """
    structuring_element = morphology.disk(structure_size)
    
    # Apply the filter multiple times
    for _ in range(iterations):
        # Morphological closing
        mask = morphology.closing(mask, structuring_element)
        # Hole filling
        mask = ndimage.binary_fill_holes(mask, structure=np.ones((structure_size, structure_size)))
    
    return mask.astype(np.uint8)

def get_largest_bbox(mask: np.ndarray, extra_margin: Tuple[int, int, int, int] = (0, 0, 0, 0)) -> Tuple[int, int, int, int]:
    """
    Finds the largest bounding box within a binary mask and optionally extends it by 
    specified margins in each direction. Adjusts the bounding box if it exceeds image 
    boundaries, showing a message when adjustments are made.

    Parameters:
    - mask (np.ndarray): A 2D binary array representing the mask where regions are defined by 1s.
    - extra_margin (tuple of int, optional): A tuple (min_row_margin, min_col_margin, 
      max_row_margin, max_col_margin) specifying the number of units to adjust the bounding 
      box in each direction. Default is (0, 0, 0, 0), meaning no extension.

    Returns:
    - largest_bbox (tuple): A tuple (min_row, min_col, max_row, max_col) defining the bounding 
      box coordinates for the largest region in the mask, with any specified margin applied and 
      adjusted to fit within the image boundaries.
    """
    # Label the connected components in the mask
    labeled_mask, _ = ndimage.label(mask == 1)
    regions = measure.regionprops(labeled_mask)
    
    # Initialize variables to find the largest bounding box
    largest_bbox = None
    max_area = 0
    
    # Iterate through each region to find the largest one by area
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        area = (max_row - min_row) * (max_col - min_col)
        
        # Update largest_bbox if the current region is larger
        if area > max_area:
            max_area = area
            largest_bbox = (min_row, min_col, max_row, max_col)
    
    # Apply the extra margin to the largest bounding box, if provided
    if largest_bbox is not None:
        min_row, min_col, max_row, max_col = largest_bbox
        min_row_margin, min_col_margin, max_row_margin, max_col_margin = extra_margin

        # Adjust the bounding box by the given margins
        min_row = max(0, min_row - min_row_margin)
        min_col = max(0, min_col - min_col_margin)
        max_row = min(mask.shape[0], max_row + max_row_margin)
        max_col = min(mask.shape[1], max_col + max_col_margin)
        
        # Final largest bbox after applying margins and adjustments
        largest_bbox = (min_row, min_col, max_row, max_col)

    return largest_bbox

def get_noise_outside_bbox(image: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray) -> np.ndarray:
    """
    Extracts noise values from an image that lie outside a specified bounding box.

    Parameters:
    - image (np.ndarray): A 2D array representing the input image from which noise values are extracted.
    - bbox (tuple[int, int, int, int]): A tuple (min_row, min_col, max_row, max_col) defining the bounding box
      coordinates. These coordinates represent the region to exclude when extracting noise.
    - mask (np.ndarray): A binary or boolean mask array with the same shape as `image` where regions of interest 
      are defined. This mask is used to identify pixels outside the bounding box.

    Returns:
    - np.ndarray: An array of pixel values from `image` that are outside the specified bounding box.
    """
    min_row, min_col, max_row, max_col = bbox
    
    # Create a mask to exclude the bounding box region
    outside_bbox_mask = np.ones_like(mask, dtype=bool)
    outside_bbox_mask[min_row:max_row, min_col:max_col] = False
    
    # Extract noise values from the image that are outside the bounding box
    noise_values = image[outside_bbox_mask]
    return noise_values

def get_kde(noise_values: List[int], h: float = 1.0, num_points: int = 1000, return_x_values: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Estimate the probability density function (PDF) of noise values using a Kernel Density Estimation (KDE) approach.

    This function computes a KDE of the provided noise values, allowing for adjustable smoothing with `sigma`
    and controlling the granularity of the output PDF with `num_points`.

    Parameters:
    - noise_values (List[int]): A list of noise values sampled from an MRI image's background or similar data.
    - h (float, optional): The bandwidth smoothing parameter for KDE. Controls the width of the Gaussian kernel.
      Higher values result in smoother KDEs. Default is 1.0.
    - num_points (int, optional): The number of points to evaluate the KDE over the range of `noise_values`.
      A higher number increases resolution of the PDF but may require more computation time. Default is 1000.

    Returns:
    - np.ndarray: The estimated PDF values across the specified number of points within the range of `noise_values`.
    - Tuple[np.ndarray, np.ndarray]: The estimated PDF values across the specified number of points within the range of `noise_values` and the x values used.
    """

    # Define the range over which to evaluate the KDE
    x_values = np.linspace(np.min(noise_values), np.max(noise_values), num_points)

    # Apply Gaussian KDE with specified bandwidth (sigma)
    kde = gaussian_kde(noise_values, bw_method=h)

    # Evaluate KDE over x_values and return
    if not return_x_values: return  kde(x_values) 
    else: return (kde(x_values), x_values)
    
def get_rician(x_values: np.ndarray, sigma: float) -> np.ndarray:
    """
    Calculate the Rician probability density function (PDF) for given values and sigma.

    Parameters:
    - x_values (np.ndarray): Values over which to calculate the Rician PDF.
    - sigma (float): Scale parameter of the Rician distribution.

    Returns:
    - np.ndarray: Calculated Rician PDF values.
    """
    return rice.pdf(x_values, b=0, scale=sigma)

def main() -> None:
    import matplotlib.pyplot as plt
    from image_processing import ImageProcessing
    import Meningioma.src.metrics.metrics as metrics
    import scienceplots
    plt.style.use(['science', 'ieee', 'std-colors'])
    plt.rcParams['font.size'] = 10
    plt.rcParams.update({'figure.dpi': '100'})
    patient = 72
    pulse = 'SUSC'
    rm_nrrd_path=f'/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition/RM/{pulse}/P{patient}/{pulse}_P{patient}.nrrd'
    tc_nrrd_path=f'/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition/TC/P{patient}/TC_P{patient}.nrrd'

    # Initialize image processing and load the image
    processing = ImageProcessing()
    image = processing.open_nrrd_file(rm_nrrd_path)

    # Extract the transversal axis and slice for the middle image
    axis = ImageProcessing.get_transversal_axis(rm_nrrd_path)
    original_im = ImageProcessing.extract_middle_transversal_slice(image, transversal_axis=axis)

    # Segment the MRI image and fill the mask with the specified parameters
    mask = get_detected_mri_image(original_im)
    mask = get_filled_mask(mask, structure_size=7, iterations=3)

    # Define bounding boxes with and without margin
    largest_bbox_nomargin = get_largest_bbox(mask)
    largest_bbox_margin = get_largest_bbox(mask, extra_margin=(5, 5, 5, 5))

    # Extract noise values outside the bounding box
    noise_outside_bbox = get_noise_outside_bbox(image=original_im, bbox=largest_bbox_margin, mask=mask)

    # Define a list of sigma values for Parzen-Rosenblatt KDE
    sigma_values = [0.5, 1, 1.5, 2]  # List of sigma values for different KDE smoothing levels


    # Inside the `main()` function, after calculating the KDE PDF:
    sigma = 1  # Choose a sigma for KDE, or loop over multiple sigma values
    kde_pdf = get_kde(noise_outside_bbox, sigma=sigma)

    # Compute KL divergence between empirical and KDE PDFs
    kl_divergence_value = metrics.kl_divergence(noise_outside_bbox, kde_pdf)
    print(f"KL Divergence (sigma={sigma}):", kl_divergence_value)

    # Visualization
    _, axes = plt.subplots(1, 5, figsize=(15, 5))

    # Show the original image on the first axis
    axes[0].imshow(original_im, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Display mask with bounding box without extra margin
    axes[1].imshow(mask, cmap='gray')
    rect_largest_bbox = plt.Rectangle((largest_bbox_nomargin[1], largest_bbox_nomargin[0]), 
                                    largest_bbox_nomargin[3] - largest_bbox_nomargin[1],
                                    largest_bbox_nomargin[2] - largest_bbox_nomargin[0], 
                                    edgecolor='red', linewidth=2, fill=False)
    axes[1].add_patch(rect_largest_bbox)
    axes[1].set_title("BBox no extra")
    axes[1].axis('off')

    # Display mask with bounding box with extra margin
    axes[2].imshow(mask, cmap='gray')
    rect_largest_bbox = plt.Rectangle((largest_bbox_margin[1], largest_bbox_margin[0]), 
                                    largest_bbox_margin[3] - largest_bbox_margin[1],
                                    largest_bbox_margin[2] - largest_bbox_margin[0], 
                                    edgecolor='red', linewidth=2, fill=False)
    axes[2].add_patch(rect_largest_bbox)
    axes[2].set_title("BBox extra")
    axes[2].axis('off')

    # Plot histogram of noise values outside the bounding box
    counts, bins = np.histogram(noise_outside_bbox)
    axes[3].hist(bins[:-1], bins, weights=counts, histtype='stepfilled')
    axes[3].set_title("Noise Histogram")

    # Plot multiple Parzen-Rosenblatt PDFs for different sigma values
    for sigma in sigma_values:
        # Calculate the PDF for the current sigma value
        pdf_parzen_rosenblatt = get_kde(noise_values=noise_outside_bbox, sigma=sigma)
        
        # Plot the PDF on the same axis
        axes[4].plot(pdf_parzen_rosenblatt, label=f'sigma={sigma}')

    # Set up legend, title, and layout for the KDE plot
    axes[4].set_title("Parzen-Rosenblatt KDE")
    axes[4].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()