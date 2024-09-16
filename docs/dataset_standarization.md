# Dataset Standarization

> Start Date: 07/09/2024

## Introduction

Upon examining each acquisition-based dataset, several differences were identified between the images. These discrepancies include:

- Pixel intensity and image data type
- Image resolution
- Brain position within the image
- Percentage of the image occupied by the brain
- Number of slices per patient (slice depth)
- Inconsistent positioning of the transverse view

The goal of this document is to address these issues related to dataset standardization, providing a detailed account of each step taken toward the deployment of the final dataset. Each of these problems will be addressed individually or as a subproblem that may arise in the process of solving others in the list.

## Inconsistent positioning of the transverse view

To isolate the transversal view from the 3-channel images where the position of this view varies across files, we can rely on the header information of the `.nrrd` file, specifically looking at the `'space directions'` and `'space'` fields. Here's how we can approach this:

```python
# Example header
OrderedDict([('type', 'short'), ('dimension', 3), 
             ('space', 'left-posterior-superior'), 
             ('sizes', array([512, 512,  20])), 
             ('space directions', 
             array([[ 4.68163789e-01, -4.56154976e-03, -2.30892829e-02],
                   [ 3.30458536e-03,  4.68049202e-01, -2.54638708e-02],
                   [ 3.23124518e-01,  3.50395609e-01,  6.48252494e+00]])), 
             ('kinds', ['domain', 'domain', 'domain']), 
             ('endian', 'little'), ('encoding', 'gzip'), 
             ('space origin', array([-130.79400635, -134.75999451,  -41.02399826]))])
```

I'm glad to hear the algorithm works well! Here's an in-depth explanation of how and why the new algorithm is effective, with reference to coordinate systems and the underlying logic:

### Background: Coordinate Systems in Medical Imaging

In neuroimaging, medical images are typically stored in 3D coordinate systems that correspond to physical directions within the body. The most common systems are:

- **Left-Right (x)**: Along the axis from the patient's left side to the right side.
- **Posterior-Anterior (y)**: Along the axis from the patient's back to the front.
- **Superior-Inferior (z)**: Along the axis from the top of the head (superior) to the feet (inferior).

Medical imaging files like **NRRD** encode how each voxel (3D pixel) is mapped to real-world coordinates using the **`space directions`** matrix. This matrix describes how the voxel grid aligns with the physical space.

In neuroimaging, images are often viewed in three orientations (planes):

1. **Axial (Transversal)**: Horizontal slices that are perpendicular to the superior-inferior (z) axis.
2. **Sagittal**: Vertical slices that divide the body into left and right.
3. **Coronal**: Vertical slices that divide the body into front and back.

### Problem: Identifying the Transversal Axis

Given that medical images can be acquired from different machines or orientations, the axes in the data don’t always correspond directly to the physical x, y, and z directions. Thus, the challenge is to automatically identify which axis of the image data corresponds to the **transversal plane** (perpendicular to the superior-inferior direction).

### Approach: Analyzing the **`space directions`** Matrix

The **`space directions`** matrix in the NRRD header is a 3x3 matrix where:

- Each row describes how one of the image's axes aligns with physical space.
- The third element in each row corresponds to the contribution of that axis to the **superior-inferior (z)** direction.

For example, a row `[a, b, c]` in the matrix indicates how much of the image's axis is aligned with the **left-right (x)**, **posterior-anterior (y)**, and **superior-inferior (z)** directions.

### Why the Simple Z-Component Selection Failed

Initially, the algorithm worked by simply choosing the axis where the z-component (third element in the space direction vector) had the largest absolute value. This worked for many cases, but it failed in some cases because:

- **Other components of the direction vector (x and y)**: Sometimes, even if an axis has a significant z-component, it also has large contributions in the x or y directions. This means that the axis is not purely aligned with the superior-inferior direction and could correspond to a sagittal or coronal view instead of the transversal view.
- **Misleading z-components**: If an axis is mostly aligned with x or y but has a small z-component, it might be incorrectly chosen as the transversal axis.

```python
# Wrong slice extracted (not transversal)
short,3,[512 512  20],left-posterior-superior,
"[[ 3.59789165e-02  4.67362157e-01 -4.99059144e-06]
 [ 4.56489845e-02 -3.51917481e-03 -4.66503649e-01]
 [-5.95366956e+00  4.58324775e-01 -5.86044494e-01]]",
 "['domain', 'domain', 'domain']",
 little,gzip,[  37.09389877 -102.84300232  121.2460022 ],P34,T1
```

```python
#Good slice extracted (transversal):

unsigned short,3,[512 512  20],left-posterior-superior,
"[[-2.58588462e-02  4.48473863e-01  5.63667452e-09]
 [ 1.95621667e-02  1.12795334e-03 -4.48791193e-01]
 [-6.48304740e+00 -3.73810244e-01 -2.83526274e-01]]",
 "['domain', 'domain', 'domain']",
 little,gzip,[  66.65032196 -140.43499756  143.04386902],P33,T1SIN
```

```python
def transversal_axis(nrrd_path: str) -> int:
    """
    Finds the transversal axis of a nrrd file

    args:
        - nrrd_path (str): Path to the nrrd file.
    
    returns:
        - transversal_axis: Tranversal axis of the nrrd file.
    """

    _, header = open_nrrd(nrrd_path=nrrd_path, return_header=True)
    space_directions = np.array(header['space directions'])

    # Identify the axis corresponding to the transversal view (superior-inferior)
    # This is typically the axis with the largest value 
    # in the third column (superior-inferior)
    # We take the absolute values to avoid sign issues
    transversal_axis = np.argmax(np.abs(space_directions[:, 2]))

    return transversal_axis
```

### Improved Approach: Dominance of the Z-Component

To make the method more robust, we introduced the concept of **dominance**. The idea is to not just look at the absolute value of the z-component but to compare it against the contributions in the other directions (x and y). This ensures that the selected axis is primarily aligned with the superior-inferior direction and has minimal influence from the left-right or anterior-posterior directions.

#### Steps in the New Algorithm

1. **Normalize the Direction Vectors**
   - Each row in the `space directions` matrix is treated as a vector. By calculating the **norm** (magnitude) of each vector, we normalize the contribution of each component. This allows for a fair comparison of the x, y, and z components, even if the vectors have different magnitudes.

2. **Calculate the Dominance Ratio**:
   - For each axis (i.e., each row in the space directions matrix), the algorithm computes the ratio of the absolute z-component to the overall magnitude of the vector. This ratio represents how dominant the z-component is compared to the other components (x and y).
   - Mathematically:
     $$
     \text{Dominance Ratio} = \frac{|z_{\text{component}}|}{\text{norm of direction vector}}
     $$
   - A high dominance ratio means that the axis is primarily aligned with the superior-inferior direction, which is what we expect for the transversal plane.

3. **Select the Axis with the Highest Dominance Ratio**:
   - The axis with the largest dominance ratio is chosen as the transversal axis because it has the most significant alignment with the superior-inferior direction and minimal influence from the other directions.

### Why It Works

- **Considering the Full Vector**: Instead of just focusing on the z-component, the algorithm takes into account how much influence each axis has from the other directions (x and y). This avoids cases where an axis is mistakenly identified as transversal just because of a small z-component.
- **Robust to Orientation Variations**: The algorithm can handle different orientations, as it adapts to each individual scan’s orientation based on the space directions provided in the header.
- **Coordinate System Independence**: By using normalization, the algorithm is not dependent on the magnitude of the components but focuses on the relative alignment of the axes with respect to the physical coordinate system.

```python
import nrrd
import numpy as np

def find_dominant_transversal_axis_from_nrrd(nrrd_path: str) -> int:
    """
    Finds the transversal axis from the NRRD file by selecting the axis where
    the z-component dominates over the other components (x, y).
    
    Args:
        nrrd_path (str): Path to the NRRD file.
    
    Returns:
        int: The index of the transversal axis (0, 1, or 2), or None if no valid axis is found.
    """
    try:
        # Load the NRRD file and read the header
        _, header = nrrd.read(nrrd_path)

        # Extract space directions from the header
        space_directions = np.array(header.get("space directions"))

        # Extract the z-components from the space directions (third element of each vector)
        z_components = space_directions[:, 2]

        # Normalize each direction vector to compare relative dominance
        normalized_directions = np.linalg.norm(space_directions, axis=1)
        
        # Calculate the dominance ratios for each axis
        dominance_ratios = [abs(z) / norm for z, norm in zip(z_components, normalized_directions)]
        
        # Find the axis with the highest dominance ratio
        transversal_axis = np.argmax(dominance_ratios)
        
        return transversal_axis

    except Exception as e:
        print(f"Error processing {nrrd_path}: {e}")
        return None

def extract_middle_transversal_slice(image_data, transversal_axis):
    """
    Extracts the middle slice along the transversal axis.
    """
    # Calculate the middle slice index along the transversal axis
    middle_slice_idx = image_data.shape[transversal_axis] // 2
    
    # Use np.take to extract the middle slice from the correct axis
    middle_slice = np.take(image_data, middle_slice_idx, axis=transversal_axis)
    
    return middle_slice
```

## Image Resolution

The primary objective of this project is the segmentation and characterization of meningioma brain tumors. Therefore, the most critical anatomical structure to preserve between the original and resized images is the meningioma itself, as its texture, characteristics and structure will play a pivotal role in determining the tumor's growth rate.

### Interpolation Methods used

### Metrics

#### Structural Similarity Index (SSIM)

The **Structural Similarity Index (SSIM)** is a perceptual metric that assesses the similarity between two images by taking into account structural information, luminance, and contrast. Unlike traditional metrics like Mean Squared Error (MSE), which only measure pixel-wise differences, SSIM is designed to better mimic human visual perception by focusing on the structure of the objects in an image and how they relate to luminance and contrast.

SSIM compares two images based on three components:

1. **Luminance (brightness)**: How much the pixel values differ in brightness.
2. **Contrast**: How much the images differ in contrast (variation of pixel intensities).
3. **Structure**: The overall structural similarity, which refers to the patterns in the pixel values and their spatial distribution.

##### SSIM Formula

$$
\text{SSIM}(x, y) = \left[\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}\right]
$$

Where:

- $x$ and $y$ are the two images being compared (original and resized).
- $\mu_x$ and $\mu_y$ are the **mean intensities** of images $x$ and $y$, respectively.
- $\sigma_x^2$ and $\sigma_y^2$ are the **variances** (representing contrast) of $x$ and $y$.
- $\sigma_{xy}$ is the **covariance** between $x$ and $y$, representing structural similarity.
- $C_1$ and $C_2$ are small constants to stabilize the division when the denominator is close to zero. These constants are typically derived from the dynamic range of the pixel values (e.g., $C_1 = (K_1 L)^2$ and $C_2 = (K_2 L)^2$, where $L$ is the dynamic range of the pixel values, and $K_1$ and $K_2$ are small constants like 0.01 and 0.03, respectively).

##### Components in Detail

1. **Luminance**:
   $$
   l(x, y) = \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}
   $$
   This compares the average luminance (brightness) between the two images.

2. **Contrast**:
   $$
   c(x, y) = \frac{2\sigma_x\sigma_y + C_2}{\sigma_x^2 + \sigma_y^2 + C_2}
   $$
   This compares the contrast (variation of intensities) between the two images.

3. **Structure**:
   $$
   s(x, y) = \frac{\sigma_{xy} + C_2}{\sigma_x\sigma_y + C_2}
   $$
   This compares the structural similarity by evaluating the correlation between the two images.

##### Characteristics

- **Range**: SSIM values range from $-1$ to $1$, where 1 indicates perfect similarity between the two images, 0 indicates no structural similarity, and negative values indicate structural dissimilarity.
- **Window-Based Calculation**: SSIM is often computed locally, using a sliding window over small image patches (e.g., 11x11 pixels) to get a localized score. The final SSIM score is the average over all windows.

#### Normalized Cross-Correlation (NCC)

**Normalized Cross-Correlation (NCC)** is a similarity measure used primarily to compare the spatial structure of two images. It calculates the correlation between two images after normalizing the pixel intensities, which helps eliminate the effect of differences in overall intensity or brightness between the two images. NCC is widely used in image registration, template matching, and structural similarity tasks because it focuses on the correlation of intensity patterns rather than the absolute intensity values.

NCC evaluates how well the pixel intensity values of two images correlate with each other, providing a structural similarity measure.

##### NCC Formula

$$
\text{NCC}(x, y) = \frac{\sum_{i=1}^{N} (x_i - \mu_x)(y_i - \mu_y)}{\sqrt{\sum_{i=1}^{N} (x_i - \mu_x)^2 \sum_{i=1}^{N} (y_i - \mu_y)^2}}
$$

Where:

- $x$ and $y$ are the two images being compared (typically one original and one resized).
- $x_i$ and $y_i$ represent the intensity values of pixel $i$ in images $x$ and $y$, respectively.
- $N$ is the total number of pixels in the image.
- $\mu_x$ and $\mu_y$ are the mean intensity values of images $x$ and $y$, respectively.
- The numerator, $\sum_{i=1}^{N} (x_i - \mu_x)(y_i - \mu_y)$, represents the cross-covariance between the two images.
- The denominator, $\sqrt{\sum_{i=1}^{N} (x_i - \mu_x)^2 \sum_{i=1}^{N} (y_i - \mu_y)^2}$, normalizes the covariance, ensuring that the correlation is independent of image intensity scaling.

##### Key Points of NCC

1. **Covariance**: Measures how pixel intensity variations in the two images correspond to each other. If similar structures exist in both images, the covariance will be high.

2. **Normalization**: By subtracting the mean intensity and dividing by the standard deviation, the NCC formula is normalized. This makes NCC invariant to changes in brightness or contrast between the two images, as it focuses purely on structural similarity.

3. **Range**:
   - NCC values range from $-1$ to $1$:
     - $1$ indicates perfect positive correlation (the two images are identical in structure).
     - $0$ indicates no correlation (no structural similarity).
     - $-1$ indicates perfect negative correlation (inverse structures, where light and dark areas are swapped between the images).

##### Properties of NCC

- **Translation Invariant**: NCC works well when the images have the same structure, even if they have undergone linear transformations such as intensity scaling or shifts in brightness.
- **Structural Preservation**: Since NCC compares patterns of pixel intensities rather than their absolute values, it captures the structural integrity of the image well.

### Methodology

NCC and SSIM cannot be computed between two images of different resolutions. Therefore, the primary course of action in this experiment will be to downsample the input image to an arbitrary size, intentionally losing some pixel information (e.g., from (512,512) to (256,256) or (128,128)), and then upscale it back to its original resolution. This allows us to compute the performance metrics between the original and upscaled images. A similar experimental setup is discussed in the paper 'Hidden Influences on Image Quality when Comparing Interpolation Methods'. **The objective of this experiment is to determine which interpolation algorithm performs best when upscaling biomedical neuroimages**.

Once we've determined the most suitable interpolation algorithm for our problem, the procedure applied to the rest of images will have some extre steps in order to preserve the anatomical structure of the brain. As noted in our earlier analysis of size variability across datasets, image resolution is far from standardized in this project, with aspect ratios that are not necessarily square. This raises the critical question of how to upscale the images while preserving the brain’s aspect ratio, to avoid introducing any anatomical distortions.

![incorrect reshaping](./figures/size_stats/incorrect_reshaping.png)
> In this image we can see how resizing an image with an aspect ratio that is not a square can affect the anatomical structures of the brain (Pulse T1; P1). Input image is (416, 512) it is standarized to a (512,512) shape using `cv2.INTER_LANCZOS`, then its downscaled to (256,256) and upscaled to (512,512) in order to compute PSNR and SSIM.

In order to keep the aspect ratio of the brain in the images while succesfully resizing it to the desired resolution the following steps have been pourposed:

1. **Rescaling with Aspect Ratio Preservation**. 

```python
   @staticmethod
   def calculate_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int]) -> float:
      """
      Calculate the scaling factor to preserve the aspect ratio.
      
      Args:
         image (np.ndarray): Input image.
         target_size (tuple): Target resolution (width, height).
      
      Returns:
         float: Scaling factor to preserve aspect ratio.
      """
      original_h, original_w = image.shape[:2]
      target_w, target_h = target_size
      scale = min(target_w / original_w, target_h / original_h)
      return scale
```

![correct reshaping](./figures/size_stats/Rescaling_and_Padding.png)

## Sources

- [NCC vs MI](https://eng.libretexts.org/Bookshelves/Industrial_and_Systems_Engineering/Chemical_Process_Dynamics_and_Controls_(Woolf)/13%3A_Statistics_and_Probability_Background/13.13%3A_Correlation_and_Mutual_Information)
- [NCC docs](https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html)
- [SSIM](https://en.wikipedia.org/wiki/Structural_similarity_index_measure)
- [Pareto front](https://en.wikipedia.org/wiki/Pareto_front)
- [Development of Improved SSIM Quality Index for Compressed Medical Images](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6707593&casa_token=GDxLE3Vdo2wAAAAA:M6DuC3s_g1QPakyG5-QuO-qDZswEMHB5YwACmI3RQMLD_B-YL_JXJ_aj1wYjpDaODrXFixa2fI0)
- [OpenCV Documentation for Interpolation Algorithms](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121ac6c578caa97f2d00f82bac879cf3c781)
- [P. Thévenaz, T. Blu, M. Unser, "Image Interpolation and Resampling", Handbook of Medical Imaging, Processing and Analysis, Academic Press, pp. 393-420, 2000](https://biblioseb.wordpress.com/wp-content/uploads/2018/03/academic-press-handbook-medical-imaging-processing-analysis.pdf)
- [Hidden Influences on Image Quality when Comparing Interpolation Methods](https://ieeexplore.ieee.org/abstract/document/4604443)
