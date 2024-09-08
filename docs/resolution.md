# Dimensionality Standarization

> Start Date: 07/09/2024

## Problem

Image resolution within the acquisition dataset is highly inconsistent, with sizes ranging from (200,200) to (1000,1000), though the most common height-width pairs are (512,512) and (256,256).

To address this issue, we will conduct a study that evaluates both the computational cost and the similarity between the original and resized images. This analysis will help determine the most suitable algorithm for each dataset.

## Metrics used

The primary objective of this project is the segmentation and characterization of meningioma brain tumors. Therefore, the most critical anatomical structure to preserve between the original and resized images is the meningioma itself, as its texture, characteristics and structure will play a pivotal role in determining the tumor's growth rate.

### Structural Similarity Index (SSIM)

The **Structural Similarity Index (SSIM)** is a perceptual metric that assesses the similarity between two images by taking into account structural information, luminance, and contrast. Unlike traditional metrics like Mean Squared Error (MSE), which only measure pixel-wise differences, SSIM is designed to better mimic human visual perception by focusing on the structure of the objects in an image and how they relate to luminance and contrast.

SSIM compares two images based on three components:

1. **Luminance (brightness)**: How much the pixel values differ in brightness.
2. **Contrast**: How much the images differ in contrast (variation of pixel intensities).
3. **Structure**: The overall structural similarity, which refers to the patterns in the pixel values and their spatial distribution.

### SSIM Formula

$$
\text{SSIM}(x, y) = \left[\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}\right]
$$

Where:

- $x$ and $y$ are the two images being compared (original and resized).
- $\mu_x$ and $\mu_y$ are the **mean intensities** of images $x$ and $y$, respectively.
- $\sigma_x^2$ and $\sigma_y^2$ are the **variances** (representing contrast) of $x$ and $y$.
- $\sigma_{xy}$ is the **covariance** between $x$ and $y$, representing structural similarity.
- $C_1$ and $C_2$ are small constants to stabilize the division when the denominator is close to zero. These constants are typically derived from the dynamic range of the pixel values (e.g., $C_1 = (K_1 L)^2$ and $C_2 = (K_2 L)^2$, where $L$ is the dynamic range of the pixel values, and $K_1$ and $K_2$ are small constants like 0.01 and 0.03, respectively).

### Components in Detail

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

### Characteristics

- **Range**: SSIM values range from $-1$ to $1$, where 1 indicates perfect similarity between the two images, 0 indicates no structural similarity, and negative values indicate structural dissimilarity.
- **Window-Based Calculation**: SSIM is often computed locally, using a sliding window over small image patches (e.g., 11x11 pixels) to get a localized score. The final SSIM score is the average over all windows.

### Normalized Cross-Correlation (NCC)

**Normalized Cross-Correlation (NCC)** is a similarity measure used primarily to compare the spatial structure of two images. It calculates the correlation between two images after normalizing the pixel intensities, which helps eliminate the effect of differences in overall intensity or brightness between the two images. NCC is widely used in image registration, template matching, and structural similarity tasks because it focuses on the correlation of intensity patterns rather than the absolute intensity values.

NCC evaluates how well the pixel intensity values of two images correlate with each other, providing a structural similarity measure.

### NCC Formula

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

### Key Points of NCC

1. **Covariance**: Measures how pixel intensity variations in the two images correspond to each other. If similar structures exist in both images, the covariance will be high.

2. **Normalization**: By subtracting the mean intensity and dividing by the standard deviation, the NCC formula is normalized. This makes NCC invariant to changes in brightness or contrast between the two images, as it focuses purely on structural similarity.

3. **Range**:
   - NCC values range from $-1$ to $1$:
     - $1$ indicates perfect positive correlation (the two images are identical in structure).
     - $0$ indicates no correlation (no structural similarity).
     - $-1$ indicates perfect negative correlation (inverse structures, where light and dark areas are swapped between the images).

### Properties of NCC

- **Translation Invariant**: NCC works well when the images have the same structure, even if they have undergone linear transformations such as intensity scaling or shifts in brightness.
- **Structural Preservation**: Since NCC compares patterns of pixel intensities rather than their absolute values, it captures the structural integrity of the image well.

## Methodology

Since NCC and SSIM metrics can't be computed between two images with different resolution, the image will be upscaled/downscaled to the desired size, and then downscaled/upscaled to its original size, in order to compute the performance metrics between the original image and the resized one. This experiment setup is also seen in "Hidden Influences on Image Quality when Comparing Interpolation Methods" reference.

In order to find the most suitable algorithm for each dataset, the pareto front will be plotted for the computational cost (X) against NCC/SSIM measure (Y).  



## Sources

- [NCC vs MI](https://eng.libretexts.org/Bookshelves/Industrial_and_Systems_Engineering/Chemical_Process_Dynamics_and_Controls_(Woolf)/13%3A_Statistics_and_Probability_Background/13.13%3A_Correlation_and_Mutual_Information)
- [NCC docs](https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html)
- [SSIM](https://en.wikipedia.org/wiki/Structural_similarity_index_measure)
- [Pareto front](https://en.wikipedia.org/wiki/Pareto_front)
- [Development of Improved SSIM Quality Index for Compressed Medical Images](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6707593&casa_token=GDxLE3Vdo2wAAAAA:M6DuC3s_g1QPakyG5-QuO-qDZswEMHB5YwACmI3RQMLD_B-YL_JXJ_aj1wYjpDaODrXFixa2fI0)
- [OpenCV Documentation for Interpolation Algorithms](https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#gga5bb5a1fea74ea38e1a5445ca803ff121ac6c578caa97f2d00f82bac879cf3c781)
- [P. Thévenaz, T. Blu, M. Unser, "Image Interpolation and Resampling", Handbook of Medical Imaging, Processing and Analysis, Academic Press, pp. 393-420, 2000](https://biblioseb.wordpress.com/wp-content/uploads/2018/03/academic-press-handbook-medical-imaging-processing-analysis.pdf)
- [Hidden Influences on Image Quality when Comparing Interpolation Methods](https://ieeexplore.ieee.org/abstract/document/4604443)