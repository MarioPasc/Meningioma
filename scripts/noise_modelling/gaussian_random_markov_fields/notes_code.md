# Notes on Markov 

## Hyperparameters to Tune

### 1. **Variogram Estimation Parameters**
- **`bins`**:  
   Defines the distance range and resolution for variogram estimation.  
   A higher resolution (more bins) provides finer spatial correlation details but increases computation time.
- **`sampling_size`**:  
   Number of pixel pairs sampled to estimate the variogram. Larger values reduce noise but increase runtime.  
   Default: `2000`.

### 2. **Covariance Model Parameters**
- **`var` (Initial Variance Guess)**:  
   Controls the amplitude of the spatial variability in the covariance model.  
- **`len_scale` (Length Scale)**:  
   Defines the spatial range of correlation. A larger value results in smoother, more slowly varying noise fields.
- **`nugget`**:  
   Small-scale noise variability. Adding a nugget parameter accounts for fine-grained, uncorrelated noise in the data.

---

## Validation Techniques for Background Noise Estimation

### 1. **Statistical Metrics**
- **Mean and Standard Deviation**:  
   Compare the mean and standard deviation of the original image and the generated synthetic noise fields:  
   $$
   \text{Mean}_{\text{real}} \approx \text{Mean}_{\text{synthetic}}, \quad \text{Std}_{\text{real}} \approx \text{Std}_{\text{synthetic}}
   $$
- **Histogram Comparison**:  
   Plot the intensity histograms of both fields to ensure similarity in their distributions.

### 2. **Spatial Structure Validation**
- **Variogram Comparison**:  
   Compute and overlay variograms for the original data and the generated noise fields. Matching variograms indicates that spatial correlation has been accurately preserved.
- **Covariance Model Fit**:  
   Evaluate the covariance model's goodness-of-fit to the estimated variogram to ensure an accurate spatial correlation representation.

### 3. **Visual Inspection**
- Visualize the real and synthetic noise fields side by side using color maps. Assess for consistency in structure, noise lumps, and smoothness.
- Overlay the synthetic noise on the original image to confirm that the distribution appears realistic and spatially consistent.

### 4. **Frequency Domain Analysis**
- Compare the frequency spectra of the original and synthetic noise fields using the **Fourier Transform**. Similar frequency content validates the noise's spatial texture.

### 5. **Rician Distribution Check**
- Verify that the magnitude of the generated noise approximates a **Rician distribution**, characteristic of MRI background noise.  
   Fit a Rician distribution to the histogram of the noise magnitude and compare it to the expected behavior.

### 6. **Peak Signal-to-Noise Ratio (PSNR)**
- If a noise-free reference image is available, compute the PSNR to evaluate the fidelity of the noise-added image:  
   $$
   \text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}^2}{\text{MSE}} \right)
   $$
   where MAX is the maximum pixel intensity, and MSE is the mean squared error.
