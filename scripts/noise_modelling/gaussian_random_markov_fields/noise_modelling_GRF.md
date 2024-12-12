# Modelling MRI Background Noise using Gaussian Random Fields (GRF)

## Methodology

This section showcases how we have used **Gaussian Random Fields (GRFs)** to model spatially correlated noise in MRI background images without assuming isotropy or stationarity. By starting with a real 2D magnitude image, the GRFs simulate spatially coherent noise lumps that reflect the variability observed in MRI data. The two essential input for GRF generation is the **spatial correlation structure**, estimated using a **variogram**.

First, we need to deconstruct the input image into its real and complex parts. To archive this goal, we approximate the **k-space** from the MRI image in order to extract the phase ($\theta$). We apply the **2D Fast Fourier Transform (FFT)**, which converts the image $I(x, y)$, the spatial domain, into its frequency domain representation $k(u, v)$, where $(u, v)$ are spatial frequencies:

$$
k(u, v) = \sum_{x=0}^{N-1} \sum_{y=0}^{M-1} I(x, y) \cdot e^{-2\pi i \left( \frac{ux}{N} + \frac{vy}{M} \right)},
$$
where $I(x, y)$ is the pixel intensity at spatial position $(x, y)$, and $k(u, v)$ is the corresponding complex frequency component.

We use the FFT to estimate the k-space because, in MRI, the raw k-space data corresponds to the Fourier domain of the measured signal. While the original k-space is not typically available, applying the FFT to the reconstructed image provides an **approximate k-space**, allowing us to extract the **phase information** embedded in its complex values. The phase of the complex k-space is defined as:  
$$
\theta(x, y) = \arctan\left(\frac{\text{Im}(k(x, y))}{\text{Re}(k(x, y))}\right),
$$
where $k(x, y)$ represents the complex Fourier coefficients. The real image is then decomposed into its **real** and **imaginary** parts using the phase $\theta$ and the original image magnitude $|I|$:  
$$
\text{Real}(I) = |I| \cdot \cos(\theta), \quad \text{Imag}(I) = |I| \cdot \sin(\theta).
$$
This decomposition is crucial because the GRF framework models real-valued fields, and separating the real and imaginary components allows for independent noise modeling.

The spatial structure of the real component is analyzed using the **variogram**, which quantifies how pixel intensity differences vary with spatial distance $h$. The variogram is defined as:  
$$
\gamma(h) = \frac{1}{2N(h)} \sum_{i=1}^{N(h)} \left( Z(x_i) - Z(x_i + h) \right)^2,
$$
where $Z(x_i)$ is the intensity at position $x_i$, and $N(h)$ is the number of pixel pairs separated by distance $h$. This statistical measure captures the spatial variability and reveals how correlations decay over distance. To smooth and generalize the variogram, a **covariance model** is fitted. The covariance model provides the variance ($\sigma^2$), length scale (correlation range), and nugget (small-scale noise).

Once the covariance model is established, **Gaussian Random Fields** are generated independently for the **real** and **imaginary** components. These fields are spatially correlated and respect the noise structure defined by the covariance model. The final noise magnitude is then computed as:  
$$
|N| = \sqrt{N_{\text{real}}^2 + N_{\text{imag}}^2}.
$$
This formulation ensures that the resulting noise behaves like a **Rician distribution**, which naturally characterizes MRI background noise. 
