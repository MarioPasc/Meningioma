# Modelling MRI Background Noise using Gaussian Random Fields (GRF)

## Methodology

This section showcases how we have used **Gaussian Random Fields (GRFs)** to model spatially correlated noise in MRI background images without assuming isotropy or stationarity. By starting with a real 2D magnitude image, the GRFs simulate spatially coherent noise lumps that reflect the variability observed in MRI data. The two essential input for GRF generation is the **spatial correlation structure**, estimated using a **variogram**.

First, we need to deconstruct the input image into its real and complex parts. To archive this goal, we approximate the **k-space** from the MRI image in order to extract the phase ($\theta$). Due to the fact that modern MRI procedures often include the pMRI approach, subsampling the k-space bellow twice the bandwidth, violating the Nyquist-Shannon criterium, and introducing aliasing in the image adquisition process. This process makes the relationship between the magnitude image and the k-space using the Fourier Transform non-bidirectional, so the Fourier Transform is not a reliable tool that we can use to approximate the k-space, at least without any additional information about the reconstruction algorithm of the k-space (e.g. GRAPPA, SENSE, SMASH, etc.), which we don't have.


Despite its limitations, we apply the **2D Fast Fourier Transform (FFT)**, which converts the image $I(x, y)$, the spatial domain, into its frequency domain representation $k(u, v)$, where $(u, v)$ are spatial frequencies:

$$
k(u, v) = \sum_{x=0}^{N-1} \sum_{y=0}^{M-1} I(x, y) \cdot e^{-2\pi i \left( \frac{ux}{N} + \frac{vy}{M} \right)},
$$
where $I(x, y)$ is the pixel intensity at spatial position $(x, y)$, and $k(u, v)$ is the corresponding complex frequency component.

This **approximate k-space** allows us to extract the **phase information** embedded in its complex values, which can be is defined as:  
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
where $Z(x_i)$ is the intensity at position $x_i$, and $N(h)$ is the number of pixel pairs separated by distance $h$. This statistical measure captures the spatial variability and reveals how correlations decay over distance. To smooth and generalize the variogram, a **covariance model** is fitted. We consider various covariance models that are provided in the `gstools` framework. The covariance models that yields the highest $r^2$ will be selected as the representative of the variogram. The covariance model provides the variance ($\sigma^2$), length scale (correlation range), and nugget (small-scale noise).

Once the covariance model is established, **Gaussian Random Fields** are generated independently for the **real ($X$)** and **imaginary ($Y$)** components. These fields respect the spatial noise structure defined by the covariance model. These fields follow a zero-mean gaussian distribution:

$$
X \sim \mathcal{N}(0, \sigma_x^2) 
\space \space \space \space \space 
Y \sim \mathcal{N}(0, \sigma_y^2)
$$

The final noise magnitude is then computed as:  
$$
|R| = \sqrt{X^2 + Y^2}
\space \space \space \space \space 
R \sim \text{Rayleigh}(\hat{\sigma})
$$

In the ideal case, where we generate two non-stationary, correlated, noise maps which each one follows the distribution of normal random variable with zero-mean, and we combine them using the module operation, the resulting distribution follows a **Rayleigh distribution**, defined by a *scale parameter ($\hat{\sigma}$)*. The distribution of the background noise of the original image fits correctly the shape of a Rayleigh distribution.

The problem is that we are generating two non-stationary, *independent* noise maps which follow a zero-mean normal distribution and with approximatly the same standard deviation. This setup follows a NC-$\chi^2$ distribution, which is also characteristic of a SoS GRAPPA processing of the subsampled k-space. 

The independent real and imaginary parts combined to generate the final noise slice do not follow a *Rice distribution* since there is no underlying signal in these fields, which, in a typical MRI slice, this signal would correspond to the anatomical body parts of the patient. Finally, we substract the mean of the generalted slice to make the noise zero-mean:

$$
R = |R| - \hat{\sigma}\sqrt{\frac{\pi}{2}} \approx |R| - 1.253\times\hat{\sigma}
$$

The potential application of this pipeline is to generate a coherent volume of noise with a desired resolution of the MRI brain images. When constructing a MRI dataset, it is usual to recieve images from different machines, having varying image resolutions. We want to standarize these resolutions while keeping a coherent noise distribution in the study (i.e. no skull-stripping). Having $N$ slices of a study, we would generate a $[D_x, D_y, N]$ noise volume with our pipeline. The next step would be to resize the brain in the original slice using the letterbox algorithm (i.e. upscale/downscale the axis of the image in a 1:1 relation until one of them reaches the desired length, then, keeping the other axis at its size, keeping the aspect-ratio of the brain), and then cutting the anatomical region using the convex hull algorithm. Then, we would paste the brain into the generated noise slice.



### Potential Issues

- We are generating independent noise slices which follow a Gaussian RV distribution with zero-mean and similar standard deviation. When combined, the final distribution follows a NC-$\chi^2$ distribution, however, our background data better fits a Rayleigh distribution. **Solution**: We must find a way to generate dependent slices with non-stationary noise.
- We are using the FFT to estimate the k-space, even though the relationship between the magnitude image and its k-space do not share a bidirectional relationship when the k-space is subsampled. **Solution**: Image preprocessing before estimating the k-space or use a different transform (Riesz).

### Notas

- [X] Quitar el cerebro y usar solo ruido de fondo
- [X] Validar la estimación de la fase mediante la FFT de la imagen completa
- [X] Comparar resultados con una fase de la FFT total de la imagen y un parche.
- [X] Comprobar las direcciones del ruido anisotrópico
- [X] Comprobar la generación de ruido tridimensional

- [X] Código: Debido a la adición de la máscara, hemos tenido que cambiar de grid type structured a unstructured:

> In structured mode, the coordinates (x, y) map directly to pixel indices, making visualization straightforward.
> In unstructured mode, positions are explicitly given as a list of coordinates, which may require re-mapping if you want to interpret or visualize results back on a regular > grid.

- [ ] Modular el código para entrar dentro del paquete de Python creado.
