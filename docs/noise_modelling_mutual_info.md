### Methodology

The objective of this methodology is to quantify the statistical dependencies between pixel values in a local neighborhood within an MRI slice by calculating the mutual information (MI) between the binary values of a central pixel and its surrounding neighbors. Mutual information measures the amount of information shared between two variables, making it a suitable metric for detecting structured patterns or dependencies in the noise distribution. If the noise in the image is purely isotropic and random, the mutual information between a pixel and its neighbors should be close to zero, as no dependencies should exist.

The process begins by preprocessing the MRI image. A specific slice is extracted, and a mean filter of size $w \times w$ is applied to calculate the local average pixel intensity for each pixel. This mean-filtered image is used to binarize the original image, where a pixel is assigned a value of $1$ if its intensity is greater than or equal to the local mean and $0$ otherwise. A small amount of Gaussian noise is added to the original image to break ties and ensure randomness in binary values.

To compute the mutual information, a local neighborhood of size $z \times z$ is defined around each pixel. For each pixel, the co-occurrence of binary values between the central pixel and its neighbors is counted to determine the joint frequencies. These joint frequencies are normalized to probabilities $P(i, j)$, where $i$ represents the value of the central pixel (\(0\) or \(1\)) and $j$ represents the value of a neighboring pixel (\(0\) or \(1\)). Marginal probabilities for the central pixel $P(i)$ and the neighbor $P(j)$ are derived by summing over the appropriate dimensions of the joint probabilities.

Finally, the mutual information for each pixel is calculated using the formula:

$$
MI = \sum_{i,j} P(i, j) \log \left( \frac{P(i, j)}{P(i)P(j)} \right),
$$

where $P(i, j)$ is the joint probability, and $P(i)$ and $P(j)$ are the marginal probabilities. This computation is performed while ensuring numerical stability by masking invalid operations (e.g., division by zero or logarithm of zero). The resulting mutual information values are visualized as a map, which highlights regions of the image where dependencies between pixels and their neighbors exist. 