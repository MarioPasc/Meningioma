# Noise Modelling characterizising noise lumps in MRI

## Methodology

To estimate the probability of each pixel in an MRI image being corrupted by impulse noise, the process begins with an initial restoration step. Since the original pixel values $z(x_i)$ are unknown, an estimate $\tilde{z}(x_i)$ is obtained using the **Iteratively Reweighted Norm (IRN)** method. This method minimizes a functional that combines fidelity to the noisy input image and regularization to suppress abrupt changes, which are indicative of noise (for now, whatever this means, this is the wikipedia explanation). The output is a restored image providing predicted pixel values $\tilde{z}(x_i)$, and the errors are approximated as:
$$
\tilde{e}_i = y_i - \tilde{z}(x_i),
$$
where $y_i$ is the observed noisy pixel value. These predicted errors are used to construct a probabilistic mixture noise model for the error distribution:
$$
p(\tilde{e}_i) = (1 - P_{\text{Im}}) N_\sigma(\tilde{e}_i) + P_{\text{Im}} \text{Tri}_v(\tilde{e}_i),
$$
where $N_\sigma(\tilde{e}_i)$ is the Gaussian density with variance $\sigma^2$, and $\text{Tri}_v(\tilde{e}_i)$ is the triangular density representing uniform impulse noise. The parameters $P_{\text{Im}}$ (impulse noise probability) and $\sigma^2$ (Gaussian noise variance) need to be estimated to quantify the likelihood of each pixel being corrupted by impulse noise.

The parameter estimation and classification of pixels into Gaussian or impulse corrupted classes are achieved using the **Expectation-Maximization (EM)** algorithm. The EM algorithm addresses the latent variable challenge (whether a pixel is Gaussian or impulse corrupted) by iteratively refining the probabilities $P(\text{Im}(x_i) | \tilde{e}_i)$ and model parameters. In the **E-Step**, the algorithm computes the posterior probabilities (responsibilities) using Bayes' theorem:
$$
P(\text{Im}(x_i) | \tilde{e}_i) = \frac{P_{\text{Im}} \text{Tri}_v(\tilde{e}_i)}{(1 - P_{\text{Im}}) N_\sigma(\tilde{e}_i) + P_{\text{Im}} \text{Tri}_v(\tilde{e}_i)},
$$
$$
P(G(x_i) | \tilde{e}_i) = \frac{(1 - P_{\text{Im}}) N_\sigma(\tilde{e}_i)}{(1 - P_{\text{Im}}) N_\sigma(\tilde{e}_i) + P_{\text{Im}} \text{Tri}_v(\tilde{e}_i)}.
$$
In the **M-Step**, these probabilities are used to update the model parameters:
$$
P_{\text{Im}}^{(t+1)} = \frac{1}{AB} \sum_{i} P(\text{Im}(x_i) | \tilde{e}_i),
$$
$$
\sigma^{(t+1)} = \sqrt{\frac{\sum_i P(G(x_i) | \tilde{e}_i) \tilde{e}_i^2}{AB(1 - P_{\text{Im}}^{(t+1)})}},
$$
where $A \times B$ is the total number of pixels in the image. These steps are repeated until convergence, ensuring the parameters and posterior probabilities are well-calibrated.