import numpy as np
from numpy.typing import NDArray

from scipy.stats import rice, rayleigh, ncx2, norm  # type: ignore
from scipy.interpolate import interp1d  # type: ignore

import pandas as pd

from Meningioma import ImageProcessing


class StatsException(Exception):
    """
    Custom exception for errors related to metric computations.
    """

    def __init__(self, message: str) -> None:
        """
        Throws the spefified message as a MetricsException

        Parameters:
        - message (str): Message to be thrown
        """
        super().__init__(message)


class Stats:
    @staticmethod
    def approximate_pmf_from_pdf(
        reference_pdf: np.ndarray,
        x_values: np.ndarray,
        epsilon: float = 1e-10,
    ) -> np.ndarray:
        """
        Discretization of a continuous probability density function into a probability mass function (PMF)
        using numerical integration (or Riemann sum approximation).

        Parameters:
        - reference_pdf (np.ndarray): Discrete PDF values evaluated at x_values.
        - x_values (np.ndarray): The x-values over which the PDFs are evaluated.
        - epsilon (float): Small value to prevent division by zero.

        Returns:
        - np.ndarray: The discretized PDF
        """
        if len(reference_pdf) < 2:
            raise StatsException(
                "Reference PDF must have at least two values to be discretized."
            )

        # Discretize the reference PDF into probability masses
        dx = x_values[1] - x_values[0]  # Width of each small interval
        prob_masses = reference_pdf * dx  # Approximate probability masses
        prob_masses += epsilon  # Add epsilon to avoid zeros
        prob_masses /= np.sum(prob_masses)  # Normalize to sum to 1

        return prob_masses

    @staticmethod
    def histogram_based_pmf_estimation(
        noise_values: np.ndarray,
        x_values: np.ndarray,
        epsilon: float = 1e-10,
    ) -> np.ndarray:
        """
        Histogram-based estimation of the empirical Probability Mass Function (PMF).
        The method is a basic form of density estimation where the data is grouped into bins to approximate the distribution.

        Parameters:
        - noise_values (np.ndarray): Discrete samples taken from the image background.
        - x_values (np.ndarray): The x-values over which the PDFs are evaluated.
        - epsilon (float): Small value to prevent division by zero.

        Returns:
        - np.ndarray: The estimated probability distribution for the noise values
        """
        if len(noise_values) < 2:
            raise StatsException(
                "Noise samples must contain at least two values in order to estimate the probability function."
            )

        # Use the same bins as x_values for histogram, this way we ensure both discretized PDF and estimated PD are the same length
        dx = x_values[1] - x_values[0]  # Width of each small interval
        hist_counts, _ = np.histogram(
            noise_values,
            bins=np.append(x_values, x_values[-1] + dx),
            density=False,
        )
        noise_prob_masses = (
            hist_counts.astype(float) + epsilon
        )  # Add epsilon to avoid zeros
        noise_prob_masses /= np.sum(noise_prob_masses)  # Normalize to sum to 1

        return noise_prob_masses

    @staticmethod
    def compute_joint_frequencies(
        binarized_image: NDArray[np.uint8], z: int
    ) -> NDArray[np.float64]:
        """
        Computes joint frequencies for binary pixel values in local neighborhoods.

        Args:
            binarized_image (NDArray[np.uint8]): The binarized MRI slice.
            z (int): Neighborhood size.

        Returns:
            NDArray[np.float64]: A 3D array storing joint frequencies for each pixel.
        """
        try:
            n, m = binarized_image.shape
            pad_width = z // 2
            binary_padded = np.pad(binarized_image, pad_width, constant_values=0)
            joint_frequencies = np.zeros((n, m, 4), dtype=np.float64)
            binary_padded_rolled = np.roll(binary_padded, 1, axis=0)

            # Fill the 3D ndarray
            for i in range(n):
                for j in range(m):
                    # Define z x z neighborhood
                    i_start, i_end = i, i + z
                    j_start, j_end = j, j + z
                    neighborhood = binary_padded[i_start:i_end, j_start:j_end]

                    # Original pixel values
                    # pixel_value = binarized_image[i, j]
                    pixel_value = binary_padded_rolled[i_start:i_end, j_start:j_end]

                    # Calculate counts
                    joint_frequencies[i, j, 0] = np.sum(
                        (neighborhood == 0) & (pixel_value == 0)
                    )
                    joint_frequencies[i, j, 1] = np.sum(
                        (neighborhood == 1) & (pixel_value == 0)
                    )
                    joint_frequencies[i, j, 2] = np.sum(
                        (neighborhood == 0) & (pixel_value == 1)
                    )
                    joint_frequencies[i, j, 3] = np.sum(
                        (neighborhood == 1) & (pixel_value == 1)
                    )

            # Normalize the probabilities
            joint_frequencies /= z * z
            return joint_frequencies
        except Exception as e:
            raise RuntimeError(f"Error during joint frequency computation: {e}")

    @staticmethod
    def compute_mutual_information(
        joint_frequencies: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Computes the mutual information for each pixel using joint frequencies.

        Args:
            joint_frequencies (NDArray[np.float64]): A 3D array of joint frequencies.

        Returns:
            NDArray[np.float64]: A 2D array of mutual information values for each pixel.
        """
        try:
            n, m, _ = joint_frequencies.shape
            mutual_info = np.zeros((n, m), dtype=np.float64)

            marginal_pixel = np.zeros((n, m, 2), dtype=np.float64)
            marginal_neighbor = np.zeros((n, m, 2), dtype=np.float64)

            marginal_pixel[:, :, 0] = (
                joint_frequencies[:, :, 0] + joint_frequencies[:, :, 1]
            )
            marginal_pixel[:, :, 1] = (
                joint_frequencies[:, :, 2] + joint_frequencies[:, :, 3]
            )
            marginal_neighbor[:, :, 0] = (
                joint_frequencies[:, :, 0] + joint_frequencies[:, :, 2]
            )
            marginal_neighbor[:, :, 1] = (
                joint_frequencies[:, :, 1] + joint_frequencies[:, :, 3]
            )

            for pixel_value in range(2):
                for neighbor_value in range(2):
                    joint_prob = joint_frequencies[
                        :, :, neighbor_value + 2 * pixel_value
                    ]
                    denom = (
                        marginal_pixel[:, :, pixel_value]
                        * marginal_neighbor[:, :, neighbor_value]
                    )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        quotient = np.divide(joint_prob, denom, where=denom > 0)
                        logarithm = np.log(quotient, where=quotient > 0)
                        mutual_info += joint_prob * logarithm

            return mutual_info
        except Exception as e:
            raise RuntimeError(f"Error during mutual information computation: {e}")

    @staticmethod
    def compute_pdf(data: np.ndarray, h: float, dist: str = "norm") -> tuple:
        """
        Estimate the probability density function (PDF) for a 1D data array by computing:
        - A Parzen–Rosenblatt KDE (using ImageProcessing.kde)
        - A theoretical PDF fitted to the data using one of several distributions.

        Parameters:
            data: 1D numpy array of noise values.
            h: Bandwidth for the KDE estimation.
            dist: A string specifying the distribution to fit. One of:
                "norm"     → Gaussian (normal) distribution.
                "rayleigh" → Rayleigh distribution.
                "rice"     → Rice distribution.
                "ncx2"     → Non-central chi-squared distribution.

        Returns:
            A tuple with:
            - x_common: Common x-axis values (numpy array).
            - kde_est: The KDE-estimated PDF (numpy array).
            - pdf_fit: The theoretical PDF evaluated on x_common (numpy array).
            - param_str: A string summarizing the fitted parameters.
            - param_series: A pandas Series with the fitted parameter values.
        """
        data = data.flatten()
        x_min = data.min()
        x_max = data.max()
        x_common = np.linspace(x_min, x_max, 1000)

        # Compute KDE using the custom function.
        kde_vals, x_kde = ImageProcessing.kde(
            data, h=h, num_points=1000, return_x_values=True
        )
        f_interp = interp1d(x_kde, kde_vals, bounds_error=False, fill_value=0)
        kde_est = f_interp(x_common)

        # Fit the theoretical distribution and compute the PDF.
        if dist.lower() == "norm":
            mu, sigma = norm.fit(data)
            pdf_fit = norm.pdf(x_common, loc=mu, scale=sigma)
            param_str = f"μ={mu:.2f}, σ={sigma:.2f}"
            param_series = pd.Series({"mu": mu, "sigma": sigma})
        elif dist.lower() == "rayleigh":
            loc, scale = rayleigh.fit(data)
            pdf_fit = rayleigh.pdf(x_common, loc=loc, scale=scale)
            # Report sigma as sqrt(scale) if desired.
            sigma_est = np.sqrt(scale)
            param_str = f"loc={loc:.2f}, σ̂={sigma_est:.2f}"
            param_series = pd.Series({"loc": loc, "scale": scale, "sigma": sigma_est})
        elif dist.lower() == "rice":
            # rice.fit returns: shape parameter b, location, and scale.
            b, loc, scale = rice.fit(data)
            pdf_fit = rice.pdf(x_common, b, loc=loc, scale=scale)
            param_str = f"b={b:.5f}, loc={loc:.2f}, scale={scale:.2f}"
            param_series = pd.Series({"b": b, "loc": loc, "scale": scale})
        elif dist.lower() == "ncx2":
            # ncx2.fit returns: degrees of freedom, noncentrality, location, and scale.
            df, nc, loc, scale = ncx2.fit(data)
            pdf_fit = ncx2.pdf(x_common, df, nc, loc=loc, scale=scale)
            sigma_est = np.sqrt(scale)
            param_str = f"L={df:.2f}, NC={nc:.2f}, σ={sigma_est:.2f}"
            param_series = pd.Series(
                {"df": df, "nc": nc, "loc": loc, "scale": scale, "sigma": sigma_est}
            )
        else:
            raise ValueError(
                f"Distribution '{dist}' not recognized. Choose among 'norm', 'rayleigh', 'rice', 'ncx2'."
            )

        return x_common, kde_est, pdf_fit, param_str, param_series
