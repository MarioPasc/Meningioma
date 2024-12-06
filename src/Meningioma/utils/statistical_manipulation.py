import numpy as np
from numpy.typing import NDArray


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

            for i in range(n):
                for j in range(m):
                    i_start, i_end = i, i + z
                    j_start, j_end = j, j + z
                    neighborhood = binary_padded[i_start:i_end, j_start:j_end]
                    central_value = binarized_image[i, j]

                    joint_frequencies[i, j, 0] = np.sum(
                        (neighborhood == 0) & (central_value == 0)
                    )
                    joint_frequencies[i, j, 1] = np.sum(
                        (neighborhood == 1) & (central_value == 0)
                    )
                    joint_frequencies[i, j, 2] = np.sum(
                        (neighborhood == 0) & (central_value == 1)
                    )
                    joint_frequencies[i, j, 3] = np.sum(
                        (neighborhood == 1) & (central_value == 1)
                    )

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
