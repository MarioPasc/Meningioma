import numpy as np

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
    def approximate_pmf_from_pdf(reference_pdf: np.ndarray, x_values: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
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
            raise StatsException("Reference PDF must have at least two values to be discretized.")

        # Discretize the reference PDF into probability masses
        dx = x_values[1] - x_values[0]  # Width of each small interval
        prob_masses = reference_pdf * dx  # Approximate probability masses
        prob_masses += epsilon  # Add epsilon to avoid zeros
        prob_masses /= np.sum(prob_masses)  # Normalize to sum to 1

        return prob_masses

    @staticmethod
    def histogram_based_pmf_estimation(noise_values: np.ndarray, x_values: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
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
            raise StatsException("Noise samples must contain at least two values in order to estimate the probability function.")
        
        # Use the same bins as x_values for histogram, this way we ensure both discretized PDF and estimated PD are the same length
        dx = x_values[1] - x_values[0]  # Width of each small interval
        hist_counts, _ = np.histogram(noise_values, bins=np.append(x_values, x_values[-1] + dx), density=False)
        noise_prob_masses = hist_counts.astype(float) + epsilon  # Add epsilon to avoid zeros
        noise_prob_masses /= np.sum(noise_prob_masses)  # Normalize to sum to 1
        
        return noise_prob_masses