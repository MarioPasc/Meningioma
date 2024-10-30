from scipy.special import kl_div
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class Metrics:

    @staticmethod
    def compute_kl_divergence(noise_values: np.ndarray, reference_pdf: np.ndarray, x_values: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Compute the Kullback-Leibler divergence between an empirical noise distribution
        and a KDE-estimated distribution, adding epsilon to avoid division by zero.

        Parameters:
        - noise_values (np.ndarray): Original noise values outside the bounding box.
        - reference_pdf (np.ndarray): Discrete KDE-estimated PDF values evaluated at x_values.
        - x_values (np.ndarray): The x-values over which the PDFs are evaluated.
        - epsilon (float): Small value to prevent division by zero.

        Returns:
        - float: KL divergence between the empirical and KDE distributions.
        """
        # Step 1: Discretize the reference PDF into probability masses
        dx = x_values[1] - x_values[0]  # Width of each small interval
        prob_masses = reference_pdf * dx  # Approximate probability masses
        prob_masses += epsilon  # Add epsilon to avoid zeros
        prob_masses /= np.sum(prob_masses)  # Normalize to sum to 1

        # Step 2: Create empirical probability distribution from noise_values
        # Use the same bins as x_values for histogram
        bins = np.append(x_values, x_values[-1] + dx)  # Ensure the last bin captures the max value
        hist_counts, _ = np.histogram(noise_values, bins=bins, density=False)
        noise_prob_masses = hist_counts.astype(float) + epsilon  # Add epsilon to avoid zeros
        noise_prob_masses /= np.sum(noise_prob_masses)  # Normalize to sum to 1

        # Step 3: Compute KL divergence
        kl_divergence = np.sum(noise_prob_masses * np.log(noise_prob_masses / prob_masses))
        
        return kl_divergence

    @staticmethod
    def compute_bhattacharyya_distance(noise_values: np.ndarray, reference_pdf: np.ndarray, x_values:np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Compute the Bhattacharyya distance between an empirical noise distribution
        and a KDE-estimated distribution.

        Parameters:
        - noise_values (np.ndarray): Original noise values outside the bounding box.
        - reference_pdf (np.ndarray): Discrete KDE-estimated PDF values.
        - epsilon (float): Small value to prevent log(0) issues.

        Returns:
        - float: Bhattacharyya distance between the empirical and KDE distributions.
        """
        # Step 1: Discretize the reference PDF into probability masses
        dx = x_values[1] - x_values[0]  # Width of each small interval
        prob_masses = reference_pdf * dx  # Approximate probability masses
        prob_masses /= np.sum(prob_masses)  # Normalize to sum to 1

        # Step 2: Create empirical probability distribution from noise_values
        # Use the same bins as x_values for histogram
        hist_counts, _ = np.histogram(noise_values, bins=np.append(x_values, x_values[-1] + dx), density=False)
        noise_prob_masses = hist_counts.astype(float) / np.sum(hist_counts)  # Normalize to sum to 1

        # Step 3: Compute the Bhattacharyya coefficient
        bc_coefficient = np.sum(np.sqrt(noise_prob_masses * prob_masses)) + epsilon  # Add epsilon for numerical stability
        bhattacharyya_distance = -np.log(bc_coefficient)
        
        return bhattacharyya_distance

    @staticmethod
    def compute_ssim(original_image: np.ndarray, resized_image: np.ndarray) -> float:
        """
        Compute the Structural Similarity Index between the original and resized image
        """
        return ssim(image1=original_image, image2=resized_image, data_range=resized_image.max() - resized_image.min())

    @staticmethod
    def compute_psnr(original_image: np.ndarray, resized_image: np.ndarray) -> float:
        """
        Compute the Peak Signal Noise Ratio (PSNR) between the original and resized image
        """
        return psnr(image1=original_image, image2=resized_image, data_range=resized_image.max() - resized_image.min())

