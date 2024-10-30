from scipy.special import kl_div
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class Metrics:

    @staticmethod
    def compute_kl_divergence(noise_values: np.ndarray, reference_pdf: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Compute the Kullback-Leibler divergence between an empirical noise distribution
        and a KDE-estimated distribution, adding epsilon to avoid division by zero.

        Parameters:
        - noise_values (np.ndarray): Original noise values outside the bounding box.
        - reference_pdf (np.ndarray): Discrete KDE-estimated PDF values.
        - epsilon (float): Small value to prevent division by zero.

        Returns:
        - float: KL divergence between the empirical and KDE distributions.
        """
        # Compute a histogram of the noise values
        hist_counts, _ = np.histogram(noise_values, bins=len(reference_pdf), density=True)
        
        # Normalize histogram to create an empirical probability distribution (PDF)
        hist_pdf = hist_counts / np.sum(hist_counts)
        
        # Ensure both hist_pdf and reference_pdf have the same length
        min_len = min(len(hist_pdf), len(reference_pdf))
        hist_pdf, reference_pdf = hist_pdf[:min_len], reference_pdf[:min_len]
        
        # Avoid zeros in reference_pdf by adding epsilon, and renormalize
        reference_pdf = reference_pdf + epsilon
        reference_pdf /= np.sum(reference_pdf)  # Normalize to ensure it remains a PDF
        hist_pdf = hist_pdf + epsilon
        hist_pdf /= np.sum(hist_pdf)  # Normalize as well

        # Compute KL divergence
        kl_divergence = np.sum(hist_pdf * np.log(hist_pdf / reference_pdf))
        
        return kl_divergence

    @staticmethod
    def compute_bhattacharyya_distance(noise_values: np.ndarray, reference_pdf: np.ndarray, epsilon: float = 1e-10) -> float:
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
        # Compute a histogram of the noise values
        hist_counts, _ = np.histogram(noise_values, bins=len(reference_pdf), density=True)
        
        # Normalize the histogram to create a probability distribution (PDF)
        hist_pdf = hist_counts / np.sum(hist_counts)
        
        # Ensure both hist_pdf and reference_pdf have the same length
        min_len = min(len(hist_pdf), len(reference_pdf))
        hist_pdf, reference_pdf = hist_pdf[:min_len], reference_pdf[:min_len]
        
        # Compute the Bhattacharyya coefficient with an epsilon to prevent log(0)
        bc_coefficient = np.sum(np.sqrt(hist_pdf * reference_pdf)) + epsilon  # Bhattacharyya coefficient with stability term
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

