from scipy.special import kl_div
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class Metrics:

    @staticmethod
    def compute_kl_divergence(noise_values: np.ndarray, kde_pdf: np.ndarray) -> float:
        """
        Compute the Kullback-Leibler divergence between an empirical noise distribution
        and a KDE-estimated distribution.

        Parameters:
        - noise_values (np.ndarray): Original noise values outside the bounding box.
        - kde_pdf (np.ndarray): Discrete KDE-estimated PDF values.

        Returns:
        - float: KL divergence between the empirical and KDE distributions.
        """
        # Compute a histogram of the noise values
        hist_counts, bin_edges = np.histogram(noise_values, bins=len(kde_pdf), density=True)
        
        # Calculate the midpoints of the bins to match kde_pdf points
        hist_pdf = hist_counts / np.sum(hist_counts)  # Normalize histogram to get a PDF
        
        # Ensure both hist_pdf and kde_pdf are of the same length
        if len(hist_pdf) != len(kde_pdf):
            min_len = min(len(hist_pdf), len(kde_pdf))
            hist_pdf, kde_pdf = hist_pdf[:min_len], kde_pdf[:min_len]
        
        # Compute KL divergence using scipy's kl_div function, and sum to get total divergence
        kl_divergence = np.sum(kl_div(hist_pdf, kde_pdf))
        
        return kl_divergence

    @staticmethod
    def compute_bhattacharyya_distance(noise_values: np.ndarray, kde_pdf: np.ndarray) -> float:
        """
        Compute the Bhattacharyya distance between an empirical noise distribution
        and a KDE-estimated distribution.

        Parameters:
        - noise_values (np.ndarray): Original noise values outside the bounding box.
        - kde_pdf (np.ndarray): Discrete KDE-estimated PDF values.

        Returns:
        - float: Bhattacharyya distance between the empirical and KDE distributions.
        """
        # Compute a histogram of the noise values
        hist_counts, _ = np.histogram(noise_values, bins=len(kde_pdf), density=True)
        
        # Normalize the histogram to create a probability distribution (PDF)
        hist_pdf = hist_counts / np.sum(hist_counts)
        
        # Ensure both hist_pdf and kde_pdf have the same length
        if len(hist_pdf) != len(kde_pdf):
            min_len = min(len(hist_pdf), len(kde_pdf))
            hist_pdf, kde_pdf = hist_pdf[:min_len], kde_pdf[:min_len]
        
        # Compute the Bhattacharyya distance
        bc_coefficient = np.sum(np.sqrt(hist_pdf * kde_pdf))  # Bhattacharyya coefficient
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

