import numpy as np
from Meningioma.utils.statistical_manipulation import Stats


class Metrics:

    @staticmethod
    def compute_kullback_leibler_divergence(
        noise_values: np.ndarray,
        reference_pdf: np.ndarray,
        x_values: np.ndarray,
        epsilon: float = 1e-10,
        only_kb_divergence: bool = True,
    ) -> float:
        """
        Compute the Kullback-Leibler divergence between an empirical noise distribution
        and a KDE-estimated distribution, adding epsilon to avoid division by zero.

        Parameters:
        - noise_values (np.ndarray): Original noise values outside the bounding box.
        - reference_pdf (np.ndarray): Discrete KDE-estimated PDF values evaluated at x_values.
        - x_values (np.ndarray): The x-values over which the PDFs are evaluated.
        - epsilon (float): Small value to prevent division by zero.
        - only_kb_divergence (bool): If True, only computes the Kullback-Leibler divergence between two given PMF
                                     If False, assumes the `noise_values` is an empirical probability mass function, and calls Stats.histogram_based_pmf_estimation
                                     also, assumes `reference_pdf` is a continuous Probability Density Function, calling Stats.approximate_pmf_from_pdf.

        Returns:
        - float: KL divergence between the empirical and KDE distributions.
        """
        if not only_kb_divergence:
            # Discretize the PDF
            reference_prob_masses = Stats.approximate_pmf_from_pdf(
                reference_pdf=reference_pdf,
                x_values=x_values,
                epsilon=epsilon,
            )

            # Estimate the empirical probability distribution from the discrete background noise values
            noise_prob_masses = Stats.histogram_based_pmf_estimation(
                noise_values=noise_values,
                x_values=x_values,
                epsilon=epsilon,
            )

        else:
            # Treat input arrays as probability distributions and normalize them
            noise_prob_masses = noise_values / np.sum(noise_values)
            reference_prob_masses = reference_pdf / np.sum(reference_pdf)

            # Avoid division by zero and ensure positivity by adding epsilon
            noise_prob_masses += epsilon
            reference_prob_masses += epsilon

            # Re-normalize to ensure both are proper PMFs
            noise_prob_masses /= np.sum(noise_prob_masses)
            reference_prob_masses /= np.sum(reference_prob_masses)

        # Compute the Kullback-Leibler divergence
        return np.sum(
            noise_prob_masses * np.log(noise_prob_masses / reference_prob_masses)
        )

    @staticmethod
    def compute_jensen_shannon_divergence(
        noise_values: np.ndarray,
        reference_pdf: np.ndarray,
        x_values: np.ndarray,
        epsilon: float = 1e-10,
    ) -> float:
        """
        Compute the Jensen-Shannon divergence between an empirical noise distribution
        and a KDE-estimated distribution, adding epsilon to avoid division by zero.

        Parameters:
        - noise_values (np.ndarray): Original noise values outside the bounding box.
        - reference_pdf (np.ndarray): Discrete KDE-estimated PDF values evaluated at x_values.
        - x_values (np.ndarray): The x-values over which the PDFs are evaluated.
        - epsilon (float): Small value to prevent division by zero.

        Returns:
        - float: Jensen-Shannon divergence divergence between the empirical and KDE distributions.

        Notes:
        - More info on Jensen-Shannon Divergence: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
        - More info on Mixture Distributions: https://en.wikipedia.org/wiki/Mixture_distribution
        """

        # Discretize the PDF
        reference_prob_masses = Stats.approximate_pmf_from_pdf(
            reference_pdf=reference_pdf, x_values=x_values, epsilon=epsilon
        )

        # Estimate the empirical probability distribution from the discrete background noise values
        noise_prob_masses = Stats.histogram_based_pmf_estimation(
            noise_values=noise_values, x_values=x_values, epsilon=epsilon
        )

        # Compute the mixture distribution
        mixture_distribution = 0.5 * (reference_prob_masses + noise_prob_masses)

        # Compute the Kullback-Leibler divergence for each PMF distribution with the mixture

        # KL(P ∥ M): Compares the distribution P (reference_prob_masses) to M (the mixture_distribution).
        # KL(Q ∥ M): Compares the distribution Q (noise_prob_masses) to M (the mixture_distribution).

        kl_reference_mixture = Metrics.compute_kullback_leibler_divergence(
            noise_values=reference_prob_masses,  # Original reference distribution
            reference_pdf=mixture_distribution,  # Mixture distribution
            x_values=x_values,
            epsilon=epsilon,
            only_kb_divergence=True,
        )

        kl_noise_mixture = Metrics.compute_kullback_leibler_divergence(
            noise_values=noise_prob_masses,  # Original noise distribution
            reference_pdf=mixture_distribution,  # Mixture distribution
            x_values=x_values,
            epsilon=epsilon,
            only_kb_divergence=True,
        )
        return 0.5 * (kl_reference_mixture + kl_noise_mixture)

    @staticmethod
    def compute_bhattacharyya_distance(
        noise_values: np.ndarray,
        reference_pdf: np.ndarray,
        x_values: np.ndarray,
        epsilon: float = 1e-10,
        only_bhattacharyya_distance: bool = True,
    ) -> float:
        """
        Compute the Bhattacharyya distance between an empirical noise distribution
        and a KDE-estimated distribution.

        Parameters:
        - noise_values (np.ndarray): Original noise values outside the bounding box.
        - reference_pdf (np.ndarray): Discrete KDE-estimated PDF values.
        - epsilon (float): Small value to prevent log(0) issues.
        - only_bhattacharyya_distance (bool): If True, only computes the Bhattacharyya distance between two given PMF
                                              If False, assumes the `noise_values` is an empirical probability mass function, and calls Stats.histogram_based_pmf_estimation
                                              also, assumes `reference_pdf` is a continuous Probability Density Function, calling Stats.approximate_pmf_from_pdf.
        Returns:
        - float: Bhattacharyya distance between the empirical and KDE distributions.
        """
        if not only_bhattacharyya_distance:
            # Discretize the PDF
            reference_prob_masses = Stats.approximate_pmf_from_pdf(
                reference_pdf=reference_pdf,
                x_values=x_values,
                epsilon=epsilon,
            )

            # Estimate the empirical probability distribution from the discrete background noise values
            noise_prob_masses = Stats.histogram_based_pmf_estimation(
                noise_values=noise_values,
                x_values=x_values,
                epsilon=epsilon,
            )
        else:
            # Treat input arrays as probability distributions and normalize them
            noise_prob_masses = noise_values / np.sum(noise_values)
            reference_prob_masses = reference_pdf / np.sum(reference_pdf)

            # Avoid division by zero and ensure positivity by adding epsilon
            noise_prob_masses += epsilon
            reference_prob_masses += epsilon

            # Re-normalize to ensure both are proper PMFs
            noise_prob_masses /= np.sum(noise_prob_masses)
            reference_prob_masses /= np.sum(reference_prob_masses)

        # Compute the Bhattacharyya coefficient
        bc_coefficient = np.sum(np.sqrt(noise_prob_masses * reference_prob_masses))
        # Compute the Bhattacharyya distance
        bhattacharyya_distance = -np.log(bc_coefficient)
        return bhattacharyya_distance
