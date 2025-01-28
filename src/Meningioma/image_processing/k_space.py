from typing import Tuple

import numpy as np

"""
This script contains all the functions needed to obtain some approximated k-space data from the original 
MRI image in order to incorporate this information to our blind estimation of the underlying noise distribution. 
Therefore, it includes:
    1. The extraction of the phase (theta) of the k-space, by computing the arctan of the imaginary part divided by the real part. 
       This estimation of the phase is naive, since it does not take into account several factors that influence the final generation
       of the image, such as the usage of subsampling techniques of the k-space to make the adquisition process of the MRI faster, such 
       as the SENSE and GRAPPA algorithms. 
    2. The extraction of the estimated real and imaginary parts, given the image.  

"""


def get_phase_from_kspace(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate k-space from an image and extract the phase data.

    Parameters
    ----------
    image : np.ndarray
        Input MRI image data (2D).

    Returns
    -------
    phase : np.ndarray
        Phase data from the FFT of the image.
    k_space : np.ndarray
        The 2D FFT (shifted) of the image.
    """
    k_space = np.fft.fftshift(np.fft.fft2(image))
    phase = np.angle(k_space)
    return phase, k_space


def get_real_imag(
    magnitude: np.ndarray, phase: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert magnitude data into synthetic complex data by applying a phase.

    Parameters
    ----------
    magnitude : np.ndarray
        Magnitude image data.
    phase : np.ndarray
        Phase data.

    Returns
    -------
    (real_part, imag_part) : Tuple[np.ndarray, np.ndarray]
        Real and imaginary parts of the complex data.
    """
    real_part = magnitude * np.cos(phase)
    imag_part = magnitude * np.sin(phase)
    return real_part, imag_part
