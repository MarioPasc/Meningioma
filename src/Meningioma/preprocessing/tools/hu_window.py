#!/usr/bin/env python3
"""
hu_window.py

Provides a function to apply a window/level transformation to a CT image
using SimpleITK.

- Brain window (80/40)
    - window_width = 80
    - window_level_value = 40

- Soft tissue window (400/40)
    - window_width = 400
    - window_level_value = 40
"""

import numpy as np
import SimpleITK as sitk

def apply_window(
    image_sitk: sitk.Image, 
    window_width: float = 400, 
    window_level: float = 40, 
    verbose: bool = False
) -> sitk.Image:
    """
    Apply window settings to a CT image.
    
    Args:
        image_sitk: SimpleITK CT image
        window_width: Width of the window in HU
        window_level: Center level of the window in HU
        verbose: Whether to print processing information
        
    Returns:
        SimpleITK image with window applied
    """
    if verbose:
        print(f"Applying window with width={window_width}, level={window_level}")
    
    # Calculate window parameters
    min_value = window_level - window_width/2
    max_value = window_level + window_width/2
    
    # Get array from SimpleITK image
    image_array = sitk.GetArrayFromImage(image_sitk)
    
    # Apply window/level transformation
    windowed_array = np.clip(image_array, min_value, max_value)
    windowed_array = (windowed_array - min_value) / (max_value - min_value)
    
    # Convert back to SimpleITK image
    result = sitk.GetImageFromArray(windowed_array)
    result.CopyInformation(image_sitk)
    
    if verbose:
        stats = sitk.StatisticsImageFilter()
        stats.Execute(result)
        print(f"  After windowing - Min: {stats.GetMinimum():.4f}, Max: {stats.GetMaximum():.4f}")
    
    return result