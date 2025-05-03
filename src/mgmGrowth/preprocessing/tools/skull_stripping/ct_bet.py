#!/usr/bin/env python3
"""
ct_bet.py

Provides a function to run a brain extraction pipeline on CT images, including
enhanced skull removal. This is a modified version of the original CT brain
extraction pipeline from the following paper:

https://www.sciencedirect.com/science/article/pii/S1053811915002700 
"""

from typing import Tuple
import time 
import numpy as np

import SimpleITK as sitk
from mgmGrowth.preprocessing.tools.skull_stripping.fsl_bet import fsl_bet_brain_extraction

def ct_brain_extraction(
    image_sitk: sitk.Image,
    apply_smoothing: bool = False,
    bet_fractional_intensity: float = 0.01,
    perform_skull_erosion: bool = False,
    erosion_radius: int = 1,
    verbose: bool = False
) -> Tuple[sitk.Image, sitk.Image]:
    """
    Enhanced CT brain extraction with better skull removal.

    Source:
        https://www.sciencedirect.com/science/article/pii/S1053811915002700
        But added the perform_skull_erosion steps to further remove skull remnants (only if needed)
    
    Steps:
    1. Threshold to brain tissue range (0-100 HU)
    2. Optional: Apply 3D Gaussian smoothing (sigma = 1 mm³)
    3. If smoothing applied, re-threshold to 0-100 HU
    4. Apply FSL BET with specified fractional intensity
    5. Create brain mask and apply morphological operations to remove skull remnants
    6. Apply the refined mask to the original image
    
    Args:
        image_sitk: SimpleITK CT image
        apply_smoothing: Whether to apply Gaussian smoothing
        bet_fractional_intensity: BET fractional intensity (default: 0.01)
        perform_skull_erosion: Whether to perform additional erosion to remove skull
        erosion_radius: Radius for erosion operation
        verbose: Whether to print processing information
        
    Returns:
        Tuple of (brain_extracted_image, brain_mask)
    """
    if verbose:
        t0 = time.time()
        print(f"Starting enhanced CT brain extraction (smoothing: {apply_smoothing}, "
              f"BET FI: {bet_fractional_intensity}, skull erosion: {perform_skull_erosion})")
    
    # Step 1: Threshold to brain tissue range (0-100 HU)
    threshold_filter = sitk.ThresholdImageFilter()
    threshold_filter.SetLower(0)
    threshold_filter.SetUpper(100)
    threshold_filter.SetOutsideValue(0)
    brain_range_image = threshold_filter.Execute(image_sitk)
    
    if verbose:
        print(f"  Initial thresholding completed in {time.time() - t0:.2f} seconds")
        t0 = time.time()
    
    # Step 2: Optional Gaussian smoothing
    if apply_smoothing:
        # Apply 3D Gaussian filter with sigma = 1mm³
        smoothing_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        smoothing_filter.SetSigma(1.0)
        smoothed_image = smoothing_filter.Execute(brain_range_image)
        
        # Step 3: Re-threshold after smoothing
        threshold_filter = sitk.ThresholdImageFilter()
        threshold_filter.SetLower(0)
        threshold_filter.SetUpper(100)
        threshold_filter.SetOutsideValue(0)
        brain_range_image = threshold_filter.Execute(smoothed_image)
        
        if verbose:
            print(f"  Smoothing and re-thresholding completed in {time.time() - t0:.2f} seconds")
            t0 = time.time()
    
    # Step 4: Apply FSL BET with specified fractional intensity
    try:
        bet_result, bet_mask = fsl_bet_brain_extraction(
            input_image_sitk=brain_range_image,
            frac=bet_fractional_intensity,
            robust=True,
            vertical_gradient=0,
            skull=False,
            verbose=verbose
        )
        
        if verbose:
            print(f"  BET completed in {time.time() - t0:.2f} seconds")
            t0 = time.time()
        
        # Step 5: Create brain mask with additional processing to remove skull
        # Threshold the result to get values > 0 HU
        binary_threshold = sitk.BinaryThresholdImageFilter()
        binary_threshold.SetLowerThreshold(0.1)  # Just above 0
        binary_threshold.SetUpperThreshold(float('inf'))
        binary_threshold.SetInsideValue(1)
        binary_threshold.SetOutsideValue(0)
        brain_mask = binary_threshold.Execute(bet_result)
        
        # Apply additional processing to remove skull remnants
        if perform_skull_erosion:
            # First erode to remove thin connections to skull
            erode_filter = sitk.BinaryErodeImageFilter()
            erode_filter.SetKernelRadius(erosion_radius)
            erode_filter.SetForegroundValue(1)
            erode_filter.SetBackgroundValue(0)
            eroded_mask = erode_filter.Execute(brain_mask)
            
            # Keep only the largest connected component (the brain)
            cc_filter = sitk.ConnectedComponentImageFilter()
            label_map = cc_filter.Execute(eroded_mask)
            
            # Find the largest component
            label_stats = sitk.LabelShapeStatisticsImageFilter()
            label_stats.Execute(label_map)
            
            largest_label = 0
            largest_size = 0
            for label in label_stats.GetLabels():
                if label > 0 and label_stats.GetPhysicalSize(label) > largest_size:
                    largest_size = label_stats.GetPhysicalSize(label)
                    largest_label = label
            
            # Create a mask with only the largest component
            binary_filter = sitk.BinaryThresholdImageFilter()
            binary_filter.SetLowerThreshold(largest_label)
            binary_filter.SetUpperThreshold(largest_label)
            binary_filter.SetInsideValue(1)
            binary_filter.SetOutsideValue(0)
            brain_core = binary_filter.Execute(label_map)
            
            # Dilate the core brain mask slightly, but stay away from edges
            dilate_filter = sitk.BinaryDilateImageFilter()
            dilate_filter.SetKernelRadius(erosion_radius)
            dilate_filter.SetForegroundValue(1)
            dilate_filter.SetBackgroundValue(0)
            dilated_core = dilate_filter.Execute(brain_core)
            
            # Combine with original mask using AND operation to keep only valid brain regions
            brain_mask = sitk.And(dilated_core, brain_mask)
            
            if verbose:
                print(f"  Enhanced skull removal completed in {time.time() - t0:.2f} seconds")
                t0 = time.time()
        
        # Fill holes in the mask
        hole_filler = sitk.BinaryFillholeImageFilter()
        hole_filler.SetForegroundValue(1)
        brain_mask = hole_filler.Execute(brain_mask)
        
        # Apply the final mask to the original image
        masking_filter = sitk.MaskImageFilter()
        brain_extracted_image = masking_filter.Execute(image_sitk, brain_mask)
        
        if verbose:
            print(f"  Final mask creation completed in {time.time() - t0:.2f} seconds")
            stats = sitk.StatisticsImageFilter()
            stats.Execute(brain_mask)
            brain_volume_ml = stats.GetSum() * np.prod(image_sitk.GetSpacing()) / 1000
            print(f"  Estimated brain volume: {brain_volume_ml:.2f} ml")
            
        return brain_extracted_image, brain_mask
        
    except Exception as e:
        if verbose:
            print(f"  Error during brain extraction: {str(e)}")
        raise

def zero_non_brain_pixels(
    image_sitk: sitk.Image, 
    mask_sitk: sitk.Image, 
    verbose: bool = False
) -> sitk.Image:
    """
    Zero out all pixels outside the brain mask.
    
    Args:
        image_sitk: SimpleITK image to be masked
        mask_sitk: SimpleITK binary brain mask
        verbose: Whether to print processing information
        
    Returns:
        SimpleITK image with all non-brain pixels set to zero
    """
    if verbose:
        t0 = time.time()
        print("Zeroing out non-brain pixels...")
        
    # Apply the mask using SimpleITK's masking operation
    masking_filter = sitk.MaskImageFilter()
    masked_image = masking_filter.Execute(image_sitk, mask_sitk)
    
    if verbose:
        print(f"  Non-brain pixels zeroed in {time.time() - t0:.2f} seconds")
        stats = sitk.StatisticsImageFilter()
        stats.Execute(masked_image)
        non_zero_count = sitk.GetArrayFromImage(masked_image).astype(bool).sum()
        total_count = np.prod(masked_image.GetSize())
        print(f"  Non-zero pixels: {non_zero_count} ({non_zero_count/total_count*100:.2f}% of total)")
    
    return masked_image

# Test the CT brain extraction pipeline with visualizations and statistics
def test_ct_brain_extraction(
    image_sitk, 
    slice_idx=None, 
    apply_smoothing=True, 
    test_all_fi=True
):
    """
    Test the CT brain extraction pipeline and visualize results.
    
    Args:
        image_sitk: SimpleITK CT image
        slice_idx: Slice index to display (None for middle slice)
        apply_smoothing: Whether to apply smoothing in the pipeline
        test_all_fi: Test all fractional intensity values from the paper
    """
    import matplotlib.pyplot as plt

    if slice_idx is None:
        slice_idx = image_sitk.GetSize()[2] // 2
    
    # Get original image statistics
    stats_orig = sitk.StatisticsImageFilter()
    stats_orig.Execute(image_sitk)
    orig_min = stats_orig.GetMinimum()
    orig_max = stats_orig.GetMaximum()
    orig_mean = stats_orig.GetMean()
    orig_std = stats_orig.GetSigma()
    
    # Get original image as numpy array for display
    orig_array = sitk.GetArrayFromImage(image_sitk)
    
    # Display statistics
    print("Original Image Statistics:")
    print(f"  Min: {orig_min:.2f}, Max: {orig_max:.2f}")
    print(f"  Mean: {orig_mean:.2f}, StdDev: {orig_std:.2f}")
    print(f"  Size: {image_sitk.GetSize()}")
    print(f"  Spacing: {image_sitk.GetSpacing()}")
    
    results = []
    
    # Test different fractional intensity values
    if test_all_fi:
        fi_values = [0.35, 0.1, 0.01]
    else:
        fi_values = [0.01]  # Just use the recommended value
        
    for fi in fi_values:
        print(f"\nTesting BET with fractional intensity {fi}:")
        try:
            brain_extracted, brain_mask = ct_brain_extraction(
                image_sitk=image_sitk,
                apply_smoothing=apply_smoothing,
                bet_fractional_intensity=fi,
                verbose=True
            )
            
            # Get statistics for extracted brain
            stats_brain = sitk.StatisticsImageFilter()
            stats_brain.Execute(brain_extracted)
            brain_min = stats_brain.GetMinimum()
            brain_max = stats_brain.GetMaximum()
            brain_mean = stats_brain.GetMean()
            brain_std = stats_brain.GetSigma()
            
            # Get statistics for brain mask
            stats_mask = sitk.StatisticsImageFilter()
            stats_mask.Execute(brain_mask)
            mask_volume = stats_mask.GetSum() * np.prod(image_sitk.GetSpacing()) / 1000  # ml
            mask_percentage = stats_mask.GetSum() / np.prod(image_sitk.GetSize()) * 100  # %
            
            # Get arrays for display
            brain_array = sitk.GetArrayFromImage(brain_extracted)
            mask_array = sitk.GetArrayFromImage(brain_mask)
            
            print(f"Brain-Extracted Image Statistics (FI={fi}):")
            print(f"  Min: {brain_min:.2f}, Max: {brain_max:.2f}")
            print(f"  Mean: {brain_mean:.2f}, StdDev: {brain_std:.2f}")
            print(f"  Brain volume: {mask_volume:.2f} ml ({mask_percentage:.2f}% of image)")
            
            results.append((fi, brain_extracted, brain_mask, brain_array, mask_array))
            
        except Exception as e:
            print(f"  Error with FI={fi}: {str(e)}")
    
    # Create visualization
    if not results:
        print("No successful extractions to visualize")
        return
    
    # Plot results
    n_results = len(results)
    fig, axs = plt.subplots(n_results + 1, 3, figsize=(15, 5 * (n_results + 1)))
    plt.subplots_adjust(hspace=0.4)
    
    # If only one result, wrap axes in list for consistent indexing
    if n_results == 1:
        if not isinstance(axs[0], np.ndarray):
            axs = [axs[0:3], axs[3:6]]
    
    # Title for the figure
    fig.suptitle(f"CT Brain Extraction Results (Slice {slice_idx})", fontsize=16)
    
    # First row: Original image
    axs[0][0].imshow(orig_array[slice_idx], cmap='gray')
    axs[0][0].set_title(f"Original CT\nRange: [{orig_min:.1f}, {orig_max:.1f}]")
    
    # Histogram of original image
    histdata, bins = np.histogram(orig_array.flatten(), bins=100)
    axs[0][1].bar((bins[:-1] + bins[1:]) / 2, histdata, width=np.diff(bins), alpha=0.7)
    axs[0][1].set_title(f"Histogram\nMean: {orig_mean:.1f}, StdDev: {orig_std:.1f}")
    axs[0][1].set_xlim(-1000, 1000)  # Common HU range for display
    
    # Histogram focused on brain tissue range
    axs[0][2].bar((bins[:-1] + bins[1:]) / 2, histdata, width=np.diff(bins), alpha=0.7)
    axs[0][2].set_title("Brain Tissue Range")
    axs[0][2].set_xlim(-20, 120)  # Focus on 0-100 HU range
    axs[0][2].axvspan(0, 100, alpha=0.2, color='green')
    
    # Each subsequent row: Results from a different FI value
    for i, (fi, _, _, brain_array, mask_array) in enumerate(results, start=1):
        # Brain-extracted image
        axs[i][0].imshow(brain_array[slice_idx], cmap='gray')
        axs[i][0].set_title(f"Brain-Extracted (FI={fi})")
        
        # Brain mask
        axs[i][1].imshow(mask_array[slice_idx], cmap='gray')
        axs[i][1].set_title(f"Brain Mask\nVolume: {stats_mask.GetSum() * np.prod(image_sitk.GetSpacing()) / 1000:.1f} ml")
        
        # Overlay of mask on original image
        axs[i][2].imshow(orig_array[slice_idx], cmap='gray')
        masked = np.ma.masked_where(mask_array[slice_idx] == 0, mask_array[slice_idx])
        axs[i][2].imshow(masked, cmap='jet', alpha=0.5)
        axs[i][2].set_title(f"Mask Overlay\n{mask_percentage:.1f}% of image")
    
    # Remove axis ticks
    for ax_row in axs:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('ct_brain_extraction_test.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results