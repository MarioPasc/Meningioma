#!/usr/bin/env python

import os
import sys
import numpy as np
import cv2

from Meningioma import ImageProcessing

def main():
    output_npz_path = "/home/mario/Python/Datasets/Meningiomas/npz"

    patient = "P50"
    pulse = "T1"
    input_filepath = os.path.join(output_npz_path, patient, f"{patient}_{pulse}.npz")
        
    # Run the segmentation – using the "li" threshold method.
    print("Performing 3D segmentation on the volume...")
    volume, mask = ImageProcessing.segment_3d_volume(input_filepath, threshold_method="li")
    print("Segmentation complete.")

    # Check that the volume and mask have the same shape.
    if volume.shape != mask.shape:
        print("Error: Volume and mask shapes do not match.")
        sys.exit(1)
    
    # Create a combined array with shape (H, W, S, 2) where:
    #   - combined[..., 0] is the original volume
    #   - combined[..., 1] is the mask (converted to the same dtype as volume)
    combined = np.stack((volume, mask.astype(volume.dtype)), axis=-1)

    # Determine output paths.
    # The output NPZ file will be saved in the same folder as the input file.
    # The PNG folder will be created in that same directory.
    base_dir = os.path.dirname(os.path.abspath(input_filepath))
    base_name = os.path.splitext(os.path.basename(input_filepath))[0]
    output_npz = os.path.join(base_dir, f"{base_name}_segmentation.npz")
    overlay_folder = os.path.join(base_dir, f"{base_name}_overlays")
    
    # Save the combined array to an NPZ file.
    np.savez_compressed(output_npz, data=combined)
    print(f"Saved segmentation data to {output_npz}")
    
    # Create the folder for the overlay PNG images if it doesn’t already exist.
    os.makedirs(overlay_folder, exist_ok=True)
    
    # Assume the volume has shape (H, W, S) where S is the number of slices.
    H, W, S = volume.shape
    print("Generating overlay images for each slice...")
    
    for i in range(S):
        # Extract the i-th slice from volume and mask.
        slice_vol = volume[..., i]
        slice_mask = mask[..., i]
        
        # Normalize the slice to the range 0-255 and convert to uint8.
        slice_norm = cv2.normalize(slice_vol, None, 0, 255, cv2.NORM_MINMAX)
        slice_norm = slice_norm.astype(np.uint8)
        
        # Convert the grayscale slice to a BGR image.
        slice_bgr = cv2.cvtColor(slice_norm, cv2.COLOR_GRAY2BGR)
        
        # Create a blended image: only for pixels where the mask is True,
        # blend the original pixel with pure green ([0, 255, 0]) at 50% transparency.
        blended = slice_bgr.copy().astype(np.float32)
        # Get the indices of the mask (a 2D boolean array) where the mask is True.
        mask_indices = np.where(slice_mask)
        green_color = np.array([0, 255, 0], dtype=np.float32)
        # For these pixels, blend the original with green.
        blended[mask_indices] = 0.5 * blended[mask_indices] + 0.5 * green_color
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Save the blended image as a PNG file.
        png_filename = os.path.join(overlay_folder, f"slice_{i:03d}.png")
        cv2.imwrite(png_filename, blended)
    
    print(f"Overlay images saved in folder: {overlay_folder}")

if __name__ == "__main__":
    main()
