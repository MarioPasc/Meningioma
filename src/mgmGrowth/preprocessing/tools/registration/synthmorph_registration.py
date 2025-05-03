#!/usr/bin/env python3
# filepath: /home/mariopasc/Python/Projects/Meningioma/Meningioma/registration/synthmorph_register.py

"""
SynthMorph-based registration module for medical images.

This module provides functionality to register 3D medical images using
the SynthMorph neural network model for deformable registration.
"""

import os
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import voxelmorph as vxm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SynthMorphRegister")


class SynthMorphRegister:
    """Class for handling SynthMorph-based 3D medical image registration."""

    SYNTHMORPH_MODEL_URL = "https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/shapes-dice-vel-3-res-8-16-32-256f.h5"
    
    def __init__(
        self,
        output_dir: str,
        common_shape: Tuple[int, int, int] = (160, 192, 160),
        model_features: Optional[Tuple[List[int], List[int]]] = None,
        int_steps: int = 5,
        int_resolution: int = 2,
        svf_resolution: int = 2,
    ) -> None:
        """
        Initialize the SynthMorph registration module.
        
        Args:
            output_dir: Directory where registered images and model weights will be stored
            common_shape: Common shape for registration (must be divisible by 16)
            model_features: UNet features for the model, defaults to ([256]*4, [256]*6)
            int_steps: Number of integration steps for the velocity field
            int_resolution: Resolution factor for the integration
            svf_resolution: Resolution factor for the SVF
        """
        self.output_dir = os.path.abspath(output_dir)
        self.common_shape = common_shape
        self.model_features = model_features or ([256] * 4, [256] * 6)
        self.int_steps = int_steps
        self.int_resolution = int_resolution
        self.svf_resolution = svf_resolution
        self.model = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Path to model weights
        self.model_path = os.path.join(self.output_dir, 'shapes-dice-vel-3-res-8-16-32-256f.h5')
        
        # Ensure model weights are available
        self._download_model_if_needed()
        
    def _download_model_if_needed(self) -> None:
        """Download the SynthMorph model weights if not already available."""
        if not os.path.exists(self.model_path):
            logger.info(f"Downloading SynthMorph model weights to {self.model_path}")
            try:
                subprocess.run(
                    ["curl", "-o", self.model_path, self.SYNTHMORPH_MODEL_URL],
                    check=True
                )
                logger.info("Model download complete")
            except subprocess.SubprocessError as e:
                logger.error(f"Failed to download model: {e}")
                raise RuntimeError(f"Failed to download SynthMorph model: {e}")
                
    def _load_model(self) -> None:
        """Load the SynthMorph neural network model."""
        if self.model is not None:
            return
            
        logger.info("Initializing SynthMorph model")
        model = vxm.networks.VxmDense(
            nb_unet_features=self.model_features,
            int_steps=self.int_steps,
            int_resolution=self.int_resolution,
            svf_resolution=self.svf_resolution,
            inshape=self.common_shape,
        )
        self.model = tf.keras.Model(model.inputs, model.references.pos_flow)
        
        try:
            self.model.load_weights(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise RuntimeError(f"Failed to load SynthMorph model weights: {e}")
            
    @staticmethod
    def normalize(image_array: np.ndarray) -> np.ndarray:
        """
        Normalize the image array to [0,1] and add batch and channel dimensions.
        
        Args:
            image_array: 3D image array
            
        Returns:
            Normalized 5D image array with shape [batch, h, w, d, channel]
        """
        image_array = np.float32(image_array)
        image_array -= image_array.min()
        image_array /= image_array.max()
        return image_array[None, ..., None]
        
    @staticmethod
    def resize_to_model_compatible(
        image: sitk.Image, 
        target_shape: Optional[Tuple[int, int, int]] = None
    ) -> Tuple[sitk.Image, Tuple[int, int, int]]:
        """
        Resize image to dimensions compatible with the model (divisible by 16).
        
        Args:
            image: SimpleITK image to resize
            target_shape: Target shape or None to automatically calculate
            
        Returns:
            Tuple of (resized SimpleITK image, target shape used)
        """
        if target_shape is None:
            # Make dimensions divisible by 16
            shape = image.GetSize()
            target_shape = tuple(((dim + 15) // 16) * 16 for dim in shape)

        resample = sitk.ResampleImageFilter()
        resample.SetSize(target_shape)
        resample.SetOutputOrigin(image.GetOrigin())
        spacing = [orig_sz * orig_sp / targ_sz for orig_sz, orig_sp, targ_sz in 
                 zip(image.GetSize(), image.GetSpacing(), target_shape)]
        resample.SetOutputSpacing(spacing)
        resample.SetOutputDirection(image.GetDirection())
        resample.SetInterpolator(sitk.sitkLinear)
        resample.SetDefaultPixelValue(0)
        
        return resample.Execute(image), target_shape
        
    def register_image_to_atlas(
        self,
        moving_image_path: str,
        atlas_image_path: str,
        segmentation_path: Optional[str] = None,
        output_prefix: str = "registered",
        file_suffix: str = "",
        two_pass: bool = True
    ) -> Dict[str, str]:
        """
        Register a moving image to an atlas image using SynthMorph.
        
        Args:
            moving_image_path: Path to the moving image
            atlas_image_path: Path to the atlas (fixed) image
            segmentation_path: Optional path to a segmentation to transform
            output_prefix: Prefix for output filenames
            file_suffix: Suffix to add to output filenames
            two_pass: Whether to use two-pass registration for better alignment
            
        Returns:
            Dictionary with paths to registered images and transformation
        """
        # Ensure model is loaded
        self._load_model()
        
        # Load images
        logger.info(f"Loading atlas image from {atlas_image_path}")
        atlas_img = sitk.ReadImage(atlas_image_path)
        
        logger.info(f"Loading moving image from {moving_image_path}")
        moving_img = sitk.ReadImage(moving_image_path)
        
        # Store original metadata
        orig_shape = moving_img.GetSize()
        orig_spacing = moving_img.GetSpacing()
        
        logger.info(f"Original moving image shape: {orig_shape}, spacing: {orig_spacing}")
        logger.info(f"Using common registration shape: {self.common_shape}")
        
        # Resize images to common shape
        atlas_resampled, _ = self.resize_to_model_compatible(atlas_img, self.common_shape)
        moving_resampled, _ = self.resize_to_model_compatible(moving_img, self.common_shape)
        
        # Convert to arrays and normalize
        atlas_array = sitk.GetArrayFromImage(atlas_resampled)
        moving_array = sitk.GetArrayFromImage(moving_resampled)
        
        atlas_norm = self.normalize(atlas_array)
        moving_norm = self.normalize(moving_array)
        
        # First registration pass
        logger.info("Running first registration pass")
        warp_field = self.model.predict((moving_norm, atlas_norm))
        
        # Apply transformation
        moved = vxm.layers.SpatialTransformer(fill_value=0)((moving_norm, warp_field))
        
        # Second registration pass if requested
        if two_pass:
            logger.info("Running second registration pass")
            resid = self.model.predict((moved, atlas_norm))
            combined_warp = vxm.layers.ComposeTransform()((warp_field, resid))
            final_moved = vxm.layers.SpatialTransformer(fill_value=0)((moving_norm, combined_warp))
        else:
            combined_warp = warp_field
            final_moved = moved
        
        # Create output paths
        output_files = {}
        registered_output_path = os.path.join(
            self.output_dir, f"{output_prefix}{file_suffix}_registered_to_atlas.nii.gz"
        )
        registered_warp_path = os.path.join(
            self.output_dir, f"{output_prefix}{file_suffix}_to_atlas_warp.nii.gz"
        )
        
        # Convert results back to SimpleITK images with metadata
        final_moved_sitk = sitk.GetImageFromArray(np.squeeze(final_moved[0]))
        final_moved_sitk.SetSpacing(moving_resampled.GetSpacing())
        final_moved_sitk.SetOrigin(moving_resampled.GetOrigin())
        final_moved_sitk.SetDirection(moving_resampled.GetDirection())
        
        # Save registered volume
        sitk.WriteImage(final_moved_sitk, registered_output_path)
        output_files["volume"] = registered_output_path
        
        # Save the warp field
        warp_field_sitk = sitk.GetImageFromArray(np.squeeze(combined_warp[0]), isVector=True)
        warp_field_sitk.SetSpacing(moving_resampled.GetSpacing())
        warp_field_sitk.SetOrigin(moving_resampled.GetOrigin())
        warp_field_sitk.SetDirection(moving_resampled.GetDirection())
        sitk.WriteImage(warp_field_sitk, registered_warp_path)
        output_files["warp"] = registered_warp_path
        
        # Process segmentation if provided
        if segmentation_path:
            logger.info(f"Applying transformation to segmentation from {segmentation_path}")
            seg_img = sitk.ReadImage(segmentation_path)
            
            # Resize segmentation to common shape
            seg_resampled, _ = self.resize_to_model_compatible(seg_img, self.common_shape)
            seg_array = sitk.GetArrayFromImage(seg_resampled)
            seg_norm = self.normalize(seg_array)
            
            # Apply transformation with nearest neighbor interpolation for segmentations
            seg_moved = vxm.layers.SpatialTransformer(
                fill_value=0, 
                interp_method='nearest'
            )((seg_norm, combined_warp))
            
            # Save registered segmentation
            seg_registered_path = os.path.join(
                self.output_dir, f"{output_prefix}{file_suffix}_seg_registered_to_atlas.nii.gz"
            )
            
            # Convert back to SimpleITK image with metadata
            seg_moved_sitk = sitk.GetImageFromArray(np.squeeze(seg_moved[0]))
            seg_moved_sitk.SetSpacing(seg_resampled.GetSpacing())
            seg_moved_sitk.SetOrigin(seg_resampled.GetOrigin())
            seg_moved_sitk.SetDirection(seg_resampled.GetDirection())
            sitk.WriteImage(seg_moved_sitk, seg_registered_path)
            output_files["segmentation"] = seg_registered_path
            
        logger.info(f"Registration complete. Files saved to {self.output_dir}")
        return output_files
        
    def register_all_to_atlas(
        self,
        data_dict: Dict[str, Dict[str, Any]],
        atlas_path: str,
        patient_id: str,
        pulse_key: str = "output_paths"
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        """
        Register all modalities in a data dictionary to an atlas.
        
        Args:
            data_dict: Dictionary with modalities and image paths
            atlas_path: Path to the atlas image
            patient_id: Patient identifier for output naming
            pulse_key: Key in data_dict that contains volume and segmentation paths
            
        Returns:
            Updated data dictionary with registration results
        """
        logger.info(f"Starting registration of all modalities to atlas")
        
        for pulse, pulse_data in data_dict.items():
            logger.info(f"Registering {pulse} to atlas")
            
            moving_path = pulse_data[pulse_key]["vol"]
            seg_path = pulse_data[pulse_key]["seg"]
            
            results = self.register_image_to_atlas(
                moving_image_path=moving_path,
                atlas_image_path=atlas_path,
                segmentation_path=seg_path,
                output_prefix=pulse,
                file_suffix=f"_P{patient_id}",
                two_pass=True
            )
            
            # Update data dictionary with registration results
            if "registered_paths" not in pulse_data:
                pulse_data["registered_paths"] = {}
                
            pulse_data["registered_paths"] = {
                "vol": results["volume"],
                "warp": results["warp"],
                "seg": results.get("segmentation")
            }
            
        return data_dict
        
    def visualize_results(
        self,
        data_dict: Dict[str, Dict[str, Any]],
        pulses: List[str],
        patient_id: str,
        slice_indices: Optional[Dict[str, int]] = None,
        output_filename: Optional[str] = None
    ) -> None:
        """
        Visualize registration results in axial, sagittal, and coronal views.
        
        Args:
            data_dict: Dictionary with registration data
            pulses: List of pulse types to visualize
            patient_id: Patient identifier
            slice_indices: Dictionary with slice indices for each view
            output_filename: Optional custom filename for the visualization
        """
        import matplotlib.pyplot as plt
        
        # Set up figure
        fig, axes = plt.subplots(3, len(pulses), figsize=(5*len(pulses), 12))
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        
        # Default to middle slices if not specified
        if slice_indices is None:
            slice_indices = {'axial': None, 'sagittal': None, 'coronal': None}
        
        # Set figure background to black
        fig.patch.set_facecolor('black')
        
        # Load all images first to determine common intensity range
        images = []
        for pulse in pulses:
            img_path = data_dict[pulse]["registered_paths"]["vol"]
            img = sitk.ReadImage(img_path)
            img_array = sitk.GetArrayFromImage(img)
            images.append(img_array)
        
        # Find global min and max for consistent windowing
        global_min = min(img.min() for img in images)
        global_max = max(img.max() for img in images)
        
        # Determine middle slices from first image if not specified
        if slice_indices['axial'] is None:
            slice_indices['axial'] = images[0].shape[0] // 2
        if slice_indices['sagittal'] is None:
            slice_indices['sagittal'] = images[0].shape[2] // 2
        if slice_indices['coronal'] is None:
            slice_indices['coronal'] = images[0].shape[1] // 2
        
        logger.info(f"Visualizing at slices - Axial: {slice_indices['axial']}, "
              f"Sagittal: {slice_indices['sagittal']}, Coronal: {slice_indices['coronal']}")
        
        # Plot each pulse type in columns
        for col, pulse in enumerate(pulses):
            img_array = images[col]
            
            # Axial view (top row)
            ax = axes[0, col]
            ax.imshow(img_array[slice_indices['axial'], :, :], 
                      cmap='gray', vmin=global_min, vmax=global_max)
            ax.set_title(f"{pulse} - Axial", color='white', fontsize=14)
            ax.axis('off')
            
            # Sagittal view (middle row)
            ax = axes[1, col]
            ax.imshow(img_array[:, :, slice_indices['sagittal']], 
                      cmap='gray', vmin=global_min, vmax=global_max)
            ax.set_title(f"{pulse} - Sagittal", color='white', fontsize=14)
            ax.axis('off')
            
            # Coronal view (bottom row)
            ax = axes[2, col]
            ax.imshow(img_array[:, slice_indices['coronal'], :], 
                      cmap='gray', vmin=global_min, vmax=global_max)
            ax.set_title(f"{pulse} - Coronal", color='white', fontsize=14)
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save the visualization
        if output_filename is None:
            output_filename = f"registered_visualization_P{patient_id}.png"
            
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, facecolor='black', bbox_inches='tight', dpi=150)
        logger.info(f"Visualization saved to: {output_path}")
        
        return fig


def main() -> None:
    """Command line interface for the module."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SynthMorph-based registration of 3D medical images")
    parser.add_argument("--moving", required=True, help="Path to moving image")
    parser.add_argument("--fixed", required=True, help="Path to fixed/atlas image")
    parser.add_argument("--segmentation", help="Optional path to segmentation to transform")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prefix", default="registered", help="Prefix for output files")
    parser.add_argument("--suffix", default="", help="Suffix for output files")
    parser.add_argument("--single-pass", action="store_true", help="Use single-pass registration")
    
    args = parser.parse_args()
    
    # Initialize registration module
    registrator = SynthMorphRegister(output_dir=args.output_dir)
    
    # Register image
    registrator.register_image_to_atlas(
        moving_image_path=args.moving,
        atlas_image_path=args.fixed,
        segmentation_path=args.segmentation,
        output_prefix=args.prefix,
        file_suffix=args.suffix,
        two_pass=not args.single_pass
    )


if __name__ == "__main__":
    main()

"""
This is an example script of how to create the data dictionary

from Meningioma.preprocessing.tools.remove_extra_channels import remove_first_channel
from Meningioma.preprocessing.tools.casting import cast_volume_and_mask
from Meningioma.preprocessing.tools.nrrd_to_nifti import nifti_write_3d
from Meningioma.preprocessing.tools.denoise_susan import denoise_susan
from Meningioma.preprocessing.tools.bias_field_corr_n4 import n4_bias_field_correction, generate_brain_mask_sitk
from Meningioma.preprocessing.tools.skull_stripping.fsl_bet import fsl_bet_brain_extraction

import os

import SimpleITK as sitk

import matplotlib.pyplot as plt

PATH = "/home/mariopasc/Python/Datasets/Meningiomas"
ATLAS = "/home/mariopasc/Python/Datasets/Meningiomas/ATLAS/sri24_spm8/templates/T1_brain.nii"
INPUT_PATH = os.path.join(PATH, "Meningioma_Adquisition")
OUTPUT_PATH = os.path.join(PATH, "output")
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
MODALITY = "RM"
PULSES = ["T1", "T2", "SUSC"]
PATIENT = "1"

data = {}
for pulse in PULSES:
    vol_path = os.path.join(INPUT_PATH, MODALITY, pulse, f"P{PATIENT}", f"{pulse}_P{PATIENT}.nrrd")
    seg_path = os.path.join(INPUT_PATH, MODALITY, pulse, f"P{PATIENT}", f"{pulse}_P{PATIENT}_seg.nrrd")
    data[pulse] = {"vol": vol_path, "seg": seg_path}

    print("=====================================")
    print(f"Processing {pulse}")
    print("=====================================")

    # Load and save paths
    vol_path = data[pulse]["vol"]
    seg_path = data[pulse]["seg"]
    
    # Remove first channel
    vol, hdr = remove_first_channel(nrrd_path=vol_path, channel=0, verbose=True)
    seg, hdr_seg = remove_first_channel(nrrd_path=seg_path, channel=0, verbose=True)
    # Cast volume and mask
    vol, seg = cast_volume_and_mask(volume_img=vol, mask_img=seg)

    # n4 bias field correction
    _, mask_brain = generate_brain_mask_sitk(volume_sitk=vol)
    vol = n4_bias_field_correction(volume_sitk=vol, verbose=True)

    # save files
    nifti_write_3d([vol, hdr], os.path.join(OUTPUT_PATH, f"{pulse}_P{PATIENT}"))
    nifti_write_3d([seg, hdr_seg], os.path.join(OUTPUT_PATH, f"{pulse}_P{PATIENT}_seg"))
    # Save
    data[pulse] = {
        "input_paths": {
            "vol": vol_path,
            "seg": seg_path,
        },
        "arrays": {
            "vol": sitk.GetArrayFromImage(vol),
            "seg": sitk.GetArrayFromImage(seg),
        },
        "output_paths": {
            "vol": os.path.join(OUTPUT_PATH, f"{pulse}_P{PATIENT}.nii.gz"),
            "seg": os.path.join(OUTPUT_PATH, f"{pulse}_P{PATIENT}_seg.nii.gz"),
        }
        
    }
"""