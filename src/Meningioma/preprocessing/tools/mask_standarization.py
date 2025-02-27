"""
This module provides utility functions for reading and processing 3D volumes 
in NRRD format using SimpleITK, including:
- Removing the first channel (or any specified channel) from multi-component volumes
- Ensuring a segmentation NRRD has the same geometry/dimensions as a reference volume
"""

import SimpleITK as sitk
import numpy as np
from typing import Tuple, Optional


def remove_first_channel(volume_path: str, channel: int = 0) -> np.ndarray:
    """
    Load a multi-channel (vector) NRRD volume and extract a single channel
    as a scalar image, returning it as a NumPy array.

    Args:
        volume_path (str):
            Path to the NRRD volume file.
        channel (int, optional):
            Which channel index to extract. Defaults to 0 (i.e., the first channel).

    Returns:
        np.ndarray:
            A 3D NumPy array (shape = [Z, Y, X]) representing the specified channel
            of the volume in float32 format.

    Example:
        >>> channel0_array = remove_first_channel("my_vector_volume.nrrd", channel=0)
        >>> channel1_array = remove_first_channel("my_vector_volume.nrrd", channel=1)
    """
    # Read the NRRD as a SimpleITK image
    img = sitk.ReadImage(volume_path)

    # Extract the specified channel
    scalar_img = sitk.VectorIndexSelectionCast(img, channel, sitk.sitkFloat32)

    # Convert to NumPy array (shape [Z, Y, X])
    array_3d = sitk.GetArrayFromImage(scalar_img)

    # Transpose accordingly to return [X, Y, Z]:

    array_3d = array_3d.transpose(1, 2, 0)

    return array_3d


def match_mask_and_volume_dimensions(
    volume_path: str,
    seg_path: str,
    out_seg_path: Optional[str] = "./output_segmentation.nrrd",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a volume NRRD and a segmentation NRRD, ensuring they match in geometry
    (dimensions, spacing, origin, direction). If mismatched, resample the segmentation
    to align with the volume. Returns both as NumPy arrays.

    Args:
        volume_path (str):
            Path to the .nrrd volume file.
        seg_path (str):
            Path to the .nrrd segmentation file.
        out_seg_path (Optional[str], optional):
            If provided, writes the (potentially resampled) segmentation
            to this path. Defaults to "./output_segmentation.nrrd".

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            (volume_array, seg_array)
            Each is a 3D NumPy array with shape [Z, Y, X].
            The segmentation is guaranteed to match the volume geometry.

    Example:
        >>> volume_array, seg_array = match_mask_and_volume_dimensions(
        ...     "T2_P2.nrrd",
        ...     "T2_P2_seg.nrrd",
        ...     out_seg_path="T2_P2_seg_matched.nrrd"
        ... )
    """
    # Read both as SimpleITK images (preserves all geometric metadata)
    volume_img = sitk.ReadImage(volume_path)
    seg_img = sitk.ReadImage(seg_path)

    # Check if size, origin, spacing, and direction match
    same_size = volume_img.GetSize() == seg_img.GetSize()
    same_spacing = volume_img.GetSpacing() == seg_img.GetSpacing()
    same_origin = volume_img.GetOrigin() == seg_img.GetOrigin()
    same_direction = volume_img.GetDirection() == seg_img.GetDirection()

    if not (same_size and same_spacing and same_origin and same_direction):
        # Resample the segmentation onto the volume's grid
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(volume_img)
        # Use NearestNeighbor for label images to preserve discrete values
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        seg_img = resampler.Execute(seg_img)

    # Optionally write out the updated segmentation
    if out_seg_path is not None:
        sitk.WriteImage(seg_img, out_seg_path)

    # Convert both images to NumPy arrays (shape [Z, Y, X])
    volume_array = sitk.GetArrayFromImage(volume_img)
    seg_array = sitk.GetArrayFromImage(seg_img)

    # Transpose accordingly to return [X, Y, Z]:
    volume_array = volume_array.transpose(1, 2, 0)
    seg_array = seg_array.transpose(1, 2, 0)

    return volume_array, seg_array
