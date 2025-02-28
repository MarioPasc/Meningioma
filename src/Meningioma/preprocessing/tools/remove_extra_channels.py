#!/usr/bin/env python3
"""
remove_channel.py
Extracts a specified channel from a multi-component (vector) SimpleITK Image
and returns it as a single-component sitk.Image.
"""

import collections
import SimpleITK as sitk
import os
import nrrd  # type: ignore
from typing import Union, Tuple
import numpy as np


def remove_first_channel(
    nrrd_path: Union[str, os.PathLike], channel: int, verbose: bool = False
) -> Tuple[sitk.Image, collections.OrderedDict]:
    """
    Reads a nrrd file from the given path, extracts the specified channel (if multi-channel),
    and updates the header to reflect a 3D scalar image (ignoring the channel dimension).

    Parameters:
        nrrd_path (Union[str, os.PathLike]): Path to the input nrrd file.
        channel (int): The index of the channel to extract from the multi-channel image.
                       If the image is single-channel, it is returned unchanged.

    Returns:
        Tuple[sitk.Image, collections.OrderedDict]: A tuple containing the extracted scalar SimpleITK image
        and the updated nrrd header.
    """
    # Read the nrrd file using pynrrd.
    data, hdr = nrrd.read(str(nrrd_path))

    if verbose:
        print(f"Input volume shape: {data.shape}")
        print(
            f"Input hdr fields:\n - dimensions: {hdr['dimension']}\n - sizes: {hdr['sizes']}\n - space directions: {hdr['space directions']}\n - kinds: {hdr['kinds']}\n - space origin: {hdr['space origin']}"
        )

    # If the image has multiple components, extract the specified channel.
    if len(data.shape) > 3:
        # Convert the numpy array to a SimpleITK image.
        data_reordered = np.moveaxis(data, 0, -1)
        volume_image = sitk.GetImageFromArray(data_reordered, isVector=True)

        if channel >= volume_image.GetNumberOfComponentsPerPixel():
            raise ValueError(
                f"The image has only {volume_image.GetNumberOfComponentsPerPixel()} channels, "
                f"but channel {channel} was requested."
            )
        scalar_img = sitk.VectorIndexSelectionCast(
            volume_image, channel, sitk.sitkVectorFloat32
        )

        # Update header to ignore the channel dimension.
        hdr["dimension"] = 3
        hdr["sizes"] = hdr["sizes"][1:]  # Remove channel dimension size.
        hdr["space directions"] = hdr["space directions"][
            1:
        ]  # Remove channel's space direction.
        hdr["kinds"] = hdr["kinds"][1:]  # Remove channel kind designation.
        if "space origin" in hdr and len(hdr["space origin"]) > 3:
            hdr["space origin"] = hdr["space origin"][1:]  # Remove channel origin
        if verbose:
            print(
                f"Output volume shape: height: {scalar_img.GetHeight()}, width: {scalar_img.GetWidth()}, depth: {scalar_img.GetDepth()}"
            )
            print(
                f"Input hdr fields:\n - dimensions: {hdr['dimension']}\n - sizes: {hdr['sizes']}\n - space directions: {hdr['space directions']}\n - kinds: {hdr['kinds']}\n - space origin: {hdr['space origin']}"
            )
    else:
        scalar_img = sitk.GetImageFromArray(data)

    return scalar_img, hdr


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python remove_channel.py <vector_image.nrrd> [channel_index]")
        sys.exit(1)

    vector_path = sys.argv[1]
    channel_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    try:
        scalar_img, _ = remove_first_channel(vector_path, channel_idx)

        # Example: write output
        out_path = "channel_extracted.nrrd"
        sitk.WriteImage(scalar_img, out_path)  # type: ignore
        print(f"Channel {channel_idx} extracted and saved to {out_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
