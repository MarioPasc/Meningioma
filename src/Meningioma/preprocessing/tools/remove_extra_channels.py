#!/usr/bin/env python3
"""
remove_channel.py
Extracts a specified channel from a multi-component (vector) SimpleITK Image
and returns it as a single-component sitk.Image.
"""

import SimpleITK as sitk


def remove_first_channel(volume_image: sitk.Image, channel: int) -> sitk.Image:
    """
    Extract a specific channel from a volume image that is provided as a sitk.Image.

    Parameters:
        volume_image (sitk.Image): The input volume image.
        channel (int): The index of the channel to extract.

    Returns:
        sitk.Image: The extracted scalar image corresponding to the specified channel, cast to sitkFloat32.
    """
    # Check if the image is multi-component.
    if volume_image.GetNumberOfComponentsPerPixel() == 1:
        # The image is already scalar; no channel extraction is needed.
        return volume_image

    # Ensure the requested channel is valid.
    if channel >= volume_image.GetNumberOfComponentsPerPixel():
        raise ValueError(
            f"The image has only {volume_image.GetNumberOfComponentsPerPixel()} channels, "
            f"but channel {channel} was requested."
        )

    # Extract the specified channel and cast the output to float32.
    scalar_img = sitk.VectorIndexSelectionCast(
        volume_image, channel, sitk.sitkVectorFloat32
    )
    volume_image = sitk.Cast(volume_image, sitk.sitkInt16)
    return scalar_img


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python remove_channel.py <vector_image.nrrd> [channel_index]")
        sys.exit(1)

    vector_path = sys.argv[1]
    channel_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    try:
        vector_img = sitk.ReadImage(vector_path)
        scalar_img = remove_first_channel(vector_img, channel_idx)

        # Example: write output
        out_path = "channel_extracted.nrrd"
        sitk.WriteImage(scalar_img, out_path)
        print(f"Channel {channel_idx} extracted and saved to {out_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
