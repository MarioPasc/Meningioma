#!/usr/bin/env python3
"""
This script provides two functions:
1. pad_in_plane_with_info: Resizes and/or pads (or letterboxes) a 3D SimpleITK image (and an optional mask)
   to a desired in-plane target size (e.g., 512x512). It returns the processed image(s) along with an
   extra_info dictionary containing the transformation parameters needed to reverse the letterboxing.
2. reverse_letterbox: Receives a 3D predicted mask (as a SimpleITK image) and the extra_info dictionary,
   and returns the mask cropped and inversely resampled to the original in-plane dimensions.

These functions are intended for pipelines where images are first resampled to a physical voxel size (e.g., 1×1×1 mm³)
and then, for CNN processing, letterboxed/padded to a fixed digital size. If physical measurements need to be made on the CNN output,
the reverse transformation can be applied.
  
Usage (command-line):
    python pad_letterbox_with_reversal.py input_image.nii.gz --target-x 512 --target-y 512 --interpolate
"""

from typing import Tuple, Optional, Union, Dict, Any
import SimpleITK as sitk


def pad_in_plane(
    image_sitk: sitk.Image,
    mask_sitk: Optional[sitk.Image] = None,
    target_size_xy: Tuple[int, int] = (512, 512),
    constant_value: float = 0.0,
    mask_constant_value: float = 0.0,
    interpolate: bool = False,
    interpolation_method: int = sitk.sitkBSpline,
    mask_interpolation_method: int = sitk.sitkNearestNeighbor,
) -> Union[
    Tuple[sitk.Image, Dict[str, Any]], Tuple[sitk.Image, sitk.Image, Dict[str, Any]]
]:
    """
    Resizes and/or pads a 3D SITK image (and optional mask) so that its in-plane dimensions match target_size_xy.

    If interpolate=False, only zero-padding is performed. If True, letterboxing is applied:
      - The image is first scaled (keeping aspect ratio) so that its larger dimension matches the target,
        then padded on the smaller dimension.

    Extra information is returned in a dictionary (extra_info) containing:
        - 'orig_size': original (X, Y) dimensions before letterboxing
        - 'scale_factor': the factor used for resampling (if interpolate=True, else 1.0)
        - 'pad_lower': (pad_x_before, pad_y_before)
        - 'pad_upper': (pad_x_after, pad_y_after)

    These parameters allow the reversal of the letterboxing (i.e. cropping and inverse resampling).

    Args:
        image_sitk (sitk.Image): A 3D image (size interpreted as (X, Y, Z)).
        mask_sitk (Optional[sitk.Image]): A 3D mask matching the image dimensions.
        target_size_xy (Tuple[int,int]): Desired in-plane size (X, Y), e.g., (512, 512).
        constant_value (float): Value for padded regions in the image.
        mask_constant_value (float): Value for padded regions in the mask.
        interpolate (bool): If True, performs letterboxing (scaling + padding). If False, only pads.
        interpolation_method: SITK interpolation method for image resampling.
        mask_interpolation_method: SITK interpolation method for mask resampling.

    Returns:
        If mask_sitk is None:
            (processed_image, extra_info)
        Otherwise:
            (processed_image, processed_mask, extra_info)
    """
    # Get current size and spacing.
    current_size = image_sitk.GetSize()  # (X, Y, Z)
    current_x, current_y, current_z = current_size
    current_spacing = image_sitk.GetSpacing()

    # Save original in-plane size.
    orig_size = (current_x, current_y)

    # Verify mask dimensions if provided.
    if mask_sitk is not None:
        if mask_sitk.GetSize() != current_size:
            raise ValueError(
                f"Mask size {mask_sitk.GetSize()} doesn't match image size {current_size}."
            )

    target_x, target_y = target_size_xy
    processed_img = image_sitk
    processed_mask = mask_sitk
    scale_factor = 1.0  # default if no interpolation

    # If interpolation (letterboxing) is requested.
    if interpolate:
        scale_x = target_x / current_x
        scale_y = target_y / current_y
        scale_factor = min(scale_x, scale_y)  # scale factor to preserve aspect ratio

        new_x = int(current_x * scale_factor)
        new_y = int(current_y * scale_factor)

        # Compute new spacing to keep the physical size similar.
        new_spacing = (
            current_spacing[0] * (current_x / new_x),
            current_spacing[1] * (current_y / new_y),
            current_spacing[2],
        )

        # Resample the image.
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetSize((new_x, new_y, current_z))
        resample_filter.SetOutputDirection(image_sitk.GetDirection())
        resample_filter.SetOutputOrigin(image_sitk.GetOrigin())
        resample_filter.SetTransform(sitk.Transform())
        resample_filter.SetDefaultPixelValue(constant_value)
        resample_filter.SetInterpolator(interpolation_method)
        processed_img = resample_filter.Execute(image_sitk)

        # Resample the mask if provided.
        if mask_sitk is not None:
            mask_resample_filter = sitk.ResampleImageFilter()
            mask_resample_filter.SetOutputSpacing(new_spacing)
            mask_resample_filter.SetSize((new_x, new_y, current_z))
            mask_resample_filter.SetOutputDirection(mask_sitk.GetDirection())
            mask_resample_filter.SetOutputOrigin(mask_sitk.GetOrigin())
            mask_resample_filter.SetTransform(sitk.Transform())
            mask_resample_filter.SetDefaultPixelValue(mask_constant_value)
            mask_resample_filter.SetInterpolator(mask_interpolation_method)
            processed_mask = mask_resample_filter.Execute(mask_sitk)

        # Update current_x, current_y for padding calculation.
        current_x, current_y = new_x, new_y

    # Calculate padding needed to reach target size.
    pad_x = max(0, target_x - current_x)
    pad_y = max(0, target_y - current_y)
    before_x = pad_x // 2
    after_x = pad_x - before_x
    before_y = pad_y // 2
    after_y = pad_y - before_y
    pad_lower = [before_x, before_y, 0]
    pad_upper = [after_x, after_y, 0]

    # Apply padding to image.
    if pad_x > 0 or pad_y > 0:
        pad_filter = sitk.ConstantPadImageFilter()
        pad_filter.SetPadLowerBound(pad_lower)
        pad_filter.SetPadUpperBound(pad_upper)
        pad_filter.SetConstant(constant_value)
        processed_img = pad_filter.Execute(processed_img)

        # Apply padding to mask if provided.
        if processed_mask is not None:
            mask_pad_filter = sitk.ConstantPadImageFilter()
            mask_pad_filter.SetPadLowerBound(pad_lower)
            mask_pad_filter.SetPadUpperBound(pad_upper)
            mask_pad_filter.SetConstant(mask_constant_value)
            processed_mask = mask_pad_filter.Execute(processed_mask)

    # Prepare extra_info dictionary.
    extra_info = {
        "orig_size": orig_size,  # original in-plane size before letterboxing
        "scale_factor": scale_factor,  # scaling factor applied (1.0 if interpolate=False)
        "pad_lower": (before_x, before_y),  # padding added before (X, Y)
        "pad_upper": (after_x, after_y),  # padding added after (X, Y)
        "target_size": target_size_xy,  # desired in-plane size (target_x, target_y)
    }

    if processed_mask is not None:
        return processed_img, processed_mask, extra_info  # type: ignore
    return processed_img, extra_info


def reverse_letterbox(
    letterboxed_img: sitk.Image,
    extra_info: Dict[str, Any],
    interpolation_method: int = sitk.sitkBSpline,
) -> sitk.Image:
    """
    Reverses the letterboxing transformation performed by pad_in_plane_with_info.

    The function does the following:
      1. Removes the padding (crops the image) using the pad_lower and pad_upper values.
      2. If letterboxing (scaling) was applied (scale_factor != 1.0), rescales the cropped image
         back to the original in-plane dimensions (orig_size) while preserving the physical spacing.

    Args:
        letterboxed_img (sitk.Image): The padded/letterboxed image (e.g. CNN output mask).
        extra_info (Dict[str, Any]): Dictionary containing:
            - 'orig_size': original (X, Y) size before letterboxing,
            - 'scale_factor': scaling factor used (1.0 if no interpolation was performed),
            - 'pad_lower': (before_x, before_y) padding added,
            - 'pad_upper': (after_x, after_y) padding added,
            - 'target_size': the final target in-plane size.
        interpolation_method: Interpolator for resampling during reverse scaling.
                              Use sitk.sitkNearestNeighbor for masks.

    Returns:
        sitk.Image: The image restored to the original in-plane dimensions.
    """
    # Get current size of the letterboxed image.
    current_size = letterboxed_img.GetSize()  # (X, Y, Z)
    current_x, current_y, current_z = current_size

    # Retrieve padding amounts.
    before_x, before_y = extra_info.get("pad_lower", (0, 0))
    after_x, after_y = extra_info.get("pad_upper", (0, 0))

    # Compute size after removing padding.
    cropped_x = current_x - (before_x + after_x)
    cropped_y = current_y - (before_y + after_y)

    # Crop the image: define region of interest.
    # We use the RegionOfInterest filter.
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex([before_x, before_y, 0])
    roi_filter.SetSize([cropped_x, cropped_y, current_z])
    cropped_img = roi_filter.Execute(letterboxed_img)

    # If a scaling factor was applied, then reverse it.
    scale_factor = extra_info.get("scale_factor", 1.0)
    orig_size = extra_info.get("orig_size", (cropped_x, cropped_y))

    if scale_factor != 1.0:
        # We need to resample the cropped image back to the original in-plane dimensions.
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetSize((orig_size[0], orig_size[1], current_z))
        resample_filter.SetOutputDirection(letterboxed_img.GetDirection())
        resample_filter.SetOutputOrigin(letterboxed_img.GetOrigin())
        # To preserve the physical size, compute new spacing.
        # The cropped image physical size is (cropped_x * new_spacing_x, cropped_y * new_spacing_y)
        # Let new_spacing = (cropped_size / orig_size)
        current_spacing = letterboxed_img.GetSpacing()
        new_spacing = (
            current_spacing[0] * (cropped_x / orig_size[0]),
            current_spacing[1] * (cropped_y / orig_size[1]),
            current_spacing[2],
        )
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetInterpolator(interpolation_method)
        restored_img = resample_filter.Execute(cropped_img)
    else:
        restored_img = cropped_img

    return restored_img


def main():
    import os
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply letterboxing (scale + pad) to a 3D image and then reverse it."
    )
    parser.add_argument("input_image", help="Path to the input image (NIfTI).")
    parser.add_argument("--mask", help="Path to the optional mask (NIfTI).")
    parser.add_argument("--output-dir", default=".", help="Output directory.")
    parser.add_argument("--target-x", type=int, default=512, help="Target X size.")
    parser.add_argument("--target-y", type=int, default=512, help="Target Y size.")
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Apply letterboxing (scale + pad) instead of pure padding.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.input_image).split(".")[0]
    padded_image_path = os.path.join(args.output_dir, f"{base_name}_padded.nii.gz")
    reversed_image_path = os.path.join(args.output_dir, f"{base_name}_reversed.nii.gz")

    # Read input image
    img = sitk.ReadImage(args.input_image)

    # If mask is provided, read it
    mask = sitk.ReadImage(args.mask) if args.mask else None

    # Apply padding/letterboxing with info
    if mask:
        processed_img, processed_mask, extra_info = pad_in_plane(
            img,
            mask,
            target_size_xy=(args.target_x, args.target_y),
            interpolate=args.interpolate,
        )
        # For demonstration, save both processed image and mask.
        sitk.WriteImage(processed_img, padded_image_path)
        sitk.WriteImage(
            processed_mask,
            os.path.join(args.output_dir, f"{base_name}_mask_padded.nii.gz"),
        )
    else:
        processed_img, extra_info = pad_in_plane(
            img,
            target_size_xy=(args.target_x, args.target_y),
            interpolate=args.interpolate,
        )
        sitk.WriteImage(processed_img, padded_image_path)

    print("Padding/letterboxing completed. Extra info:")
    print(extra_info)

    # Now, simulate receiving a CNN prediction in the padded space.
    # For this demonstration, we'll simply use the processed image as a placeholder.
    predicted_img = processed_img  # In practice, this is your CNN output mask.

    # Reverse the letterboxing/padding.
    restored_img = reverse_letterbox(
        predicted_img, extra_info, interpolation_method=sitk.sitkNearestNeighbor
    )
    sitk.WriteImage(restored_img, reversed_image_path)
    print(
        f"Restored image (with padding removed and inverse scaling) saved to: {reversed_image_path}"
    )


if __name__ == "__main__":
    main()
