import numpy as np
import nibabel as nib
from degrade.degrade import fwhm_units_to_voxel_space, fwhm_needed


def normalize(x, a=-1, b=1):
    orig_min = x.min()
    orig_max = x.max()

    numer = (x - orig_min) * (b - a)
    denom = orig_max - orig_min

    return a + numer / denom, orig_min, orig_max


def inv_normalize(x, orig_min, orig_max, a=-1, b=1):
    tmp = x - a
    tmp = tmp * (orig_max - orig_min)
    tmp = tmp / (b - a)
    tmp += orig_min
    return tmp


def parse_image(image_file, slice_thickness=None, normalize_image=False):
    """
    Open the image volume file, and return pertinent information:
    - The image array as a numpy array
    - The "scale" of the anisotropy (this is the slice separation)
    - The LR axis
    - The FWHM of the PSF (this is the slice thickness, which can be provided as an argument)
    - The header of the image file
    - The affine matrix of the image file
    """
    obj = nib.load(image_file)
    voxel_size = tuple(float(v) for v in obj.header.get_zooms())
    image = obj.get_fdata(dtype=np.float32)

    # x, y, and z are the spatial physical measurement sizes
    lr_axis = np.argmax(voxel_size)
    z = voxel_size[lr_axis]
    xy = list(voxel_size)
    xy.remove(z)
    xyz = (xy[0], xy[1], z)
    x, y, z = xyz

    # Exit if the provided image is isotropic through-plane
    assert (
        x != z and y != z
    ), f'Worst resolution found {z} matches one of the better resolutions {x} or {y}; image is "isotropic" and cannot be run through SMORE.'
    # Exit if the provided image is anisotropic in-plane
    assert np.isclose(
        x, y, atol=1e-2
    ), f"In-plane resolution {x} and {y} are not close to equal; anisotropic in-plane is not supported by SMORE"

    slice_separation = float(z / x)

    if slice_thickness is None:
        slice_thickness = z
    blur_fwhm_voxels = fwhm_units_to_voxel_space(fwhm_needed(x, slice_thickness), x)

    if normalize_image:
        image, orig_min, orig_max = normalize(image, 0, 1)
    else:
        orig_min = None
        orig_max = None

    return (
        image,
        slice_separation,
        lr_axis,
        blur_fwhm_voxels,
        obj.header,
        obj.affine,
        orig_min,
        orig_max,
    )


def lr_axis_to_z(img, lr_axis):
    """
    Orient the image volume such that the low-resolution axis
    is in the "z" axis.
    """
    if lr_axis == 0:
        return img.transpose(1, 2, 0)
    elif lr_axis == 1:
        return img.transpose(2, 0, 1)
    elif lr_axis == 2:
        return img


def z_axis_to_lr_axis(img, lr_axis):
    """
    Orient the image volume such that the "z" axis
    is back to the original low-resolution axis
    """
    if lr_axis == 0:
        return img.transpose(2, 0, 1)
    elif lr_axis == 1:
        return img.transpose(1, 2, 0)
    elif lr_axis == 2:
        return img
