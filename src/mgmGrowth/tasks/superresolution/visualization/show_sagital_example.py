import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

def load_nifti_file(filepath):
    """Load a NIfTI file and return the data array."""
    try:
        nii = nib.load(filepath)
        return nii.get_fdata()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_sagittal_slice(data, slice_idx):
    """Extract sagittal slice from 3D data."""
    if data is None or len(data.shape) < 3:
        return None
    return data[slice_idx, :, :]

import logging
from typing import Tuple

import numpy as np

def replicate_to_isotropic(
    vol: np.ndarray,
    zooms: Tuple[float, float, float],
    target: float = 1.0,
) -> np.ndarray:
    """
    Repeat voxels along each axis so that the resulting grid *appears* to have
    `target` mm isotropic spacing (default 1 mm).

    Parameters
    ----------
    vol : np.ndarray
        3-D array ordered exactly like `nibabel` yields it.
    zooms : (Δx, Δy, Δz)
        Physical voxel size in mm, as given by `nii.header.get_zooms()[:3]`.
    target : float, optional
        Desired isotropic spacing in mm. **Must** divide all zooms exactly.

    Returns
    -------
    np.ndarray
        Upsampled volume; values are *copied*, never interpolated.

    Raises
    ------
    ValueError
        If any zoom is not an integer multiple of `target`.
    """
    reps = [int(round(z / target)) for z in zooms]
    if not np.allclose(np.array(reps) * target, zooms, atol=1e-3):
        raise ValueError(f"{zooms} are not integer multiples of {target} mm")

    logging.debug("Replication factors: %s", reps)
    out = vol
    for ax, k in enumerate(reps):
        out = np.repeat(out, k, axis=ax) if k > 1 else out
    return out


def create_individual_visualization(file_path, data, resolution, pulse, subject_name, slice_number, output_path):
    """
    Create a single visualization for one resolution and pulse sequence.
    
    Args:
        data: 3D numpy array of the medical image
        resolution: Resolution name (hr, 3mm, 5mm, 7mm)
        pulse: Pulse sequence name (t1c, t2w, t2f)
        subject_name: Subject name
        slice_number: Sagittal slice number
        output_path: Full path for the output PDF file
    """
    data = load_nifti_file(file_path)
    if data is not None:
        zooms = nib.load(file_path).header.get_zooms()[:3]

        # -- extract sagittal slice *first* --
        sag = get_sagittal_slice(data, slice_number)

        # -- replicate only in-plane (ax=0: rows, ax=1: cols) --
        #    For a sagittal slice, axes are (Y, Z) in BraTS convention.
        sag_iso = replicate_to_isotropic(
            sag,
            zooms[1:],          # Y- and Z-resolution
            target=1.0,
        )

        fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
        ax.imshow(
            np.fliplr(np.rot90(sag_iso, k=3)),
            cmap="gray",
            origin="lower",
            interpolation="nearest",   # <— keep pixels sharp
        )
            
        # Remove all axes, labels, and margins
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        # Remove all padding and margins
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        
        # Save as PDF with no padding
        plt.savefig(output_path, format='pdf', facecolor='black', 
                    bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Saved: {output_path}")
        plt.close(fig)
        return True
    else:
        print(f"Invalid slice {slice_number} for {resolution}-{pulse}")
        return False

def create_visualization(subject_name, super_resolution_path, slice_number, output_dir):
    """
    Create visualization for a subject across different resolutions and pulse sequences.
    
    Args:
        subject_name: Name of the subject (e.g., 'BraTS-MEN-00018-000')
        super_resolution_path: Path to super_resolution folder
        slice_number: Sagittal slice number to visualize
        output_dir: Output directory for PDF files
    """
    
    # Create subject-specific output directory
    subject_slice_dir = os.path.join(output_dir, f"{subject_name}_slice_{slice_number}")
    os.makedirs(subject_slice_dir, exist_ok=True)
    
    # Define resolutions and pulse sequences
    resolutions = ['hr', '3mm', '5mm', '7mm']
    pulse_sequences = ['t1c', 't2w', 't2f']
    
    # Map resolution names to actual paths
    resolution_paths = {
        'hr': 'subset',
        '3mm': 'low_res/3mm',
        '5mm': 'low_res/5mm',
        '7mm': 'low_res/7mm'
    }
    
    successful_saves = 0
    total_expected = len(resolutions) * len(pulse_sequences)
    
    for resolution in resolutions:
        for pulse in pulse_sequences:
            # Construct file path
            file_path = os.path.join(
                super_resolution_path,
                resolution_paths[resolution],
                subject_name,
                f"{subject_name}-{pulse}.nii.gz"
            )
            
            # Load data
            print(f"Loading: {file_path}")
            data = load_nifti_file(file_path)
            
            # Create output filename and path
            output_filename = f"{resolution}_{pulse}.pdf"
            output_path = os.path.join(subject_slice_dir, output_filename)
            
            # Create individual visualization
            if create_individual_visualization(file_path, data, resolution, pulse, subject_name, slice_number, output_path):
                successful_saves += 1
            else:
                print(f"Failed to create visualization for {resolution}-{pulse}")
    
    print(f"\nSummary: {successful_saves}/{total_expected} visualizations created successfully")
    print(f"Output directory: {subject_slice_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize meningioma data across resolutions')
    parser.add_argument('--subject', help='Subject name (e.g., BraTS-MEN-00018-000)', default="BraTS-MEN-00231-000")
    parser.add_argument('--super_resolution_path', help='Path to super_resolution folder',
                        default="/home/mpascual/research/datasets/meningiomas/BraTS/super_resolution")
    parser.add_argument('--slice_number', type=int, help='Sagittal slice number', default=115)
    parser.add_argument('--output_dir', 
                        default='/home/mpascual/research/datasets/meningiomas/BraTS/super_resolution/results/metrics/example', 
                        help='Output directory (default: ./visualization_output)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.super_resolution_path):
        print(f"Error: Super resolution path does not exist: {args.super_resolution_path}")
        return
    
    if args.slice_number < 0:
        print("Error: Slice number must be non-negative")
        return
    
    print(f"Processing subject: {args.subject}")
    print(f"Super resolution path: {args.super_resolution_path}")
    print(f"Slice number: {args.slice_number}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 70)
    
    create_visualization(args.subject, args.super_resolution_path, args.slice_number, args.output_dir)
    
    print("-" * 70)
    print("Visualization complete!")

if __name__ == "__main__":
    main()