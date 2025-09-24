import matplotlib.pyplot as plt
import numpy as np
import os
import nibabel as nib

def plot_axial_slices(patient_id="P40", n_slices=5, mask=True, white=False, grey=False, csf=False):
    """
    Generate a 3xN plot showing multiple axial slices of a patient with one pulse per row.
    
    Args:
        patient_id: Patient identifier
        n_slices: Number of axial slices to display
        mask: Whether to overlay the tumor segmentation mask
        white: Whether to overlay white matter regions
        grey: Whether to overlay grey matter regions
        csf: Whether to overlay CSF regions
    """
    # Paths
    PATH = "/home/mariopasc/Python/Datasets/Meningiomas/meningioma"
    ATLAS_PATH = "/home/mariopasc/Python/Datasets/Meningiomas/ATLAS/sri24_spm8/tpm"
    ASSETS_PATH = "/home/mariopasc/Python/Datasets/Meningiomas/assets"
    PULSES = ["T1", "T2", "SUSC"]
    
    # Load patient data
    data = {}
    for pulse in PULSES:
        file = os.path.join(PATH, patient_id, f"{pulse}_{patient_id}.nii.gz")
        seg_file = os.path.join(PATH, patient_id, f"{pulse}_{patient_id}_seg.nii.gz")
        img = nib.load(file)
        seg = nib.load(seg_file)

        data[pulse] = {
            "vol": img.get_fdata(),
            "seg": seg.get_fdata(),
        }
    
    # Load tissue maps if needed
    tissue_maps = {}
    if white or grey or csf:
        # Create a descriptive string for output filename
        overlay_str = ""
        
        if white:
            white_img = nib.load(os.path.join(ATLAS_PATH, "white.nii"))
            tissue_maps["white"] = white_img.get_fdata()
            overlay_str += "_white"
            print("Loaded white matter map")
        
        if grey:
            grey_img = nib.load(os.path.join(ATLAS_PATH, "grey.nii"))
            tissue_maps["grey"] = grey_img.get_fdata()
            overlay_str += "_grey"
            print("Loaded grey matter map")
        
        if csf:
            csf_img = nib.load(os.path.join(ATLAS_PATH, "csf.nii"))
            tissue_maps["csf"] = csf_img.get_fdata()
            overlay_str += "_csf"
            print("Loaded CSF map")
    else:
        overlay_str = "_no_overlay"
    
    # Get volume dimensions
    vol_shape = data[PULSES[0]]["vol"].shape
    
    # Calculate evenly spaced slice indices
    z_start = int(vol_shape[2] * 0.05)  # Start at 5% of volume
    z_end = int(vol_shape[2] * 0.9)     # End at 90% of volume
    slice_indices = np.linspace(z_start, z_end, n_slices, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(len(PULSES), n_slices, figsize=(n_slices * 3, len(PULSES) * 3))
    fig.patch.set_facecolor('black')
    
    # For each pulse (row)
    for row, pulse in enumerate(PULSES):
        vol = data[pulse]["vol"]
        seg = data[pulse]["seg"]
        
        # Normalize volume data for better visualization
        vol_norm = (vol - np.min(vol)) / (np.max(vol) - np.min(vol))
        
        # For each slice (column)
        for col, slice_idx in enumerate(slice_indices):
            ax = axes[row, col]
            
            # Display axial slice
            ax.imshow(np.rot90(vol_norm[:, :, slice_idx]), cmap='gray')
            
            # Overlay segmentation mask for tumor if requested
            if mask:
                ax.imshow(np.rot90(np.ma.where(seg[:, :, slice_idx] == 1, 1, np.nan)), 
                         cmap='Reds_r', alpha=0.5)
            
            # Overlay tissue maps if available
            # We'll use different colors for each tissue type
            if "white" in tissue_maps:
                # Normalize and threshold the tissue map
                white_slice = tissue_maps["white"][:, :, slice_idx]
                white_slice = (white_slice - np.min(white_slice)) / (np.max(white_slice) - np.min(white_slice))
                white_slice = np.ma.masked_where(white_slice < 0.5, white_slice)  # Threshold at 0.5
                ax.imshow(np.rot90(white_slice), cmap='Blues_r', alpha=0.5, interpolation='none')
            
            if "grey" in tissue_maps:
                grey_slice = tissue_maps["grey"][:, :, slice_idx]
                grey_slice = (grey_slice - np.min(grey_slice)) / (np.max(grey_slice) - np.min(grey_slice))
                grey_slice = np.ma.masked_where(grey_slice < 0.5, grey_slice)
                ax.imshow(np.rot90(grey_slice), cmap='Greens_r', alpha=0.5, interpolation='none')
            
            if "csf" in tissue_maps:
                csf_slice = tissue_maps["csf"][:, :, slice_idx]
                csf_slice = (csf_slice - np.min(csf_slice)) / (np.max(csf_slice) - np.min(csf_slice))
                csf_slice = np.ma.masked_where(csf_slice < 0.5, csf_slice)
                ax.imshow(np.rot90(csf_slice), cmap='Purples_r', alpha=0.5, interpolation='none')
            
            # Configure axis
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor('black')
            
            # Add slice index to first row
            if row == 0:
                ax.set_title(f"Slice {slice_idx}", color='white', fontsize=10)
                
            # Add pulse label to first column
            if col == 0:
                ax.set_ylabel(pulse, color='white', fontsize=12, rotation=90, labelpad=15)
                
            # Set border color
            for spine in ax.spines.values():
                spine.set_color('black')
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    
    # Create a string for the mask option
    mask_str = "_with_mask" if mask else "_no_mask"
    
    # Save figure
    output_file = f"axial_slices_{patient_id}{mask_str}{overlay_str}.png"
    plt.savefig(os.path.join(ASSETS_PATH, output_file), 
                facecolor='black', bbox_inches='tight', dpi=150)
    
    plt.show()
    
    return f"Saved to {os.path.join(ASSETS_PATH, output_file)}"

# Execute the function - showing white and grey matter with mask
plot_axial_slices(patient_id="P1", n_slices=10, mask=False, white=False, grey=False, csf=True)