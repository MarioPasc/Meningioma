#!/usr/bin/env python
import argparse
import shutil
from pathlib import Path
from typing import List

def copy_and_rename_cases(
    input_root: Path,
    output_root: Path,
    sequences_to_use: List[str],
    dataset_name: str = "BraTSMen",
    dataset_id: int = 501
):
    """
    Merges BraTS Men 2023 'Train' and 'Validation' subfolders into a single dataset,
    selecting only the given sequences, and renames them to the nnUNet format.

    Args:
        input_root (Path): Path to the folder 'BraTS_Men_Train'. 
        output_root (Path): Path to where the nnUNet_raw folder will be created. The dataset
                            will go to `nnUNet_raw/Dataset<dataset_id>_<dataset_name>`.
        sequences_to_use (List[str]): List of MRI sequence suffixes to include (e.g. ["t1c", "t2f", "t2w"]).
        dataset_name (str): A name identifier for your dataset folder inside nnUNet_raw.
        dataset_id (int): A numeric ID for your dataset folder inside nnUNet_raw.

    Example:
        copy_and_rename_cases(
            Path("/home/user/BraTS"),
            Path("/home/user/nnUNet_raw"),
            ["t1c", "t2f", "t2w"],
            dataset_name="BraTSMen",
            dataset_id=501
        )
    """
    # Construct the final nnUNet dataset folder name, e.g. "Dataset501_BraTSMen"
    dataset_folder_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    dataset_folder = output_root / dataset_folder_name
    
    # Subfolders for imagesTr and labelsTr
    imagesTr_folder = dataset_folder / "imagesTr"
    labelsTr_folder = dataset_folder / "labelsTr"
    imagesTr_folder.mkdir(parents=True, exist_ok=True)
    labelsTr_folder.mkdir(parents=True, exist_ok=True)
    
    
    all_cases = sorted(input_root.iterdir())
    
    # For each case folder, copy the relevant sequences + seg
    for case_dir in all_cases:
        if not case_dir.is_dir():
            continue
        
        # Example case_dir name: "BraTS-MEN-00008-000"
        # We'll parse the patient ID from that name. Alternatively, just use the entire folder name minus suffix.
        case_id = case_dir.name  # e.g. "BraTS-MEN-00008-000"

        # Typically for nnUNet, we'd want a shorter ID like "BraTS_MEN_00008"
        # We'll remove the trailing '-xxx' to get a stable ID across sequences.
        # But if you prefer to keep it as is, adjust accordingly.
        base_id = "-".join(case_id.split("-")[:3])  # "BraTS-MEN-00008"
        base_id = base_id.replace("-", "_")         # "BraTS_MEN_00008"
        
        # Collect all .nii.gz in this folder
        all_nii_files = list(case_dir.glob("*.nii.gz"))
        
        # Sort or group by sequence
        # We only copy sequences that appear in sequences_to_use, plus the segmentation
        # that typically ends with '-seg.nii.gz'.
        
        # Create a list to store the actual sequence files we find
        chosen_files = []
        seg_file = None
        
        for nii_file in all_nii_files:
            filename = nii_file.name
            # Example filename: "BraTS-MEN-00008-000-t1c.nii.gz"
            
            if filename.endswith("-seg.nii.gz"):
                seg_file = nii_file
            else:
                # Identify which sequence (t1c, t2w, etc.) by splitting on '-'
                # Another approach is to use regex or parse the final part before .nii.gz
                seq_candidate = filename.split(".nii.gz")[0].split("-")[-1]  # e.g. "t1c"
                if seq_candidate in sequences_to_use:
                    chosen_files.append(nii_file)
        
        # Sort chosen_files by the order in sequences_to_use so channel numbering is consistent
        # (0000 -> first in sequences_to_use, 0001 -> second, etc.)
        # This ensures correct channel ordering for nnUNet
        chosen_files_sorted = []
        for seq in sequences_to_use:
            for f in chosen_files:
                seq_candidate = f.name.split(".nii.gz")[0].split("-")[-1]
                if seq_candidate == seq:
                    chosen_files_sorted.append(f)
        
        if len(chosen_files_sorted) == 0:
            # No matching sequences found, skip case
            continue
        
        # Copy each chosen sequence to imagesTr, using the naming convention:
        #   base_id_0000.nii.gz, base_id_0001.nii.gz, etc.
        for idx, seq_file in enumerate(chosen_files_sorted):
            destination_name = f"{base_id}_{idx:04d}.nii.gz"
            destination_path = imagesTr_folder / destination_name
            shutil.copy(seq_file, destination_path)
        
        # Copy the segmentation to labelsTr, named base_id.nii.gz
        if seg_file is not None:
            seg_destination = labelsTr_folder / f"{base_id}.nii.gz"
            shutil.copy(seg_file, seg_destination)

def main():
    parser = argparse.ArgumentParser(description="Merge BraTS Men 2023 training+validation into nnUNet format.")
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Path to folder containing BraTS_Men_Train and BraTS_Men_Validation."
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Where to create the nnUNet_raw/DatasetXXX_Foo structure."
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["t1c", "t2f", "t2w"],
        help="List of sequences to include (e.g. t1c t2f t2w)."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="BraTSMen",
        help="Name of the dataset folder, e.g. 'BraTSMen'."
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        default=501,
        help="Integer ID for the dataset folder, e.g. 501 => Dataset501_BraTSMen."
    )
    
    args = parser.parse_args()

    copy_and_rename_cases(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        sequences_to_use=args.sequences,
        dataset_name=args.dataset_name,
        dataset_id=args.dataset_id
    )

if __name__ == "__main__":
    main()
