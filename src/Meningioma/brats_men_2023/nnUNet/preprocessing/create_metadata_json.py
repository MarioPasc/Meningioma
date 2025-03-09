#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path

def generate_dataset_json(
    dataset_folder: Path,
    sequences_to_use,
    labels=None,
    dataset_name="BraTSMen",
    file_ending=".nii.gz",
    reader_writer=None
):
    """
    Generates a minimal dataset.json for an nnUNet dataset.

    Args:
        dataset_folder (Path): Path to DatasetXXX_* folder.
        sequences_to_use (List[str]): Channel sequences, e.g. ["t1c", "t2f", "t2w"].
        labels (dict): Label name -> int. Example: {"background": 0, "tumor": 1}.
        dataset_name (str): Name of the dataset (only for documentation).
        file_ending (str): Typically ".nii.gz" for MRI data.
        reader_writer (str): Optional override for how nnU-Net reads the data (e.g. "SimpleITKIO").

    The "numTraining" in dataset.json is simply the number of unique files in imagesTr / (number_of_modalities).
    """
    if labels is None:
        # Default label scheme: 0 = background, 1 = tumor
        labels = {
            "background": 0,
            "tumor": 1
        }
    
    imagesTr_folder = dataset_folder / "imagesTr"
    labelsTr_folder = dataset_folder / "labelsTr"
    
    # Count how many unique cases by counting .nii.gz in labelsTr (or imagesTr)
    # Each training case in nnUNet has a single segmentation in labelsTr with the pattern <case_id>.nii.gz
    seg_files = list(labelsTr_folder.glob(f"*{file_ending}"))
    num_training_cases = len(seg_files)
    
    # Build channel_names dict: "0": "t1c", "1": "t2f", ...
    channel_names_dict = {str(i): seq for i, seq in enumerate(sequences_to_use)}
    
    dataset_dict = {
        "channel_names": channel_names_dict,
        "labels": labels,
        "numTraining": num_training_cases,
        "file_ending": file_ending,
    }
    
    if reader_writer is not None:
        dataset_dict["overwrite_image_reader_writer"] = reader_writer
    
    # Save dataset.json
    dataset_json_path = dataset_folder / "dataset.json"
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_dict, f, indent=4)
    
    print(f"dataset.json created at: {dataset_json_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Generate a minimal dataset.json for nnUNet.")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to the dataset folder (e.g. /path/to/Dataset501_BraTSMen)."
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=["t1c", "t2f", "t2w"],
        help="List of sequences in the order they appear as channels (e.g. t1c t2f t2w)."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="BraTSMen",
        help="Name of the dataset, used only for documentation in dataset.json."
    )
    parser.add_argument(
        "--file_ending",
        type=str,
        default=".nii.gz",
        help="File format extension (default .nii.gz)."
    )
    parser.add_argument(
        "--reader_writer",
        type=str,
        default=None,
        help="Optional overwrite_image_reader_writer (e.g. SimpleITKIO)."
    )
    args = parser.parse_args()
    
    # You can define your custom label mapping here if needed
    custom_labels = {
        "background": 0,
        "tumor": 1
    }
    
    generate_dataset_json(
        dataset_folder=Path(args.dataset_folder),
        sequences_to_use=args.sequences,
        labels=custom_labels,
        dataset_name=args.dataset_name,
        file_ending=args.file_ending,
        reader_writer=args.reader_writer
    )

if __name__ == "__main__":
    main()
