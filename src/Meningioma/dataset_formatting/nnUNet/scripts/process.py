from typing import List

from pathlib import Path
import time
import os

from Meningioma.brats_men_2023.nnUNet.preprocessing.merge_and_convert import copy_and_rename_cases
from Meningioma.brats_men_2023.nnUNet.preprocessing.create_metadata_json import generate_dataset_json

# User-defined variables

# Paths
INPUT_ROOT: str = "/home/mariopasc/Python/Datasets/Meningiomas/BraTS/BraTS_Men_Train"
OUTPUT_ROOT: str = "/home/mariopasc/Python/Datasets/Meningiomas/BraTS/nnUNet/nnUNet_raw"

# Sequences to use
SEQUENCES: List[str] = ["t1c", "t2w", "t2f"]

# Custom names
DATASET_NAME: str = "BraTSMen"
DATASET_ID: int = 501

def main() -> None:
    print(f"Starting BraTS 2023 dataset conversion to nnUNet format ...")
    t0 = time.time()
    copy_and_rename_cases(input_root=Path(INPUT_ROOT),
                          output_root=Path(OUTPUT_ROOT),
                          sequences_to_use=SEQUENCES,
                          dataset_name=DATASET_NAME,
                          dataset_id=DATASET_ID)
    generate_dataset_json(dataset_folder=Path(os.path.join(OUTPUT_ROOT, f"Dataset{DATASET_ID}_{DATASET_NAME}")),
                          sequences_to_use=SEQUENCES,
                          dataset_name=DATASET_NAME,
                          file_ending=".nii.gz")
    tf = time.time() - t0
    print(f"Dataset conversion ended in {tf:.3f} seconds")

if __name__ == "__main__":
    main()