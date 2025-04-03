from typing import List

from pathlib import Path
import time
import os

from Meningioma.dataset_formatting.nnUNet.tools.merge_and_convert import copy_and_rename_cases
from Meningioma.dataset_formatting.nnUNet.tools.create_metadata_json  import generate_dataset_json
from Meningioma.dataset_formatting.nnUNet.tools.plan_and_preprocess_dataset import plan_and_preprocess
from Meningioma.dataset_formatting.nnUNet.tools.setup_folders import setup_folders

# User-defined variables

# Paths
INPUT_ROOT: str = "/home/mariopasc/Python/Datasets/Meningiomas/BraTS/BraTS_Men_Train"
OUTPUT_ROOT: str = "/home/mariopasc/Python/Datasets/Meningiomas/BraTS/nnUNet"

# Sequences to use
SEQUENCES: List[str] = ["t1c", "t2w", "t2f"]

# Custom names
DATASET_NAME: str = "BraTSMen"
DATASET_ID: int = 501

# Plan and preprocess configuration
CONFIGURATION: str = "3d_fullres"



def main() -> None:
    print(f"Starting BraTS 2023 dataset conversion to nnUNet format ...")
    t0 = time.time()

    paths_dict = setup_folders(root_path=OUTPUT_ROOT)

    nnUNet_raw_path = os.path.join(OUTPUT_ROOT, "nnUNet_raw")
    copy_and_rename_cases(input_root=Path(INPUT_ROOT),
                          output_root=Path(nnUNet_raw_path),
                          sequences_to_use=SEQUENCES,
                          dataset_name=DATASET_NAME,
                          dataset_id=DATASET_ID)
    generate_dataset_json(dataset_folder=Path(os.path.join(nnUNet_raw_path, f"Dataset{DATASET_ID}_{DATASET_NAME}")),
                          sequences_to_use=SEQUENCES,
                          file_ending=".nii.gz")

    print("Applying plan and preprocessing nnUNet plan ...")
    plan_and_preprocess(paths_dict= paths_dict,
                        dataset_id=DATASET_ID,
                        verify_dataset_integrity=True,
                        configurations=CONFIGURATION)
    tf = time.time() - t0
    print(f"Dataset conversion ended in {tf:.3f} seconds")

if __name__ == "__main__":
    main()