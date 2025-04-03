#!/usr/bin/env python3

import argparse
import subprocess
import time
from typing import List, Optional


def train_all_folds(
    dataset_id: int,
    configuration: str,
    folds: int,
    gpu_ids: List[int],
    save_npz: bool = False,
    sleep_before_spawning_others: int = 120
) -> None:
    """
    Trains nnUNet models in a 5-fold (or custom #fold) cross-validation setup, 
    assigning each fold to a different GPU (if provided).
    
    Args:
        dataset_id (int): The dataset ID or dataset name (e.g., '501' or 'Dataset501_BraTSMen').
        configuration (str): One of nnUNet's recognized configurations (e.g. "2d", "3d_fullres", "3d_lowres", etc.).
        folds (int): Number of folds to train (commonly 5 for 5-fold cross-validation).
        gpu_ids (List[int]): List of GPU IDs on which to run each fold. For example, [0,1,2,3,4].
                             If fewer GPUs than folds are provided, subsequent folds may reuse GPUs (round-robin).
        save_npz (bool): Whether to include the '--npz' flag for saving softmax probability maps 
                         during validation (useful for ensembling).
        sleep_before_spawning_others (int): Time in seconds to wait after starting the first fold, 
                                            allowing data extraction to finish before spawning more folds.
    """
    if folds < 1:
        raise ValueError("Number of folds must be >= 1.")
    
    # If there are more folds than GPU IDs, we cycle through GPU IDs in a round-robin manner.
    # If you must strictly match 'one fold per GPU' with the same number, ensure len(gpu_ids) == folds.
    
    processes = []
    for fold_idx in range(folds):
        gpu_id = gpu_ids[fold_idx % len(gpu_ids)]  # round-robin if fewer GPUs than folds
        
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} nnUNetv2_train {dataset_id} {configuration} {fold_idx}"
        )
        if save_npz:
            cmd += " --npz"
        
        print(f"[INFO] Starting training for fold {fold_idx} on GPU {gpu_id}...")
        
        # Start process in background via shell (&). You could also use subprocess.Popen 
        # for more control, but we'll show a simple approach here:
        cmd_background = cmd + " &"
        processes.append(cmd_background)
        
        # If this is the first fold, we do not immediately spawn the rest.
        # We'll wait a bit to let data be extracted into memory (per nnUNet's advice).
        if fold_idx == 0 and folds > 1:
            subprocess.run(cmd_background, shell=True, check=False)
            print(f"[INFO] Sleeping {sleep_before_spawning_others} seconds to let data extraction finish.")
            time.sleep(sleep_before_spawning_others)
        else:
            subprocess.run(cmd_background, shell=True, check=False)
    
    # "wait" for all background processes to complete
    # We can just run a final wait in the shell:
    print("[INFO] All folds spawned. Waiting for them to finish...")
    subprocess.run("wait", shell=True, check=False)
    print("[INFO] All folds completed.")


def main() -> None:

    description: str = " Script to train nnUNet in a cross-validation setup, one fold per GPU." \
    " Typical scenario: 5-fold cross-validation on dataset 501 using the 3d_fullres config," \
    "running each fold on one of the 5 GPUs [0, 1, 2, 3, 4], saving npz for ensembling: " \
    ""\
    "python src/Meningioma/engines/nnUNet_trainer.py " \
    "--dataset_id 501 " \
    "--configuration 3d_fullres" \
    "--folds 5" \
    "--gpus 0 1 2 3 4" \
    "--save_npz" \
    ""\
    "If you only have 1 GPU but still want 5 folds (if 2, folds will be round-robin across GPU 0 and GPU 1):" \
    ""\
    "python src/Meningioma/engines/nnUNet_trainer.py" \
    "--dataset_id 501" \
    "--configuration 3d_fullres" \
    "--folds 0"  \
    "--gpus 0"


    parser = argparse.ArgumentParser(
        description=description
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="The dataset ID (e.g. 501) or name (e.g. 'Dataset501_BraTSMen')."
    )
    parser.add_argument(
        "--configuration",
        type=str,
        required=True,
        help="The nnUNet configuration, e.g. '2d', '3d_fullres', '3d_lowres', '3d_cascade_fullres'."
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of folds to train (default: 5)."
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0],
        help="List of GPU IDs to use (default: [0]). E.g. --gpus 0 1 2 3 4"
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help="Include the --npz flag to store predicted softmax (large files!). Needed for ensembling."
    )
    parser.add_argument(
        "--sleep_before_spawning_others",
        type=int,
        default=120,
        help="Time in seconds to wait after starting the first fold, "
             "allowing data extraction to complete (default: 120)."
    )
    args = parser.parse_args()
    
    train_all_folds(
        dataset_id=int(args.dataset_id) if args.dataset_id.isdigit() else args.dataset_id,
        configuration=args.configuration,
        folds=args.folds,
        gpu_ids=args.gpus,
        save_npz=args.save_npz,
        sleep_before_spawning_others=args.sleep_before_spawning_others
    )

if __name__ == "__main__":
    main()
