#!/usr/bin/env python3

from os import path
import subprocess
from typing import Optional, Dict


def plan_and_preprocess(
    paths_dict: Dict[str, str],
    dataset_id: int,
    verify_dataset_integrity: bool = False,
    configurations: Optional[str] = None
) -> None:
    """
    Runs nnUNetv2_plan_and_preprocess with environment variables prefixed in the shell command,
    e.g.:
      nnUNet_raw="/some/path" nnUNet_preprocessed="/other/path" nnUNet_results="/another/path" nnUNetv2_plan_and_preprocess -d 501
    """
    # For convenience
    raw_path = paths_dict["nnUNet_raw"]
    prep_path = paths_dict["nnUNet_preprocessed"]
    res_path = paths_dict["nnUNet_results"]
    
    # Build the base command string with environment prefixes
    cmd_str = (
        f"nnUNet_raw='{raw_path}' "
        f"nnUNet_preprocessed='{prep_path}' "
        f"nnUNet_results='{res_path}' "
        f"nnUNetv2_plan_and_preprocess -d {dataset_id}"
    )

    # Add optional flags
    if verify_dataset_integrity:
        cmd_str += " --verify_dataset_integrity"
    if configurations:
        # e.g. "3d_fullres" or "2d 3d_fullres"
        # The -c needs to precede them. If multiple configs: -c 3d_fullres 2d
        cmd_str += f" -c {configurations}"
    
    print("[INFO] Running preprocessing command (prefix style) with shell=True:")
    print("       ", cmd_str)

    # Run the command using shell=True so the variable prefix is interpreted
    subprocess.run(cmd_str, shell=True, check=True)