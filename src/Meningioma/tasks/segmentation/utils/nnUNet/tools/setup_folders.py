import os
from typing import Dict

def setup_folders(root_path: str) -> Dict[str, str]:
    """
    Setting up folders under root_path and returning a dictionary with the paths.
    """
    paths = {}
    for folder in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        full_path = os.path.join(root_path, folder)
        os.makedirs(full_path, exist_ok=True)
        paths[folder] = full_path

    return paths
