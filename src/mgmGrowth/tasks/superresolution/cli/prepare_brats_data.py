#!/usr/bin/env python3
# filepath: Meningioma/src/mgmGrowth/tasks/superresolution/cli/prepare_brats_data.py
"""
prepare_brats_data.py
=====================

Create a subset of BraTS data containing the top N patients with all tumor labels
present, then downsample this subset to create multiple lower-resolution datasets
(3mm, 5mm, 7mm).

Example
-------
python -m src.mgmGrowth.tasks.superresolution.cli.prepare_brats_data \
    --src-root  /home/mpascual/research/datasets/meningiomas/BraTS/source/BraTS_Men_Train \
    --out-root  /home/mpascual/research/datasets/meningiomas/BraTS/super_resolution \
    --num-patients 50 \
    --pulses t1c t2w t2f t1n \
    --jobs 8
"""
from __future__ import annotations

import argparse
import pathlib
from pathlib import Path
from typing import List

from src.mgmGrowth.tasks.superresolution import LOGGER
from src.mgmGrowth.tasks.superresolution.utils.sort_brats_volume import select_top_brats_patients
from src.mgmGrowth.tasks.superresolution.cli.downsample import _process_patient
from src.mgmGrowth.tasks.superresolution.tools.parallel import run_parallel


def create_brats_subset(
    src_root: Path,
    out_root: Path,
    num_patients: int,
    pulses: List[str],
    skip_if_exists: bool = False,
) -> Path:
    """
    Create a subset of the BraTS dataset with the top N patients having all tumor labels.
    
    Parameters
    ----------
    src_root : Path
        Source directory containing BraTS-MEN-* folders
    out_root : Path
        Destination root directory
    num_patients : int
        Number of patients to select
    pulses : List[str]
        Pulse types to include (e.g., ["t1c", "t2w"])
    skip_if_exists : bool
        If True, skip subset creation if the destination already exists
        
    Returns
    -------
    Path
        Path to the created subset directory
    """
    subset_dir = out_root / "subset"
    
    if skip_if_exists and subset_dir.exists() and any(subset_dir.iterdir()):
        LOGGER.info(f"Subset directory {subset_dir} already exists, skipping creation")
        return subset_dir
    
    LOGGER.info(f"Creating BraTS subset with {num_patients} patients in {subset_dir}")
    subset_dir.mkdir(parents=True, exist_ok=True)
    
    select_top_brats_patients(
        root_dir=src_root,
        num_patients=num_patients,
        pulses=pulses,
        output_dir=subset_dir,
    )
    
    return subset_dir


def create_downsampled_datasets(
    subset_dir: Path,
    out_root: Path,
    resolutions_mm: List[int],
    jobs: int,
) -> None:
    """
    Create downsampled versions of the BraTS subset at specified resolutions.
    
    Parameters
    ----------
    subset_dir : Path
        Directory containing the BraTS subset
    out_root : Path
        Root directory where downsampled datasets will be stored
    resolutions_mm : List[int]
        List of z-axis resolutions to create (e.g., [3, 5, 7])
    jobs : int
        Number of parallel jobs for downsampling
    """
    for res_mm in resolutions_mm:
        res_dir = out_root / "low_res" / f"{res_mm}mm"
        res_dir.mkdir(parents=True, exist_ok=True)
        
        LOGGER.info(f"Creating {res_mm}mm downsampled dataset in {res_dir}")
        
        patient_dirs = [d for d in subset_dir.iterdir() if d.is_dir()]
        
        import functools
        worker = functools.partial(
            _process_patient,
            out_root=res_dir,
            target_mm=res_mm,
        )
        
        run_parallel(worker, patient_dirs, jobs=jobs, 
                     desc=f"Downsampling to {res_mm}mm")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create BraTS subset and downsampled datasets"
    )
    parser.add_argument(
        "--src-root", 
        type=Path, 
        required=True,
        help="Source directory containing BraTS-MEN-* folders"
    )
    parser.add_argument(
        "--out-root", 
        type=Path, 
        required=True,
        help="Output root directory"
    )
    parser.add_argument(
        "--num-patients", 
        type=int, 
        default=50,
        help="Number of patients to include in the subset (default: 50)"
    )
    parser.add_argument(
        "--pulses", 
        nargs="+", 
        default=["t1c", "t2w", "t2f"],
        help="Pulse types to include (default: t1c t2w t2f)"
    )
    parser.add_argument(
        "--resolutions", 
        nargs="+", 
        type=int,
        default=[3, 5, 7],
        help="List of resolutions in mm to create (default: 3 5 7)"
    )
    parser.add_argument(
        "--skip-subset", 
        action="store_true",
        help="Skip subset creation if it already exists"
    )
    parser.add_argument(
        "--jobs", 
        type=int, 
        default=8,
        help="Number of parallel jobs for downsampling (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Create subset
    subset_dir = create_brats_subset(
        args.src_root,
        args.out_root,
        args.num_patients,
        args.pulses,
        args.skip_subset
    )
    
    # Create downsampled datasets
    create_downsampled_datasets(
        subset_dir,
        args.out_root,
        args.resolutions,
        args.jobs
    )
    
    LOGGER.info("All done! Dataset preparation complete.")
    LOGGER.info(f"Subset: {subset_dir}")
    for res in args.resolutions:
        LOGGER.info(f"{res}mm dataset: {args.out_root}/low_res/{res}mm")


if __name__ == "__main__":
    main()