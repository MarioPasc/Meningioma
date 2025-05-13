# file: src/mgmGrowth/tasks/superresolution/cli/downsample.py
"""
Down-sample BraTS in z only, with optional N-way patient-level parallelism.

Example
-------
python -m src.mgmGrowth.tasks.superresolution.cli.downsample \
    --src-root  ~/Datasets/BraTS_Men_Train \
    --out-root  ~/Datasets/downsampled_brats_5mm \
    --target-dz 5 \
    --jobs 4
"""
from __future__ import annotations

import argparse
import functools
from pathlib import Path

from src.mgmGrowth.tasks.superresolution import LOGGER
from src.mgmGrowth.tasks.superresolution.tools import (
    downsample_z,
    load_nifti,
    save_nifti,
    ensure_dir,
    run_parallel,
)


# ---------------------------------------------------------------------
#  Top-level, picklable worker  
# ---------------------------------------------------------------------
def _process_patient(
    pat_dir: Path,
    out_root: Path,
    target_mm: float,
) -> None:
    """Down-sample **all** NIfTIs inside *pat_dir* and save to *out_root/patient*."""
    dst_dir = ensure_dir(out_root / pat_dir.name)
    for nii in pat_dir.glob("*.nii.gz"):
        if "seg" in nii.name:
            LOGGER.debug("Skipping segmentation file %s", nii.name)
            continue
        arr, spacing = load_nifti(nii)
        arr_ds, spacing_ds = downsample_z(arr, spacing, target_mm)
        save_nifti(arr_ds, spacing_ds, dst_dir / nii.name, reference=nii)
    LOGGER.info("✓ %s", pat_dir.name)


# ---------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--target-dz", type=float, required=True)
    p.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Patient folders processed in parallel "
        "(1 = sequential, 0 or <0 = use all CPUs).",
    )
    args = p.parse_args()

    patients = [d for d in args.src_root.iterdir() if d.is_dir()]

    # functools.partial produces a picklable callable with bound args
    worker = functools.partial(
        _process_patient,
        out_root=args.out_root,
        target_mm=args.target_dz,
    )

    run_parallel(worker, patients, jobs=args.jobs, desc="patients")
    LOGGER.info("Finished down-sampling %d patients → %s", len(patients), args.out_root)


if __name__ == "__main__":
    main()