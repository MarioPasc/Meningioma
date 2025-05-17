#!/usr/bin/env python3
"""
super_res_pipeline.py
=====================

Batch-upsample the BraTS-Meningioma low-resolution volumes to *1 mm³* isotropic
spacing, using SimpleITK and the resampling helpers defined in
``mgmGrowth.tasks.superresolution.engine.interpolation_runner``

Folder layout
-------------
Input  : /home/mariopasc/Python/Datasets/Meningiomas/BraTS/SR/low_res/{3mm,5mm,7mm}/<case>/<mod>.nii.gz
Output : /home/mariopasc/Python/Results/Meningioma/super_resolution/{ALGO}/{3mm,5mm,7mm}/output_volumes/<same-name>.nii.gz

The script is restart-safe: existing output files are skipped unless
``--overwrite`` is given.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Tuple

import SimpleITK as sitk
from mgmGrowth.tasks.superresolution.engine import interpolation_runner as ir

# ───────────────────────── configuration constants ────────────────────── #
IN_ROOT = Path("/home/mariopasc/Python/Datasets/Meningiomas/BraTS/SR/low_res")
OUT_ROOT = Path("/home/mariopasc/Python/Results/Meningioma/super_resolution")
RES_DIRS = ("3mm", "5mm", "7mm")
TARGET_SPACING: Tuple[float, float, float] = (1.0, 1.0, 1.0)

_LOG = logging.getLogger("super_res_pipeline")


# ────────────────────────────── helpers ───────────────────────────────── #
def _upsample(in_file: Path, out_file: Path,
              spacing: Tuple[float, float, float],
              interpolator: str, overwrite: bool) -> None:
    """
    Upsample one NIfTI volume and write both the image and provenance.

    Parameters
    ----------
    in_file
        Source ``*.nii`` or ``*.nii.gz`` volume.
    out_file
        Target path (parent directories must already exist).
    spacing
        Desired isotropic spacing in millimetres.
    interpolator
        Interpolator key accepted by
        ``mgmGrowth.tasks.superresolution.engine.interpolation_runner``.
    overwrite
        Whether to recompute if *out_file* already exists.
    """
    if out_file.exists() and not overwrite:
        _LOG.info("Skipping existing %s", out_file.name)
        return

    _LOG.info("Reading %s", in_file)
    img = sitk.ReadImage(str(in_file))
    iso = ir.resample_isotropic(img, spacing, interpolator)

    _LOG.info("Writing %s", out_file)
    sitk.WriteImage(iso, str(out_file), useCompression=True)
    ir.save_provenance(out_file, in_file, spacing, interpolator)


def _gather_jobs(res_dir: Path, algo_dir: Path,
                 spacing: Tuple[float, float, float]) -> list[tuple[Path, Path]]:
    """Return (input, output) pairs for every volume in *res_dir*."""
    out_volumes = algo_dir / res_dir.name / "output_volumes"
    out_volumes.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path]] = []
    for nii in res_dir.rglob("*.nii*"):
        jobs.append((nii, out_volumes / nii.name))
    return jobs


# ──────────────────────────── main routine ────────────────────────────── #
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--interp", default="bspline",
                   choices=ir.INTERPOLATORS.keys(),
                   help="Interpolation algorithm (default: bspline).")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if the target file already exists.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable info-level logging.")
    return p.parse_args()


def main() -> None:
    args = _parse_cli()
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")

    algo_dir = OUT_ROOT / args.interp.upper()
    algo_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[Path, Path]] = []
    for res_name in RES_DIRS:
        res_path = IN_ROOT / res_name
        if not res_path.exists():
            _LOG.warning("Input resolution folder %s not found – skipping", res_path)
            continue
        jobs.extend(_gather_jobs(res_path, algo_dir, TARGET_SPACING))

    _LOG.info("Discovered %d volumes to process.", len(jobs))

    with mp.get_context("spawn").Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(_upsample,
                     [(src, dst, TARGET_SPACING, args.interp, args.overwrite)
                      for src, dst in jobs])

    _LOG.info("All done.")


if __name__ == "__main__":
    main()
