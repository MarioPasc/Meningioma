#!/usr/bin/env python3
"""
resample_iso.py
===============

Resample a medical volume to isotropic spacing (default 1 mm³) using
SimpleITK, preserving origin & orientation.

Usage (CLI)
-----------
    python resample_iso.py input.nii.gz output.nii.gz \
           --spacing 1 1 1 --interp bspline

Interpolation options
---------------------
    nn          → sitkNearestNeighbor
    linear      → sitkLinear                (default)
    bspline     → sitkBSpline               (cubic)
    lanczos     → sitkLanczosWindowedSinc
    gaussian    → sitkGaussian
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from datetime import datetime
from typing import Dict, Tuple, Sequence, Any

import SimpleITK as sitk

_LOG = logging.getLogger(__name__)


# ────────────────────────────────────────── core logic ──────────────────── #
INTERPOLATORS: Dict[str, Any] = {
    "nn": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
    "lanczos": sitk.sitkLanczosWindowedSinc,
    "gaussian": sitk.sitkGaussian,
}


def resample_isotropic(
        image: sitk.Image,
        target_spacing: Tuple[float, float, float],
        interpolator: str = "linear",
) -> sitk.Image:
    """
    Resample *image* to isotropic spacing using SimpleITK.

    Parameters
    ----------
    image
        Input SimpleITK image.
    target_spacing
        Desired spacing in mm (dx, dy, dz).
    interpolator
        Key defined in :data:`INTERPOLATORS`.

    Returns
    -------
    sitk.Image
        Resampled image with spacing `target_spacing`.
    """
    if interpolator not in INTERPOLATORS:
        raise ValueError(f"Unknown interpolator '{interpolator}'. "
                         f"Choose from: {', '.join(INTERPOLATORS)}")

    src_spacing = image.GetSpacing()
    src_size = image.GetSize()

    new_size = [
        int(round(sz * sp / tgt))
        for sz, sp, tgt in zip(src_size, src_spacing, target_spacing)
    ]

    _LOG.info("Resampling from spacing %s → %s vox, %s → %s vox",
              src_spacing, target_spacing, src_size, new_size)

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(INTERPOLATORS[interpolator])
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetTransform(sitk.Transform())        # identity

    return resampler.Execute(image)


def save_provenance(path: pathlib.Path,
                    in_path: pathlib.Path,
                    spacing: Sequence[float],
                    interpolator: str) -> None:
    """Write a tiny JSON file alongside *path* describing the operation."""
    meta = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "source": str(in_path),
        "target_spacing_mm": list(spacing),
        "interpolator": interpolator,
        "version": "1.0",
    }
    json_path = path.with_suffix(path.suffix + ".json")
    json_path.write_text(json.dumps(meta, indent=2))
    _LOG.info("Provenance written to %s", json_path.name)


# ──────────────────────────────────────────── CLI ───────────────────────── #
def _parse_cli(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resample a 3-D medical volume to isotropic spacing "
                    "using SimpleITK."
    )
    p.add_argument("input", type=pathlib.Path,
                   help="Input volume (.nii/.nii.gz/...).")
    p.add_argument("output", type=pathlib.Path,
                   help="Output filename.")
    p.add_argument("--spacing", nargs=3, type=float, default=(1.0, 1.0, 1.0),
                   metavar=("SX", "SY", "SZ"),
                   help="Target spacing in mm (default: 1 1 1).")
    p.add_argument("--interp", default="bspiline", type=str,
                   choices=INTERPOLATORS.keys(),
                   help="Interpolation algorithm (default: linear).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Enable info-level logging.")
    return p.parse_args(argv)


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level,
                        format="%(levelname)s: %(message)s")


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_cli(argv)
    _configure_logging(args.verbose)

    if not args.input.exists():
        _LOG.error("Input file %s does not exist", args.input)
        sys.exit(1)

    _LOG.info("Reading %s", args.input)
    img = sitk.ReadImage(str(args.input))

    iso = resample_isotropic(img,
                             target_spacing=tuple(args.spacing),
                             interpolator=args.interp)

    _LOG.info("Writing %s", args.output)
    sitk.WriteImage(iso, str(args.output), useCompression=True)

    save_provenance(args.output, args.input, args.spacing, args.interp)
    _LOG.info("Done")


if __name__ == "__main__":
    main()
