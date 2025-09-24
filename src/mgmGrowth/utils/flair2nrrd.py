"""
flair2nrrd.py
=============

Batch-convert FLAIR DICOM series (files may have no *.dcm* extension) into
*.nrrd* while

* deduplicating anonymiser clones (e.g. *_an, *_anon*);
* preserving slice order via DICOM geometry (works for axial, oblique,
  sagittal, coronal …);
* enforcing a consistent LPS orientation;
* writing a neat cohort summary (.csv).

Usage
-----
$ python flair2nrrd.py <DATASET_ROOT> <OUTPUT_ROOT> [--overwrite]
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

DUP_SUFFIXES: List[str] = ["_an", "_anon"]  # accepted duplicate suffixes

KEYWORDS = ("flair", "t2 flair", "t2flair")  # strings that flag a FLAIR series

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def _slice_key(path: str) -> tuple[float, int]:
    """
    Return a sortable key per DICOM-spec:

    1. position along the slice normal  d = n • P
    2. *InstanceNumber* as a tie-breaker
    """
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)

        pos = np.asarray(ds.ImagePositionPatient, dtype=float)        # (x, y, z)
        orient = np.asarray(ds.ImageOrientationPatient, dtype=float)  # (6,)
        normal = np.cross(orient[:3], orient[3:])                     # n = r x c
        loc = float(np.dot(pos, normal))

        inst = int(getattr(ds, "InstanceNumber", 0))
        return (loc, inst)
    except Exception:
        return (np.inf, 0)  # unreadable → push to end but keep deterministic


def strip_dup_suffix(name: str, suffixes: Sequence[str]) -> str:
    """Remove *one* occurrence of any suffix in *suffixes* from *name*."""
    for suf in suffixes:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def discover_flair_folders(dataset_root: Path, flair_dirname: str = "flair") \
        -> Dict[str, Path]:
    """
    Walk *dataset_root* and return {PatientID: Path-to-FLAIR-folder} where
    *flair_dirname* exists.
    """
    out: Dict[str, Path] = {}
    for pdir in sorted(dataset_root.iterdir()):
        flair_path = pdir / flair_dirname
        if flair_path.is_dir():
            out[pdir.name] = flair_path
    return out


def collect_series_files(folder: Path,
                         suffixes: Sequence[str]) -> Dict[str, List[str]]:
    """
    Group DICOM slices into series and return {SeriesUID|NOUID: [file,…]}.

    * deduplicates anonymiser copies;
    * sorts slices by physical location using *_slice_key*;
    * drops scout series (<3 slices).
    """
    series_map: Dict[str, List[str]] = {}

    # first pass: plain pydicom
    for f in folder.iterdir():
        if f.suffix:      # skip *.nrrd, *.png, …
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            uid = getattr(ds, "SeriesInstanceUID", None) or "NOUID"
            series_map.setdefault(uid, []).append(str(f))
        except Exception:
            continue

    # fallback: SimpleITK + GDCM if we found nothing
    if not series_map:
        try:
            ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(folder)) or []
            for uid in ids:
                series_map[uid] = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
                    str(folder), uid)
        except RuntimeError:
            pass

    # deduplicate + geometry sort
    for uid, files in series_map.items():
        uniq: Dict[str, str] = {}
        for fp in files:
            stem = strip_dup_suffix(Path(fp).name, suffixes)
            uniq.setdefault(stem, fp)
        series_map[uid] = sorted(uniq.values(), key=_slice_key)

    # drop scout series
    return {k: v for k, v in series_map.items() if len(v) >= 3}


def choose_series(series_map: Dict[str, List[str]]) \
        -> Tuple[str, List[str]]:
    """
    Pick the most likely FLAIR series.

    1. prefer metadata that contains any *KEYWORDS*;
    2. otherwise choose the largest slice count.
    """

    def keyword_score(files: List[str]) -> int:
        hdr = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
        for tag in ("SeriesDescription", "ProtocolName", "SequenceName"):
            if any(k in str(getattr(hdr, tag, "")).lower() for k in KEYWORDS):
                return 1
        return 0

    return max(series_map.items(),
               key=lambda kv: (keyword_score(kv[1]), len(kv[1])))


def write_nrrd(files: List[str], out_path: Path, compress: bool = True) -> None:
    """Stack *files* → NRRD with LPS orientation."""
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    img = reader.Execute()

    orient = sitk.DICOMOrientImageFilter()
    orient.SetDesiredCoordinateOrientation("LPS")
    img = orient.Execute(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path), useCompression=compress)


def determinant_direction(img: sitk.Image) -> float:
    """Return det(3x3 direction matrix)."""
    d = img.GetDirection()
    return (d[0] * (d[4] * d[8] - d[5] * d[7])
            - d[1] * (d[3] * d[8] - d[5] * d[6])
            + d[2] * (d[3] * d[7] - d[4] * d[6]))


def copy_preexisting_nrrd(candidates: List[Path],
                          dst: Path,
                          overwrite: bool) -> Path:
    """Copy largest NRRD among *candidates* to *dst* (unless dst exists)."""
    best = max(candidates, key=lambda p: p.stat().st_size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        logging.info(f"{dst} exists – keeping (overwrite=False)")
        return dst
    shutil.copy2(best, dst)
    logging.info(f"{dst.parent.name}: copied existing → {dst.name}")
    return dst

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def batch_convert_flair(dataset_root: Path,
                        output_root: Path,
                        flair_dirname: str = "flair",
                        duplicate_suffixes: Sequence[str] = DUP_SUFFIXES,
                        overwrite: bool = False) -> pd.DataFrame:
    """
    Convert every patient’s FLAIR series under *dataset_root* and write to
    *output_root/FLAIR/<PatientID>/FLAIR_<PatientID>.nrrd*.

    Returns a dataframe with QA information.
    """
    rows = []
    patients = discover_flair_folders(dataset_root, flair_dirname)

    for pid, flair_path in patients.items():
        # -------------------------------------------------- pre-existing NRRD
        nrrd_candidates = [p for p in flair_path.glob("*.nrrd")
                           if re.search(r"flair", p.name, flags=re.I)]
        out_path = output_root / "FLAIR" / pid / f"FLAIR_{pid}.nrrd"

        if nrrd_candidates:
            try:
                copy_preexisting_nrrd(nrrd_candidates, out_path, overwrite)
                rows.append(dict(PatientID=pid, Written=True,
                                 Copied=True, OutFile=str(out_path)))
            except Exception as e:
                logging.exception(f"{pid}: copy failed")
                rows.append(dict(PatientID=pid, Written=False,
                                 Reason=str(e), OutFile=str(out_path)))
            continue

        # -------------------------------------------------- discover series
        series_map = collect_series_files(flair_path, duplicate_suffixes)
        if not series_map:
            logging.error(f"{pid}: no usable DICOM series")
            rows.append(dict(PatientID=pid, Written=False,
                             Reason="no_dicom", OutFile=None))
            continue

        uid, files = choose_series(series_map)
        logging.info(f"{pid}: {len(files)} slices chosen (UID={uid})")

        # -------------------------------------------------- write NRRD
        try:
            write_nrrd(files, out_path)
            img = sitk.ReadImage(str(out_path))

            rows.append(dict(PatientID=pid, Written=True,
                             Slices=len(files),
                             VoxelSpacing=img.GetSpacing(),
                             DirectionDet=round(determinant_direction(img), 3),
                             OutFile=str(out_path)))
        except Exception as e:
            logging.exception(f"{pid}: conversion failed")
            rows.append(dict(PatientID=pid, Written=False,
                             Reason=str(e), OutFile=str(out_path)))

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Convert FLAIR DICOM series to NRRD.")
    ap.add_argument("dataset_root", type=Path,
                    help="Root folder containing P1/, P2/, …")
    ap.add_argument("output_root", type=Path,
                    help="Destination root; subfolder 'FLAIR/' is created")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing NRRD files")
    args = ap.parse_args()

    summary = batch_convert_flair(args.dataset_root,
                                  args.output_root,
                                  overwrite=args.overwrite)
    csv_path = args.output_root / "flair_conversion_summary.csv"
    summary.to_csv(csv_path, index=False)
    logging.info(f"Summary written to {csv_path}")
