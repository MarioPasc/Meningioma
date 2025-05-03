"""
flair2nrrd.py

Batch-convert FLAIR DICOM series (files may have no .dcm extension)
into *.nrrd* while keeping geometry, deduplicating anonymiser clones
(e.g. *_an, *_anon), and writing a neat cohort summary.

Author : ChatGPT (OpenAI) for Mario Researcher
Date   : 2025-05-03
"""

from __future__ import annotations
from pathlib import Path
from typing   import Dict, List, Sequence, Tuple, Optional
import logging
import re

import pydicom
import pandas as pd
import SimpleITK as sitk


# ---------------------------------------------------------------------
# Global configuration ------------------------------------------------
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s | %(message)s")

# accepted duplicate suffixes (editable by the user)
DUP_SUFFIXES: List[str] = ["_an", "_anon"]

# DICOM tags that carry geometry; used for QA bookkeeping
DICOM_GEOMETRY_KEYS: Tuple[str, ...] = (
    "0020|0032",  # Image Position (Patient)
    "0020|0037",  # Image Orientation (Patient)
    "0028|0030",  # Pixel Spacing
    "0018|0050",  # Slice Thickness
    "0018|0088",  # Spacing Between Slices
)

# Series whose metadata contains any of the keywords
# {"FLAIR", "T2 FLAIR", "T2FLAIR"} in *SeriesDescription*,
# *ProtocolName* or *SequenceName* (case-insensitive).
KEYWORDS = ("flair", "t2 flair", "t2flair")

# ---------------------------------------------------------------------
# Helper functions ----------------------------------------------------
# ---------------------------------------------------------------------
def discover_flair_folders(dataset_root: Path, flair_dirname: str = "flair") \
        -> Dict[str, Path]:
    """
    Walk *dataset_root* and return {PatientID: Path-to-FLAIR-folder}.
    A patient folder is any immediate subdirectory that contains a
    subdirectory *flair_dirname*.
    """
    out: Dict[str, Path] = {}
    for pdir in sorted(dataset_root.iterdir()):
        flair_path = pdir / flair_dirname
        if flair_path.is_dir():
            out[pdir.name] = flair_path
    return out


def strip_dup_suffix(name: str, suffixes: Sequence[str]) -> str:
    """
    Remove *one* occurrence of any suffix in *suffixes* from *name*.
    """
    for suf in suffixes:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def collect_series_files(folder: Path,
                         suffixes: Sequence[str]) -> Dict[str, List[str]]:
    """
    Group DICOM slices into series.

    Returns
    -------
    dict  UID/NOUID-k → list[path strings]  (≥ 3 slices each)
    """
    series_map: Dict[str, List[str]] = {}
    for f in folder.iterdir():
        if f.suffix:                          # ignore *.nrrd, *.png, …
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            uid = getattr(ds, "SeriesInstanceUID", None)
            key = uid if uid else "NOUID"
            series_map.setdefault(key, []).append(str(f))
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

    # deduplicate by stripping anonymiser suffixes
    for uid, files in series_map.items():
        uniq: Dict[str, str] = {}
        for fp in files:
            stem = Path(fp).name
            stem = strip_dup_suffix(stem, suffixes)
            uniq.setdefault(stem, fp)         # keep first occurrence
        series_map[uid] = sorted(uniq.values())

    # drop scout series (<3 slices)
    series_map = {k: v for k, v in series_map.items() if len(v) >= 3}
    return series_map


def choose_series(series_map: Dict[str, List[str]]) -> Tuple[str, List[str]]:
    """
    Pick the most likely FLAIR series.

    Priority
    --------
    1. Series whose metadata contains any of the keywords
       {"FLAIR", "T2 FLAIR", "T2FLAIR"} in *SeriesDescription*,
       *ProtocolName* or *SequenceName* (case-insensitive).
    2. If none found, largest slice count.
    """

    def keyword_score(files: List[str]) -> int:
        head = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
        for tag in ("SeriesDescription", "ProtocolName", "SequenceName"):
            val = getattr(head, tag, "")
            if any(k in str(val).lower() for k in KEYWORDS):
                return 1
        return 0

    scored = sorted(series_map.items(),
                    key=lambda kv: (keyword_score(kv[1]), len(kv[1])),
                    reverse=True)
    return scored[0]


def write_nrrd(files: List[str], out_path: Path, compress: bool = True) -> None:
    """
    Load *files* with SimpleITK and write a *.nrrd* volume to *out_path*.
    """
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    img = reader.Execute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path), useCompression=compress)


def determinant_direction(img: sitk.Image) -> float:
    """
    Compute det(DirMatrix) where DirMatrix is 3×3 direction cosines.
    """
    dir_flat = img.GetDirection()
    det = (
        dir_flat[0] * (dir_flat[4] * dir_flat[8] - dir_flat[5] * dir_flat[7])
        - dir_flat[1] * (dir_flat[3] * dir_flat[8] - dir_flat[5] * dir_flat[6])
        + dir_flat[2] * (dir_flat[3] * dir_flat[7] - dir_flat[4] * dir_flat[6])
    )
    return det


# ---------------------------------------------------------------------
# Public API ----------------------------------------------------------
# ---------------------------------------------------------------------
def batch_convert_flair(dataset_root: Path,
                        output_root: Path,
                        flair_dirname: str = "flair",
                        duplicate_suffixes: Sequence[str] = DUP_SUFFIXES,
                        overwrite: bool = False) -> pd.DataFrame:
    """
    Convert every patient’s FLAIR series under *dataset_root* to NRRD and
    write it to *output_root/FLAIR/<PatientID>/FLAIR_<PatientID>.nrrd*.

    A patient is **skipped** if an NRRD with “FLAIR” in its name already
    exists in their input FLAIR folder.
    """
    rows = []
    patients = discover_flair_folders(dataset_root, flair_dirname)

    for pid, flair_path in patients.items():
        # --- pre-existing NRRD? --------------------------------------
        existing = [p for p in flair_path.glob("*.nrrd")
                    if re.search(r"flair", p.name, flags=re.I)]
        if existing and not overwrite:
            logging.info(f"{pid}: found existing NRRD → skip")
            rows.append(dict(PatientID=pid,
                             Written=False,
                             Reason="already_exists",
                             OutFile=str(existing[0])))
            continue

        # --- discover series ----------------------------------------
        series_map = collect_series_files(flair_path, duplicate_suffixes)
        if not series_map:
            logging.error(f"{pid}: no usable DICOM series")
            rows.append(dict(PatientID=pid,
                             Written=False,
                             Reason="no_dicom",
                             OutFile=None))
            continue

        uid, files = choose_series(series_map)
        logging.info(f"{pid}: {len(files)} slices chosen (series key={uid})")

        # --- write ---------------------------------------------------
        out_path = (output_root / "FLAIR" / pid /
                    f"FLAIR_{pid}.nrrd")
        try:
            write_nrrd(files, out_path)
            img = sitk.ReadImage(str(out_path))
            det = determinant_direction(img)

            # QA fields
            rows.append(dict(PatientID=pid,
                             Written=True,
                             Slices=len(files),
                             VoxelSpacing=img.GetSpacing(),
                             DirectionDet=round(det, 3),
                             OutFile=str(out_path)))
        except Exception as e:
            logging.exception(f"{pid}: conversion failed")
            rows.append(dict(PatientID=pid,
                             Written=False,
                             Reason=str(e),
                             OutFile=str(out_path)))

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------
# Guard-block (optional) ---------------------------------------------
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Convert FLAIR DICOM series to NRRD.")
    ap.add_argument("dataset_root", type=Path,
                    help="Root folder that contains P1/, P2/, …")
    ap.add_argument("output_root", type=Path,
                    help="Destination root; subfolder 'FLAIR/' is created")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing NRRD files")
    args = ap.parse_args()

    summary = batch_convert_flair(args.dataset_root,
                                  args.output_root,
                                  overwrite=args.overwrite)
    csv_path = args.output_root / "flair_conversion_summary.csv"
    summary.to_csv(csv_path, index=False)
    logging.info(f"Summary written to {csv_path}")
