#!/usr/bin/env python3
"""
possible_pulse.py · v9 – 19 May 2025

v9 additions
------------
• --logging-file : duplicate loggingger stream to a file
• _sitk_stack_and_write now drops unreadable DICOM slices
  and skips a volume when < 3 usable slices remain.
"""

from __future__ import annotations
import argparse, csv, json, logging, shutil, sys, re, numpy as np
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Sequence, Tuple

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError
import SimpleITK as sitk

# ──────────────────────── logging set-up ─────────────────────────────── #
def _config_logging(logging_file: Path | None) -> None:
    handlers = [logging.StreamHandler(sys.stdout)]
    if logging_file:
        handlers.append(logging.FileHandler(logging_file, mode="w"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        handlers=handlers,
    )
pydicom.config.enforce_valid_values = False

ACCEPTED_PULSES = {"SWI", "T2", "T1", "TC", "CT", "UNKNOWN"}
pydicom.config.enforce_valid_values = False            # tolerate bad UIDs

# ── simple helpers ────────────────────────────────────────────────────── #
def _is_nrrd(p: Path)  -> bool: return p.suffix.lower() == ".nrrd"
def _is_nifti(p: Path) -> bool: return p.suffix.lower() in {".nii"} or p.name.lower().endswith(".nii.gz")

# ── skip lists ────────────────────────────────────────────────────────── #
NON_IMAGE = {"DICOMDIR", "LOCKFILE", "VERSION", "CDEPXYVS"}
_NON_AXIAL_RE = re.compile(r"\b(sagitt?al|sag\b|coronal|cor\b)", re.I)
_PRE_STUDY_RE = re.compile(r"imageorientationpatient", re.I)

def should_skip_name(name: str) -> bool:
    """Return True if *name* should be ignored outright."""
    if name.upper() in NON_IMAGE:
        return True
    if _PRE_STUDY_RE.search(name):
        return True
    if _NON_AXIAL_RE.search(name):
        return True
    return False

# ── robust pulse dictionary (with precedence) ─────────────────────────── #
_SPECIAL_SWI_RE = re.compile(r"t2.*gre|gre.*t2",             re.I)  # rule 1
_SPECIAL_T2_RE  = re.compile(r"t2.*(fse|tse|frfse)",         re.I)  # rule 1b

_RX: dict[str, re.Pattern[str]] = {
    # ordinary keywords (evaluated after the specials)
    "T1Gd": re.compile(r"(gad|gd|contr|post[- ]?g?d)",        re.I),
    "DWI":  re.compile(r"(dwi|diff|difusion|b[0-9]{3,4})",    re.I),
    "FLAIR":re.compile(r"(flair|cube ?flair|fame)",           re.I),
    "SWI":  re.compile(r"(swi|t2\*|gre|swan|medic)",          re.I),
    "T2":   re.compile(r"(t2|frfse|tse)",                    re.I),
    "T1":   re.compile(r"(t1|bravo|mprage|spgr|fspgr|sesag)", re.I),
}

_TR_SHORT, _TE_LONG, _TI_FLAIR = 800, 70, (2000, 3000)

# ── heuristic helpers ─────────────────────────────────────────────────── #
def _guess_from_text(txt: str) -> str | None:
    # --- special precedence rules -------------------------------------- #
    if _SPECIAL_SWI_RE.search(txt):
        return "SWI"
    if _SPECIAL_T2_RE.search(txt):
        return "T2"
    if "craneo" in txt.lower():
        return "TC"   # CT of the skull
    # --- normal keyword scan ------------------------------------------- #
    for pulse, rx in _RX.items():
        if rx.search(txt):
            return pulse
    return None

def _guess_from_numeric(tr: float | None, te: float | None, ti: float | None) -> str | None:
    if tr is None or te is None:
        return None
    if ti and _TI_FLAIR[0] <= ti <= _TI_FLAIR[1]:
        return "FLAIR"
    if tr <= _TR_SHORT and te < 20:
        return "T1"
    if tr >= 1500 and te >= _TE_LONG:
        return "T2"
    return None

# ── DICOM classifier ──────────────────────────────────────────────────── #
def guess_pulse_from_dicom(ds: pydicom.Dataset) -> str:
    if ds.get("Modality", "").upper() == "CT":
        return "TC"

    txt = " ".join(str(ds.get(t, "")) for t in ("SeriesDescription", "SequenceName"))
    pulse = _guess_from_text(txt)
    if pulse:
        return pulse

    # DWI via b-value
    b_val = ds.get(("0018", "9087")) or ds.get("DiffusionBValue")
    try:
        if b_val and float(b_val) >= 500:
            return "DWI"
    except (TypeError, ValueError):
        pass

    # Post-contrast T1
    if ds.get("ContrastBolusAgent") or ds.get("ContrastBolusVolume"):
        return "T1Gd"

    tr = float(ds.get("RepetitionTime", 0) or 0)
    te = float(ds.get("EchoTime", 0) or 0)
    ti = float(ds.get("InversionTime", 0) or 0)
    return _guess_from_numeric(tr, te, ti) or "UNKNOWN"

def guess_pulse_from_filename(name: str) -> str | None:
    return _guess_from_text(name)

# ── directory scanning (skip loggingic added) ─────────────────────────────── #
def representative_dicom(series_dir: Path) -> Path | None:
    for f in series_dir.iterdir():
        if f.is_file() and not f.name.startswith("."):
            return f
    return None

def classify_dir_series(series_dir: Path) -> tuple[str, list[str]]:
    if should_skip_name(series_dir.name):
        return "SKIP", []
    rep = representative_dicom(series_dir)
    if rep:
        try:
            ds = pydicom.dcmread(rep, stop_before_pixels=True, force=True)
            pulse = guess_pulse_from_dicom(ds)
        except Exception as exc:
            logging.warning("cannot read %s (%s)", rep, exc)
            pulse = "UNKNOWN"
    else:
        pulse = "UNKNOWN"
    return pulse, [series_dir.name]

def classify_file_series(first_file: Path, files: list[Path]) -> tuple[str, list[str]]:
    if should_skip_name(first_file.name):
        return "SKIP", []
    try:
        ds = pydicom.dcmread(first_file, stop_before_pixels=True, force=True)
        pulse = guess_pulse_from_dicom(ds)
    except Exception:
        pulse = guess_pulse_from_filename(first_file.name) or "UNKNOWN"
    return pulse, [f.name for f in files]

def scan_control_dir(ctrl_dir: Path) -> Dict[str, list[str]]:
    pulse_map: dict[str, list[str]] = defaultdict(list)

    # 1) sub-directories
    for d in (p for p in ctrl_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        pulse, names = classify_dir_series(d)
        if pulse != "SKIP":
            pulse_map[pulse].extend(names)

    # 2a) export files
    for f in (p for p in ctrl_dir.iterdir() if p.is_file()
              and not p.name.lower().endswith((".mrml", ".png", ".json"))
              and not should_skip_name(p.name)
              and (_is_nrrd(p) or _is_nifti(p))):
        pulse = guess_pulse_from_filename(f.name) or "UNKNOWN"
        pulse_map[pulse].append(f.name)

    # 2b) loose DICOM slices
    slices = [p for p in ctrl_dir.iterdir() if p.is_file()
              and not _is_nrrd(p) and not _is_nifti(p) and not p.name.lower().endswith((".mrml", ".png", ".json"))
              and not p.name.startswith(".") and not should_skip_name(p.name)]
    groups: dict[str, list[Path]] = defaultdict(list)
    headers: dict[str, Path] = {}
    for f in slices:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True,
                                 specific_tags=[
                                     "SeriesInstanceUID", "StudyInstanceUID", "Modality",
                                     "SeriesDescription", "SequenceName",
                                     "RepetitionTime", "EchoTime", "InversionTime",
                                     ("0018","9087"), "ContrastBolusAgent",
                                 ])
        except (InvalidDicomError, Exception) as exc:
            logging.warning("Skipping %s (%s)", f.name, exc); continue
        uid = ds.get("SeriesInstanceUID") or ds.get("StudyInstanceUID") or f.name
        groups[uid].append(f); headers.setdefault(uid, f)

    for uid, files in groups.items():
        pulse, names = classify_file_series(headers[uid], files)
        if pulse != "SKIP":
            pulse_map[pulse].extend(names)

    return pulse_map

def scan_men_root(men_root: Path) -> Sequence[Dict[str, str]]:
    rows: list[dict[str, str]] = []
    for patient_dir in sorted(p for p in men_root.iterdir() if p.is_dir()):
        pid = patient_dir.name
        for ctrl_dir in sorted(d for d in patient_dir.iterdir()
                               if d.is_dir() and d.name.startswith("control")):
            pmap = scan_control_dir(ctrl_dir)
            for pulse, names in pmap.items():
                rows.append({"patient": pid, "control": ctrl_dir.name,
                             "old_names": json.dumps(sorted(names)),
                             "possible_pulse": pulse})
            logging.info("%s/%s : %d pulse classes (%s)",
                     pid, ctrl_dir.name, len(pmap), ", ".join(pmap))
    return rows

# ─────────────────────────── CSV helper ───────────────────────────────── #
def write_csv(rows: Sequence[Dict[str, str]], out_csv: Path) -> None:
    rows = [r for r in rows if r["possible_pulse"] != "SKIP"]
    with out_csv.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["patient", "control", "old_names", "possible_pulse"]
        )
        writer.writeheader()
        writer.writerows(rows)
    logging.info("CSV written to %s (%d rows)", out_csv, len(rows))

# ────────────────────── slice ordering helper (from flair2nrrd) ───────── #
def _slice_key(path: Path) -> Tuple[float, int]:
    """
    Return sortable key (distance along slice-normal, InstanceNumber).
    Invalid files get (inf, 0) so they sort last.
    """
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        pos = np.asarray(ds.ImagePositionPatient, dtype=float)
        orient = np.asarray(ds.ImageOrientationPatient, dtype=float)
        normal = np.cross(orient[:3], orient[3:])
        loc = float(np.dot(pos, normal))
        inst = int(getattr(ds, "InstanceNumber", 0))
        return (loc, inst)
    except Exception:
        return (np.inf, 0)
    
def _is_readable_dicom(path: Path) -> bool:
    """True if pydicom can read header AND PixelData."""
    try:
        ds = pydicom.dcmread(path, force=True)
        _ = ds.pixel_array  # forces GDCM / pylibjpeg to touch PixelData
        return True
    except Exception:
        return False
# ────────────────────── improved grouping / logging  ──────────────────── #
def _slice_signature(ds: pydicom.Dataset) -> Tuple[str, int, int, str]:
    uid = str(ds.get("SeriesInstanceUID", "")).strip()
    rows = int(ds.get("Rows", 0) or 0)
    cols = int(ds.get("Columns", 0) or 0)
    desc = str(ds.get("SeriesDescription", "")).strip()
    return uid, rows, cols, desc

def _group_loose_slices(files: List[Path],
                        pid: str, ctrl: str, pulse: str) -> Dict[str, List[Path]]:
    groups: DefaultDict[str, List[Path]] = defaultdict(list)

    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True,
                                 specific_tags=["SeriesInstanceUID",
                                                "Rows", "Columns",
                                                "SeriesDescription"])
            sig = _slice_signature(ds)
        except Exception as exc:
            logging.warning("Skipping non-DICOM %s (%s)", f.name, exc)
            continue
        key = "|".join(map(str, sig))
        groups[key].append(f)

    # sort & logging
    for key, sl in groups.items():
        uid, r, c, desc = key.split("|", 3)
        logging.info("%s/%s %s  UID=%s  %sx%s  slices=%d  desc='%s'",
                 pid, ctrl, pulse, uid or "—", r, c, len(sl), desc[:30])
        groups[key] = sorted(sl, key=_slice_key)

    return groups

# ────────────────────── SimpleITK helpers  ────────────────────────────── #
def _sitk_stack_and_write(slices: List[Path], dst: Path) -> None:
    good = [p for p in slices if _is_readable_dicom(p)]
    bad  = set(slices) - set(good)
    for b in bad:
        logging.warning("Unreadable slice skipped: %s", b.name)

    if len(good) < 3:
        logging.error("Volume %s has <3 readable slices – skipped", dst)
        return

    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.SetFileNames([str(p) for p in sorted(good, key=_slice_key)])
    img = reader.Execute()
    sitk.WriteImage(img, str(dst))

def _copy_nrrd(src: Path, dst_stem: Path) -> None:
    if src.suffix.lower() == ".nrrd":
        shutil.copy2(src, dst_stem.with_suffix(".nrrd"))
    else:
        shutil.copy2(src, dst_stem.with_suffix(".nhdr"))
        twin = src.with_suffix(".raw.gz") if src.with_suffix(".raw.gz").exists() \
               else src.with_suffix(".raw")
        if twin.exists():
            shutil.copy2(twin, dst_stem.with_suffix(twin.suffix))

# ────────────────────── dataset builder (unchanged except order) ──────── #
def _series_to_nrrd(series: List[Path], dst_stem: Path) -> None:
    if len(series) == 1 and series[0].suffix.lower() in {".nrrd", ".nhdr"}:
        _copy_nrrd(series[0], dst_stem)
        return

    files: List[Path] = []
    for p in series:
        files.extend([p] if p.is_file() else [q for q in p.iterdir() if q.is_file()])

    by_shape: DefaultDict[Tuple[int, int], List[Path]] = defaultdict(list)
    for f in files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True,
                                 specific_tags=["Rows", "Columns"])
            shape = (int(ds.get("Rows", 0) or 0), int(ds.get("Columns", 0) or 0))
        except Exception:
            shape = (-1, -1)
        by_shape[shape].append(f)

    for suffix, fset in enumerate(by_shape.values(), start=1):
        out = dst_stem.with_suffix(".nrrd") if len(by_shape) == 1 \
              else dst_stem.with_name(f"{dst_stem.name}_{suffix:02d}.nrrd")
        _sitk_stack_and_write(sorted(fset, key=_slice_key), out)

def build_dataset(csv_path: Path, men_root: Path, out_root: Path) -> None:
    rows = pd.read_csv(csv_path)
    rows = rows[rows["possible_pulse"].isin(ACCEPTED_PULSES)]
    counter: DefaultDict[Tuple[str, str, str], int] = defaultdict(int)

    for _, row in rows.iterrows():
        pid, ctrl, pulse = row.patient, row.control, row.possible_pulse
        src_ctrl, dst_ctrl = men_root / pid / ctrl, out_root / pid / ctrl
        dst_ctrl.mkdir(parents=True, exist_ok=True)

        names = json.loads(row.old_names)
        paths = [src_ctrl / n for n in names]

        ready = [p for p in paths if p.suffix.lower() in {".nrrd", ".nhdr"}]
        dirs  = [p for p in paths if p.is_dir()]
        slices = [p for p in paths if p.is_file() and p.suffix == "" and p not in ready]

        for ser in ready + dirs:
            idx = counter[(pid, ctrl, pulse)] = counter[(pid, ctrl, pulse)] + 1
            _series_to_nrrd([ser], dst_ctrl / f"{pulse}_{pid}_{idx:02d}")

        for _, slice_set in _group_loose_slices(slices, pid, ctrl, pulse).items():
            idx = counter[(pid, ctrl, pulse)] = counter[(pid, ctrl, pulse)] + 1
            _series_to_nrrd(slice_set, dst_ctrl / f"{pulse}_{pid}_{idx:02d}")

    logging.info("Dataset written to %s", out_root)


def _arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Classify pulses and (optionally) build dataset.")
    ap.add_argument(
        "--men-root",
        type=Path,
        default=Path("/home/mpascual/research/datasets/meningiomas/raw/controls"),
        help="Folder that contains P*/control* trees",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/home/mpascual/research/datasets/meningiomas/raw/misc/pulse_guess.csv"),
        help="Destination CSV file",
    )
    ap.add_argument("--dataset-out", type=Path,
                    help="If given, create a copy of the controls here "
                         "(no flair folders, only accepted pulses)",
                         default=Path("/home/mpascual/research/datasets/meningiomas/raw/control_cured"))
    ap.add_argument("--rescan",       action="store_true",
                    help="Force rescanning men-root even if out-csv exists")
    ap.add_argument("--log-file",    type=Path, help="Write logging to file",
                    default=Path("./logging_possible_pulse.txt"))
    return ap.parse_args()

def main() -> None:
    args = _arg_parser()
    _config_logging(args.log_file)

    if not args.men_root.is_dir():
        logging.error("%s is not a directory", args.men_root); sys.exit(1)

    if args.rescan or not args.out_csv.exists():
        rows = scan_men_root(args.men_root)  # from previous version
        write_csv(rows, args.out_csv)
    else:
        logging.info("CSV %s exists – reuse (pass --rescan to regenerate)", args.out_csv)

    if args.dataset_out:
        build_dataset(args.out_csv, args.men_root, args.dataset_out)  # unchanged


if __name__ == "__main__":
    main()
