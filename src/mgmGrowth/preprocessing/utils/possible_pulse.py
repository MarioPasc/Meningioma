#!/usr/bin/env python3
"""
possible_pulse.py  –  v4  (adds axial-only & extra text rules)

New heuristics requested 2025-05-18
──────────────────────────────────
1.  “T2” **&** “GRE”  →  SWI
2.  “T2” **&** “FSE”/“TSE”/“FRFSE”  →  T2       (override SWI)
3.  filename contains “imageOrientationPatient” → ignore series
4.  filename contains “craneo”                  → TC (CT modality)
5.  exclude non-axial: any of  sag, sagittal, cor, coronal  (case-ins.)
"""

from __future__ import annotations
import argparse, csv, json, logging, re, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence
import pydicom
from pydicom.errors import InvalidDicomError

# ── pydicom tolerant mode ─────────────────────────────────────────────── #
pydicom.config.enforce_valid_values = False
pydicom.config.debug(False)

# ── logging ───────────────────────────────────────────────────────────── #
logging.basicConfig(stream=sys.stdout,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    level=logging.INFO)
LOG = logging.getLogger("pulse_guess")

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

# ── directory scanning (skip logic added) ─────────────────────────────── #
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
            LOG.warning("cannot read %s (%s)", rep, exc)
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
            LOG.warning("Skipping %s (%s)", f.name, exc); continue
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
            LOG.info("%s/%s : %d pulse classes (%s)",
                     pid, ctrl_dir.name, len(pmap), ", ".join(pmap))
    return rows

# ── CSV IO and CLI ─────────────────────────────────────────────────────── #
def write_csv(rows: Sequence[Dict[str, str]], out_csv: Path) -> None:
    rows = [r for r in rows if r["possible_pulse"] != "SKIP"]
    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["patient", "control", "old_names", "possible_pulse"])
        w.writeheader(); w.writerows(rows)
    LOG.info("CSV written to %s (%d rows)", out_csv, len(rows))

# ─────────────────────────────── CLI ─────────────────────────────────────── #
def parse_args() -> argparse.Namespace:
    pr = argparse.ArgumentParser(description="Guess pulses for control visits.")
    pr.add_argument(
        "--men-root",
        type=Path,
        default=Path("/home/mariopasc/Python/Datasets/Meningiomas/raw/men"),
        help="Folder that contains P*/control* trees",
    )
    pr.add_argument(
        "--out-csv",
        type=Path,
        default=Path("pulse_guess.csv"),
        help="Destination CSV file",
    )
    return pr.parse_args()


def main() -> None:
    args = parse_args()
    if not args.men_root.is_dir():
        LOG.error("men-root %s does not exist or is not a directory", args.men_root)
        sys.exit(1)

    rows = scan_men_root(args.men_root)
    write_csv(rows, args.out_csv)


if __name__ == "__main__":
    main()
