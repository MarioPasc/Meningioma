#!/usr/bin/env python3
"""
Re-organise meningioma longitudinal data *and* convert each NRRD to NIfTI-1.

New behaviour (May-2025)
------------------------
1.  Every attempt to locate a file (baseline & follow-up) is logged.
2.  Found NRRDs are converted with `nifti_write_3d` and written as .nii.gz.
3.  The logger is the package-level LOGGER defined in
       mgmGrowth.preprocessing.__init__
4.  In controls, detect these filename‐pulse mappings:
     - "T1-PRE_*"   → T1
     - "T1W-FS_*"   → T1
     - "T2-FLAR_*"  → FLAIR
     - otherwise, use the prefix before the first underscore as pulse
5.  **Each control folder (visit) now gets one single control index** so that
    all pulses from the same visit share the same `{controlid}`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from mgmGrowth.preprocessing import LOGGER
from mgmGrowth.preprocessing.tools.nrrd_to_nifti import nifti_write_3d

# map filename prefixes to canonical pulses
ALIAS_TO_PULSE: Dict[str, str] = {
    "T1":       "T1",
    "T1-Gd":   "T1",
    "T1W-FS":   "T1",
    "T2":       "T2",
    "T2-FLAR":  "FLAIR",
    "FLAIR":    "FLAIR",
    "SWI":      "SWI",
    "SUSC":     "SWI",
    "TC":       "TC",
    "T1SIN":    "T1SIN",
}

def enumerate_patients(followup_root: Path) -> List[str]:
    """Sorted list of patient folder names (e.g. ['P1', 'P15'])."""
    return sorted(p.name for p in followup_root.iterdir() if p.is_dir())

def build_id_map(patient_dirs: Sequence[str]) -> Dict[str, str]:
    """Injective mapping old-id → zero-padded 5-digit new-id."""
    return {old: f"{idx:05d}" for idx, old in enumerate(patient_dirs, start=1)}

def baseline_file(baseline_root: Path, pulse: str, old_id: str) -> Path:
    """Return full path to the expected baseline NRRD."""
    if pulse == "TC":
        return baseline_root / "TC" / old_id / f"{pulse}_{old_id}.nrrd"
    return baseline_root / "RM" / pulse / old_id / f"{pulse}_{old_id}.nrrd"

def convert_if_exists(src: Path, dst: Path) -> bool:
    """
    Convert *src* NRRD to NIfTI-1 at *dst* (must end with '.nii.gz').

    Returns
    -------
    True  – success (file written)
    False – file missing or conversion failed (already logged)
    """
    if not src.exists():
        LOGGER.warning("Missing file: %s", src)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        nifti_write_3d(str(src), str(dst), verbose=False)
        LOGGER.info("Converted %s → %s", src, dst)
        return True
    except Exception as exc:
        LOGGER.error("Conversion failed for %s : %s", src, exc)
        return False

def find_control_dirs(patient_root: Path) -> List[Path]:
    """Return control* directories (sorted alphanumerically)."""
    return sorted(d for d in patient_root.iterdir() if d.is_dir() and d.name.startswith("control"))

# --------------------------------------------------------------------------- #
#                           core orchestration                                #
# --------------------------------------------------------------------------- #

def reorg_patient(
    old_id: str,
    new_idx: str,
    pulses: Sequence[str],
    baseline_root: Path,
    followup_root: Path,
    out_root: Path,
) -> None:
    """Handle one patient (baseline + all follow-ups)."""

    # prepare output folder for this patient
    patient_out = out_root / f"MenGrowth-{new_idx}"
    patient_out.mkdir(parents=True, exist_ok=True)

    # ---------- baseline --------------------------------------------------- #
    for pulse in pulses:
        src = baseline_file(baseline_root, pulse, old_id)
        dst = patient_out / f"MenGrowth-{new_idx}-0000-{pulse}.nii.gz"
        LOGGER.info("Searching baseline %s for %s → %s", pulse, old_id, src)
        convert_if_exists(src, dst)

    # ---------- longitudinal controls ------------------------------------- #
    patient_follow = followup_root / old_id
    control_dirs   = find_control_dirs(patient_follow)
    LOGGER.info("Control visits for %s : %s", old_id, [c.name for c in control_dirs])

    control_idx = 1
    # loop once per control visit folder
    for control_dir in control_dirs:
        LOGGER.info("Processing visit %s (will use control ID %04d)", control_dir.name, control_idx)
        found_any = False

        # scan all modalities in this one visit
        for src in control_dir.glob("*.nrrd"):
            if src.name.endswith("_seg.nrrd"):
                continue

            # detect prefix and map to canonical pulse
            prefix = src.name.split("_", 1)[0]
            pulse = ALIAS_TO_PULSE.get(prefix)
            if pulse is None:
                LOGGER.warning("Unrecognized pulse prefix '%s' in %s; skipping", prefix, src)
                continue

            LOGGER.info("Assigning %s → pulse '%s'", src.name, pulse)
            dst = patient_out / f"MenGrowth-{new_idx}-{control_idx:04d}-{pulse}.nii.gz"
            if convert_if_exists(src, dst):
                found_any = True

        # only bump control_idx if this visit actually had at least one file
        if found_any:
            control_idx += 1
        else:
            LOGGER.warning("No valid NRRD found in %s; control index unchanged", control_dir)

# --------------------------------------------------------------------------- #
#                               CLI wrapper                                   #
# ---------------------------------------------------------------------------

def run(baseline: Path, followup: Path, out_root: Path, pulses: Sequence[str]) -> None:
    """End-to-end execution."""
    patients = enumerate_patients(followup)
    id_map   = build_id_map(patients)

    out_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Patients to process: %s", patients)

    # save mapping old→new IDs
    with open(out_root / "patient_id_map.json", "w") as fp:
        json.dump(id_map, fp, indent=2)
    LOGGER.info("Wrote patient_id_map.json")

    # process each patient
    for old_id, new_idx in id_map.items():
        LOGGER.info("=== Patient %s → %s ===", old_id, new_idx)
        reorg_patient(
            old_id, new_idx, pulses,
            baseline_root=baseline,
            followup_root=followup,
            out_root=out_root,
        )

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Re-organise meningioma dataset.")
    parser.add_argument("--baseline", type=Path, default=Path(
        "/home/mpascual/research/datasets/meningiomas/raw/baseline"
    ))
    parser.add_argument("--followup", type=Path, default=Path(
        "/home/mpascual/research/datasets/meningiomas/raw/controls"
    ))
    parser.add_argument("--out", type=Path, default=Path(
        "/home/mpascual/research/datasets/meningiomas/raw/MenGrowth-2025"
    ))
    parser.add_argument(
        "--pulses", default="T1,T2,SWI,FLAIR,TC",
        help="Comma-separated pulse list"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(
        baseline=args.baseline,
        followup=args.followup,
        out_root=args.out,
        pulses=[p.strip() for p in args.pulses.split(",") if p.strip()],
    )
