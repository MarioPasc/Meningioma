#!/usr/bin/env python3
"""
Re-organise meningioma longitudinal data *and* convert each NRRD to NIfTI-1.

New behaviour (May-2025)
------------------------
1.  Every attempt to locate a file (baseline & follow-up) is logged.
2.  Found NRRDs are converted with `nifti_write_3d` and written as .nii.gz.
3.  The logger is the package-level LOGGER defined in
       mgmGrowth.preprocessing.__init__
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from mgmGrowth.preprocessing import LOGGER
from mgmGrowth.preprocessing.tools.nrrd_to_nifti import nifti_write_3d   # conversion routine

# --------------------------------------------------------------------------- #
#                         helper / pure functions                             #
# --------------------------------------------------------------------------- #

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


def followup_pattern(pulse: str, old_id: str) -> str:
    """Filename pattern searched for inside each control visit."""
    return f"{pulse}_{old_id}.nrrd"


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
    except Exception as exc:                        # pylint: disable=broad-except
        LOGGER.error("Conversion failed for %s : %s", src, exc)
        return False


def find_control_dirs(patient_root: Path) -> List[Path]:
    """Return control* directories (sorted alphanumerically)."""
    return sorted(d for d in patient_root.iterdir() if d.is_dir() and d.name.startswith("control"))


def find_followup_files(control_dir: Path, pulse: str, old_id: str) -> Iterable[Path]:
    """Yield NRRDs inside *control_dir* matching the strict naming rule."""
    pattern = followup_pattern(pulse, old_id)
    return control_dir.glob(pattern)


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
    for control_dir in control_dirs:
        for pulse in pulses:
            # -> pattern & explicit log
            pat = followup_pattern(pulse, old_id)
            LOGGER.info("Searching %s for pattern '%s'", control_dir, pat)

            any_found = False
            for src in find_followup_files(control_dir, pulse, old_id):
                any_found = True
                dst = patient_out / f"MenGrowth-{new_idx}-{control_idx:04d}-{pulse}.nii.gz"
                if convert_if_exists(src, dst):
                    control_idx += 1
            if not any_found:
                LOGGER.warning("No %s files found in %s", pulse, control_dir)


# --------------------------------------------------------------------------- #
#                               CLI wrapper                                   #
# --------------------------------------------------------------------------- #

def run(baseline: Path, followup: Path, out_root: Path, pulses: Sequence[str]) -> None:
    """End-to-end execution."""
    patients = enumerate_patients(followup)
    id_map   = build_id_map(patients)

    out_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Patients to process: %s", patients)

    with open(out_root / "patient_id_map.json", "w") as fp:
        json.dump(id_map, fp, indent=2)
    LOGGER.info("Wrote patient_id_map.json")

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
        "/home/mariopasc/Python/Datasets/Meningiomas/raw/Meningioma_Adquisition"
    ))
    parser.add_argument("--followup", type=Path, default=Path(
        "/home/mariopasc/Python/Datasets/Meningiomas/raw/men"
    ))
    parser.add_argument("--out", type=Path, default=Path(
        "/home/mariopasc/Python/Datasets/Meningiomas/raw/MenGrowth-2025"
    ))
    parser.add_argument(
        "--pulses", default="T1,T2,SUSC,FLAIR,TC",
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
