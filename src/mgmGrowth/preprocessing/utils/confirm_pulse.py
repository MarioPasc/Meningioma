#!/usr/bin/env python3
"""
Pulse-assignment for meningioma controls via Bhattacharyya histogram similarity,
with robust handling to skip unreadable or invalid NRRD files, and an added
'File' column in the output CSV.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import nrrd
from tqdm import tqdm

# ---------- low-level utilities ------------------------------------------------
def load_volume(path: Path) -> np.ndarray:
    """
    Load a NRRD file and return the volume as float32.
    Raises any I/O or format errors to caller.
    """
    data, _ = nrrd.read(str(path))
    return data.astype(np.float32)

def histogram_uint8(vol: np.ndarray) -> np.ndarray:
    """
    Min-max normalise vol to [0,255], cast to uint8, and return
    a 256-bin probability histogram as float32.
    """
    vmin, vmax = vol.min(), vol.max()
    if vmax == vmin:
        # flat image: return delta at zero
        h = np.zeros(256, dtype=np.float32)
        h[0] = 1.0
        return h
    scaled = ((vol - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    hist = np.bincount(scaled.ravel(), minlength=256).astype(np.float32)
    return hist / hist.sum()

def bhattacharyya(q: np.ndarray, p: np.ndarray) -> float:
    """
    Compute Bhattacharyya coefficient between two probability histograms.
    """
    return float(np.sum(np.sqrt(q * p)))


# ---------- building reference histograms -------------------------------------
def build_reference_histograms(baseline_root: Path) -> dict[str, np.ndarray]:
    """
    Walk through baseline_root and aggregate one histogram per pulse.
    Skip any files that raise errors during load or processing.
    """
    pulses = {
        "T1": baseline_root / "RM" / "T1",
        "T2": baseline_root / "RM" / "T2",
        "SWI": baseline_root / "RM" / "SWI",
        "TC": baseline_root / "TC",
    }

    ref_hist: dict[str, np.ndarray] = {}
    for pulse, p_dir in pulses.items():
        agg = np.zeros(256, dtype=np.float64)
        if not p_dir.exists():
            continue
        for vol_path in tqdm(
            list(p_dir.rglob("*.nrrd")),
            desc=f"Aggregating {pulse}", unit="scan"
        ):
            if vol_path.name.endswith("_seg.nrrd"):
                continue
            try:
                vol = load_volume(vol_path)
                hist = histogram_uint8(vol)
                agg += hist
            except Exception as e:
                print(f"Warning: skipping invalid baseline file {vol_path}: {e}", file=sys.stderr)
                continue
        if agg.sum() > 0:
            ref_hist[pulse] = (agg / agg.sum()).astype(np.float32)
    return ref_hist


# ---------- classification of controls ----------------------------------------
PATTERN_PREFIX = re.compile(r"^([A-Za-z0-9\-]+)[_\-]")  # e.g. "T2_P12_01" → T2

def classify_controls(controls_root: Path,
                      ref_hist: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Traverse every patient/control; compute Bhattacharyya coefficients per image,
    skip unreadable files, and decide the most likely pulse for each image.
    Returns a DataFrame with columns:
      Patient, Control, File, <each pulse>, old_pulse, new_pulse
    """
    rows = []
    for patient_dir in sorted(controls_root.iterdir()):
        if not patient_dir.is_dir():
            continue
        patient = patient_dir.name
        for control_dir in sorted(patient_dir.iterdir()):
            if not control_dir.is_dir():
                continue
            control = control_dir.name
            for img_path in control_dir.glob("*.nrrd"):
                if img_path.name.endswith("_seg.nrrd"):
                    continue
                try:
                    vol = load_volume(img_path)
                    q_hist = histogram_uint8(vol)
                except Exception as e:
                    print(f"Warning: skipping invalid control file {img_path}: {e}", file=sys.stderr)
                    continue
                coeffs = {p: bhattacharyya(q_hist, h) for p, h in ref_hist.items()}
                best_pulse = max(coeffs, key=coeffs.get) if coeffs else "UNKNOWN"
                m = PATTERN_PREFIX.match(img_path.stem)
                old_pulse = m.group(1) if m else "UNKNOWN"
                row = {
                    "Patient": patient,
                    "Control": control,
                    "File": img_path.name,
                    **coeffs,
                    "old_pulse": old_pulse,
                    "new_pulse": best_pulse
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    # ensure all pulse columns exist
    for pulse in sorted(ref_hist.keys()):
        if pulse not in df.columns:
            df[pulse] = np.nan
    # order columns
    cols = ["Patient", "Control", "File"] + sorted(ref_hist.keys()) + ["old_pulse", "new_pulse"]
    return df.loc[:, cols]


# ---------- command-line interface --------------------------------------------
def main() -> None:
    """
    Parse CLI arguments, build reference histograms, classify controls,
    and write the resulting CSV.
    """

    parser = argparse.ArgumentParser(
        description="Assign MRI pulses to control studies via Bhattacharyya histograms."
    )
    parser.add_argument("--baseline", type=Path, help="Path to baseline root folder", default="/home/mpascual/research/datasets/meningiomas/raw/baseline")
    parser.add_argument("--controls", type=Path, help="Path to controls root folder", default="/home/mpascual/research/datasets/meningiomas/raw/controls")
    parser.add_argument("-o", "--output", type=Path, default="/home/mpascual/research/datasets/meningiomas/raw/pulse_assignment.csv",
                        help="Output CSV filename")
    args = parser.parse_args()

    ref_hist = build_reference_histograms(args.baseline)
    if not ref_hist:
        print("Error: no valid baseline histograms found, aborting.", file=sys.stderr)
        sys.exit(1)

    df = classify_controls(args.controls, ref_hist)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")

if __name__ == "__main__":
    main()

