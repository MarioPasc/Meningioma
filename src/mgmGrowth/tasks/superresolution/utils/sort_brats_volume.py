from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np


LABEL_NAMES = {1: "Tumor Core", 2: "Edema", 3: "Surrounding Tumor"}


@dataclass
class PatientInfo:
    """Container for the per-patient information that will be serialized to JSON."""
    volumes: Dict[str, int]
    mean_volume: float
    paths: Dict[str, str]


def _load_label_volumes(seg_path: Path) -> Dict[int, int]:
    """
    Count voxels for labels 1, 2, 3 in a segmentation.
    
    Parameters
    ----------
    seg_path
        Path to the *.nii.gz segmentation.
    
    Returns
    -------
    Dict[int, int]
        Mapping label → voxel count.
    """
    data = nib.load(seg_path).get_fdata().astype(np.uint8)
    return {lab: int(np.sum(data == lab)) for lab in (1, 2, 3)}


def _copy_requested_pulses(
    patient_dir: Path, pulses: List[str], dst_root: Path
) -> Dict[str, str]:
    """
    Copy only the files ending with the requested pulses into *dst_root/patient_dir.name/*.
    
    Returns a mapping pulse → copied file path (or "" if not found).
    """
    dst_dir = dst_root / patient_dir.name
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied_paths: Dict[str, str] = {}
    pulses.append("seg")  # add segmentation to the list
    for pulse in pulses:
        pattern = f"*-{pulse}.nii.gz"
        try:
            src_file = next(patient_dir.glob(pattern))
            dst_file = dst_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            copied_paths[pulse] = str(dst_file)
        except StopIteration:
            copied_paths[pulse] = ""  # pulse missing
    return copied_paths


def select_top_brats_patients(
    root_dir: str | Path,
    num_patients: int,
    pulses: List[str],
    output_dir: str | Path,
    json_name: str = "selected_patients.json",
) -> Dict[str, PatientInfo]:
    """
    Select BraTS-MEN patients that contain **all three** segmentation labels (1, 2, 3),
    rank them by the mean tumor volume, copy the requested pulse files, and save summary JSON.
    
    Parameters
    ----------
    root_dir
        Root directory containing *BraTS-MEN-XXXX-000/* folders.
    num_patients
        Number of top patients to keep.
    pulses
        List of pulse suffixes (e.g. ``["t1c", "t2w"]``).
    output_dir
        Destination folder where the new patient folders and JSON will be written.
    json_name
        Filename of the JSON manifest (default ``selected_patients.json``).
    
    Returns
    -------
    Dict[str, PatientInfo]
        Mapping patient folder name → information stored in JSON.
    """
    root = Path(root_dir)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    patient_info: Dict[str, PatientInfo] = {}

    # 1 & 2 — enumerate patients and compute volumes ---------------------------
    qualified = []
    for pdir in root.glob("BraTS-MEN-*"):
        seg_files = list(pdir.glob("*-seg.nii.gz"))
        if not seg_files:
            print(f"⚠ Segmentation not found in {pdir.name}, skipping.")
            continue

        volumes = _load_label_volumes(seg_files[0])

        # Require labels 1,2,3 all present (volume > 0)
        if all(volumes[label] > 0 for label in (1, 2, 3)):
            mean_vol = float(np.mean(list(volumes.values())))
            qualified.append((pdir, volumes, mean_vol))

    # 3 — sort and slice -------------------------------------------------------
    qualified.sort(key=lambda x: x[2], reverse=True)  # descending by mean
    selected = qualified[: num_patients]

    # 4 — copy pulses & build JSON dict ---------------------------------------
    for pdir, volumes, mean_vol in selected:
        copied = _copy_requested_pulses(pdir, pulses, dest)

        vol_dict = {LABEL_NAMES[k]: v for k, v in volumes.items()}
        patient_info[pdir.name] = PatientInfo(
            volumes={**vol_dict, "Mean": mean_vol},
            mean_volume=mean_vol,
            paths={
                "original_path": str(pdir),
                "copied_path": str(dest / pdir.name),
                **copied,
            },
        )

    # 5 — persist JSON ---------------------------------------------------------
    json_path = dest / json_name
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({k: asdict(v) for k, v in patient_info.items()}, f, indent=4)

    print(f"✅ Saved manifest to {json_path}")
    return patient_info

if __name__ == "__main__":
    select_top_brats_patients(
        root_dir="/media/mario/PortableSSD/Meningiom/BraTS_Men_Train",
        num_patients=50,
        pulses=["t1c", "t2w"],
        output_dir="/home/mario/x2go_shared/sr/original_volumes",
    )
