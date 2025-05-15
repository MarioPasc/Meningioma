# file: src/mgmGrowth/tasks/superresolution/pipelines/smore_full_experiment.py
"""
Run SMORE *per volume*, compute PSNR/SSIM/MI vs original HR, save results.

Outputs
-------
<out-root>/
    weights/           each patient has its own folder with best_weights.pt
    output_volumes/    *_SR.nii.gz
    metrics_<tag>.npz  metrics  (N, 4, 3) + patient_ids
"""
from __future__ import annotations

import argparse
from os import name
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from src.mgmGrowth.tasks.superresolution import LOGGER
from src.mgmGrowth.tasks.superresolution.config import SmoreConfig
from src.mgmGrowth.tasks.superresolution.engine.smore_runner import run_smore
from src.mgmGrowth.tasks.superresolution.tools import ensure_dir, metrics_regions, matching_gt_seg


# ---------------- helpers -----------------------------------------------------
def _iter_vols(root: Path, pulses: Sequence[str]) -> Iterable[Path]:
    LOGGER.debug("Searching for volumes in %s", root)
    for patient in root.iterdir():
        LOGGER.debug("- %s", patient)
        for f in patient.rglob("*.nii.gz"):
            if f.name.endswith("-seg.nii.gz"):
                continue
            LOGGER.debug("  -  %s", f.stem)

            if pulses:
                # Check if the pulse is in the filename
                if f.stem.rstrip(".nii").rsplit("-", 1)[-1] not in pulses:
                    continue
            yield f

# ---------------- main --------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr-root", type=Path, required=True,
                    help="Down-sampled LR volumes (input to SMORE).")
    ap.add_argument("--orig-root", type=Path, required=True,
                    help="Original HR images (+ seg masks).")
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--pulses", nargs="+", default=[],
                    help="Pulse suffixes to include; empty = all.")
    ap.add_argument("--slice-dz", type=float, required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--suffix", default="_smore")
    args = ap.parse_args()

    vols = sorted(_iter_vols(args.lr_root, args.pulses))
    if not vols:
        LOGGER.error("No matching volumes in %s", args.lr_root)
        return

    tag = "_".join(args.pulses) or "all"
    weights_root = ensure_dir(args.out_root / "weights")
    sr_root = ensure_dir(args.out_root / "output_volumes")

    patient_ids: List[str] = []

    cfg = SmoreConfig(gpu_id=args.gpu_id)

    for v in vols:
        patient   = v.parent.name                      # BraTS-MEN-XXXXX-000
        name_pulse = v.stem.rstrip(".nii")                           # e.g. BraTS-MEN-xxxx-t1c
        out_dir   = ensure_dir(weights_root / patient)

        LOGGER.info("Processing %s (%s)", patient, name_pulse)

        # ----------------------------------------------------------------
        # SMORE directory tree:
        #   <out_dir>/
        #       weights/<patient>/<name_pulse>/weights/best_weights.pt
        #       weights/<patient>/<name_pulse>/<name_pulse>.nii.gz
        # We now relocate those two artefacts into flat folders.
        # ----------------------------------------------------------------
        best_pt  = (
            out_dir
            / name_pulse
            / "weights"
            / "best_weights.pt"
        )
        sr_vol   = (
            out_dir
            / name_pulse
            / f"{name_pulse}{args.suffix}.nii.gz"
        )


        LOGGER.info(f"Expected weights path: {best_pt}")
        LOGGER.info(f"Expected SR volume path: {sr_vol}")

        # -------------- run SMORE (per-volume) --------------------------
        run_smore(
            v,
            out_dir,
            cfg=cfg,
            slice_thickness=args.slice_dz,
            gpu_id=args.gpu_id,
            suffix=args.suffix,
        )

        # destination paths in the *flat* structure
        flat_pt  = weights_root / f"{name_pulse}.pt"
        flat_vol = sr_root / f"{name_pulse}.nii.gz"

        # move or copy – use .replace() for atomic rename if on same filesystem
        import shutil, os

        shutil.move(best_pt, flat_pt)        # weight file
        shutil.move(sr_vol,  flat_vol)       # SR volume

        
        patient_ids.append(patient)
        LOGGER.info("✓ %s", patient)


    npz_path = args.out_root / f"metrics_{tag}.npz"
    np.savez_compressed(
        npz_path,
        patient_ids=np.array(patient_ids),
    )
    LOGGER.info("Saved → %s  (%d patients)", npz_path.name, len(patient_ids))
    # ----------------------------------------------------------------


if __name__ == "__main__":
    main()
