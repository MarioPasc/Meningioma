# file: src/mgmGrowth/tasks/superresolution/pipelines/superresolution_brats_experiment.py
"""
Incremental SMORE fine-tuning & validation.

Outputs
-------
train_metrics_<pulse>.npz
    metrics      : float32  (n_train_patients, 4, 2)
    patient_ids  : str      (n_train_patients,)

val_metrics_<pulse>.npz
    metrics      : float32  (n_batches, n_val_patients, 4, 2)
    patient_ids  : str      (n_val_patients,)

"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from src.mgmGrowth.tasks.superresolution import LOGGER
from src.mgmGrowth.tasks.superresolution.config import SmoreConfig
from src.mgmGrowth.tasks.superresolution.engine.smore_runner import train_volume, infer_volume
from src.mgmGrowth.tasks.superresolution.tools import ensure_dir, psnr_ssim_regions, matching_gt_seg


# ------------------------------------------------------------------ helpers


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


# ------------------------------------------------------------------ pipeline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Down-sampled LR images used for training.")
    ap.add_argument("--orig-root", type=Path, required=True,
                    help="Folder with the ORIGINAL 1 mm³ BraTS images + seg masks.")
    ap.add_argument("--pulses", nargs="+", default=[])
    ap.add_argument("--slice-dz", type=float, required=True)
    ap.add_argument("--val-frequency", type=int, default=4)
    ap.add_argument("--holdout-ratio", type=float, default=0.2)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    pulses = args.pulses
    cfg = SmoreConfig(gpu_id=args.gpu)

    vols_all = sorted(_iter_vols(args.data_root, pulses))
    if not vols_all:
        LOGGER.error("No matching volumes.")
        return

    rng = random.Random(0)
    rng.shuffle(vols_all)
    n_hold = int(round(len(vols_all) * args.holdout_ratio))
    hold_vols, train_vols = vols_all[:n_hold], vols_all[n_hold:]
    LOGGER.info("Train %d  Hold-out %d", len(train_vols), len(hold_vols))

    tag = "_".join(pulses) or "all"
    weights_root = ensure_dir(args.data_root / f"_smore_weights_{tag}")
    preds_root = ensure_dir(args.data_root / f"preds_{tag}")

    train_results: dict[str, np.ndarray] = {}      # patient_id -> (4,2)
    val_metrics_batches: list[np.ndarray] = []     # (n_val,4,2) per batch
    patient_ids_val = np.array([p.parts[-2] for p in hold_vols])

    batch: list[Path] = []
    for idx, vol in enumerate(train_vols, 1):
        batch.append(vol)
        w_dir = train_volume(vol, cfg, weights_root, args.slice_dz)

        reached = idx % args.val_frequency == 0 or idx == len(train_vols)
        if not reached:
            continue

        # --------─ evaluate last batch ─------------------------------------------
        for v in batch:
            sr = preds_root / f"{v.stem}_SR.nii.gz"
            infer_volume(v, w_dir, cfg, sr)
            hr, seg = matching_gt_seg(v, args.orig_root)

            train_results[v.parts[-2]] = psnr_ssim_regions(hr, sr, seg)  # (4,2)

        # --------─ evaluate hold-out ---------------------------------------------
        val_arr = []
        for v in hold_vols:
            sr = preds_root / f"holdout_{v.stem}_SR.nii.gz"
            infer_volume(v, w_dir, cfg, sr)
            hr, seg = matching_gt_seg(v, args.orig_root)
            val_arr.append(psnr_ssim_regions(hr, sr, seg))
        val_metrics_batches.append(np.stack(val_arr, axis=0))            # (n_val,4,2)

        LOGGER.info("Validated batch %d / %d", len(train_results), len(train_vols))
        batch.clear()

    # ---------------- save outputs -----------------------------------------------
    # ---- train (per-patient) -----------------------------------------------------
    train_ids = np.array(sorted(train_results))
    train_metrics_arr = np.stack([train_results[id_] for id_ in train_ids], axis=0)
    np.savez_compressed(
        args.data_root / f"train_metrics_{tag}.npz",
        metrics=train_metrics_arr.astype(np.float32),
        patient_ids=train_ids,
    )
    LOGGER.info("Saved train metrics (%d patients)", len(train_ids))

    # ---- validation (unchanged) --------------------------------------------------
    np.savez_compressed(
        args.data_root / f"val_metrics_{tag}.npz",
        metrics=np.stack(val_metrics_batches, axis=0).astype(np.float32),
        patient_ids=patient_ids_val,
    )
    LOGGER.info("Saved val metrics (%d batches × %d patients)",
                len(val_metrics_batches), len(patient_ids_val))
    LOGGER.info("Final weights → %s", w_dir)


if __name__ == "__main__":
    main()
