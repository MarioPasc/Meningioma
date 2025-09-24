#!/usr/bin/env python3
"""
compute_pvalues.py
==================

★ Author: Mario Pascual González
★ Purpose: For each (model, ROI, resolution) triplet, test whether the
  voxel-wise intensity distribution inside the ROI is identical between
  the HR ground-truth volume and the super-resolved volume.

Method
------
1. Per patient: two-sided KS test on the ROI voxels.
2. Combine patient p-values with Fisher’s method (fixed effect).
3. Correct across all triplets (Holm–Bonferroni  *and*  Benjamini–Hochberg).

Output
------
`pvalues.npz` containing

    p_raw       (M, 3, 4)   float64   – combined but *uncorrected*
    p_holm      (M, 3, 4)   float64   – Holm-Bonferroni FWER
    p_fdr       (M, 3, 4)   float64   – BH-FDR (q)
    models      list[str]
    resolutions_mm (3,)
    roi_labels  ("all","core","edema","surround")

where axis order = (model, resolution, roi).
"""

from __future__ import annotations

# ---------------------------------------------------------------- imports
import argparse
import logging
import multiprocessing as mp
import pathlib
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import nibabel as nib
from scipy.stats import ks_2samp, chi2
from statsmodels.stats.multitest import multipletests  # pip install statsmodels

# ---------- reuse helper functions & constants from metrics.py ------------
import mgmGrowth.tasks.superresolution.statistics.metrics as m  # <– same directory

logger = logging.getLogger("pvalue")
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(message)s")

# ----------------------------------------------------------------- types --
ROI_LABELS = tuple(m.ROI_LABELS)          # ("all","core","edema","surround")
RESOLUTIONS_MM = tuple(m.RESOLUTIONS)     # (3,5,7)


@dataclass(frozen=True, slots=True)
class PatientItem:
    """All file paths for a single (patient, model, resolution)."""
    pid:     str
    hr_path: pathlib.Path
    sr_path: pathlib.Path
    seg_path: pathlib.Path
    model:   str
    res_mm:  int


# ---------------------------------------------------------------- helpers -
def discover_items(hr_root: pathlib.Path,
                   res_root: pathlib.Path,
                   pulse: str = "t1c"
                   ) -> list[PatientItem]:
    """Traverse the file system exactly as `metrics.collect_paths` does."""
    items: list[PatientItem] = []
    model_dirs = sorted(d.name for d in res_root.iterdir() if d.is_dir())

    for model in model_dirs:
        for res in RESOLUTIONS_MM:
            for vp in m.collect_paths(hr_root, res_root, pulse, model, res):                
                items.append(PatientItem(pid=vp.hr.parent.name,
                                         hr_path=vp.hr,
                                         sr_path=vp.sr,
                                         seg_path=vp.seg,
                                         model=model,
                                         res_mm=res))
    return items


def ks_pvalue_single_patient(hr_path: pathlib.Path,
                             sr_path: pathlib.Path,
                             seg_path: pathlib.Path,
                             roi_label: int | None) -> float:
    """
    Two-sample KS on the voxel vectors inside *one* ROI.

    Returns NaN if the ROI is empty.
    """
    # --- load volumes (float32 to save RAM) ------------------------------
    hr = m.load_lps(hr_path, dtype=np.float32)     # shape z,y,x
    sr = m.load_lps(sr_path, like=nib.load(str(hr_path)), order=1)
    seg = m.load_lps(seg_path, like=nib.load(str(hr_path)), order=0)

    # --- shape safety (≤2 voxels mismatch) -------------------------------
    try:
        hr, sr = m.match_slices(hr, sr)
        hr, sr, seg = m.unify_shapes(hr, sr, seg)
    except ValueError as e:
        logger.warning("Geometry mismatch → skip patient (%s)", e)
        return np.nan

    # --- mask ------------------------------------------------------------
    mask = np.ones_like(seg, dtype=bool) if roi_label is None else seg == roi_label
    if not mask.any():
        return np.nan

    hr_vec = hr[mask].astype(np.float64, copy=False)
    sr_vec = sr[mask].astype(np.float64, copy=False)

    if hr_vec.size == 0 or sr_vec.size == 0:
        return np.nan

    # --- KS --------------------------------------------------------------
    return ks_2samp(hr_vec, sr_vec, alternative="two-sided", mode="auto").pvalue


def combine_pvalues_fisher(pvals: Sequence[float]) -> float:
    """
    Fisher’s method: X² = -2 ∑ ln p_i  ~ χ²(2n).

    Ignores NaN entries; returns NaN if < 2 valid p-values.
    """
    pvals = np.asarray(pvals, dtype=float)
    pvals = pvals[np.isfinite(pvals) & (pvals > 0)]
    n = pvals.size
    if n < 2:
        return np.nan
    X2 = -2.0 * np.log(pvals).sum()
    return 1.0 - chi2.cdf(X2, df=2 * n)


def adjust_pvalues(p_raw: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) corrections.

    Shapes preserved.
    """
    flat = p_raw.ravel()
    ok = np.isfinite(flat)

    p_holm = np.full_like(flat, np.nan)
    p_fdr  = np.full_like(flat, np.nan)

    if ok.any():
        p_holm[ok] = multipletests(flat[ok], method="holm")[1]
        p_fdr[ok]  = multipletests(flat[ok], method="fdr_bh")[1]

    return p_holm.reshape(p_raw.shape), p_fdr.reshape(p_raw.shape)


def ks_pipeline(hr_root: pathlib.Path,
                res_root: pathlib.Path,
                pulse: str,
                workers: int = max(mp.cpu_count() - 1, 1)
                ) -> tuple[np.ndarray, list[str]]:
    """
    Main engine. Returns:
        p_raw  (M, 3, 4)   float64
        model_names
    """
    all_items = discover_items(hr_root, res_root, pulse)
    if not all_items:
        raise RuntimeError("No volumes found – check paths.")

    model_names = sorted({it.model for it in all_items})
    model_idx = {m: i for i, m in enumerate(model_names)}

    # Allocate array: (models, resolutions, rois)
    p_raw = np.full((len(model_names), len(RESOLUTIONS_MM), len(ROI_LABELS)),
                    np.nan, dtype=np.float64)

    # --- group items by (model, res) for efficient mapping ---------------
    grouped: dict[tuple[str, int], list[PatientItem]] = {}
    for it in all_items:
        grouped.setdefault((it.model, it.res_mm), []).append(it)

    # --- pool ------------------------------------------------------------
    with mp.Pool(processes=workers) as pool:
        for (model, res_mm), items in grouped.items():
            # structure: dict[roi_label → list[p_i]]
            roi_pvals: dict[int | None, list[float]] = {None: [], 1: [], 2: [], 3: []}

            # Prepare async jobs (patient x roi)
            async_results = []
            for it in items:
                for roi_label in (None, 1, 2, 3):
                    async_results.append(
                        pool.apply_async(
                            ks_pvalue_single_patient,
                            (it.hr_path, it.sr_path, it.seg_path, roi_label)
                        )
                    )

            # Collect results in submission order
            itr = iter(async_results)
            for roi_label in (None, 1, 2, 3):
                for _ in items:                    # one per patient
                    roi_pvals[roi_label].append(next(itr).get())

            # Combine patient-level p-values
            for roi_label, roi_list in roi_pvals.items():
                pr = combine_pvalues_fisher(roi_list)
                p_raw[model_idx[model],
                      RESOLUTIONS_MM.index(res_mm),
                      (0 if roi_label is None else ROI_LABELS.index(
                          ["core", "edema", "surround"][roi_label - 1]))
                      ] = pr

    return p_raw, model_names


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute KS-test p-values.")
    ap.add_argument("--hr_root", type=pathlib.Path,
                    default=pathlib.Path(
                        "/home/mpascual/research/datasets/meningiomas/BraTS/super_resolution/subset"))
    ap.add_argument("--results_root", type=pathlib.Path,
                    default=pathlib.Path(
                        "/home/mpascual/research/datasets/meningiomas/BraTS/super_resolution/results/models"))

    # -------- pulse --------------------------------------------------------
    ap.add_argument("--pulse", choices=("t1c", "t2w", "t2f"),
                    default="t1c",
                    help="Evaluate given pulse or all (default)")
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(
                        "/home/mpascual/research/datasets/meningiomas/BraTS/super_resolution/results/metrics/p_values.npz"))
    ap.add_argument("--workers", type=int, default=mp.cpu_count() - 1)
    args = ap.parse_args()

    logger.info("Scanning data …")
    p_raw, model_names = ks_pipeline(args.hr_root, args.results_root,
                                     args.pulse, args.workers)

    logger.info("Adjusting for multiplicity …")
    p_holm, p_fdr = adjust_pvalues(p_raw)

    # ----------------------------- save ----------------------------------
    np.savez(f"{args.out}_{args.pulse}",
             p_raw=p_raw,
             p_holm=p_holm,
             p_fdr=p_fdr,
             models=np.asarray(model_names, dtype=object),
             resolutions_mm=np.asarray(RESOLUTIONS_MM),
             roi_labels=np.asarray(ROI_LABELS, dtype=object))
    logger.info("Saved p-values to %s", args.out)


# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
