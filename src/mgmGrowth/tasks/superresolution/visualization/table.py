#!/usr/bin/env python3
"""
make_roi_tables.py
==================

Create LaTeX tables with SR-quality metrics, one file per
   ▸ pulse (t1c, t2w, t2f)
   ▸ ROI   (Whole-Volume, Core, Edema, Surround)
Rows   : Resolution ▸ Model
Columns: PSNR | SSIM | BC | LPIPS  (mean ± std across patients)

Best model per metric/resolution is typeset in \\textbf{}.
"""

from __future__ import annotations
import logging
import pathlib
from dataclasses import dataclass
from typing import List, Dict

import numpy as np


# ───────────────────────────── configuration ──────────────────────────────
@dataclass(frozen=True, slots=True)
class Config:
    METRICS_PATH: pathlib.Path = pathlib.Path(
        "/home/mpascual/research/datasets/meningiomas/BraTS/"
        "super_resolution/results/metrics/metrics.npz"
    )
    OUT_DIR: pathlib.Path = pathlib.Path(
        "/home/mpascual/research/datasets/meningiomas/BraTS/"
        "super_resolution/results/metrics/vis/table/roi_tables"
    )
    STAT_IDX: int = 0               # 0 → slice-mean
    PREC: int = 2                   # decimals in ‘µ ± σ’
    ARRAY_STRETCH: float = 1.2      # row spacing


# ROI mapping: axis-index → (slug, pretty label)
ROI_INFO: Dict[int, tuple[str, str]] = {
    0: ("volume",   "Whole\\,Volume"),
    1: ("core",     "Enhancing\\,Tumour\\,Core"),
    2: ("edema",    "FLAIR\\,Hyperintensity"),
    3: ("surround", "Surrounding\\,Tumour"),
}


# ───────────────────────────── helpers ────────────────────────────────────
def fmt(mean: float, std: float, prec: int, bold: bool) -> str:
    """Return ‘µ ± σ’ (dash if NaN) and optional bold wrapper."""
    txt = "—" if np.isnan(mean) else f"{mean:.{prec}f} ± {std:.{prec}f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def patient_stats(block: np.ndarray) -> np.ndarray:
    """
    Mean ± std across patients.

    Parameters
    ----------
    block : (patients, 4 metrics) ndarray

    Returns
    -------
    stats : (4, 2) ndarray
        [:,0] = mean ,  [:,1] = std
    """
    μ = np.nanmean(block, axis=0)
    σ = np.nanstd(block, axis=0)
    return np.stack([μ, σ], axis=1)


def build_table(res_mm: List[int],
                models: List[str],
                stats: np.ndarray,
                prec: int,
                stretch: float) -> str:
    """
    Parameters
    ----------
    res_mm : list[int]                       length R (3)
    models : list[str]                       length M (4)
    stats  : ndarray (R, M, 4 metrics, 2)   mean/std already computed
    """
    maximise = np.array([True, True, False, False])  # per metric (len=4)

    hdr = (
        f"\\renewcommand{{\\arraystretch}}{{{stretch}}}%\n"
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}"
        " c l c c c c}\n"
        "  \\toprule\n"
        "  Res. (mm) & Model"
        " & PSNR $\\uparrow$ & SSIM $\\uparrow$"
        " & BC $\\downarrow$ & LPIPS $\\downarrow$ \\\\\n"
        "  \\midrule\n"
    )

    rows: list[str] = []
    R, M = stats.shape[:2]

    for r in range(R):
        means = stats[r, :, :, 0]                  # (M, 4)
        best = np.zeros_like(means, dtype=bool)    # (M, 4)

        # best per metric for this resolution
        for j in range(4):
            col = means[:, j]
            good = np.isfinite(col)
            if not good.any():
                continue
            best_val = (np.nanmax if maximise[j] else np.nanmin)(col)
            best[:, j] = np.isclose(col, best_val, atol=1e-8)

        for m, model in enumerate(models):
            res_cell = f"\\multirow{{{M}}}{{*}}{{{res_mm[r]}}}" if m == 0 else ""
            cells = " & ".join(fmt(*stats[r, m, j], prec, best[m, j])
                               for j in range(4))
            rows.append(f"  {res_cell} & {model} & {cells} \\\\")
        # put space after the last model of this resolution (except final res.)
        if r < R - 1 and m == M - 1:
            rows.append("  \\addlinespace[0.6em]")

    ftr = "  \\bottomrule\n\\end{tabular*}\n"
    return hdr + "\n".join(rows) + ftr


# ───────────────────────────── main ───────────────────────────────────────
def main(cfg: Config = Config()) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    logging.info("Loading %s", cfg.METRICS_PATH)
    d = np.load(cfg.METRICS_PATH, allow_pickle=True)

    # axes: (patients, pulse, res, model, metric, ROI, stat)
    metrics = d["metrics"]

    pulses:  List[str] = [p.decode() if isinstance(p, bytes) else str(p)
                          for p in d["pulses"]]
    res_mm:  List[int] = list(d["resolutions_mm"])
    models:  List[str] = [m.decode() if isinstance(m, bytes) else str(m)
                          for m in d["models"]]
    metric_names: List[str] = [m.decode() if isinstance(m, bytes) else str(m)
                               for m in d["metric_names"]]

    # metric order: PSNR, SSIM, BC, LPIPS
    metric_order = [metric_names.index(m) for m in ("PSNR", "SSIM", "BC", "LPIPS")]

    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p_idx, pulse in enumerate(pulses):
        logging.info("Pulse %s …", pulse)
        for roi_idx, (slug, pretty) in ROI_INFO.items():
            # ── slice 7-D array correctly ────────────────────────────────
            # keep metric axis intact to avoid NumPy’s advanced-index quirk
            data = metrics[:, p_idx, :, :, :, roi_idx, cfg.STAT_IDX]   # (P, R, M, 4)
            data = data[..., metric_order]                             # reorder metrics

            # aggregate over patients
            R, M = len(res_mm), len(models)
            stats = np.empty((R, M, 4, 2))
            for r in range(R):
                for m in range(M):
                    stats[r, m] = patient_stats(data[:, r, m, :])

            tex = build_table(res_mm, models, stats,
                              prec=cfg.PREC, stretch=cfg.ARRAY_STRETCH)

            out = cfg.OUT_DIR / f"metrics_table_{pulse}_{slug}.tex"
            with out.open("w") as f:
                f.write(f"% Pulse: {pulse} | ROI: {pretty}\n")
                f.write(tex)
            logging.info("  wrote %s", out.relative_to(cfg.OUT_DIR))

    logging.info("Done – tables in %s", cfg.OUT_DIR)


if __name__ == "__main__":  # pragma: no cover
    main()
