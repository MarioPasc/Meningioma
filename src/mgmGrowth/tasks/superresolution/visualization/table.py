#!/usr/bin/env python3
"""
make_roi_tables.py
==================

• One LaTeX table per  (pulse, ROI)  pair.
• Rows   : Resolution ▸ Model [+ KS-test stars]
• Columns: PSNR ↑, SSIM ↑, BC ↓, LPIPS ↓   (mean ± std across patients)

KS two-sample test
------------------
Holm-corrected p-values (already computed in p_values_<pulse>.npz) are
translated into significance stars and shown next to each *model*:

    p ≤ 0.001 : ***     0.001 < p ≤ 0.01 : **     0.01 < p ≤ 0.05 : *

File layout
-----------
metrics.npz          – 7-D quality metrics  (see earlier script)
p_values_<pulse>.npz – KS p-values          (model, res, roi)

Output
------
roi_tables/metrics_table_<pulse>_<roi>.tex
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
    PVALUE_DIR: pathlib.Path = pathlib.Path(      # directory with p_values_*.npz
        "/home/mpascual/research/datasets/meningiomas/BraTS/"
        "super_resolution/results/metrics/pvalues"
    )
    OUT_DIR: pathlib.Path = pathlib.Path(
        "/home/mpascual/research/datasets/meningiomas/BraTS/"
        "super_resolution/results/metrics/vis/table/roi_tables"
    )
    STAT_IDX: int = 0               # 0 → slice-mean
    PREC: int = 2                   # decimals in ‘µ ± σ’
    ARRAY_STRETCH: float = 1.2      # row spacing
    GAP_EM: float = 0.6             # gap between resolutions
    PVAL_FIELD: str = "p_holm"      # which array in *.npz to use


# ROI mapping: axis-index → (slug, pretty-label)
ROI_INFO: Dict[int, tuple[str, str]] = {
    0: ("volume",   "Whole\\,Volume"),
    1: ("core",     "Enhancing\\,Tumour\\,Core"),
    2: ("edema",    "FLAIR\\,Hyperintensity"),
    3: ("surround", "Surrounding\\,Tumour"),
}


# ───────────────────────────── helpers ────────────────────────────────────
def p_to_star(p: float) -> str:
    """Return LaTeX superscript with *, **, ***."""
    if not np.isfinite(p) or p > 0.05:
        return ""
    if p <= 1e-3:
        return "$^{***}$"
    if p <= 1e-2:
        return "$^{**}$"
    return "$^{*}$"


def fmt(mean: float, std: float, prec: int, bold: bool) -> str:
    txt = "—" if np.isnan(mean) else f"{mean:.{prec}f} ± {std:.{prec}f}"
    return f"\\textbf{{{txt}}}" if bold else txt


def patient_stats(block: np.ndarray) -> np.ndarray:
    """Mean ± std across patients ⇒ (4,2)."""
    μ, σ = np.nanmean(block, axis=0), np.nanstd(block, axis=0)
    return np.stack([μ, σ], axis=1)


def build_table(res_mm: List[int], models: List[str],
                stats: np.ndarray, stars: np.ndarray,
                cfg: Config) -> str:
    """
    stats shape : (R, M, 4, 2)   stars shape : (R, M)
    """
    maximise = np.array([True, True, False, False])
    hdr = (
        f"\\renewcommand{{\\arraystretch}}{{{cfg.ARRAY_STRETCH}}}%\n"
        "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}"
        " c l c c c c}\n"
        "  \\toprule\n"
        "  Res. (mm) & Model & PSNR $\\uparrow$ & SSIM $\\uparrow$"
        " & BC $\\downarrow$ & LPIPS $\\downarrow$ \\\\\n"
        "  \\midrule\n"
    )

    rows: list[str] = []
    R, M = stats.shape[:2]

    for r in range(R):
        means = stats[r, :, :, 0]                     # (M,4)
        best = np.zeros_like(means, bool)
        for j in range(4):
            col = means[:, j]
            good = np.isfinite(col)
            if good.any():
                best_val = (np.nanmax if maximise[j] else np.nanmin)(col)
                best[:, j] = np.isclose(col, best_val, atol=1e-8)

        for m_idx, model in enumerate(models):
            res_cell = f"\\multirow{{{M}}}{{*}}{{{res_mm[r]}}}" if m_idx == 0 else ""
            model_cell = f"{model}{stars[r, m_idx]}"
            metrics_cells = " & ".join(
                fmt(*stats[r, m_idx, j], cfg.PREC, best[m_idx, j])
                for j in range(4)
            )
            rows.append(f"  {res_cell} & {model_cell} & {metrics_cells} \\\\")
        if r < R - 1:
            rows.append(f"  \\addlinespace[{cfg.GAP_EM}em]")

    ftr = "  \\bottomrule\n\\end{tabular*}\n"
    return hdr + "\n".join(rows) + ftr


# ───────────────────────────── main ───────────────────────────────────────
def main(cfg: Config = Config()) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    logging.info("Loading quality metrics …")
    d = np.load(cfg.METRICS_PATH, allow_pickle=True)
    metrics = d["metrics"]  # (P, pulse, res, model, metric, ROI, stat)

    pulses  = [p.decode() if isinstance(p, bytes) else str(p) for p in d["pulses"]]
    res_mm  = list(d["resolutions_mm"])
    models  = [m.decode() if isinstance(m, bytes) else str(m) for m in d["models"]]
    metric_names = [m.decode() if isinstance(m, bytes) else str(m)
                    for m in d["metric_names"]]
    order = [metric_names.index(m) for m in ("PSNR", "SSIM", "BC", "LPIPS")]

    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p_idx, pulse in enumerate(pulses):
        # ── load p-values for this pulse ─────────────────────────────────
        pfile = cfg.PVALUE_DIR / f"p_values_{pulse}.npz"
        logging.info("Pulse %s – reading %s", pulse, pfile.name)
        pdat = np.load(pfile, allow_pickle=True)
        p_corr = pdat[cfg.PVAL_FIELD]      # shape (M, R, ROI)

        for roi_idx, (slug, pretty) in ROI_INFO.items():
            # quality metrics (patients, res, model, metrics)
            data = metrics[:, p_idx, :, :, :, roi_idx, cfg.STAT_IDX][..., order]

            # stars array (R, M)
            stars = np.vectorize(p_to_star)(
                p_corr[:, :, roi_idx].T  # → (R, M) to match loops
            )

            # aggregate patient stats
            R, M = len(res_mm), len(models)
            stats = np.empty((R, M, 4, 2))
            for r in range(R):
                for m in range(M):
                    stats[r, m] = patient_stats(data[:, r, m, :])

            tex = build_table(res_mm, models, stats, stars, cfg)

            out = cfg.OUT_DIR / f"metrics_table_{pulse}_{slug}.tex"
            out.write_text(f"% Pulse: {pulse} | ROI: {pretty}\n{tex}")
            logging.info("  wrote %s", out.relative_to(cfg.OUT_DIR))

    logging.info("All tables generated in %s", cfg.OUT_DIR)


if __name__ == "__main__":  # pragma: no cover
    main()
