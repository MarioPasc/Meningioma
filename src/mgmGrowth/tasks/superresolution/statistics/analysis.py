# file: analysis/slice_metrics_and_plots.py
"""
Compute slice-wise PSNR/SSIM/MI per ROI and create three figures:

1. Violin plots of the three metrics.
2. PDF overlays HR vs SR + stats (KS p, Cohen-d, FDR).
3. HR / SR / difference slice montage (best slice per ROI).

Author : mgmGrowth tasks.superresolution
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde, ks_2samp
from statsmodels.stats.multitest import multipletests
from skimage.metrics import structural_similarity as ssim

from mgmGrowth.tasks.superresolution import LOGGER

ROI_NAMES = ("Whole", "Tumour Core", "Edema", "Surrounding")
METRIC_NAMES = ("PSNR", "SSIM", "MI")
MIN_VOXELS = 30

# ------------------------------------------------------------------ I/O helpers
def _load(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def _norm(v: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(v, (0.5, 99.5))
    return np.clip((v - lo) / (hi - lo + 1e-8), 0.0, 1.0)


# ------------------------------------------------------------------ metrics
def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2, dtype=np.float64)
    if mse == 0:
        return np.nan      # identical vectors → ignore in violin
    return 10 * np.log10(1.0 / mse)


def _ssim_vec(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 and np.std(b) == 0:
        return 1.0
    try:
        return ssim(a, b, data_range=1.0)
    except ValueError:      # skimage needs >1 pixel
        return np.nan



def _mi(a: np.ndarray, b: np.ndarray, bins: int = 64) -> float:
    ha, _ = np.histogram(a, bins=bins, range=(0, 1), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(0, 1), density=True)
    hab, _, _ = np.histogram2d(a, b, bins=bins, range=((0, 1), (0, 1)), density=True)
    ha += 1e-12
    hb += 1e-12
    hab += 1e-12
    return float(np.sum(hab * np.log(hab / (ha[:, None] * hb[None, :]))))


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    return (x.mean() - y.mean()) / np.sqrt(pooled + 1e-12)


# ------------------------------------------------------------------ main class
class SliceMetricAnalysis:
    """Compute metrics and render figures for one HR/SR/SEG triplet."""

    def __init__(self, hr_path: Path, sr_path: Path, seg_path: Path | None):
        self.hr = _norm(_load(hr_path))
        self.sr = _norm(_load(sr_path))
        self.seg = _load(seg_path).astype(int) if seg_path and seg_path.exists() else None

        assert self.hr.shape == self.sr.shape, "HR / SR must match shape"

        self.metrics = self._compute_slice_metrics()  # (n_slices, 4, 3)

    # ---------------------------------------------------------------- metrics
    def _compute_slice_metrics(self) -> np.ndarray:
        n_slices = self.hr.shape[0]
        out = np.full((n_slices, 4, 3), np.nan, dtype=np.float32)

        for z in range(n_slices):
            hr_z, sr_z = self.hr[z], self.sr[z]
            seg_z = self.seg[z] if self.seg is not None else None

            for ridx, rid in enumerate([None, 1, 2, 3]):  # whole + 3 ROIs
                if rid is None:
                    mask = np.ones_like(hr_z, bool)
                else:
                    if seg_z is None or not np.any(seg_z == rid):
                        continue
                    mask = seg_z == rid

                if np.sum(mask) < MIN_VOXELS:
                    continue
                a, b = hr_z[mask], sr_z[mask]
                out[z, ridx, 0] = _psnr(a, b)
                out[z, ridx, 1] = _ssim_vec(a, b)
                out[z, ridx, 2] = _mi(a, b)

        return out

    # ---------------------------------------------------------------- plots
    def make_violin_plot(self) -> plt.Figure:
        """
        1 × 3 violin plot: x = metric value, y = ROI.
        ROIs that have no valid slices are skipped.
        """
        # ---- long-form DataFrame -------------------------------------------------
        rows = []
        for m_idx, m_name in enumerate(METRIC_NAMES):
            for r_idx, r_name in enumerate(ROI_NAMES):
                vals = self.metrics[:, r_idx, m_idx]
                vals = vals[np.isfinite(vals)]          # drop NaN / inf
                rows.extend((m_name, r_name, v) for v in vals)
        df = pd.DataFrame(rows, columns=["metric", "roi", "value"])

        # ---- which ROIs actually contain data?  (use the first metric as reference)
        roi_with_data = [
            roi
            for roi in ROI_NAMES
            if not df[(df.metric == METRIC_NAMES[0]) & (df.roi == roi)].empty
        ]
        if not roi_with_data:   # should never happen, but be safe
            raise ValueError("No ROI contains any valid measurements.")

        # ---- figure ----------------------------------------------------------------
        fig = plt.figure(figsize=(10, 4))
        gs = gridspec.GridSpec(1, 3, wspace=0.35)

        for i, m_name in enumerate(METRIC_NAMES):
            ax = fig.add_subplot(gs[0, i])

            data = [
                df[(df.metric == m_name) & (df.roi == roi)]["value"].values
                for roi in roi_with_data
            ]
            ax.violinplot(data, vert=False, showmeans=True)

            ax.set_yticks(range(1, len(roi_with_data) + 1))
            ax.set_yticklabels(roi_with_data if i == 0 else [""] * len(roi_with_data))
            ax.set_xlabel(m_name)

        fig.suptitle("Slice-wise metrics (ROIs with ≥ MIN_VOXELS slices)")
        return fig


    def make_pdf_plot(self, ax: plt.Axes | None = None) -> plt.Figure:
        """1×4 PDF overlay of HR vs SR per ROI, + KS p-val + Cohen-d."""
        fig = plt.figure(figsize=(14, 3))
        gs = gridspec.GridSpec(1, 4, wspace=0.3)

        p_vals = []
        for r_idx, r_name in enumerate(ROI_NAMES):
            vals_hr = self.hr[self._mask_roi(r_idx)]
            vals_sr = self.sr[self._mask_roi(r_idx)]
            if len(vals_hr) == 0:
                continue

            kde_hr = gaussian_kde(vals_hr)
            kde_sr = gaussian_kde(vals_sr)
            x = np.linspace(0, 1, 512)

            p = ks_2samp(vals_hr, vals_sr).pvalue
            d = _cohen_d(vals_hr, vals_sr)
            p_vals.append(p)

            ax = fig.add_subplot(gs[0, r_idx])
            ax.plot(x, kde_hr(x), label="HR", lw=1.5)
            ax.plot(x, kde_sr(x), label="SR", lw=1.5, ls="--")
            ax.set_title(r_name)
            ax.set_xlim(0, 1)

        # FDR correction
        p_vals = [p for p in p_vals if not np.isnan(p)]
        if p_vals:
            _, p_adj, _, _ = multipletests(p_vals, method="fdr_bh")
        for r_idx, ax in enumerate(fig.axes):
            ax.legend(
                title=f"p={p_adj[r_idx]:.3g}\nd={_cohen_d(self.hr[self._mask_roi(r_idx)], self.sr[self._mask_roi(r_idx)]):.2f}",
                fontsize=8,
            )
        fig.suptitle("Intensity PDF HR vs SR")
        return fig

    def make_difference_montage(self) -> plt.Figure:
        """Select best slice per ROI and show HR/SR/diff crop."""
        idx_best = self._best_slice_per_roi()
        fig = plt.figure(figsize=(6, 8))
        gs = gridspec.GridSpec(4, 3, wspace=0.02, hspace=0.15)

        for row, (r_idx, z) in enumerate(idx_best.items()):
            hr, sr = self.hr[z], self.sr[z]
            diff = hr - sr
            mask = self._mask_roi(r_idx, slice_idx=z)

            # crop tight around ROI (or full brain for whole)
            if r_idx == 0:
                bbox = (slice(None), slice(None))
            else:
                ys, xs = np.where(mask)
                bbox = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))

            for col, img, cm in zip(
                range(3),
                (hr, sr, diff),
                ("gray", "gray", "bwr"),
            ):
                ax = fig.add_subplot(gs[row, col])
                im = ax.imshow(img[bbox], cmap=cm, vmin=0, vmax=1 if col < 2 else None)
                ax.axis("off")
                if row == 0:
                    ax.set_title(("HR", "SR", "HR-SR")[col])
            fig.text(0.01, 0.88 - 0.22 * row, ROI_NAMES[r_idx], fontsize=10, va="center")

        fig.suptitle("Best slice per ROI – HR / SR / diff")
        return fig

    # ---------------------------------------------------------------- helpers
    def _mask_roi(self, r_idx: int, *, slice_idx: int | None = None) -> np.ndarray:
        if r_idx == 0:
            return np.ones_like(self.hr, bool)[slice_idx] if slice_idx else np.ones_like(self.hr, bool)
        if self.seg is None:
            return np.zeros_like(self.hr, bool)[slice_idx] if slice_idx else np.zeros_like(self.hr, bool)
        return (self.seg == r_idx)[slice_idx] if slice_idx else self.seg == r_idx

    def _best_slice_per_roi(self) -> Dict[int, int]:
        """Return {ROI index -> slice index} with best balance."""
        scores = {}
        for r_idx in range(1, 4):  # tumour ROIs only
            # rank-sum: lower is better (PSNR high, SSIM & MI high)
            ranks = np.argsort(
                np.argsort(-self.metrics[:, r_idx, 0])  # PSNR (descending)
                + np.argsort(self.metrics[:, r_idx, 1])  # SSIM (ascending for tie-break)
                + np.argsort(self.metrics[:, r_idx, 2])  # MI   (ascending)
            )
            scores[r_idx] = int(ranks[0])
        # whole brain: choose middle slice (visual context)
        scores[0] = self.hr.shape[0] // 2
        return scores


# ---------------------------------------------------------------- driver
def run_analysis(hr: Path, sr: Path, seg: Path | None, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ana = SliceMetricAnalysis(hr, sr, seg)

    ana.make_violin_plot().savefig(out_dir / "violin_metrics.png", dpi=300, bbox_inches="tight")
    ana.make_pdf_plot().savefig(out_dir / "pdf_overlay.png", dpi=300, bbox_inches="tight")
    ana.make_difference_montage().savefig(out_dir / "difference_montage.png", dpi=300, bbox_inches="tight")

    # raw metrics saved for programmatic use
    np.save(out_dir / "slice_metrics.npy", ana.metrics)


# ---------------------------------------------------------------- CLI (optional)
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--hr", type=Path, required=True)
    p.add_argument("--sr", type=Path, required=True)
    p.add_argument("--seg", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    run_analysis(args.hr, args.sr, args.seg, args.out_dir)
    LOGGER.info("Analysis complete.")
    LOGGER.info("✓ Results saved to %s", args.out_dir)
