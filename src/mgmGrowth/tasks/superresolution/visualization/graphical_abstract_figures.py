#!/usr/bin/env python3
"""
Generate white-background PDF figures for the paper's graphical abstract:

1) Octants per pulse:
   - HR octant.
   - Nearest-neighbor (NN) degraded 3/5/7 mm, resampled back to HR grid.
   - Super-resolved (SR) octants for every model, at 3/5/7 mm.

2) Residual octants (SR − HR) for every pulse, model, and scale.

3) Per-ROI radiomics showcase PDFs (not for analysis): for each pulse and
   ROI (all, core, edema, surround), render a 3-panel PDF: an axial slice
   of the ROI overlay, the intensity histogram, and a small bar plot with
   simple FO/GLCM features.

Design goals: reuse I/O/conventions from existing visual tools; cache and
resample once; serialize plots to one-PDF-per-figure.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

try:
    import nibabel as nib
except Exception as e:  # pragma: no cover
    raise RuntimeError("This script requires nibabel to be installed.") from e

# Local project helpers
try:
    # Prefer importing shared utilities to avoid duplication
    from mgmGrowth.tasks.superresolution.visualization.show_octant_example import (
        data_LPS,
        hr_pulse_path,
        hr_seg_path,
        sr_model_path,
        resample_like,
        center_of_mass,
        compute_head_mask_from_hr,
        apply_background_black,
    )
except Exception:
    # Minimal fallbacks if the module path changes
    from typing import Any as _Any  # local alias to relax types in fallback

    def data_LPS(nii: _Any) -> np.ndarray:
        ras = nib.as_closest_canonical(nii)
        arr = ras.get_fdata(dtype=np.float32)
        arr = np.flip(arr, axis=0)  # R→L
        arr = np.flip(arr, axis=1)  # A→P
        return arr

    def hr_pulse_path(hr_dir: Path, subject: str, pulse: str) -> Path:
        return hr_dir / subject / f"{subject}-{pulse}.nii.gz"

    def hr_seg_path(hr_dir: Path, subject: str) -> Path:
        return hr_dir / subject / f"{subject}-seg.nii.gz"

    def sr_model_path(models_dir: Path, model: str, res_mm: int, subject: str, pulse: str) -> Path:
        return models_dir / model / f"{res_mm}mm" / "output_volumes" / f"{subject}-{pulse}.nii.gz"

    def resample_like(src: _Any, like: _Any, order: int = 1) -> _Any:
        from nibabel.processing import resample_from_to  # type: ignore
        return resample_from_to(src, (like.shape, like.affine), order=order)

    def center_of_mass(mask: np.ndarray) -> Tuple[int, int, int]:
        idx = np.argwhere(mask > 0)
        if idx.size == 0:
            nz = np.array(mask.shape) // 2
            return int(nz[2]), int(nz[0]), int(nz[1])
        mean_ijk = idx.mean(axis=0)  # (i,j,k)
        i = int(round(mean_ijk[0])); j = int(round(mean_ijk[1])); k = int(round(mean_ijk[2]))
        return k, i, j

    def compute_head_mask_from_hr(hr_vol_LPS: np.ndarray) -> np.ndarray:
        v = np.abs(hr_vol_LPS.astype(np.float32))
        v[~np.isfinite(v)] = 0.0
        nz = v[v > 0]
        if nz.size == 0:
            return np.zeros_like(v, dtype=bool)
        thr = max(1e-6, float(np.percentile(nz, 0.5)))
        return v > thr

    def apply_background_black(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = vol.copy()
        out[~mask] = 0.0
        return out

from mgmGrowth.tasks.superresolution.visualization.octant import plot_octant


# ----------------------------- CLI and config -----------------------------

RES_ROWS: Tuple[int, int, int] = (3, 5, 7)


@dataclass(frozen=True)
class Args:
    subject: str
    highres_dir: Path
    models_dir: Path
    pulses: Tuple[str, ...]
    models: Tuple[str, ...]
    coords: Optional[Tuple[int, int, int]]
    coord_mode: str
    out_dir: Path
    workers: int
    # radiomics
    roi_core_label: int
    roi_edema_label: int
    # styling
    fig_size: Tuple[int, int]
    log: str


def parse_args() -> Args:
    """
    
    Example:
    python src/mgmGrowth/tasks/superresolution/visualization/graphical_abstract_figures.py \
        --subject BraTS-MEN-00231-000 \
        --highres_dir /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/high_resolution \
        --models_dir /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/models \
        --pulses t1c t1n t2f t2w \
        --coords 65 120 135 \
        --out  /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/graphical_abstract \
        --workers 8 \
        --roi_core_label 1 \
        --roi_edema_label 2 
    """
    p = argparse.ArgumentParser(description="Create graphical abstract figures (octants, residuals, radiomics)")
    p.add_argument("--subject", required=True)
    p.add_argument("--highres_dir", required=True, type=Path)
    p.add_argument("--models_dir", required=True, type=Path)
    p.add_argument("--pulses", nargs="+", default=["t1c", "t1n", "t2f", "t2w"])
    p.add_argument("--models", nargs="+", default=None, help="Defaults to subdirs in models_dir")
    p.add_argument("--coords", nargs=3, type=int, default=None, metavar=("k", "i", "j"))
    p.add_argument("--coord_mode", choices=["auto", "com"], default="auto",
                   help="auto: COM if seg present else center; com: force center-of-mass")
    p.add_argument("--out", type=Path, default=Path("figures/graphical_abstract"))
    p.add_argument("--workers", type=int, default=0, help="If >0, parallelize SR loading/resampling")
    p.add_argument("--roi_core_label", type=int, default=3)
    p.add_argument("--roi_edema_label", type=int, default=2)
    p.add_argument("--fig_w", type=float, default=6.0)
    p.add_argument("--fig_h", type=float, default=6.0)
    p.add_argument("--log", default="INFO")

    ns = p.parse_args()
    models = tuple(sorted(d.name for d in ns.models_dir.iterdir() if d.is_dir())) if ns.models is None else tuple(ns.models)

    return Args(
        subject=ns.subject,
        highres_dir=ns.highres_dir,
        models_dir=ns.models_dir,
        pulses=tuple(ns.pulses),
        models=models,
        coords=tuple(ns.coords) if ns.coords is not None else None,
        coord_mode=str(ns.coord_mode),
        out_dir=ns.out,
        workers=int(ns.workers),
        roi_core_label=int(ns.roi_core_label),
        roi_edema_label=int(ns.roi_edema_label),
        fig_size=(int(ns.fig_w), int(ns.fig_h)),
        log=str(ns.log).upper(),
    )


def configure_matplotlib_white() -> None:
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'font.size': 12,
    })


# ----------------------------- I/O utilities ------------------------------

def load_nii(path: Path):
    try:
        logging.debug("Loading NIfTI: %s", path)
        return nib.load(str(path))
    except Exception as e:
        raise FileNotFoundError(f"Failed to load NIfTI: {path}") from e


def load_hr_and_seg(args: Args, pulse: str) -> Tuple[object, np.ndarray, Optional[np.ndarray]]:
    logging.info("[HR] Loading subject=%s pulse=%s", args.subject, pulse)
    hr_path = hr_pulse_path(args.highres_dir, args.subject, pulse)
    hr_nii = load_nii(hr_path)
    hr_LPS = data_LPS(hr_nii)
    seg_path = hr_seg_path(args.highres_dir, args.subject)
    if seg_path.exists():
        seg_LPS = data_LPS(load_nii(seg_path))
        logging.debug(
            "[SEG] Found segmentation for subject=%s at %s (hr shape=%s)",
            args.subject,
            seg_path,
            tuple(hr_LPS.shape),
        )
    else:
        seg_LPS = None
        logging.debug("[SEG] No segmentation found at %s", seg_path)
    try:
        zooms = tuple(map(float, nib.as_closest_canonical(hr_nii).header.get_zooms()[:3]))
    except Exception:
        zooms = (np.nan, np.nan, np.nan)
    logging.info("[HR] Loaded %s | shape=%s | zooms=%s", hr_path.name, tuple(hr_LPS.shape), zooms)
    return hr_nii, hr_LPS, seg_LPS


def compute_coords(args: Args, hr_LPS: np.ndarray, seg_LPS: Optional[np.ndarray]) -> Tuple[int, int, int]:
    logging.debug(
        "Computing coords | explicit=%s | mode=%s | seg_present=%s",
        args.coords is not None,
        args.coord_mode,
        seg_LPS is not None,
    )
    if args.coords is not None:
        coords = args.coords
    elif seg_LPS is not None and args.coord_mode in ("auto", "com"):
        coords = center_of_mass(seg_LPS > 0)
    else:
        nz = np.array(hr_LPS.shape) // 2
        coords = (int(nz[2]), int(nz[0]), int(nz[1]))
    logging.info("Using coordinates (k,i,j)=%s", coords)
    return coords


def nn_degrade_and_restore_to_hr(hr_vol: np.ndarray, hr_zooms: Tuple[float, float, float], target_mm: float) -> np.ndarray:
    """Degrade HR to target_mm using nearest neighbor, then restore to HR shape (also NN).

    Handles anisotropic voxel sizes by computing per-axis factors.
    """
    from scipy.ndimage import zoom  # type: ignore

    logging.debug("NN degrade+restore | target=%.2f mm | hr_zooms=%s | hr_shape=%s", target_mm, tuple(hr_zooms), tuple(hr_vol.shape))
    tz = np.asarray(hr_zooms, dtype=float)
    tz[tz <= 0] = 1.0
    # downsample factors to achieve approx target_mm spacing
    down = tz / float(target_mm)
    # zoom < 1 shrinks; for downsampling we want factor = 1/down
    shrink = 1.0 / np.maximum(down, 1e-6)
    low = zoom(hr_vol, zoom=shrink, order=0, prefilter=False)
    # restore to original shape
    back = zoom(low, zoom=np.array(hr_vol.shape) / np.array(low.shape), order=0, prefilter=False)
    # ensure exact shape
    back = _center_crop_or_pad_to(back, hr_vol.shape)
    logging.debug("NN degrade+restore done | low_shape=%s | back_shape=%s", tuple(low.shape), tuple(back.shape))
    return back.astype(np.float32, copy=False)


def _center_crop_or_pad_to(arr: np.ndarray, target_shape: Sequence[int]) -> np.ndarray:
    out = arr
    for ax, (n, t) in enumerate(zip(arr.shape, target_shape)):
        if n == t:
            continue
        if n > t:
            # crop centered
            start = (n - t) // 2
            end = start + t
            slicer = [slice(None)] * out.ndim
            slicer[ax] = slice(start, end)
            out = out[tuple(slicer)]
        else:
            # pad centered
            pad_before = (t - n) // 2
            pad_after = t - n - pad_before
            pad_width = [(0, 0)] * out.ndim
            pad_width[ax] = (pad_before, pad_after)
            out = np.pad(out, pad_width, mode="constant", constant_values=0)
    return out


def signed_residual(sr: np.ndarray, hr: np.ndarray) -> np.ndarray:
    a, b = sr.astype(np.float32), hr.astype(np.float32)
    # crop to common minimal shape to avoid mismatch after resampling
    tgt = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
    def crop(x, t):
        return x[tuple(slice(0, s) for s in t)]
    a = crop(a, tgt); b = crop(b, tgt)
    logging.debug("Residual | shape=%s", tgt)
    return a - b


def symmetric_rmax(residuals: Sequence[np.ndarray], q: float = 99.0) -> float:
    vals: List[float] = []
    for r in residuals:
        m = np.isfinite(r)
        if m.any():
            v = np.percentile(np.abs(r[m]).ravel(), q)
            vals.append(float(v))
    r = float(max(vals)) if vals else 1.0
    logging.debug("Symmetric residual vmax (q=%.1f) => %.6f", q, r)
    return r


# ------------------------------- Radiomics ---------------------------------

def first_order_stats(a: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    vals = a[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": np.nan, "var": np.nan, "skew": np.nan, "kurt": np.nan, "entropy": np.nan}
    hist, _ = np.histogram(vals, bins=256, density=True)
    p = hist / max(np.sum(hist), 1.0)
    p = p[p > 0]
    return {
        "mean": float(np.mean(vals)),
        "var": float(np.var(vals, ddof=1)) if vals.size > 1 else np.nan,
        "skew": float(((vals - vals.mean())**3).mean() / (vals.std(ddof=1)**3 + 1e-12)) if vals.size > 2 else np.nan,
        "kurt": float(((vals - vals.mean())**4).mean() / (vals.var(ddof=1)**2 + 1e-12) - 3.0) if vals.size > 3 else np.nan,
        "entropy": float(-np.sum(p * np.log2(p)))
    }


def glcm_props_slice(im2d: np.ndarray, m2d: np.ndarray) -> Dict[str, float]:
    try:
        from skimage.feature import graycomatrix, graycoprops
    except Exception:
        return {"contrast": np.nan, "homogeneity": np.nan, "correlation": np.nan}
    if not np.any(m2d):
        return {"contrast": np.nan, "homogeneity": np.nan, "correlation": np.nan}
    vals = im2d[m2d]
    vmin, vmax = np.percentile(vals, [1, 99]) if vals.size else (0.0, 1.0)
    if vmax <= vmin:
        vmax = vmin + 1.0
    # quantize to 32 gray-levels
    q = np.clip(((im2d - vmin) / (vmax - vmin + 1e-12) * 31).round().astype(np.uint8), 0, 31)
    dists = [1]
    angles = [0]
    P = graycomatrix(q, distances=dists, angles=angles, symmetric=True, normed=True)
    return {
        "contrast": float(graycoprops(P, 'contrast')[0, 0]),
        "homogeneity": float(graycoprops(P, 'homogeneity')[0, 0]),
        "correlation": float(graycoprops(P, 'correlation')[0, 0]),
    }


# ------------------------------- Rendering ---------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def render_octant_pdf(vol: np.ndarray, coords: Tuple[int, int, int], out_path: Path,
                      segmentation: Optional[np.ndarray] = None,
                      cmap: str = "gray",
                      fig_size: Tuple[int, int] = (6, 6)) -> None:
    ensure_dir(out_path.parent)
    logging.info("Render octant -> %s | coords=%s | cmap=%s", out_path, coords, cmap)
    fig = plot_octant(vol, coords, cmap=cmap, alpha=0.95, segmentation=segmentation,
                      seg_alpha=0.5, only_line=True, figsize=fig_size, save=out_path)
    plt.close(fig)


def render_residual_octant_pdf(res: np.ndarray, coords: Tuple[int, int, int], out_path: Path,
                               vmax_abs: float, fig_size: Tuple[int, int] = (6, 6)) -> None:
    ensure_dir(out_path.parent)
    # Diverging cmap centered at 0 using standard 'coolwarm'
    vmin, vmax = -vmax_abs, vmax_abs
    logging.info("Render residual octant -> %s | coords=%s | window=[%.6f, %.6f]", out_path, coords, vmin, vmax)
    # inject sentinels to enforce global window used by plot_octant
    k, i, j = coords
    vol = res.copy()
    nx, ny, nz = vol.shape
    ia = min(max(i + 1, 1), nx - 1)
    ja = min(max(j + 1, 1), ny - 1)
    ka = min(max(k + 1, 1), nz - 1)
    ib = min(ia + 1, nx - 1); jb = min(ja + 1, ny - 1); kb = min(ka + 1, nz - 1)
    vol[ia, ja, ka] = vmax
    vol[ib, jb, kb] = vmin
    fig = plot_octant(vol, coords, cmap="coolwarm", alpha=0.95, segmentation=None,
                      seg_alpha=0.0, only_line=False, figsize=fig_size, save=out_path)
    plt.close(fig)


def render_radiomics_pdf(hr_LPS: np.ndarray, seg_LPS: Optional[np.ndarray], coords: Tuple[int, int, int],
                         roi_label: Optional[int], mask_override: Optional[np.ndarray],
                         out_path: Path, title: str,
                         fig_size: Tuple[int, int] = (9, 3)) -> None:
    ensure_dir(out_path.parent)
    logging.info("Render radiomics -> %s | ROI=%s | roi_label=%s", out_path, title, str(roi_label))
    if seg_LPS is None and roi_label is not None and mask_override is None:
        logging.warning("No segmentation found; skipping ROI '%s' for %s", title, out_path.name)
        return

    # Build an explicit mask array (never None for type safety)
    if mask_override is not None:
        mask_roi: np.ndarray = mask_override.astype(bool)
    else:
        if seg_LPS is not None:
            if roi_label is None:
                mask_roi = (seg_LPS > 0)
            else:
                mask_roi = (seg_LPS == roi_label)
        else:
            # No segmentation available; start empty and fallback to head later
            mask_roi = np.zeros_like(hr_LPS, dtype=bool)
    if not np.any(mask_roi):
        # With no seg available or empty mask, make a crude head mask to showcase
        mask_roi = compute_head_mask_from_hr(hr_LPS)

    # FO stats and histogram
    fo = first_order_stats(hr_LPS, mask_roi)
    k, i, j = coords
    axial = hr_LPS[:, :, k]
    axial_mask = mask_roi[:, :, k]
    glcm = glcm_props_slice(axial, axial_mask)
    logging.debug("Radiomics stats | FO=%s | GLCM=%s", fo, glcm)

    fig, axs = plt.subplots(1, 3, figsize=fig_size)

    # 1) Axial slice with ROI outline
    axs[0].imshow(axial.T, cmap='gray', origin='lower')
    if np.any(axial_mask):
        # Ensure 'edge' is always a boolean ndarray
        edge: np.ndarray = axial_mask.astype(bool)
        try:
            import cv2  # type: ignore
            edge_u8 = axial_mask.astype(np.uint8)
            eroded = cv2.erode(edge_u8, np.ones((3, 3), np.uint8), iterations=1)
            edge = (edge_u8 - eroded).astype(bool)
        except Exception:
            try:
                from scipy.ndimage import binary_erosion  # type: ignore
                edge = axial_mask & (~binary_erosion(axial_mask, structure=np.ones((3, 3)), iterations=1))
            except Exception:
                pass  # keep fallback 'edge'
        overlay = np.zeros((*edge.T.shape, 4), dtype=np.float32)
        overlay[edge.T, :] = (1.0, 0.2, 0.2, 1.0)
        axs[0].imshow(overlay, origin='lower')
    axs[0].set_title(f"ROI: {title}")
    axs[0].axis('off')

    # 2) Histogram
    vals = hr_LPS[mask_roi]
    vals = vals[np.isfinite(vals)]
    axs[1].hist(vals, bins=64, color='#4477AA', alpha=0.9)
    axs[1].set_title("Intensity histogram")
    axs[1].set_xlabel("Intensity")
    axs[1].set_ylabel("Count")

    # 3) FO/GLCM bars
    keys = ["mean", "var", "skew", "kurt", "contrast", "homogeneity", "correlation"]
    data = [fo.get("mean", np.nan), fo.get("var", np.nan), fo.get("skew", np.nan), fo.get("kurt", np.nan),
            glcm.get("contrast", np.nan), glcm.get("homogeneity", np.nan), glcm.get("correlation", np.nan)]
    axs[2].bar(range(len(keys)), data, color=['#66CCEE']*4 + ['#EE6677']*3)
    axs[2].set_xticks(range(len(keys)))
    axs[2].set_xticklabels(keys, rotation=30, ha='right')
    axs[2].set_title("FO + GLCM (slice)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    logging.info("Saved radiomics figure: %s", out_path)
    plt.close(fig)


# --------------------------------- Main -----------------------------------

def _prefetch_sr_for_pulse(args: Args, pulse: str, hr_nii, hr_LPS: np.ndarray) -> Tuple[Dict[Tuple[str, int], np.ndarray], List[np.ndarray]]:
    """Load and resample SR volumes (optionally in parallel) for a pulse."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    logging.info("[SR] Prefetching SR volumes for pulse=%s (models=%d, scales=%s)", pulse, len(args.models), RES_ROWS)

    def _load_one(model: str, rmm: int) -> Tuple[Tuple[str, int], Optional[np.ndarray]]:
        sr_path = sr_model_path(args.models_dir, model, rmm, args.subject, pulse)
        if not sr_path.exists():
            logging.warning("Missing SR volume: %s", sr_path)
            return (model, rmm), None
        try:
            sr_nii = nib.load(str(sr_path))
            sr_r = resample_like(sr_nii, nib.as_closest_canonical(hr_nii), order=1)
            sr_LPS = data_LPS(sr_r)
            logging.debug("[SR] Loaded %s | shape=%s", sr_path.name, tuple(sr_LPS.shape))
            return (model, rmm), sr_LPS
        except Exception as e:
            logging.error("Failed SR resample %s: %s", sr_path.name, e)
            return (model, rmm), None

    sr_pre: Dict[Tuple[str, int], np.ndarray] = {}
    residuals_for_window: List[np.ndarray] = []

    if args.workers and args.workers > 0:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_load_one, m, r) for m in args.models for r in RES_ROWS]
            for fut in as_completed(futs):
                key, vol = fut.result()
                if vol is not None:
                    sr_pre[key] = vol
                    residuals_for_window.append(signed_residual(vol, hr_LPS))
    else:
        for m in args.models:
            for r in RES_ROWS:
                key, vol = _load_one(m, r)
                if vol is not None:
                    sr_pre[key] = vol
                    residuals_for_window.append(signed_residual(vol, hr_LPS))

    logging.info("[SR] Prefetch complete for pulse=%s | loaded=%d volumes | residuals=%d", pulse, len(sr_pre), len(residuals_for_window))
    return sr_pre, residuals_for_window


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log, logging.INFO), format="%(levelname)s: %(message)s")
    configure_matplotlib_white()
    logging.info(
        "Start graphical abstract generation | subject=%s | pulses=%s | models=%s | coord_mode=%s | workers=%d | out=%s",
        args.subject,
        ",".join(args.pulses),
        ",".join(args.models),
        args.coord_mode,
        args.workers,
        args.out_dir,
    )

    # Prepare output structure
    out_oct_hr = args.out_dir / "octants" / "hr"
    out_oct_nn = args.out_dir / "octants" / "nn"
    out_oct_sr = args.out_dir / "octants" / "sr"
    out_resid = args.out_dir / "residuals"
    out_radio = args.out_dir / "radiomics"
    for p in (out_oct_hr, out_oct_nn, out_oct_sr, out_resid, out_radio):
        ensure_dir(p)

    # Cache HR and SEG per pulse to avoid reloading
    hr_cache: Dict[str, Tuple[object, np.ndarray, Optional[np.ndarray]]] = {}
    coords: Optional[Tuple[int, int, int]] = None

    for pulse in args.pulses:
        try:
            hr_nii, hr_LPS, seg_LPS = load_hr_and_seg(args, pulse)
        except FileNotFoundError as e:
            logging.error("%s", e)
            continue
        hr_cache[pulse] = (hr_nii, hr_LPS, seg_LPS)
        if coords is None:
            coords = compute_coords(args, hr_LPS, seg_LPS)

    if coords is None:
        logging.error("No pulses loaded; aborting.")
        return

    # 1) HR and NN-degraded octants
    logging.info("Step 1/3: Rendering HR and NN-degraded octants")
    for pulse, (hr_nii, hr_LPS, seg_LPS) in hr_cache.items():
        # HR octant
        render_octant_pdf(hr_LPS, coords, out_oct_hr / f"{args.subject}_{pulse}_HR.pdf",
                          segmentation=seg_LPS, fig_size=args.fig_size)

        # NN degraded 3/5/7 mm → back to HR grid
        zooms = nib.as_closest_canonical(hr_nii).header.get_zooms()[:3]
        for rmm in RES_ROWS:
            nn_vol = nn_degrade_and_restore_to_hr(hr_LPS, zooms, rmm)
            render_octant_pdf(nn_vol, coords, out_oct_nn / f"{args.subject}_{pulse}_{rmm}mm_NN.pdf",
                              segmentation=seg_LPS, fig_size=args.fig_size)

    # 2) SR octants and residuals
    logging.info("Step 2/3: Rendering SR octants and residuals")
    for pulse, (hr_nii, hr_LPS, seg_LPS) in hr_cache.items():
        # Preload SR volumes (resampled to HR geometry)
        sr_pre, residuals_for_window = _prefetch_sr_for_pulse(args, pulse, hr_nii, hr_LPS)

        # Global residual window for this pulse
        rmax = max(symmetric_rmax(residuals_for_window, q=99.0), 1e-6)
        logging.info("[SR] Pulse=%s | residual window=±%.6f", pulse, rmax)

        # Render SR octants + residuals
        for (model, rmm), sr_LPS in sr_pre.items():
            # SR octant
            out_sr = out_oct_sr / pulse / model
            ensure_dir(out_sr)
            render_octant_pdf(sr_LPS, coords, out_sr / f"{args.subject}_{pulse}_{model}_{rmm}mm_SR.pdf",
                              segmentation=seg_LPS, fig_size=args.fig_size)

            # Residual octant
            res = signed_residual(sr_LPS, hr_LPS)
            out_re = out_resid / pulse / model
            ensure_dir(out_re)
            render_residual_octant_pdf(res, coords, out_re / f"{args.subject}_{pulse}_{model}_{rmm}mm_RES.pdf",
                                       vmax_abs=rmax, fig_size=args.fig_size)

    # 3) Radiomics per ROI (using HR only)
    # ROI map: all (union), core (label args.roi_core_label), edema (args.roi_edema_label), surround (head − tumor)
    logging.info("Step 3/3: Rendering radiomics panels (HR)")
    for pulse, (_, hr_LPS, seg_LPS) in hr_cache.items():
        head = compute_head_mask_from_hr(hr_LPS)
        tumor = (seg_LPS > 0) if seg_LPS is not None else np.zeros_like(hr_LPS, dtype=bool)
        surround = np.logical_and(head, ~tumor)
        roi_defs: List[Tuple[str, Optional[int], Optional[np.ndarray]]] = [
            ("all", None, head),
            ("core", args.roi_core_label, None),
            ("edema", args.roi_edema_label, None),
            ("surround", None, surround),
        ]

        for roi_name, roi_label, explicit_mask in roi_defs:
            out_p = out_radio / pulse
            ensure_dir(out_p)
            render_radiomics_pdf(hr_LPS, seg_LPS, coords,
                                 roi_label, explicit_mask,
                                 out_p / f"{args.subject}_{pulse}_radiomics_{roi_name}.pdf",
                                 title=roi_name, fig_size=(9, 3))

    logging.info("Done. Output under: %s", args.out_dir)


if __name__ == "__main__":
    main()

