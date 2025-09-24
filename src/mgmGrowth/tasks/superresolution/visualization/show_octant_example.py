#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_model_comparison_octants.py

Octant-based visualizations for super-resolution model comparison.

Outputs:
1) HR-only figure: 1xP grid (columns = pulses) with octant views, optional
   segmentation outline.
2) Per-pulse model comparison: for each pulse (and each requested resolution),
   a 2xM grid with:
     - Row 1: SR octant per model
     - Row 2: Residual octant per model (SR − HR), diverging cmap centered at 0.

Colormap:
- Diverging blue↔white↔red centered at 0 for signed residuals.
  Symmetric scaling via the 99th percentile of |SR − HR| across models.

Geometry:
- Volumes are reoriented to LPS.
- SR is resampled to HR geometry when needed (requires SciPy via nibabel).
- If resampling is unavailable and shapes differ, an error is raised.

Usage example:
python src/mgmGrowth/tasks/superresolution/visualization/show_octant_example.py \
  --subject BraTS-MEN-00231-000 \
  --highres_dir /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/high_resolution \
  --models_dir  /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/models \
  --pulses t1c t2f t2w t1n \
  --models UNIRES SMORE ECLARE BSPLINE \
  --out /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/figures \
  --coords 65 120 135   # or three ints: k i j


"""

from __future__ import annotations
import argparse
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, ListedColormap

# --- import project/local octant renderer -----------------------------------
try:
    from mgmGrowth.tasks.superresolution.visualization.octant import plot_octant
except Exception:
    from octant import plot_octant  # type: ignore


# -------------------- styling (copied from sagittal) ------------------------

def configure_matplotlib() -> None:
    try:
        import scienceplots  # noqa
        plt.style.use(['science'])
    except Exception as e:
        logging.warning("scienceplots not available: %s", e)
    plt.rcParams.update({
        'figure.dpi': 600,
        'font.size': 16,
        'font.family': 'serif',
        'font.serif': ['Times'],
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'black',
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
    })
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    except Exception as e:
        logging.warning("LaTeX not available: %s", e)
        plt.rcParams['text.usetex'] = False

def build_black_center_diverging() -> ListedColormap:
    """
    Custom diverging map:
      negative: #762A83  →  0.0
      zero    : #000000  →  0.5
      positive: #ffe945  →  1.0
    """
    def _hex_rgba(h: str) -> np.ndarray:
        h = h.lstrip('#')
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return np.array([r, g, b, 1.0], dtype=float)

    neg = _hex_rgba("ffe945")  # low end (SR−HR < 0)
    pos = _hex_rgba("762A83")  # high end (SR−HR > 0)
    blk = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    n = 256
    colors = np.empty((n, 4), dtype=float)
    # 0..127: neg → black
    for t in range(128):
        a = t / 127.0
        colors[t] = (1.0 - a) * neg + a * blk
    # 128..255: black → pos
    for t in range(128, 256):
        a = (t - 128) / 127.0
        colors[t] = (1.0 - a) * blk + a * pos

    return ListedColormap(colors, name="bb5566_black_004488")


# -------------------- I/O and geometry --------------------------------------

def load_nii(path: Path) -> Any:
    try:
        return nib.load(str(path))  # type: ignore[arg-type]
    except Exception as e:
        raise FileNotFoundError(f"Failed to load NIfTI: {path}") from e


def data_LPS(nii: Any) -> np.ndarray:
    ras = nib.as_closest_canonical(nii)
    arr = ras.get_fdata(dtype=np.float32)
    arr = np.flip(arr, axis=0)  # R→L
    arr = np.flip(arr, axis=1)  # A→P
    return arr


def resample_like(src: Any, like: Any, order: int = 1) -> Any:
    try:
        from nibabel.processing import resample_from_to  # type: ignore
    except Exception as e:
        raise RuntimeError("Resampling requires nibabel[scipy]. Install SciPy.") from e
    return resample_from_to(src, (like.shape, like.affine), order=order)


def hr_pulse_path(hr_dir: Path, subject: str, pulse: str) -> Path:
    return hr_dir / subject / f"{subject}-{pulse}.nii.gz"


def hr_seg_path(hr_dir: Path, subject: str) -> Path:
    return hr_dir / subject / f"{subject}-seg.nii.gz"


def sr_model_path(models_dir: Path, model: str, res_mm: int, subject: str, pulse: str) -> Path:
    return models_dir / model / f"{res_mm}mm" / "output_volumes" / f"{subject}-{pulse}.nii.gz"


def bbox_center(mask: np.ndarray) -> Tuple[int, int, int]:
    pos = np.argwhere(mask > 0)
    if pos.size == 0:
        nz = np.array(mask.shape) // 2
        return int(nz[2]), int(nz[0]), int(nz[1])
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    ctr = ((mins + maxs) / 2.0).round().astype(int)  # (i, j, k)
    return int(ctr[2]), int(ctr[0]), int(ctr[1])


def center_of_mass(mask: np.ndarray) -> Tuple[int, int, int]:
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        nz = np.array(mask.shape) // 2
        return int(nz[2]), int(nz[0]), int(nz[1])
    mean_ijk = idx.mean(axis=0)  # (i,j,k)
    i = int(round(mean_ijk[0])); j = int(round(mean_ijk[1])); k = int(round(mean_ijk[2]))
    return k, i, j

def _otsu_threshold(x: np.ndarray) -> float:
    """1D Otsu on finite, positive values."""
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size == 0:
        return 0.0
    hist, edges = np.histogram(x, bins=256)
    p = hist.astype(np.float64)
    p /= p.sum() + 1e-12
    bins = (edges[:-1] + edges[1:]) * 0.5
    omega = np.cumsum(p)
    mu = np.cumsum(p * bins)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = np.nan
    sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    k = np.nanargmax(sigma_b2)
    return float(bins[k])


def compute_head_mask_from_hr(hr_vol_LPS: np.ndarray) -> np.ndarray:
    """
    Head/background mask from HR magnitude using the largest 3D connected component.

    Steps:
      1) Robust low-percentile threshold to drop speckle (no Otsu).
      2) Keep the largest connected component (18-connectivity).
      3) Fill internal holes and lightly close gaps.

    Returns
    -------
    np.ndarray
        Boolean mask with True for head/skull/brain and False for air/background.
    """
    v = np.abs(hr_vol_LPS.astype(np.float32))
    v[~np.isfinite(v)] = 0.0

    # Robust tiny threshold to suppress numerical noise
    nz = v[v > 0]
    if nz.size == 0:
        return np.zeros_like(v, dtype=bool)
    thr = max(1e-6, float(np.percentile(nz, 0.5)))  # 0.5th percentile is conservative
    fg = v > thr

    try:
        from scipy.ndimage import (
            binary_opening, binary_closing, binary_fill_holes,
            label, generate_binary_structure,
        )
        st = generate_binary_structure(3, 2)  # 18-connectivity
        fg = binary_opening(fg, structure=st, iterations=1)

        labels, nlab = label(fg, structure=st)
        if nlab == 0:
            return np.zeros_like(fg, dtype=bool)

        counts = np.bincount(labels.ravel())
        counts[0] = 0  # ignore background
        keep = int(counts.argmax())
        mask = labels == keep

        mask = binary_fill_holes(mask)
        mask = binary_closing(mask, structure=st, iterations=1)
        return mask
    except Exception:
        # Fallback without SciPy: return foreground; still zeros-out air.
        return fg
    
def apply_background_black(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out everything outside the head mask."""
    out = vol.copy()
    out[~mask] = 0.0
    return out




# -------------------- residuals and colormap --------------------------------

def signed_residual(sr: np.ndarray, hr: np.ndarray) -> np.ndarray:
    a, b = sr.astype(np.float32), hr.astype(np.float32)
    if a.shape != b.shape:
        tgt = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))

        def crop(x, t):
            sx, sy, sz = x.shape
            cx, cy, cz = (sx - t[0]) // 2, (sy - t[1]) // 2, (sz - t[2]) // 2
            return x[cx:cx+t[0], cy:cy+t[1], cz:cz+t[2]]

        a, b = crop(a, tgt), crop(b, tgt)
    return a - b


def symmetric_rmax(residuals: Sequence[np.ndarray], q: float = 99.0) -> float:
    vals = []
    for r in residuals:
        m = np.isfinite(r)
        if m.any():
            vals.append(np.percentile(np.abs(r[m]), q))
    return float(max(vals)) if vals else 1.0


# -------------------- render octant to raster figure ------------------------

def _with_global_minmax_for_plot_octant(
    vol: np.ndarray,
    k: int, i: int, j: int,
    vmin: float,
    vmax: float
) -> np.ndarray:
    """
    Force plot_octant to use global vmin/vmax by planting two sentinel voxels
    inside the displayed octant but away from the three faces at x=0,y=0,z=0.
    """
    out = vol.copy()
    nx, ny, nz = out.shape
    ia = min(max(i+1, 1), nx-1)
    ja = min(max(j+1, 1), ny-1)
    ka = min(max(k+1, 1), nz-1)
    ib = min(ia+1, nx-1); jb = min(ja+1, ny-1); kb = min(ka+1, nz-1)
    out[ia, ja, ka] = vmax
    out[ib, jb, kb] = vmin
    return out


def render_octant_png(
    volume: np.ndarray,
    coords: Tuple[int, int, int],
    *,
    segmentation: Optional[np.ndarray],
    cmap,
    seg_alpha: float,
    only_line: bool,
    enforce_minmax: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (4, 4),
) -> np.ndarray:
    """
    Call plot_octant(volume, ...) with save=tmpfile, close fig, then read PNG as array.
    If enforce_minmax given, inject sentinels so plot_octant uses the intended window.
    """
    k, i, j = coords
    vol = volume if enforce_minmax is None else _with_global_minmax_for_plot_octant(volume, k, i, j, enforce_minmax[0], enforce_minmax[1])
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        fig = plot_octant(
            vol, coords, cmap=cmap, alpha=0.95,
            segmentation=segmentation, seg_alpha=seg_alpha,
            only_line=only_line, figsize=figsize, save=tmp_path
        )
        plt.close(fig)
        img = plt.imread(tmp_path)
        # If PNG has alpha channel, composite over black to avoid white matting
        if img.ndim == 3 and img.shape[-1] == 4:
            rgb = img[..., :3]
            alpha = img[..., 3:4]
            img = rgb * alpha  # black background: (1-alpha)*0 added implicitly
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return img


# -------------------- layout spec -------------------------------------------

RES_ROWS = (3, 5, 7)  # mm

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
    # layout controls borrowed from sagittal script
    row_gap: float
    fig_width: float
    fig_height: float
    pair_gap: float
    group_gap: float
    first_left_gap: float
    top_row_gap: Optional[float]
    left_margin: float
    right_margin: float
    top_margin: float
    bottom_margin: float
    wspace: float
    fmt: str
    log: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Per-pulse octant grids: rows=3/5/7mm, cols=2xmodels (SR|Residual)")
    p.add_argument("--subject", required=True)
    p.add_argument("--highres_dir", required=True, type=Path)
    p.add_argument("--models_dir", required=True, type=Path)
    p.add_argument("--pulses", nargs="+", default=["t1c", "t1n", "t2f", "t2w"])
    p.add_argument("--models", nargs="+", default=None, help="Defaults to subdirs in models_dir")
    p.add_argument("--coords", nargs=3, type=int, default=None, metavar=("k", "i", "j"))
    p.add_argument("--coord_mode", choices=["auto", "bbox", "com"], default="auto")

    # style/spacing cloned from sagittal
    p.add_argument('--row_gap', type=float, default=0.0015)
    p.add_argument('--fig_width', type=float, default=11.5)
    p.add_argument('--fig_height', type=float, default=6.9)
    p.add_argument('--pair_gap', type=float, default=0.010)
    p.add_argument('--group_gap', type=float, default=0.055)
    p.add_argument('--first_left_gap', type=float, default=0.025)
    p.add_argument('--top_row_gap', type=float, default=None)
    p.add_argument('--left_margin', type=float, default=0.06)
    p.add_argument('--right_margin', type=float, default=0.995)
    p.add_argument('--top_margin', type=float, default=0.955)
    p.add_argument('--bottom_margin', type=float, default=0.08)
    p.add_argument('--wspace', type=float, default=0.02)

    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--fmt", choices=["pdf", "png"], default="pdf")
    p.add_argument("--log", default="INFO")

    ns = p.parse_args()

    # infer models
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
        row_gap=ns.row_gap,
        fig_width=ns.fig_width,
        fig_height=ns.fig_height,
        pair_gap=ns.pair_gap,
        group_gap=ns.group_gap,
        first_left_gap=ns.first_left_gap,
        top_row_gap=ns.top_row_gap,
        left_margin=ns.left_margin,
        right_margin=ns.right_margin,
        top_margin=ns.top_margin,
        bottom_margin=ns.bottom_margin,
        wspace=ns.wspace,
        fmt=ns.fmt,
        log=str(ns.log).upper(),
    )


# -------------------- coord selection ---------------------------------------

def compute_coords(args: Args) -> Tuple[int, int, int]:
    if args.coords is not None:
        return args.coords
    seg_path = hr_seg_path(args.highres_dir, args.subject)
    if seg_path.exists():
        seg = data_LPS(load_nii(seg_path))
        if args.coord_mode in ("bbox", "auto"):
            k, i, j = bbox_center(seg)
            if args.coord_mode == "auto":
                return k, i, j
            return k, i, j
        if args.coord_mode == "com":
            return center_of_mass(seg)
    # fallback center from any HR pulse
    hr_any = load_nii(hr_pulse_path(args.highres_dir, args.subject, args.pulses[0]))
    vol = data_LPS(hr_any)
    nz = np.array(vol.shape) // 2
    return int(nz[2]), int(nz[0]), int(nz[1])


# -------------------- main per-pulse figure ---------------------------------

def make_pulse_figure(
    args: Args,
    pulse: str,
    coords: Tuple[int, int, int],
    res_rows: Sequence[int] = RES_ROWS,
) -> None:
    # Load HR reference and segmentation
    hr_nii = load_nii(hr_pulse_path(args.highres_dir, args.subject, pulse))
    hr_LPS = data_LPS(hr_nii)
    seg_path = hr_seg_path(args.highres_dir, args.subject)
    seg_LPS = data_LPS(load_nii(seg_path)) if seg_path.exists() else None

    # --- build head mask (from HR only) and apply to HR ---
    head_mask = compute_head_mask_from_hr(hr_LPS)
    masked_hr = apply_background_black(hr_LPS, head_mask)

    # Cache SR (masked) and collect residuals for global window
    sr_cache: Dict[Tuple[str, int], np.ndarray] = {}
    all_residuals: List[np.ndarray] = []

    for m in args.models:
        for rmm in res_rows:
            sr_nii = load_nii(sr_model_path(args.models_dir, m, rmm, args.subject, pulse))
            if (sr_nii.shape != hr_nii.shape) or not np.allclose(sr_nii.affine, hr_nii.affine):
                sr_nii = resample_like(sr_nii, hr_nii, order=1)
            sr_LPS = data_LPS(sr_nii)
            masked_sr = apply_background_black(sr_LPS, head_mask)
            sr_cache[(m, rmm)] = masked_sr

            all_residuals.append(signed_residual(masked_sr, masked_hr))

    # Error window (99th percentile of |SR−HR|) and sequential colormap for error
    rmax = max(symmetric_rmax(all_residuals, q=99.0), 1e-6)
    err_cmap = plt.get_cmap('afmhot')

    # Grid: rows = 3/5/7mm; cols = 2 x models (SR | Residual)
    n_rows = len(res_rows)
    n_cols = 2 * len(args.models)

    A4_W = 11.69
    fig_w = min(args.fig_width, A4_W); fig_h = args.fig_height
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_w, fig_h), squeeze=True)
    fig.patch.set_facecolor('black')

    for r_idx, rmm in enumerate(res_rows):
        for m_idx, m in enumerate(args.models):
            masked_sr = sr_cache[(m, rmm)]

            # SR octant (masked)
            sr_img = render_octant_png(
                masked_sr, coords,
                segmentation=seg_LPS, cmap='gray',
                seg_alpha=0.30, only_line=True,
                enforce_minmax=None, figsize=(4, 4),
            )
            axL = axes[r_idx, 2*m_idx]
            axL.imshow(sr_img, origin='upper', interpolation='nearest'); axL.set_xticks([]); axL.set_yticks([])
            for s in axL.spines.values(): s.set_visible(False)
            axL.set_facecolor('black')

            # Error octant: |SR − HR| normalized by rmax → [0, 100]
            err_vol = np.abs(signed_residual(masked_sr, masked_hr))
            err_vol = (100.0 * np.clip(err_vol / rmax, 0.0, 1.0)).astype(np.float32)
            # Ensure background is exact 0
            err_vol = apply_background_black(err_vol, head_mask)
            err_img = render_octant_png(
                err_vol, coords,
                segmentation=None, cmap=err_cmap,
                seg_alpha=0.0, only_line=False,
                enforce_minmax=(0.0, 100.0), figsize=(4, 4),
            )
            axR = axes[r_idx, 2*m_idx + 1]
            axR.imshow(err_img, origin='upper', interpolation='nearest'); axR.set_xticks([]); axR.set_yticks([])
            for s in axR.spines.values(): s.set_visible(False)
            axR.set_facecolor('black')

    # Row labels
    left_x = 0.03
    for r_idx, rmm in enumerate(res_rows):
        box = axes[r_idx, 0].get_position()
        y_mid = (box.y0 + box.y1) / 2.0
        fig.text(left_x, y_mid, f"{rmm}mm", color='white', ha='left', va='center')

    # Spacing
    plt.subplots_adjust(
        left=args.left_margin, right=args.right_margin,
        top=args.top_margin, bottom=args.bottom_margin,
        wspace=args.wspace, hspace=args.row_gap
    )

    # Pair/group packing
    try:
        G, R = len(args.models), n_rows
        x0s, x1s = [], []
        for c in range(n_cols):
            b = axes[0, c].get_position()
            x0s.append(b.x0); x1s.append(b.x1)
        inner_left, inner_right = min(x0s), max(x1s)
        inner_width = inner_right - inner_left

        within_gap = args.pair_gap
        between_gap = args.group_gap
        extra_left = args.first_left_gap
        total_gaps = extra_left + G*within_gap + (G-1)*between_gap
        w_panel = max(0.0001, (inner_width - total_gaps) / (2*G))

        for g in range(G):
            start_x = inner_left + extra_left + g * (2*w_panel + within_gap + between_gap)
            for r in range(R):
                bL = axes[r, 2*g].get_position()
                axes[r, 2*g].set_position([start_x, bL.y0, w_panel, bL.height])
                bR = axes[r, 2*g + 1].get_position()
                axes[r, 2*g + 1].set_position([start_x + w_panel + within_gap, bR.y0, w_panel, bR.height])

        if R >= 2 and args.top_row_gap is not None:
            row_boxes = [axes[r, 0].get_position() for r in range(R)]
            current_gap = row_boxes[0].y0 - (row_boxes[1].y0 + row_boxes[1].height)
            if current_gap > args.top_row_gap:
                delta = current_gap - args.top_row_gap
                for c in range(n_cols):
                    b = axes[0, c].get_position()
                    axes[0, c].set_position([b.x0, b.y0 - delta, b.width, b.height])
    except Exception as e:
        logging.warning("Custom spacing adjustment failed: %s", e)

    # Model headers
    top_y = 0.975
    for m_idx, m in enumerate(args.models):
        boxL = axes[0, 2*m_idx].get_position()
        boxR = axes[0, 2*m_idx + 1].get_position()
        x_center = 0.5 * ((boxL.x0 + boxL.x1) + (boxR.x0 + boxR.x1)) / 2.0
        fig.text(x_center, top_y, m, color='white', ha='center', va='top')

    # Colorbar for error [0, 100] (100 ≈ 99th percentile of |SR−HR|)
    sm = plt.cm.ScalarMappable(norm=Normalize(0.0, 100.0), cmap=err_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.035, pad=0.05)
    cbar.set_label(r"Absolute Residual Error $\,|\mathrm{SR} - \mathrm{HR}|$", color='white')
    outline = getattr(cbar, 'outline', None)
    if outline is not None:
        try:
            outline.set_edgecolor('white')  # type: ignore[attr-defined]
        except Exception:
            pass
    cbar.ax.tick_params(color='white', labelcolor='white')

    # Save
    out_dir = args.out_dir / "octants_by_model" / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.subject}_{pulse}_octants_models.{args.fmt}"
    if args.fmt == "pdf":
        with PdfPages(out_file) as pdf:
            pdf.savefig(fig, facecolor=fig.get_facecolor())
    else:
        fig.savefig(out_file, dpi=600, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    logging.info("Saved: %s", out_file)



# -------------------- main ---------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log, logging.INFO),
                        format="%(levelname)s: %(message)s")
    configure_matplotlib()

    coords = compute_coords(args)

    for pulse in args.pulses:
        make_pulse_figure(args, pulse, coords, RES_ROWS)


if __name__ == "__main__":
    main()
