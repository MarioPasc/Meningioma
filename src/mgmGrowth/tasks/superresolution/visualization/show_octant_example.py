#!/usr/bin/env python3
"""
Wide 4×(2×pulses) octant grid with residual maps and single PDF output.

This replicates the layout and styling of `show_sagital_example.py` but renders
3D octant views (anterior–right–cranial) instead of 2D sagittal slices.

Per pulse (two columns):
- Row 'High-Res.': centered HR octant between the two columns (no right panel)
- Rows lower (3mm/5mm/7mm): left = NN-aligned octant; right = residual (NN − HR)

Residuals use a diverging colormap with a global symmetric window and a
horizontal colorbar at the bottom labeled "Residual ΔI = NN − HR (a.u.)".

Coordinates can be provided as three integers with --coords k i j matching
the octant utility order: (k_axial, i_coronal, j_sagittal).
"""

from __future__ import annotations
import os
import argparse
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, ListedColormap
import tempfile
from pathlib import Path

# Octant renderer and volume loader (LPS + resample to reference)
import mgmGrowth.tasks.superresolution.visualization.octant as oc
from mgmGrowth.tasks.superresolution.utils.imio import load_lps


# -------------------- styling --------------------

def configure_matplotlib() -> None:
    """Matplotlib config. Falls back if LaTeX/scienceplots unavailable."""
    try:
        import scienceplots  # noqa: F401
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


# -------------------- helpers --------------------

def robust_window3d(vol: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> Tuple[float, float]:
    m = np.isfinite(vol)
    if not m.any():
        return float(np.nanmin(vol)), float(np.nanmax(vol))
    lo = float(np.percentile(vol[m], q_low))
    hi = float(np.percentile(vol[m], q_high))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def build_file_path(base: str, subdir: str, subject: str, pulse: str) -> str:
    return os.path.join(base, subdir, subject, f"{subject}-{pulse}.nii.gz")


RESOLUTION_DIR = {
    'hr':  'subset',
    '3mm': 'low_res/3mm',
    '5mm': 'low_res/5mm',
    '7mm': 'low_res/7mm',
}


def render_octant_to_image(volume: np.ndarray,
                           coords: Tuple[int, int, int],
                           cmap: Any = 'gray',
                           seg: Optional[np.ndarray] = None,
                           seg_alpha: float = 0.0,
                           only_line: bool = False,
                           figsize: Tuple[int, int] = (2, 2)) -> np.ndarray:
    """Render an octant as a small RGB image using the oc.plot_octant routine.

    Returns H×W×3 uint8 image suitable for imshow.
    """
    # oc.plot_octant creates its own fig; we capture the canvas as an image
    # Save to a temporary file to avoid oc.plot_octant calling plt.show()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()

    fig = oc.plot_octant(
        volume, coords,
        segmentation=seg,
        seg_alpha=seg_alpha,
        only_line=only_line,
        cmap=cmap,
        grid=False,
        xticks=[], yticks=[], zticks=[],
        figsize=figsize,
        save=tmp_path,
    )
    # Ensure black background to blend with our black figure
    fig.patch.set_facecolor('black')
    # Draw and extract RGB buffer
    canvas: Any = fig.canvas
    canvas.draw()
    w, h = canvas.get_width_height()
    if hasattr(canvas, 'tostring_rgb'):
        buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        img = buf.reshape(h, w, 3)
    else:
        rgba = np.asarray(canvas.buffer_rgba())
        img = rgba[..., :3]
    plt.close(fig)
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
    return img


# -------------------- data structures --------------------

@dataclass(frozen=True)
class Paths:
    super_resolution: str
    output_dir: str


@dataclass(frozen=True)
class Spec:
    subject: str
    coords: Tuple[int, int, int]
    resolutions: Tuple[str, ...] = ('hr', '3mm', '5mm', '7mm')
    pulses:     Tuple[str, ...] = ('t1c', 't2w', 't2f', 't1n')

    @property
    def n_rows(self) -> int:
        return len(self.resolutions)

    @property
    def n_cols(self) -> int:
        return 2 * len(self.pulses)


@dataclass
class Cell:
    vol: Optional[np.ndarray]    # 3D volume; for HR/NN/residual
    kind: str                    # 'hr_center' | 'nn' | 'residual' | 'blank'


def load_hr_per_pulse(spec: Spec, paths: Paths) -> Dict[str, Tuple[np.ndarray, Any]]:
    """Load HR volumes (LPS) per pulse and keep nib image for geometry."""
    out: Dict[str, Tuple[np.ndarray, Any]] = {}
    for pulse in spec.pulses:
        fp = build_file_path(paths.super_resolution, RESOLUTION_DIR['hr'], spec.subject, pulse)
        nii = nib.load(fp)
        vol = load_lps(fp)  # LPS orientation
        out[pulse] = (vol, nii)
    return out


def collect_grid_volumes(spec: Spec, paths: Paths) -> Tuple[List[List[Cell]], List[np.ndarray]]:
    """Build a grid of volume cells and collect residual volumes for global scaling."""
    hr_by_pulse = load_hr_per_pulse(spec, paths)

    grid: List[List[Cell]] = [[Cell(None, 'blank') for _ in range(spec.n_cols)] for _ in range(spec.n_rows)]
    residuals: List[np.ndarray] = []

    for p_idx, pulse in enumerate(spec.pulses):
        left_c  = 2 * p_idx
        right_c = 2 * p_idx + 1

        hr_vol, hr_img = hr_by_pulse[pulse]
        grid[0][left_c] = Cell(hr_vol, 'hr_center')
        grid[0][right_c] = Cell(None, 'blank')

        for r in range(1, spec.n_rows):
            res_label = spec.resolutions[r]
            fp = build_file_path(paths.super_resolution, RESOLUTION_DIR[res_label], spec.subject, pulse)
            try:
                lr_vol = load_lps(fp, like=hr_img, order=1)
            except Exception as e:
                logging.warning("Failed loading/resampling %s: %s", fp, e)
                grid[r][left_c]  = Cell(None, 'blank')
                grid[r][right_c] = Cell(None, 'blank')
                continue

            # Subtract after alignment; keep signed residual for diverging cmap
            res_vol = lr_vol.astype(np.float32) - hr_vol.astype(np.float32)
            residuals.append(res_vol)

            grid[r][left_c]  = Cell(lr_vol, 'nn')
            grid[r][right_c] = Cell(res_vol, 'residual')

    return grid, residuals


def draw_grid_pdf(spec: Spec,
                  paths: Paths,
                  grid: List[List[Cell]],
                  residuals: List[np.ndarray]) -> str:
    os.makedirs(paths.output_dir, exist_ok=True)
    out_name = f"{spec.subject}_octant_k{spec.coords[0]}_i{spec.coords[1]}_j{spec.coords[2]}_wide_pairs.pdf"
    out_path = os.path.join(paths.output_dir, out_name)

    # Figure size and layout same as sagittal script
    A4_W = 11.69
    fig_w = min(11.5, A4_W)
    fig_h = 6.9
    fig, axes = plt.subplots(
        nrows=spec.n_rows,
        ncols=spec.n_cols,
        figsize=(fig_w, fig_h),
        sharex=False,
        sharey=False,
        squeeze=True,
    )
    fig.patch.set_facecolor('black')

    # Global symmetric window for residuals (99th of |.|)
    if residuals:
        q = [np.percentile(np.abs(r[np.isfinite(r)]), 99.0) for r in residuals]
        rmax = float(max(q)) if q else 1.0
    else:
        rmax = 1.0
    res_norm = Normalize(vmin=-rmax, vmax=rmax)
    try:
        base_cmap = plt.get_cmap('berlin_r')
    except Exception:
        base_cmap = plt.get_cmap('seismic')
    colors = base_cmap(np.linspace(0, 1, 256))
    colors[127] = [0, 0, 0, 1]
    colors[128] = [0, 0, 0, 1]
    res_cmap = ListedColormap(colors, name='black_center_div')

    # Top headers centered over each pair of columns
    top_y = 0.975
    for p_idx, pulse in enumerate(spec.pulses):
        axL = axes[0, 2*p_idx]
        axR = axes[0, 2*p_idx + 1]
        boxL = axL.get_position(); boxR = axR.get_position()
        x_center = 0.5 * (boxL.x0 + boxL.x1 + boxR.x0 + boxR.x1) / 2.0
        fig.text(x_center, top_y, pulse.upper(), color='white', ha='center', va='top')

    # Render and plot panels
    for r in range(spec.n_rows):
        for c in range(spec.n_cols):
            ax = axes[r, c]
            ax.set_facecolor('black')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            cell = grid[r][c]
            if cell.vol is None or cell.kind in ('blank', 'hr_center'):
                continue

            vol = cell.vol
            if cell.kind == 'residual':
                # Symmetric scaling: clip to ±rmax and draw with diverging cmap
                vol = np.clip(vol, -rmax, rmax)
                img = render_octant_to_image(vol, spec.coords, cmap=res_cmap)
            else:
                # Robust window per volume in grayscale
                vmin, vmax = robust_window3d(vol, 1.0, 99.0)
                vol = np.clip(vol, vmin, vmax)
                img = render_octant_to_image(vol, spec.coords, cmap='gray')

            ax.imshow(img)

    # Row labels on the left
    left_x = 0.03
    for r, res in enumerate(spec.resolutions):
        label = 'High-Res.' if res == 'hr' else res
        box = axes[r, 0].get_position()
        y_mid = (box.y0 + box.y1) / 2.0
        fig.text(left_x, y_mid, label, color='white', ha='left', va='center')

    # Spacing
    plt.subplots_adjust(left=0.06, right=0.995, top=0.955, bottom=0.08, wspace=0.08, hspace=0.003)

    # Insert centered HR images between column pairs on the top row
    for p_idx, pulse in enumerate(spec.pulses):
        hr_cell = grid[0][2*p_idx]
        if hr_cell.vol is None:
            continue
        axL = axes[0, 2*p_idx]; axR = axes[0, 2*p_idx + 1]
        # Clear any artists on the original top-row axes to avoid overlap
        axL.cla(); axR.cla()
        axL.set_facecolor('black'); axR.set_facecolor('black')
        axL.set_xticks([]); axL.set_yticks([])
        axR.set_xticks([]); axR.set_yticks([])
        for s in axL.spines.values(): s.set_visible(False)
        for s in axR.spines.values(): s.set_visible(False)

        boxL = axL.get_position(); boxR = axR.get_position()
        # HR same size as a single column and centered between the two columns
        hr_w = boxL.width; hr_h = boxL.height
        gap_center_x = 0.5 * (boxL.x1 + boxR.x0)
        hr_x = gap_center_x - hr_w/2
        hr_y = boxL.y0
        hr_ax = fig.add_axes((hr_x, hr_y, hr_w, hr_h))
        hr_ax.set_facecolor('black'); hr_ax.set_xticks([]); hr_ax.set_yticks([])
        for s in hr_ax.spines.values(): s.set_visible(False)

        # Robust window for HR
        vmin, vmax = robust_window3d(hr_cell.vol, 1.0, 99.0)
        vol_hr = np.clip(hr_cell.vol, vmin, vmax)
        img_hr = render_octant_to_image(vol_hr, spec.coords, cmap='gray')
    hr_ax.imshow(img_hr)

    # Horizontal colorbar for residuals
    if residuals:
        sm = plt.cm.ScalarMappable(norm=res_norm, cmap=res_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.035, pad=0.05)
        cbar.set_label(r"Residual $\Delta I =$ NN $-$ HR (a.u.)", color='white')
        outline = getattr(cbar, 'outline', None)
        if outline is not None:
            try:
                outline.set_edgecolor('white')
            except Exception:
                pass
        cbar.ax.tick_params(color='white', labelcolor='white')

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, facecolor=fig.get_facecolor())

    plt.close(fig)
    logging.info("Saved PDF: %s", out_path)
    return out_path


# -------------------- CLI --------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Render wide 4×(2×pulses) octant grid with residuals.')
    p.add_argument('--subject', default="BraTS-MEN-00231-000", help='Subject ID')
    p.add_argument('--super_resolution_path',
                   default="/home/mpascual/research/datasets/meningiomas/BraTS/super_resolution",
                   help='Base path with resolution folders')
    p.add_argument('--coords', nargs=3, type=int, default=(65, 120, 135),
                   metavar=('K', 'I', 'J'), help='Octant intersection voxel (k axial, i coronal, j sagittal)')
    p.add_argument('--output_dir',
                   default='/home/mpascual/research/datasets/meningiomas/BraTS/super_resolution/results/metrics/example',
                   help='Output directory')
    p.add_argument('--log', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    if not os.path.exists(args.super_resolution_path):
        logging.error("Super resolution path does not exist: %s", args.super_resolution_path)
        return

    configure_matplotlib()

    spec = Spec(subject=args.subject, coords=tuple(args.coords))
    paths = Paths(super_resolution=args.super_resolution_path, output_dir=args.output_dir)

    grid, residuals = collect_grid_volumes(spec, paths)
    draw_grid_pdf(spec, paths, grid, residuals)


if __name__ == "__main__":
    main()
