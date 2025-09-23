#!/usr/bin/env python3
"""
WIDE 4x(2×pulses) sagittal grid with residual maps and single PDF output.

Layout per pulse (two columns):
- Row 'hr':   left = HR slice;             right = blank
- Rows lower: left = residual (NN − HR);   right = NN-upsampled slice

Top headers are centered over each 2-column pulse group.
Row labels use lowercase 'mm'. No suptitle.

Residuals use a diverging colormap with a global symmetric window and a
horizontal colorbar at the bottom labeled "Residual ΔI = NN − HR (a.u.)".

Optional orientation labels from NIfTI affine: --orientation_labels
"""

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


# -------------------- I/O and helpers --------------------

def load_nifti_file(filepath: str) -> Optional[Any]:
    """Load NIfTI or return None."""
    try:
        return nib.load(filepath)
    except Exception as e:
        logging.error("Error loading %s: %s", filepath, e)
        return None


def get_sagittal_slice(data: np.ndarray, slice_idx: int) -> Optional[np.ndarray]:
    """Return sagittal slice from 3D data or None."""
    if data is None or data.ndim < 3:
        return None
    if not (0 <= slice_idx < data.shape[0]):
        return None
    return data[slice_idx, :, :]


def replicate_to_isotropic(vol2d: np.ndarray, zooms_yz: Tuple[float, float], target: float = 1.0) -> np.ndarray:
    """Copy-only NN upsampling for 2D sagittal slice so in-plane spacing appears isotropic."""
    reps = [int(round(z / target)) for z in zooms_yz]
    if not np.allclose(np.array(reps, float) * target, np.array(zooms_yz), atol=1e-3):
        raise ValueError(f"{zooms_yz} are not integer multiples of {target} mm")
    out = vol2d
    for ax, k in enumerate(reps):
        if k > 1:
            out = np.repeat(out, k, axis=ax)
    return out


def prepare_sagittal_display(nii: Any, slice_idx: int) -> Optional[np.ndarray]:
    """Slice → NN-replicate to ~1 mm in-plane (Y,Z) → rotate/flip for display with origin='lower'."""
    data = nii.get_fdata()
    sag = get_sagittal_slice(data, slice_idx)
    if sag is None:
        return None
    zooms = nii.header.get_zooms()[:3]
    sag_iso = replicate_to_isotropic(sag, (zooms[1], zooms[2]), target=1.0)
    disp = np.fliplr(np.rot90(sag_iso, k=3))  # rows→+Z, cols→+Y
    return disp


def robust_window(img: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> Tuple[float, float]:
    """Percentile window for magnitude images."""
    m = np.isfinite(img)
    lo = np.percentile(img[m], q_low) if m.any() else float(np.nanmin(img))
    hi = np.percentile(img[m], q_high) if m.any() else float(np.nanmax(img))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def center_crop_to_match(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center-crop a and b to a common shape = min(a.shape, b.shape)."""
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])

    def crop(x):
        dh = (x.shape[0] - h) // 2
        dw = (x.shape[1] - w) // 2
        return x[dh:dh+h, dw:dw+w]

    return crop(a), crop(b)


def opposite(code: str) -> str:
    """Return anatomical opposite label."""
    pairs = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'S': 'I', 'I': 'S'}
    return pairs.get(code.upper(), code)


def overlay_orientation_labels(ax: plt.Axes, axcodes: Tuple[str, str, str]) -> None:
    """
    L/R and S/I labels consistent with the display transform used here.

    After rotate(270)+fliplr:
      horizontal → +Y ; vertical → +Z
    """
    y_plus = axcodes[1].upper()
    z_plus = axcodes[2].upper()
    ax.text(0.02, 0.50, opposite(y_plus), color='white', ha='left',  va='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.98, 0.50, y_plus,          color='white', ha='right', va='center', fontsize=12, transform=ax.transAxes)
    ax.text(0.50, 0.98, z_plus,          color='white', ha='center', va='top',    fontsize=12, transform=ax.transAxes)
    ax.text(0.50, 0.02, opposite(z_plus),color='white', ha='center', va='bottom', fontsize=12, transform=ax.transAxes)


# -------------------- orchestration --------------------

@dataclass(frozen=True)
class Paths:
    super_resolution: str
    output_dir: str


@dataclass(frozen=True)
class Spec:
    subject: str
    slice_idx: int
    resolutions: Tuple[str, ...] = ('hr', '3mm', '5mm', '7mm')
    pulses:     Tuple[str, ...] = ('t1c', 't2w', 't2f', 't1n')  # adjust to your set

    @property
    def n_rows(self) -> int:
        return len(self.resolutions)

    @property
    def n_cols(self) -> int:
        return 2 * len(self.pulses)  # two columns per pulse


RESOLUTION_DIR = {
    'hr':  'subset',
    '3mm': 'low_res/3mm',
    '5mm': 'low_res/5mm',
    '7mm': 'low_res/7mm',
}


def build_file_path(base: str, resolution: str, subject: str, pulse: str) -> str:
    return os.path.join(base, RESOLUTION_DIR[resolution], subject, f"{subject}-{pulse}.nii.gz")


@dataclass
class Cell:
    img: Optional[np.ndarray]
    axcodes: Optional[Tuple[str, str, str]]


def collect_hr_slices(spec: Spec, paths: Paths) -> Dict[str, Cell]:
    """Load HR per pulse once."""
    out: Dict[str, Cell] = {}
    for pulse in spec.pulses:
        fp = build_file_path(paths.super_resolution, 'hr', spec.subject, pulse)
        nii = load_nifti_file(fp)
        if nii is None:
            out[pulse] = Cell(None, None)
            continue
        img = prepare_sagittal_display(nii, spec.slice_idx)
        axcodes = nib.orientations.aff2axcodes(nii.affine)  # type: ignore[attr-defined]
        out[pulse] = Cell(img, axcodes)
    return out


def collect_grid(spec: Spec, paths: Paths, hr_by_pulse: Dict[str, Cell]):
    """
    Prepare plotting payload:
    - imgs[r][c] holds image or None
    - kind[r][c] in {'hr','residual','nn','blank'}
    - residuals list for global color scaling
    """
    imgs: List[List[Optional[np.ndarray]]] = [[None]*(spec.n_cols) for _ in range(spec.n_rows)]
    kinds: List[List[str]] = [[ 'blank']*(spec.n_cols) for _ in range(spec.n_rows)]
    axcodes_map: Dict[Tuple[int,int], Optional[Tuple[str,str,str]]] = {}

    residuals: List[np.ndarray] = []

    for p_idx, pulse in enumerate(spec.pulses):
        left_c  = 2*p_idx
        right_c = 2*p_idx + 1

        # Row 0: HR centered between columns (rendered later) -> mark left as holder, right blank
        hr_cell = hr_by_pulse.get(pulse, Cell(None,None))
        if hr_cell.img is not None:
            imgs[0][left_c] = hr_cell.img
            kinds[0][left_c] = 'hr_center'  # special holder; drawn between columns later
            axcodes_map[(0,left_c)] = hr_cell.axcodes
        kinds[0][right_c] = 'blank'
        axcodes_map[(0,right_c)] = None

        # Rows 1..: NN slice left; residual right
        for r in range(1, spec.n_rows):
            res_label = spec.resolutions[r]
            fp = build_file_path(paths.super_resolution, res_label, spec.subject, pulse)
            nii_lr = load_nifti_file(fp)
            if nii_lr is None or hr_cell.img is None:
                kinds[r][left_c]  = 'blank'
                kinds[r][right_c] = 'blank'
                axcodes_map[(r,left_c)]  = None
                axcodes_map[(r,right_c)] = None
                continue

            lr_img = prepare_sagittal_display(nii_lr, spec.slice_idx)
            if lr_img is None:
                kinds[r][left_c]  = 'blank'
                kinds[r][right_c] = 'blank'
                axcodes_map[(r,left_c)]  = None
                axcodes_map[(r,right_c)] = None
                continue

            # Center-crop both to common shape (safety)
            hr_c, lr_c = center_crop_to_match(hr_cell.img, lr_img)

            # Residual = NN − HR (signed)
            res_img = lr_c.astype(np.float32) - hr_c.astype(np.float32)
            residuals.append(res_img)

            # Place NN on left
            imgs[r][left_c]  = lr_c
            kinds[r][left_c] = 'nn'
            axcodes_map[(r,left_c)] = nib.orientations.aff2axcodes(nii_lr.affine)  # type: ignore[attr-defined]

            # Residual on right
            imgs[r][right_c]  = res_img
            kinds[r][right_c] = 'residual'
            axcodes_map[(r,right_c)] = None  # orientation overlay not needed on residuals

    return imgs, kinds, axcodes_map, residuals


def draw_grid_pdf(
    spec: Spec,
    paths: Paths,
    imgs: List[List[Optional[np.ndarray]]],
    kinds: List[List[str]],
    axcodes_map: Dict[Tuple[int,int], Optional[Tuple[str,str,str]]],
    residuals: List[np.ndarray],
    orientation_labels: bool = False,
) -> str:
    """Render and save a single-page PDF."""
    os.makedirs(paths.output_dir, exist_ok=True)
    out_name = f"{spec.subject}_sag{spec.slice_idx}_wide_pairs.pdf"
    out_path = os.path.join(paths.output_dir, out_name)

    # Landscape A4 width cap
    A4_W = 11.69
    fig_w = min(11.5, A4_W)
    fig_h = 6.9  # wider than tall

    fig, axes = plt.subplots(
        nrows=spec.n_rows,
        ncols=spec.n_cols,
        figsize=(fig_w, fig_h),
        sharex=False,
        sharey=False,
        squeeze=True,
    )
    fig.patch.set_facecolor('black')

    # Compute global symmetric window for residuals (robust 99th of |.|, then symmetric)
    if residuals:
        q = [np.percentile(np.abs(r[np.isfinite(r)]), 99.0) for r in residuals]
        rmax = float(max(q)) if q else 1.0
    else:
        rmax = 1.0
    res_norm = Normalize(vmin=-rmax, vmax=rmax)
    # Build a residual colormap with pure black at the center (value 0)
    try:
        base_cmap = plt.get_cmap('berlin_r')
    except Exception:
        base_cmap = plt.get_cmap('seismic')
    colors = base_cmap(np.linspace(0, 1, 256))
    # Force exact midpoint to black so 0 residual maps to black
    colors[127] = [0, 0, 0, 1]
    colors[128] = [0, 0, 0, 1]
    res_cmap = ListedColormap(colors, name='black_center_div')

    # Top headers centered over each pair of columns
    top_y = 0.975
    for p_idx, pulse in enumerate(spec.pulses):
        axL = axes[0, 2*p_idx]
        axR = axes[0, 2*p_idx + 1]
        boxL = axL.get_position()
        boxR = axR.get_position()
        x_center = 0.5 * (boxL.x0 + boxL.x1 + boxR.x0 + boxR.x1) / 2.0 
        if pulse.lower() == 't1c':
            x_center -= 0.04
        elif pulse.lower() == 't2f':
            x_center += 0.042
        elif pulse.lower() == 't1n':
            x_center += 0.08
        fig.text(x_center, top_y, pulse.upper(), color='white', ha='center', va='top')

    # Plot panels
    for r in range(spec.n_rows):
        for c in range(spec.n_cols):
            ax = axes[r, c]
            ax.set_facecolor('black')
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

            im = imgs[r][c]
            kind = kinds[r][c]

            if im is None or kind in ('blank', 'hr_center'):
                continue

            if kind == 'residual':
                ax.imshow(im, cmap=res_cmap, origin='lower', interpolation='nearest', norm=res_norm)
            else:
                vmin, vmax = robust_window(im, 1.0, 99.0)
                ax.imshow(im, cmap='gray', origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
                if orientation_labels and axcodes_map.get((r, c)) is not None:
                    overlay_orientation_labels(ax, axcodes_map[(r, c)])  # type: ignore[arg-type]

    # Row labels on the left, vertically centered per row; 'mm' lowercase
    left_x = 0.03
    for r, res in enumerate(spec.resolutions):
        label = 'High-Res.' if res == 'hr' else res  # ensure 'mm' is lowercase
        box = axes[r, 0].get_position()
        y_mid = (box.y0 + box.y1) / 2.0
        fig.text(left_x, y_mid, label, color='white', ha='left', va='center')

    # Compact spacing to bring slices closer
    # Increase horizontal space between columns to make room for centered HR
    plt.subplots_adjust(left=0.06, right=0.995, top=0.955, bottom=0.08, wspace=0.08, hspace=0.003)

    # Insert centered HR images between column pairs on the top row (just below headers)
    for p_idx, pulse in enumerate(spec.pulses):
        im_hr = imgs[0][2*p_idx]
        if im_hr is None:
            continue
        axL = axes[0, 2*p_idx]
        axR = axes[0, 2*p_idx + 1]
        # Clear any artists on the original top-row axes to avoid overlap
        axL.cla(); axR.cla()
        axL.set_facecolor('black'); axR.set_facecolor('black')
        axL.set_xticks([]); axL.set_yticks([])
        axR.set_xticks([]); axR.set_yticks([])
        for s in axL.spines.values():
            s.set_visible(False)
        for s in axR.spines.values():
            s.set_visible(False)
        boxL = axL.get_position()
        boxR = axR.get_position()
        # Make HR the same size as a single column and center it between the two columns
        hr_w = boxL.width
        hr_h = boxL.height
        # center x at the midpoint of the gap between left and right columns
        gap_center_x = 0.5 * (boxL.x1 + boxR.x0)
        hr_x = gap_center_x - hr_w/2
        hr_y = boxL.y0
        hr_ax = fig.add_axes((hr_x, hr_y, hr_w, hr_h))
        hr_ax.set_facecolor('black')
        hr_ax.set_xticks([])
        hr_ax.set_yticks([])
        for s in hr_ax.spines.values():
            s.set_visible(False)
        vmin, vmax = robust_window(im_hr, 1.0, 99.0)
        hr_ax.imshow(im_hr, cmap='gray', origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)

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

@dataclass(frozen=True)
class PathsSpec:
    super_resolution_path: str
    output_dir: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Render wide 4×(2×pulses) sagittal grid with residuals.')
    p.add_argument('--subject', default="BraTS-MEN-00231-000", help='Subject ID')
    p.add_argument('--super_resolution_path',
                   default="/home/mpascual/research/datasets/meningiomas/BraTS/super_resolution",
                   help='Base path with resolution folders')
    p.add_argument('--slice_number', type=int, default=115, help='Sagittal slice index')
    p.add_argument('--output_dir',
                   default='/home/mpascual/research/datasets/meningiomas/BraTS/super_resolution/results/metrics/example',
                   help='Output directory')
    p.add_argument('--orientation_labels', action='store_true',
                   help='Overlay L/R and S/I labels from NIfTI affine on image panels')
    p.add_argument('--log', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    if not os.path.exists(args.super_resolution_path):
        logging.error("Super resolution path does not exist: %s", args.super_resolution_path)
        return
    if args.slice_number < 0:
        logging.error("Slice number must be non-negative.")
        return

    configure_matplotlib()

    spec = Spec(subject=args.subject, slice_idx=args.slice_number)
    paths = Paths(super_resolution=args.super_resolution_path, output_dir=args.output_dir)

    hr_by_pulse = collect_hr_slices(spec, paths)
    imgs, kinds, axcodes_map, residuals = collect_grid(spec, paths, hr_by_pulse)

    draw_grid_pdf(spec, paths, imgs, kinds, axcodes_map, residuals, orientation_labels=args.orientation_labels)


if __name__ == "__main__":
    main()
