from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path

try:
    # Prefer project import path if available
    from mgmGrowth.tasks.superresolution.visualization.octant import plot_octant  # type: ignore
except Exception:
    # Fallback to colocated module
    try:
        from octant import plot_octant  # type: ignore
    except Exception:
        plot_octant = None  # late check

try:
    import nibabel as nib  # type: ignore
except Exception:
    nib = None

from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, binary_fill_holes, label, generate_binary_structure  # type: ignore
from skimage.measure import marching_cubes  # noqa: F401  # kept for completeness


# -------------------------------
# Data classes and parameters
# -------------------------------

@dataclass
class Spacing:
    dx: float
    dy: float
    dz: float


@dataclass
class StageProducts:
    vol_hr: np.ndarray
    spacing_hr: Spacing
    sigma_mm: float
    vol_blur: np.ndarray
    f: int
    vol_lr: np.ndarray
    spacing_lr: Spacing
    mask_hr: np.ndarray
    mask_lr: np.ndarray


# -------------------------------
# Logging
# -------------------------------

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[Downsampling-v2] %(asctime)s | %(levelname)-7s | %(message)s",
    )


# -------------------------------
# Core maths
# -------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fwhm_to_sigma(fwhm_mm: float) -> float:
    return fwhm_mm / 2.355


def sigma_mm_to_vox(sigma_mm: float, dz_mm: float) -> float:
    return sigma_mm / dz_mm


def block_average_z(vol: np.ndarray, f: int, *, agg: str = "mean") -> np.ndarray:
    """Downsample along z by factor f using block aggregation (mean or any)."""
    Z = vol.shape[2]
    Z_trim = (Z // f) * f
    if Z_trim != Z:
        vol = vol[:, :, :Z_trim]
    X, Y, Zt = vol.shape
    vol = vol.reshape(X, Y, Zt // f, f)
    if agg == "mean":
        return vol.mean(axis=3)
    elif agg == "any":
        return (vol.any(axis=3)).astype(vol.dtype)
    else:
        raise ValueError("agg must be 'mean' or 'any'")


def gaussian_blur_z(vol: np.ndarray, sigma_vox_z: float) -> np.ndarray:
    return gaussian_filter(vol, sigma=(0.0, 0.0, float(sigma_vox_z)))


# -------------------------------
# Masking utilities
# -------------------------------

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

    nz = v[v > 0]
    if nz.size == 0:
        return np.zeros_like(v, dtype=bool)
    thr = max(1e-6, float(np.percentile(nz, 0.5)))  # 0.5th percentile is conservative
    fg = v > thr

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


def apply_background_black(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = vol.copy()
    out[~mask] = 0.0
    return out


# -------------------------------
# I/O
# -------------------------------

def load_volume(path: Optional[str]) -> tuple[np.ndarray, Spacing]:
    if path and nib is not None and os.path.exists(path):
        img = nib.load(path)
        data = np.asanyarray(img.dataobj).astype(np.float32)
        zooms = img.header.get_zooms()[:3]
        spacing = Spacing(float(zooms[0]), float(zooms[1]), float(zooms[2]))
        data = data - data.min()
        if data.max() > 0:
            data = data / data.max()
        logging.info(f"Loaded {path} with shape {data.shape} and spacing {spacing}")
        return data, spacing

    # Synthetic phantom
    spacing = Spacing(1.0, 1.0, 1.0)
    vol = make_phantom((160, 192, 128))
    return vol, spacing


def make_phantom(shape: tuple[int, int, int]) -> np.ndarray:
    X, Y, Z = shape
    x = np.linspace(-1, 1, X)
    y = np.linspace(-1, 1, Y)
    z = np.linspace(-1, 1, Z)
    Xg, Yg, Zg = np.meshgrid(x, y, z, indexing="ij")

    base = ((Xg / 0.9) ** 2 + (Yg / 1.0) ** 2 + (Zg / 0.8) ** 2) <= 1.0
    l1 = (((Xg + 0.25) / 0.35) ** 2 + (Yg / 0.6) ** 2 + (Zg / 0.6) ** 2) <= 1.0
    l2 = (((Xg - 0.2) / 0.4) ** 2 + ((Yg + 0.1) / 0.55) ** 2 + (Zg / 0.5) ** 2) <= 1.0

    vol = 0.6 * base.astype(float) + 0.25 * l1.astype(float) + 0.15 * l2.astype(float)
    vol = gaussian_filter(vol, sigma=(1.0, 1.0, 1.0))
    vol -= vol.min()
    if vol.max() > 0:
        vol /= vol.max()
    return vol


# -------------------------------
# Stage computations
# -------------------------------

def compute_products(vol_hr: np.ndarray, spacing_hr: Spacing, target_dz_mm: float) -> StageProducts:
    if target_dz_mm <= spacing_hr.dz:
        logging.warning("Target dz' ≤ native dz; diagram assumes dz' ≥ dz.")
    f_float = target_dz_mm / spacing_hr.dz
    f = int(round(f_float))
    if not math.isclose(f * spacing_hr.dz, target_dz_mm, abs_tol=1e-5):
        raise ValueError("Require |f*dz - dz'| ≤ 1e-5 mm for integer block averaging.")

    sigma_mm = fwhm_to_sigma(target_dz_mm)
    sigma_vox_z = sigma_mm_to_vox(sigma_mm, spacing_hr.dz)

    logging.info(f"Params: d=({spacing_hr.dx:.2f},{spacing_hr.dy:.2f},{spacing_hr.dz:.2f}) mm  "
                 f"dz'={target_dz_mm:.2f}  f={f}  σ={sigma_mm:.2f} mm ({sigma_vox_z:.2f} vox)")

    # Mask from HR
    mask_hr = compute_head_mask_from_hr(vol_hr)
    vol_hr_m = apply_background_black(vol_hr, mask_hr)

    # Blur along z
    vol_blur = gaussian_blur_z(vol_hr_m, sigma_vox_z)
    vol_blur_m = apply_background_black(vol_blur, mask_hr)

    # Downsample image and mask
    vol_lr = block_average_z(vol_blur_m, f=f, agg="mean")
    mask_lr = block_average_z(mask_hr.astype(bool), f=f, agg="any").astype(bool)
    vol_lr_m = apply_background_black(vol_lr, mask_lr)

    spacing_lr = Spacing(spacing_hr.dx, spacing_hr.dy, target_dz_mm)

    return StageProducts(
        vol_hr=vol_hr_m,
        spacing_hr=spacing_hr,
        sigma_mm=sigma_mm,
        vol_blur=vol_blur_m,
        f=f,
        vol_lr=vol_lr_m,
        spacing_lr=spacing_lr,
        mask_hr=mask_hr,
        mask_lr=mask_lr,
    )


# -------------------------------
# Plotting helpers
# -------------------------------

def central_indices(vol: np.ndarray) -> tuple[int, int, int]:
    x = vol.shape[0] // 2
    y = vol.shape[1] // 2
    z = vol.shape[2] // 2
    # octant.plot expects (k_axial, i_coronal, j_sagittal) → (z, x, y)
    return (z, x, y)


# -------------------------------
# Octant → PNG (no extra windows)
# -------------------------------

def _with_global_minmax_for_plot_octant(
    vol: np.ndarray,
    k: int, i: int, j: int,
    vmin: float,
    vmax: float
) -> np.ndarray:
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
    segmentation: Optional[np.ndarray] = None,
    cmap='gray',
    seg_alpha: float = 0.30,
    only_line: bool = True,
    enforce_minmax: Optional[Tuple[float, float]] = None,
    figsize: Tuple[int, int] = (4, 4),
) -> np.ndarray:
    if plot_octant is None:
        raise RuntimeError("octant.plot_octant is not available.")
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
        if img.ndim == 3 and img.shape[-1] == 4:
            rgb = img[..., :3]
            alpha = img[..., 3:4]
            img = rgb * alpha
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return img


def plot_S0(ax, vol: np.ndarray, spacing: Spacing) -> None:
    fig = ax.get_figure()
    bbox = ax.get_position()
    fig.delaxes(ax)  # remove placeholder axes
    slice_idx = central_indices(vol)
    # re-create a 2D axes at the same position and draw PNG
    ax2 = fig.add_axes(bbox)
    img = render_octant_png(
        vol, slice_idx,
        segmentation=None, cmap='gray', seg_alpha=0.0,
        only_line=True, enforce_minmax=None, figsize=(4, 4),
    )
    ax2.imshow(img)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f"S0 · HR input\nd=({spacing.dx:.2f},{spacing.dy:.2f},{spacing.dz:.2f}) mm", pad=10)


def plot_stack(ax, vol: np.ndarray, spacing: Spacing, n_planes: int = 10, title: str = "") -> None:
    Z = vol.shape[2]
    if Z <= 2:
        indices = np.arange(Z)
    else:
        # Skip first and last slice
        indices = np.linspace(1, Z - 2, n_planes).astype(int)
        indices = np.unique(indices)

    X = np.arange(vol.shape[0]) * spacing.dx
    Y = np.arange(vol.shape[1]) * spacing.dy
    Xg, Yg = np.meshgrid(X, Y, indexing="ij")
    norm = matplotlib.colors.Normalize(vmin=vol.min(), vmax=vol.max())
    for k, iz in enumerate(indices):
        Zg = np.full_like(Xg, iz * spacing.dz + k * 0.4 * spacing.dz)
        slice2d = vol[:, :, iz]
        facecolors = plt.cm.gray(norm(slice2d))
        ax.plot_surface(Xg, Yg, Zg, rstride=1, cstride=1, facecolors=facecolors,
                        linewidth=0, antialiased=False, shade=False)

    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.view_init(elev=8, azim=-60)  # lower camera
    ax.grid(False)


def plot_S1(ax, vol: np.ndarray, spacing: Spacing) -> None:
    plot_stack(ax, vol, spacing, n_planes=10, title="S1 · Thin-slice stack\n(native, ends skipped)")


def plot_kernel(ax, sigma_mm: float, dz_mm: float) -> None:
    sigma_vox = sigma_mm / dz_mm
    extent = int(4 * sigma_vox) + 1
    z = np.arange(-extent, extent + 1) * dz_mm
    kernel = np.exp(-(z ** 2) / (2 * sigma_mm ** 2))
    kernel /= kernel.sum() * dz_mm
    ax.plot(z, kernel)
    ax.set_title(f"S2 · z-Gaussian (1D)\nσ={sigma_mm:.2f} mm")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("G(z) [a.u.]")


def plot_fft_compare(ax, vol_hr: np.ndarray, vol_blur: np.ndarray, dz_mm: float) -> None:
    s_hr = vol_hr.mean(axis=(0, 1))
    s_bl = vol_blur.mean(axis=(0, 1))
    s_hr = s_hr - s_hr.mean()
    s_bl = s_bl - s_bl.mean()

    F_hr = np.abs(np.fft.rfft(s_hr))
    F_bl = np.abs(np.fft.rfft(s_bl))
    f = np.fft.rfftfreq(s_hr.size, d=dz_mm)

    F_hr /= (F_hr.max() + 1e-12)
    F_bl /= (F_hr.max() + 1e-12)

    ax.plot(f, F_hr, label="HR")
    ax.plot(f, F_bl, label="Blurred")
    ax.set_xlim(0, f.max())
    ax.set_xlabel("frequency [cycles/mm]")
    ax.set_ylabel("|F(z)| (norm.)")
    ax.set_title("S2 · |F(z)| before vs after blur")
    ax.legend(loc="upper right", frameon=False)


def plot_S2(ax_kernel, ax_fft, ax_stack3d, vol_blur: np.ndarray, spacing: Spacing, sigma_mm: float, vol_hr: np.ndarray) -> None:
    plot_kernel(ax_kernel, sigma_mm, spacing.dz)
    plot_fft_compare(ax_fft, vol_hr, vol_blur, spacing.dz)
    plot_stack(ax_stack3d, vol_blur, spacing, n_planes=10, title="S2 · Blurred stack")


def plot_S3(ax, vol_blur: np.ndarray, spacing: Spacing, f: int) -> None:
    Z = vol_blur.shape[2]
    # Choose two central groups around the mid-plane
    mid_group = max(0, (Z // f) // 2 - 1)
    groups = [mid_group, min(mid_group + 1, max(0, Z // f - 1))]
    groups = sorted(set(groups))

    X = np.arange(vol_blur.shape[0]) * spacing.dx
    Y = np.arange(vol_blur.shape[1]) * spacing.dy
    Xg, Yg = np.meshgrid(X, Y, indexing="ij")
    norm = matplotlib.colors.Normalize(vmin=vol_blur.min(), vmax=vol_blur.max())

    z_offset = 0.0
    for g in groups:
        start = g * f
        stop = min(start + f, Z)
        # plot f semi-transparent planes
        for k in range(start, stop):
            Zg = np.full_like(Xg, k * spacing.dz + z_offset)
            slice2d = vol_blur[:, :, k]
            facecolors = plt.cm.gray(norm(slice2d))
            ax.plot_surface(Xg, Yg, Zg, rstride=1, cstride=1, facecolors=facecolors,
                            linewidth=0, antialiased=False, shade=False, alpha=0.6)

        # averaged thick slice at group center
        thick = vol_blur[:, :, start:stop].mean(axis=2)
        z_center = ((start + stop - 1) / 2.0) * spacing.dz + z_offset
        Zg_center = np.full_like(Xg, z_center)
        facecolors = plt.cm.gray(norm(thick))
        ax.plot_surface(Xg, Yg, Zg_center, rstride=1, cstride=1, facecolors=facecolors,
                        linewidth=0, antialiased=False, shade=False, alpha=1.0)

        # left-side “brace” and label
        z_top = (start) * spacing.dz + z_offset
        z_bot = (stop - 1) * spacing.dz + z_offset
        ax.plot([X[0], X[0]], [Y[0], Y[0]], [z_top, z_bot], lw=2)
        ax.text(X[0], Y[0], z_center, "f×dz", fontsize=10, ha="left", va="center")

        z_offset += (stop - start) * spacing.dz + 0.6 * spacing.dz

    ax.set_title("S3 · Block-average by f (central groups)")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")
    ax.view_init(elev=8, azim=-60)
    ax.grid(False)


def plot_S4(ax, vol_lr: np.ndarray, spacing_lr: Spacing) -> None:
    fig = ax.get_figure()
    bbox = ax.get_position()
    fig.delaxes(ax)
    slice_idx = central_indices(vol_lr)
    ax2 = fig.add_axes(bbox)
    img = render_octant_png(
        vol_lr, slice_idx,
        segmentation=None, cmap='gray', seg_alpha=0.0,
        only_line=True, enforce_minmax=None, figsize=(4, 4),
    )
    ax2.imshow(img)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f"S4 · LR output\nd'=({spacing_lr.dx:.2f},{spacing_lr.dy:.2f},{spacing_lr.dz:.2f}) mm", pad=10)


# -------------------------------
# Figure assembly
# -------------------------------

def save_stage_images(outdir: str, prods: StageProducts) -> None:
    # S0
    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    plot_S0(ax, prods.vol_hr, prods.spacing_hr)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "S0_HR_input.png"), dpi=250)
    plt.close(fig)

    # S1
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    plot_S1(ax, prods.vol_hr, prods.spacing_hr)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "S1_Thin_slice_stack.png"), dpi=250)
    plt.close(fig)

    # S2 (kernel + FFT + blurred stack)
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.2, 1.8])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2], projection="3d")
    plot_S2(ax0, ax1, ax2, prods.vol_blur, prods.spacing_hr, prods.sigma_mm, prods.vol_hr)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "S2_Gaussian_FFT_and_blurred.png"), dpi=250)
    plt.close(fig)

    # S3
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    plot_S3(ax, prods.vol_blur, prods.spacing_hr, prods.f)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "S3_Block_average.png"), dpi=250)
    plt.close(fig)

    # S4
    fig = plt.figure(figsize=(6.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    plot_S4(ax, prods.vol_lr, prods.spacing_lr)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "S4_LR_output.png"), dpi=250)
    plt.close(fig)


def save_panel(outdir: str, prods: StageProducts) -> None:
    fig = plt.figure(figsize=(22, 5))
    gs = gridspec.GridSpec(1, 5, width_ratios=[1.1, 1.1, 1.4, 1.2, 1.1])

    ax0 = fig.add_subplot(gs[0, 0], projection="3d")
    plot_S0(ax0, prods.vol_hr, prods.spacing_hr)

    ax1 = fig.add_subplot(gs[0, 1], projection="3d")
    plot_S1(ax1, prods.vol_hr, prods.spacing_hr)

    ax2a = fig.add_subplot(gs[0, 2])
    ax2b = fig.add_subplot(gs[0, 3])
    ax2c = fig.add_subplot(gs[0, 4], projection="3d")
    plot_S2(ax2a, ax2b, ax2c, prods.vol_blur, prods.spacing_hr, prods.sigma_mm, prods.vol_hr)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "Panel_S0_to_S4.png"), dpi=280)
    plt.close(fig)


# -------------------------------
# CLI
# -------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate stage-like figures for HR→LR downsampling (v2).")
    ap.add_argument("--outdir", type=str, required=True, help="Output directory.")
    ap.add_argument("--input_nii", type=str, default=None, help="Optional NIfTI path.")
    ap.add_argument("--dx", type=float, default=None, help="Override dx [mm] for phantom.")
    ap.add_argument("--dy", type=float, default=None, help="Override dy [mm] for phantom.")
    ap.add_argument("--dz", type=float, default=None, help="Override dz [mm] for phantom.")
    ap.add_argument("--target_dz", type=float, required=True, help="Target through-plane spacing dz' [mm].")
    ap.add_argument("--loglevel", type=str, default="INFO", help="Logging level.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.loglevel)
    ensure_dir(args.outdir)

    vol, spacing = load_volume(args.input_nii)
    if args.input_nii is None:
        if args.dx: spacing.dx = float(args.dx)
        if args.dy: spacing.dy = float(args.dy)
        if args.dz: spacing.dz = float(args.dz)

    prods = compute_products(vol, spacing, target_dz_mm=float(args.target_dz))
    save_stage_images(args.outdir, prods)
    save_panel(args.outdir, prods)

    meta = {
        "native_spacing_mm": {"dx": prods.spacing_hr.dx, "dy": prods.spacing_hr.dy, "dz": prods.spacing_hr.dz},
        "target_dz_mm": prods.spacing_lr.dz,
        "factor_f": prods.f,
        "sigma_mm": prods.sigma_mm,
        "vol_hr_shape": list(prods.vol_hr.shape),
        "vol_lr_shape": list(prods.vol_lr.shape),
    }
    with open(os.path.join(args.outdir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logging.info(f"Wrote images and metadata to {args.outdir}")


if __name__ == "__main__":
    main()