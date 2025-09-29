from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Tuple, Any, Callable, cast, TYPE_CHECKING

import numpy as np
import matplotlib
# Non-interactive backend for headless, faster rendering
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
# add near the other imports
from matplotlib.patches import Patch
from scipy.ndimage import zoom
# prefer local octant_surface, then package path
try:
    from mgmGrowth.tasks.superresolution.visualization.octant_surface import plot_cutaway_octant, CutawayConfig  # type: ignore
except Exception:
    from mgmGrowth.tasks.superresolution.visualization.octant_surface import (  # type: ignore
        plot_cutaway_octant, CutawayConfig
    )
plot_octant_fn: Optional[Callable[..., plt.Figure]] = None
try:
    # Prefer project import path if available
    from mgmGrowth.tasks.superresolution.visualization.octant import plot_octant as _plot_octant  # type: ignore
    plot_octant_fn = _plot_octant
except Exception:
    # Fallback to colocated module
    try:
        from octant import plot_octant as _plot_octant  # type: ignore
        plot_octant_fn = _plot_octant
    except Exception:
        plot_octant_fn = None  # late check

try:
    import nibabel as _nib  # type: ignore
    nib: Any = _nib
except Exception:
    nib = cast(Any, None)

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



def _strip_axes(ax) -> None:
    """Remove axes, ticks, and titles for a clean IEEE-style figure."""
    ax.set_title("")
    if hasattr(ax, "set_axis_off"):
        ax.set_axis_off()
    else:
        ax.axis("off")


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
        try:
            ax_pre = getattr(nib.orientations, "aff2axcodes")(img.affine)
        except Exception:
            ax_pre = None
        # Reorient to canonical RAS+
        try:
            img = nib.as_closest_canonical(img)
        except Exception:
            pass
        try:
            ax_post = getattr(nib.orientations, "aff2axcodes")(img.affine)
        except Exception:
            ax_post = None
        if ax_pre is not None and ax_post is not None and ax_pre != ax_post:
            logging.info(f"Reoriented input from {ax_pre} to canonical {ax_post} (RAS+)")

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
    """Return center coordinates as (k_axial=z, i_coronal=x, j_sagittal=y)."""
    x = vol.shape[0] // 2
    y = vol.shape[1] // 2
    z = vol.shape[2] // 2
    return (z, x, y)


def resolve_coords(
    vol: np.ndarray,
    k_axial: Optional[int] = None,
    i_coronal: Optional[int] = None,
    j_sagittal: Optional[int] = None,
) -> tuple[int, int, int]:
    """Resolve requested coords (allowing negative indices) with clamping.

    If any of k/i/j is None, the center index along that axis is used.
    """
    nx, ny, nz = vol.shape
    kc, ic, jc = central_indices(vol)

    def _norm(idx: Optional[int], dim: int, center: int) -> int:
        if idx is None:
            return center
        if idx < 0:
            idx = dim + idx
        return max(0, min(idx, dim - 1))

    k = _norm(k_axial, nz, kc)
    i = _norm(i_coronal, nx, ic)
    j = _norm(j_sagittal, ny, jc)
    return (k, i, j)


# -------------------------------
# NIfTI saving utilities
# -------------------------------

def _affine_from_spacing(sp: Spacing) -> np.ndarray:
    """Build a simple RAS-like affine with voxel spacing on the diagonal."""
    aff = np.eye(4, dtype=float)
    aff[0, 0] = float(sp.dx)
    aff[1, 1] = float(sp.dy)
    aff[2, 2] = float(sp.dz)
    return aff


def save_nifti(volume: np.ndarray, spacing: Spacing, out_path: str, *, dtype: Optional[np.dtype] = None) -> str:
    """Save 3D volume to NIfTI (.nii.gz) if nibabel is available; otherwise save .npy with a warning.

    Returns the path actually written.
    """
    arr = volume.astype(dtype) if dtype is not None else volume
    # Ensure parent dir exists
    ensure_dir(os.path.dirname(out_path) or ".")
    if nib is None:
        # Fallback to .npy
        alt = os.path.splitext(out_path)[0] + ".npy"
        logging.warning(f"Nibabel not available; saving array to {alt} instead of NIfTI.")
        np.save(alt, arr)
        return alt

    aff = _affine_from_spacing(spacing)
    img = nib.Nifti1Image(arr, aff)
    # Try to set header zooms, and sform/qform for better compatibility
    try:
        img.header.set_zooms((float(spacing.dx), float(spacing.dy), float(spacing.dz)))
        img.header.set_xyzt_units('mm')
    except Exception:
        pass
    try:
        # code=1 (scanner anat) is a common default; RAS implied by the affine
        img.set_sform(aff, code=1)
        img.set_qform(aff, code=1)
    except Exception:
        pass
    if not out_path.endswith(".nii.gz"):
        out_path = os.path.splitext(out_path)[0] + ".nii.gz"
    nib.save(img, out_path)
    return out_path


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


def _fig_to_rgb_image(fig: plt.Figure) -> np.ndarray:
    """Convert a Matplotlib figure to an RGB image (premultiplied alpha)."""
    fig.canvas.draw()
    # Prefer buffer_rgba when available (Agg), fallback to ARGB string
    try:
        canvas_any = cast(Any, fig.canvas)
        buf = np.asarray(canvas_any.buffer_rgba())  # (H,W,4) uint8
        rgba = buf
    except Exception:
        canvas_any = cast(Any, fig.canvas)
        w, h = canvas_any.get_width_height()
        argb = np.frombuffer(canvas_any.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        rgba = np.empty_like(argb)
        rgba[..., 0] = argb[..., 1]  # R
        rgba[..., 1] = argb[..., 2]  # G
        rgba[..., 2] = argb[..., 3]  # B
        rgba[..., 3] = argb[..., 0]  # A
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return rgb * a


def render_cutaway_png_from_array(
    volume: np.ndarray,
    coords: Tuple[int, int, int],
    save_png: Optional[str] = None,
    figsize: Tuple[float, float] = (4.5, 4.0),
) -> np.ndarray:
    """
    Render the cut-away octant surface as an RGBA image using nearest-neighbour
    faces (one quad per voxel). Uses your plot_cutaway_octant(). Axes/titles off.

    Parameters
    ----------
    volume : (nx,ny,nz) float array
    coords : (k,i,j) indices
    save_png : optional path to also persist the PNG
    figsize : figure size in inches

    Returns
    -------
    (H,W,3) float image for imshow
    """
    k, i, j = coords
    cfg = CutawayConfig(mesh_alpha=1.0, slice_alpha=0.95, cmap="gray")
    
    fig = plot_cutaway_octant(volume, k, i, j, cfg=cfg, use_mask_extent=False)  # axes already off
    fig.set_size_inches(*figsize)
    img = _fig_to_rgb_image(fig)
    if save_png:
        plt.imsave(save_png, img)
    plt.close(fig)
    return img


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
    if plot_octant_fn is None:
        raise RuntimeError("octant.plot_octant is not available.")
    k, i, j = coords
    vol = volume if enforce_minmax is None else _with_global_minmax_for_plot_octant(volume, k, i, j, enforce_minmax[0], enforce_minmax[1])
    fig = plot_octant_fn(
        vol, coords, cmap=cmap, alpha=0.95,
        segmentation=segmentation, seg_alpha=seg_alpha,
        only_line=only_line, figsize=figsize, save=None
    )
    img = _fig_to_rgb_image(fig)
    plt.close(fig)
    return img


def plot_stack(ax, vol: np.ndarray, spacing: Spacing, mask: Optional[np.ndarray] = None,
               n_planes: int = 10, vis_stride: int = 1) -> None:
    """
    Plot evenly spaced axial slices as semi-transparent surfaces with background removed.

    Parameters
    ----------
    vol : (X,Y,Z) float
    spacing : voxel spacing in mm
    mask : optional boolean head mask; outside gets alpha=0
    n_planes : number of axial slices to display
    vis_stride : subsampling stride in X/Y to speed 3D plotting
    """
    Z = vol.shape[2]
    idx = np.linspace(1, max(Z - 2, 1), n_planes).astype(int)
    idx = np.unique(np.clip(idx, 0, Z - 1))

    X = np.arange(vol.shape[0]) * spacing.dx
    Y = np.arange(vol.shape[1]) * spacing.dy
    Xg, Yg = np.meshgrid(X, Y, indexing="ij")
    if vis_stride > 1:
        Xg = Xg[::vis_stride, ::vis_stride]
        Yg = Yg[::vis_stride, ::vis_stride]
    norm = matplotlib.colors.Normalize(vmin=float(vol.min()), vmax=float(vol.max()))
    cmap = plt.get_cmap("gray")

    for k in idx:
        Zg = np.full_like(Xg, k * spacing.dz)
        sl = vol[:, :, k]
        if vis_stride > 1:
            sl = sl[::vis_stride, ::vis_stride]
        fc = cmap(norm(sl))               # (X,Y,4) RGBA
        if mask is not None:
            ma = mask[:, :, k].astype(float)     # 1 inside, 0 outside
            if vis_stride > 1:
                ma = ma[::vis_stride, ::vis_stride]
            fc[..., -1] = ma                     # set alpha
        ax.plot_surface(Xg, Yg, Zg, rstride=1, cstride=1,
                        facecolors=fc, linewidth=0, antialiased=False,
                        shade=False)
    ax.view_init(elev=8, azim=-60)
    ax.grid(False)
    _strip_axes(ax)


def save_kernel_figure(outdir: str, sigma_mm: float, dz_mm: float) -> str:
    """
    Save a standalone Gaussian kernel figure (filled, no axes). Returns path.
    """
    sigma_vox = sigma_mm / dz_mm
    half = int(4 * sigma_vox) + 1
    z = np.arange(-half, half + 1) * dz_mm
    G = np.exp(-(z ** 2) / (2 * sigma_mm ** 2))
    G = G / (G.sum() * dz_mm)  # discrete normalization

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.fill_between(z, 0.0, G, linewidth=0, alpha=1.0)
    ax.plot(z, G, linewidth=1.8)
    #ax.legend([f"σ={sigma_mm:.2f} mm • FWHM={2.355*sigma_mm:.2f} mm • Δz={dz_mm:.2f} mm • ΣG·Δz=1"],
              #loc="upper right", frameon=False)
    _strip_axes(ax)

    ensure_dir(os.path.join(outdir, "tmp"))
    pth = os.path.join(outdir, "S2a_Gaussian_kernel.png")
    fig.tight_layout()
    fig.savefig(pth, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pth


def plot_S2_cutaway(ax, vol_blur: np.ndarray, coords: Optional[Tuple[int, int, int]] = None) -> None:
    """
    Show the blurred volume as a cut-away octant surface. Axes off.
    """
    fig = ax.get_figure()
    bbox = ax.get_position(); fig.delaxes(ax)
    ax2 = fig.add_axes(bbox)
    if coords is None:
        k, i, j = central_indices(vol_blur)
    else:
        k, i, j = coords
    img = render_cutaway_png_from_array(vol_blur, (k, i, j), save_png=None)
    ax2.imshow(img)
    ax2.set_xticks([]); ax2.set_yticks([])
    _strip_axes(ax2)


def plot_S3_grouping(ax, vol_blur: np.ndarray, mask_hr: np.ndarray,
                     spacing: Spacing, f: int, vis_stride: int = 1) -> None:
    """
    Visualize how r=f consecutive slices are averaged into one thick slice.
    For each block:
      - draw the f input slices with low alpha and background removed,
      - overlay a single opaque surface at the block center with the mean.
    """
    Z = vol_blur.shape[2]
    X = np.arange(vol_blur.shape[0]) * spacing.dx
    Y = np.arange(vol_blur.shape[1]) * spacing.dy
    Xg, Yg = np.meshgrid(X, Y, indexing="ij")
    if vis_stride > 1:
        Xg = Xg[::vis_stride, ::vis_stride]
        Yg = Yg[::vis_stride, ::vis_stride]
    norm = matplotlib.colors.Normalize(vmin=float(vol_blur.min()),
                                       vmax=float(vol_blur.max()))
    cmap = plt.get_cmap("gray")

    for start in range(0, Z, f):
        stop = min(start + f, Z)
        # input slices, faint, background removed
        for k in range(start, stop):
            Zg = np.full_like(Xg, k * spacing.dz)
            sl = vol_blur[:, :, k]
            if vis_stride > 1:
                sl = sl[::vis_stride, ::vis_stride]
            fc = cmap(norm(sl))
            if mask_hr is not None:
                ma = mask_hr[:, :, k].astype(float)
                if vis_stride > 1:
                    ma = ma[::vis_stride, ::vis_stride]
                fc[..., -1] = 0.25 * ma       # faint inputs
            ax.plot_surface(Xg, Yg, Zg, rstride=1, cstride=1,
                            facecolors=fc, linewidth=0, antialiased=False, shade=False)
        # averaged thick slice at block center, opaque
        thick = vol_blur[:, :, start:stop].mean(axis=2)
        if vis_stride > 1:
            thick = thick[::vis_stride, ::vis_stride]
        zc = ((start + stop - 1) / 2.0) * spacing.dz
        Zg = np.full_like(Xg, zc)
        fc = cmap(norm(thick))
        if mask_hr is not None:
            ma = mask_hr[:, :, min(stop-1, mask_hr.shape[2]-1)].astype(float)
            if vis_stride > 1:
                ma = ma[::vis_stride, ::vis_stride]
            fc[..., -1] = ma
        ax.plot_surface(Xg, Yg, Zg, rstride=1, cstride=1,
                        facecolors=fc, linewidth=0, antialiased=False, shade=False)

    ax.view_init(elev=8, azim=-60)
    ax.grid(False)
    _strip_axes(ax)




def plot_kernel(ax, sigma_mm: float, dz_mm: float) -> None:
    # discrete z-grid, ~±4σ
    sigma_vox = sigma_mm / dz_mm
    half = int(4 * sigma_vox) + 1
    z = np.arange(-half, half + 1) * dz_mm
    # normalized discrete kernel: sum(G)·Δz = 1
    G = np.exp(-(z**2) / (2 * sigma_mm**2))
    G = G / (G.sum() * dz_mm)

    # darker edge, lighter fill
    ax.fill_between(z, 0.0, G, linewidth=0, alpha=1.0)
    ax.plot(z, G, linewidth=1.75)

    # legend text with distribution facts
    fwhm = 2.355 * sigma_mm  # ties to SimpleITK pipeline as well :contentReference[oaicite:3]{index=3}
    txt = (r"$G(z)=\frac{1}{\sqrt{2\pi}\sigma}\exp\!\left(-\frac{z^2}{2\sigma^2}\right)$"
           f"\nσ={sigma_mm:.2f} mm, FWHM={fwhm:.2f} mm"
           f"\nΔz={dz_mm:.2f} mm,  Σ G·Δz=1")
    ax.legend(handles=[Patch(label=txt)], loc="upper right", frameon=False)
    _strip_axes(ax)


def plot_S0(ax, vol: np.ndarray, spacing: Spacing, tmpdir: Optional[str] = None,
            coords: Optional[Tuple[int, int, int]] = None) -> None:
    fig = ax.get_figure()
    bbox = ax.get_position(); fig.delaxes(ax)
    ax2 = fig.add_axes(bbox)
    if coords is None:
        k, i, j = central_indices(vol)
    else:
        k, i, j = coords
    png_path = None if tmpdir is None else os.path.join(tmpdir, "S0_cutaway.png")
    img = render_cutaway_png_from_array(vol, (k,i,j), save_png=png_path)
    ax2.imshow(img); ax2.set_xticks([]); ax2.set_yticks([]); _strip_axes(ax2)

def plot_S1(ax, vol: np.ndarray, spacing: Spacing, mask: np.ndarray,
            n_planes: int = 10, vis_stride: int = 1) -> None:
    plot_stack(ax, vol, spacing, mask=mask, n_planes=n_planes, vis_stride=vis_stride)

def plot_fft_compare(ax, vol_hr: np.ndarray, vol_blur: np.ndarray, dz_mm: float) -> None:
    s_hr = vol_hr.mean(axis=(0, 1)) - vol_hr.mean()
    s_bl = vol_blur.mean(axis=(0, 1)) - vol_blur.mean()
    F_hr = np.abs(np.fft.rfft(s_hr)); F_bl = np.abs(np.fft.rfft(s_bl))
    F_hr /= (F_hr.max() + 1e-12); F_bl /= (F_hr.max() + 1e-12)
    f = np.fft.rfftfreq(s_hr.size, d=dz_mm)
    ax.plot(f, F_hr, label="HR"); ax.plot(f, F_bl, label="Blurred")
    ax.legend(loc="upper right", frameon=False)
    _strip_axes(ax)

def plot_S2(ax_kernel, ax_fft, ax_stack3d, vol_blur: np.ndarray,
            spacing: Spacing, sigma_mm: float, vol_hr: np.ndarray) -> None:
    plot_kernel(ax_kernel, sigma_mm, spacing.dz)
    plot_fft_compare(ax_fft, vol_hr, vol_blur, spacing.dz)
    plot_stack(ax_stack3d, vol_blur, spacing, n_planes=10)

def plot_S3(ax, vol_blur: np.ndarray, spacing: Spacing, f: int) -> None:
    # just show the *averaged* thick-slice stacks at the new effective spacing
    Z = vol_blur.shape[2]
    X = np.arange(vol_blur.shape[0]) * spacing.dx
    Y = np.arange(vol_blur.shape[1]) * spacing.dy
    Xg, Yg = np.meshgrid(X, Y, indexing="ij")
    norm = matplotlib.colors.Normalize(vmin=vol_blur.min(), vmax=vol_blur.max())

    for start in range(0, Z, f):
        stop = min(start + f, Z)
        thick = vol_blur[:, :, start:stop].mean(axis=2)
        z_center = ((start + stop - 1) / 2.0) * spacing.dz
        Zg_center = np.full_like(Xg, z_center)
        facecolors = plt.get_cmap("gray")(norm(thick))
        ax.plot_surface(Xg, Yg, Zg_center, rstride=1, cstride=1,
                        facecolors=facecolors, linewidth=0,
                        antialiased=False, shade=False, alpha=1.0)
    ax.view_init(elev=8, azim=-60); ax.grid(False); _strip_axes(ax)

def resample_iso_nearest(vol: np.ndarray, spacing: Spacing,
                         target_mm: float = 1.0) -> tuple[np.ndarray, Spacing]:
    """
    Nearest-neighbour resample to isotropic target_mm spacing.

    Returns
    -------
    vol_iso, new_spacing
    """
    zx = spacing.dx / target_mm
    zy = spacing.dy / target_mm
    zz = spacing.dz / target_mm
    vol_iso = zoom(vol, zoom=(zx, zy, zz), order=0, mode="nearest",
                   grid_mode=True, prefilter=False)
    return vol_iso, Spacing(target_mm, target_mm, target_mm)

def plot_S4(ax, vol_lr: np.ndarray, spacing_lr: Spacing, tmpdir: Optional[str] = None,
            coords: Optional[Tuple[int, int, int]] = None) -> None:
    """
    Resample LR to 1×1×1 mm with nearest neighbour, save, then cut-away.
    """
    vol_iso, sp_iso = resample_iso_nearest(vol_lr, spacing_lr, target_mm=1.0)
    if tmpdir is not None:
        ensure_dir(tmpdir)
        # Save as NIfTI with isotropic spacing
        save_nifti(vol_iso.astype(np.float32), sp_iso, os.path.join(tmpdir, "vol_lr_iso_1mm.nii.gz"))

    fig = ax.get_figure()
    bbox = ax.get_position(); fig.delaxes(ax)
    ax2 = fig.add_axes(bbox)
    if coords is None:
        k, i, j = central_indices(vol_iso)
    else:
        k, i, j = resolve_coords(vol_iso, coords[0], coords[1], coords[2])
    img = render_cutaway_png_from_array(vol_iso, (k, i, j), save_png=None)
    ax2.imshow(img); ax2.set_xticks([]); ax2.set_yticks([])
    _strip_axes(ax2)


# -------------------------------
# Figure assembly
# -------------------------------

def save_stage_images(outdir: str, prods: StageProducts,
                      *, coords: Optional[Tuple[int, int, int]] = None,
                      n_planes: int = 10, vis_stride: int = 1) -> None:
    tmpdir = os.path.join(outdir, "tmp"); ensure_dir(tmpdir)
    # save intermediates for reuse in NIfTI format (fallback to .npy if nibabel unavailable)
    save_nifti(prods.vol_hr.astype(np.float32), prods.spacing_hr, os.path.join(tmpdir, "vol_hr.nii.gz"))
    save_nifti(prods.vol_blur.astype(np.float32), prods.spacing_hr, os.path.join(tmpdir, "vol_blur.nii.gz"))
    save_nifti(prods.vol_lr.astype(np.float32), prods.spacing_lr, os.path.join(tmpdir, "vol_lr.nii.gz"))
    save_nifti(prods.mask_hr.astype(np.uint8), prods.spacing_hr, os.path.join(tmpdir, "mask_hr.nii.gz"))
    save_nifti(prods.mask_lr.astype(np.uint8), prods.spacing_lr, os.path.join(tmpdir, "mask_lr.nii.gz"))

    # S0
    fig = plt.figure(figsize=(6.5, 6)); ax = fig.add_subplot(111, projection="3d")
    plot_S0(ax, prods.vol_hr, prods.spacing_hr, tmpdir=tmpdir, coords=coords)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "S0_HR_input.png"), dpi=250); plt.close(fig)

    # S1
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    plot_S1(ax, prods.vol_hr, prods.spacing_hr, prods.mask_hr, n_planes=n_planes, vis_stride=vis_stride)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "S1_Thin_slice_stack.png"), dpi=250); plt.close(fig)

    # S2a kernel (separate file)
    save_kernel_figure(outdir, prods.sigma_mm, prods.spacing_hr.dz)

    # S2b blurred as cut-away
    fig = plt.figure(figsize=(6.5, 6)); ax = fig.add_subplot(111, projection="3d")
    plot_S2_cutaway(ax, prods.vol_blur, coords=coords)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "S2b_Blur_cutaway.png"), dpi=500); plt.close(fig)

    # S3
    fig = plt.figure(figsize=(7, 6)); ax = fig.add_subplot(111, projection="3d")
    plot_S3_grouping(ax, prods.vol_blur, prods.mask_hr, prods.spacing_hr, prods.f, vis_stride=vis_stride)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "S3_Block_grouping.png"), dpi=250); plt.close(fig)

    # S4
    fig = plt.figure(figsize=(6.5, 6)); ax = fig.add_subplot(111, projection="3d")
    plot_S4(ax, prods.vol_lr, prods.spacing_lr, tmpdir=tmpdir, coords=coords)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "S4_LR_iso1mm_cutaway.png"), dpi=250); plt.close(fig)

def save_panel(outdir: str, prods: StageProducts,
               *, coords: Optional[Tuple[int, int, int]] = None,
               n_planes: int = 10, vis_stride: int = 1) -> None:
    """
    Compact 1×4 panel: S0 cut-away | S1 stack | S2b blurred cut-away | S4 iso-1mm cut-away.
    Kernel is saved separately.
    """
    tmpdir = os.path.join(outdir, "tmp"); ensure_dir(tmpdir)
    fig = plt.figure(figsize=(22, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.1, 1.3, 1.1, 1.1])

    ax0 = fig.add_subplot(gs[0, 0], projection="3d"); plot_S0(ax0, prods.vol_hr, prods.spacing_hr, tmpdir=tmpdir, coords=coords)
    ax1 = fig.add_subplot(gs[0, 1], projection="3d"); plot_S1(ax1, prods.vol_hr, prods.spacing_hr, prods.mask_hr, n_planes=n_planes, vis_stride=vis_stride)
    ax2 = fig.add_subplot(gs[0, 2], projection="3d"); plot_S2_cutaway(ax2, prods.vol_blur, coords=coords)
    ax3 = fig.add_subplot(gs[0, 3], projection="3d"); plot_S4(ax3, prods.vol_lr, prods.spacing_lr, tmpdir=tmpdir, coords=coords)

    fig.tight_layout(); fig.savefig(os.path.join(outdir, "Panel_S0_S1_S2b_S4.png"), dpi=280); plt.close(fig)



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
    # Visualization controls
    ap.add_argument("--slice_axial", type=int, default=None, help="Axial index k (z) for octant cut-away. Negative allowed.")
    ap.add_argument("--slice_coronal", type=int, default=None, help="Coronal index i (x) for octant cut-away. Negative allowed.")
    ap.add_argument("--slice_sagittal", type=int, default=None, help="Sagittal index j (y) for octant cut-away. Negative allowed.")
    ap.add_argument("--n_planes", type=int, default=10, help="Number of axial planes for stack rendering.")
    ap.add_argument("--vis_stride", type=int, default=3, help="Pixel stride for 3D surfaces (>=1). Use 2+ for speed.")
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

    # Resolve plotting coordinates if any were provided
    coords = None
    if any(v is not None for v in (args.slice_axial, args.slice_coronal, args.slice_sagittal)):
        coords = resolve_coords(
            prods.vol_hr,
            k_axial=args.slice_axial,
            i_coronal=args.slice_coronal,
            j_sagittal=args.slice_sagittal,
        )

    vis_stride = max(1, int(args.vis_stride))
    n_planes = max(1, int(args.n_planes))

    save_stage_images(args.outdir, prods, coords=coords, n_planes=n_planes, vis_stride=vis_stride)
    save_panel(args.outdir, prods, coords=coords, n_planes=n_planes, vis_stride=vis_stride)

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