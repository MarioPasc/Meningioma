#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
show_surface_octant_example.py

Octant + surface visualizations (PyVista) for super-resolution model comparison.

This mirrors the layout and logic of show_octant_example.py, but each panel is
rendered with the octant+surface 3D renderer (slices + head surface cut-away)
from the PyVista pipeline rather than the 2D octant renderer.

Outputs per pulse:
- A grid with rows = resolutions (3/5/7 mm) and columns = 2 × models:
  [ SR octant+surface | Residual (SR−HR) octant+surface ] for each model.

Notes
-----
- Volumes are operated in LPS for IO/processing, but converted to RAS to render.
- SR volumes are resampled to HR geometry when needed (nibabel[scipy]).
- Residuals use a diverging colormap centered at 0 with symmetric window ±rmax,
  where rmax is the 99th percentile of |SR−HR| across all models/resolutions.
- Rendering uses off-screen PyVista and returns screenshots for placement in a
  Matplotlib figure.

Example
-------
python src/mgmGrowth/tasks/superresolution/visualization/show_surface_octant_example.py \
  --subject BraTS-MEN-00231-000 \
  --highres_dir /path/high_resolution \
  --models_dir /path/results/models \
  --pulses t1c t2f t2w t1n \
  --models UNIRES SMORE ECLARE BSPLINE \
  --out /path/results/figures \
  --coords 65 120 135
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union, Any, cast

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, ListedColormap, Colormap

try:
    import nibabel as nib
except Exception as e:
    raise RuntimeError("This script requires nibabel.") from e

try:
    import pyvista as pv
    pv.global_theme.allow_empty_mesh = True
except Exception as e:
    raise RuntimeError("This script requires pyvista (>=0.43).") from e

from skimage import measure

# Local mask/smoothing utilities used by the PyVista renderer
from mgmGrowth.tasks.superresolution.visualization.mask_brain import (
    compute_head_mask_from_hr,
    laplacian_smooth,
)

# Logger
logger = logging.getLogger(__name__)


# -------------------- styling ----------------------------------------------

def configure_matplotlib() -> None:
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


def build_black_center_diverging() -> ListedColormap:
    """
    Diverging colormap with a black center (0) and contrasting ends.
    Negative: yellow-ish, Positive: purple-ish (mirrors prior custom).
    """
    def _hex_rgba(h: str) -> np.ndarray:
        h = h.lstrip('#')
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return np.array([r, g, b, 1.0], dtype=float)

    neg = _hex_rgba("ffe945")  # SR−HR < 0
    pos = _hex_rgba("762A83")  # SR−HR > 0
    blk = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    n = 256
    colors = np.empty((n, 4), dtype=float)
    for t in range(128):
        a = t / 127.0
        colors[t] = (1.0 - a) * neg + a * blk
    for t in range(128, 256):
        a = (t - 128) / 127.0
        colors[t] = (1.0 - a) * blk + a * pos

    return ListedColormap(colors, name="black_center_div")


# -------------------- I/O and geometry -------------------------------------

def load_nii(path: Path):
    try:
        img = nib.load(str(path))
        logger.debug("Loaded NIfTI %s | shape=%s", path, getattr(img, 'shape', None))
        return img
    except Exception as e:
        raise FileNotFoundError(f"Failed to load NIfTI: {path}") from e


def data_LPS(nii) -> np.ndarray:
    ras = nib.as_closest_canonical(nii)
    arr = ras.get_fdata(dtype=np.float32)
    arr = np.flip(arr, axis=0)  # R→L
    arr = np.flip(arr, axis=1)  # A→P
    logger.debug("Converted to LPS | shape=%s | dtype=%s", arr.shape, arr.dtype)
    return arr


def to_RAS_from_LPS(vol_LPS: np.ndarray) -> np.ndarray:
    return np.flip(np.flip(vol_LPS, axis=0), axis=1)


def resample_like(src, like, order: int = 1):
    try:
        from nibabel.processing import resample_from_to  # type: ignore
    except Exception as e:
        raise RuntimeError("Resampling requires nibabel[scipy] (SciPy installed).") from e
    logger.info("Resampling volume | from=%s to=%s | order=%d", getattr(src, 'shape', None), getattr(like, 'shape', None), order)
    out = resample_from_to(src, (like.shape, like.affine), order=order)
    logger.debug("Resampled shape=%s", getattr(out, 'shape', None))
    return out


def hr_pulse_path(hr_dir: Path, subject: str, pulse: str) -> Path:
    return hr_dir / subject / f"{subject}-{pulse}.nii.gz"


def hr_seg_path(hr_dir: Path, subject: str) -> Path:
    return hr_dir / subject / f"{subject}-seg.nii.gz"


def sr_model_path(models_dir: Path, model: str, res_mm: int, subject: str, pulse: str) -> Path:
    return models_dir / model / f"{res_mm}mm" / "output_volumes" / f"{subject}-{pulse}.nii.gz"


def save_abs_residual_nii(
    hr_nii,
    residual_LPS: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save |SR−HR| as NIfTI in HR canonical RAS geometry.
    Background is left as 0.0 in the file; visualization masks it out.
    """
    hr_can = nib.as_closest_canonical(hr_nii)
    res_ras = to_RAS_from_LPS(residual_LPS)
    img = nib.Nifti1Image(res_ras.astype(np.float32), hr_can.affine, header=hr_can.header)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(out_path))


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


def compute_coords(subject: str, highres_dir: Path, pulses: Sequence[str], coord_mode: str,
                   coords: Optional[Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if coords is not None:
        logger.info("Using provided coords (k,i,j)=%s", coords)
        return coords
    seg_p = hr_seg_path(highres_dir, subject)
    if seg_p.exists():
        seg = data_LPS(load_nii(seg_p))
        if coord_mode in ("bbox", "auto"):
            k, i, j = bbox_center(seg)
            if coord_mode == "auto":
                # Nudge axial index slightly towards cranial if possible
                k = min(k + 3, seg.shape[2] - 2)
            logger.info("Coords from %s center: (k,i,j)=(%d,%d,%d)", coord_mode, k, i, j)
            return k, i, j
        if coord_mode == "com":
            out = center_of_mass(seg)
            logger.info("Coords from center-of-mass: %s", out)
            return out
    # Fallback: center of any HR pulse
    hr_any = load_nii(hr_pulse_path(highres_dir, subject, pulses[0]))
    vol = data_LPS(hr_any)
    nz = np.array(vol.shape) // 2
    logger.info("Coords from fallback center: (k,i,j)=(%d,%d,%d)", int(nz[2]), int(nz[0]), int(nz[1]))
    return int(nz[2]), int(nz[0]), int(nz[1])


# -------------------- residuals and ranges ---------------------------------

def signed_residual(sr: np.ndarray, hr: np.ndarray) -> np.ndarray:
    a, b = sr.astype(np.float32), hr.astype(np.float32)
    if a.shape != b.shape:
        tgt = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
        def crop(x, t):
            sx, sy, sz = x.shape
            cx = (sx - t[0]) // 2; cy = (sy - t[1]) // 2; cz = (sz - t[2]) // 2
            return x[cx:cx+t[0], cy:cy+t[1], cz:cz+t[2]]
        a, b = crop(a, tgt), crop(b, tgt)
        logger.debug("Residual shapes differ; cropped to %s", tgt)
    return a - b


def symmetric_rmax(residuals: Sequence[np.ndarray], q: float = 99.0) -> float:
    vals: List[float] = []
    for r in residuals:
        m = np.isfinite(r)
        if m.any():
            vals.append(float(np.percentile(np.abs(r[m]), q)))
    rmax_val: float = float(max(vals)) if vals else 1.0
    logger.info("Global residual rmax (q=%.1f) = %.6f from %d volumes", q, rmax_val, len(vals))
    return rmax_val


# -------------------- PyVista helpers (adapted) ----------------------------

def _marching_surface_from_mask(mask: np.ndarray, iso: float) -> tuple[np.ndarray, np.ndarray]:
    mask_f = mask.astype(np.float32)
    ntrue = int(np.count_nonzero(mask))
    logger.debug("Marching cubes on mask | shape=%s | true=%d | iso=%.3f", mask.shape, ntrue, iso)
    verts, faces, _, _ = measure.marching_cubes(mask_f, level=iso)
    logger.debug("Surface verts=%d | faces=%d", len(verts), len(faces))
    return verts, faces


def _faces_to_pv(faces_tri: np.ndarray) -> np.ndarray:
    f = np.hstack([np.full((faces_tri.shape[0], 1), 3, dtype=np.int64), faces_tri.astype(np.int64)])
    return f.ravel()


def _make_uniform_grid_from_volume(vol: np.ndarray) -> pv.ImageData:
    grid = pv.ImageData()
    # Explicit tuple to keep type-checkers happy
    sx, sy, sz = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
    grid.dimensions = (sx + 1, sy + 1, sz + 1)  # type: ignore[assignment]
    grid.spacing = (1.0, 1.0, 1.0)  # type: ignore[assignment]
    grid.origin = (-0.5, -0.5, -0.5)  # type: ignore[assignment]
    grid.cell_data["I"] = vol.ravel(order="F")
    grid = grid.cell_data_to_point_data(pass_cell_data=False)
    # ensure no duplicate lingering in cell_data
    if "I" in grid.cell_data:
        del grid.cell_data["I"]
    logger.debug("Uniform grid | dims=%s | n_points=%d | n_cells=%d", grid.dimensions, grid.n_points, grid.n_cells)
    return grid


def _add_surface(plotter: pv.Plotter, mask: np.ndarray, i0: int, j0: int, k0: int,
                 *, iso_level: float = 0.1,
                 mesh_color=(0.65, 0.65, 0.65), mesh_alpha: float = 1.0,
                 specular: float = 0.3, specular_power: float = 20.0) -> pv.PolyData:
    nx, ny, nz = mask.shape
    verts, faces = _marching_surface_from_mask(mask, iso_level)
    try:
        verts = laplacian_smooth(verts, faces, iterations=5, lam=0.45)
    except Exception:
        pass
    mesh_full = pv.PolyData(verts, _faces_to_pv(faces))
    bounds = (i0 - 0.5, nx - 0.5, j0 - 0.5, ny - 0.5, k0 - 0.5, nz - 0.5)
    logger.debug("Clipping surface with bounds=%s", bounds)
    mesh_clip = mesh_full.clip_box(bounds=bounds, invert=True, merge_points=True)
    logger.debug("Clipped surface | n_points=%d | n_cells=%d", mesh_clip.n_points, mesh_clip.n_cells)
    plotter.add_mesh(
        mesh_clip,
        color=mesh_color,
        opacity=mesh_alpha,
        smooth_shading=True,
        specular=float(np.clip(specular, 0.0, 1.0)),
        specular_power=float(specular_power),
        show_edges=False,
    )
    return mesh_clip


def _add_slices(plotter: pv.Plotter, grid: pv.ImageData,
                i: int, j: int, k: int,
                *, vmin: float, vmax: float,
                cmap: Any,
                slice_alpha: float = 0.95,
                plane_bias: float = 0.01) -> None:
    # Axial (XY) at z=k
    z0 = float(k) + plane_bias
    slc = grid.slice(normal=(0, 0, 1), origin=(0.0, 0.0, z0))
    slc = slc.clip(normal=(1, 0, 0), origin=(float(i), 0.0, 0.0), invert=False)
    slc = slc.clip(normal=(0, 1, 0), origin=(0.0, float(j), 0.0), invert=False)
    logger.debug("Add axial slice at k=%d (bias=%.3f) | clip x>=%d, y>=%d", k, plane_bias, i, j)
    plotter.add_mesh(
        slc, scalars="I", cmap=cmap, clim=(vmin, vmax), opacity=slice_alpha, nan_opacity=0.0,
        show_scalar_bar=False,
    )

    # Coronal (YZ) at x=i
    x0 = float(i) + plane_bias
    slc = grid.slice(normal=(1, 0, 0), origin=(x0, 0.0, 0.0))
    slc = slc.clip(normal=(0, 1, 0), origin=(0.0, float(j), 0.0), invert=False)
    slc = slc.clip(normal=(0, 0, 1), origin=(0.0, 0.0, float(k)), invert=False)
    logger.debug("Add coronal slice at i=%d | clip y>=%d, z>=%d", i, j, k)
    plotter.add_mesh(
        slc, scalars="I", cmap=cmap, clim=(vmin, vmax), opacity=slice_alpha, nan_opacity=0.0,
        show_scalar_bar=False,
    )

    # Sagittal (XZ) at y=j
    y0 = float(j) + plane_bias
    slc = grid.slice(normal=(0, 1, 0), origin=(0.0, y0, 0.0))
    slc = slc.clip(normal=(1, 0, 0), origin=(float(i), 0.0, 0.0), invert=False)
    slc = slc.clip(normal=(0, 0, 1), origin=(0.0, 0.0, float(k)), invert=False)
    logger.debug("Add sagittal slice at j=%d | clip x>=%d, z>=%d", j, i, k)
    plotter.add_mesh(
        slc, scalars="I", cmap=cmap, clim=(vmin, vmax), opacity=slice_alpha, nan_opacity=0.0,
        show_scalar_bar=False,
    )


def _lps_indices_to_ras(i: int, j: int, k: int, shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    nx, ny, _ = shape
    return (nx - 1 - i, ny - 1 - j, k)


def render_cutaway_octant_image(
    volume_LPS: np.ndarray,
    coords_kij: Tuple[int, int, int],
    *,
    cmap: Any = 'gray',
    enforce_minmax: Optional[Tuple[float, float]] = None,
    window_size: Tuple[int, int] = (600, 600),
    mesh_alpha: float = 1.0,
    slice_alpha: float = 0.95,
    mesh_color=(0.65, 0.65, 0.65),
    specular: float = 0.3,
    specular_power: float = 20.0,
    plane_bias: float = 0.01,
    head_mask_LPS: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Render an octant+surface cutaway. If head_mask_LPS is given, use it."""
    k_lps, i_lps, j_lps = coords_kij

    # Convert to RAS for PyVista
    vol_ras = to_RAS_from_LPS(volume_LPS)
    nx, ny, nz = vol_ras.shape
    i_ras, j_ras, k_ras = _lps_indices_to_ras(i_lps, j_lps, k_lps, (nx, ny, nz))

    # Use external mask if provided; otherwise derive from this volume
    if head_mask_LPS is not None:
        mask = to_RAS_from_LPS(head_mask_LPS.astype(bool))
    else:
        try:
            mask = compute_head_mask_from_hr(vol_ras)
        except Exception:
            # Simple fallback: non-zero as foreground
            mask = np.isfinite(vol_ras) & (np.abs(vol_ras) > 0)
    vol_masked = vol_ras.copy()
    vol_masked[~mask] = np.nan
    mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    logger.info("SR render: shape=%s | mask true=%.2f%%", vol_ras.shape, 100.0 * mask_ratio)

    # Determine intensity window
    if enforce_minmax is not None:
        vmin, vmax = float(enforce_minmax[0]), float(enforce_minmax[1])
    else:
        inside = vol_masked[np.isfinite(vol_masked)]
        vmin, vmax = (float(np.nanmin(vol_masked)), float(np.nanmax(vol_masked))) if inside.size == 0 \
                     else np.percentile(inside, [1.0, 99.0])
    logger.debug("SR window: vmin=%.6f vmax=%.6f", vmin, vmax)

    # Off-screen plotter
    pv.global_theme.background = "black"  # type: ignore[assignment]
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    try:
        cast(Any, plotter).enable_anti_aliasing("msaa")
    except Exception:
        pass
    try:
        edp = getattr(plotter, "enable_depth_peeling", None)
        if callable(edp):
            edp()
    except Exception:
        pass

    # Surface
    _add_surface(plotter, mask, i_ras, j_ras, k_ras,
                 iso_level=0.1, mesh_color=mesh_color, mesh_alpha=mesh_alpha,
                 specular=specular, specular_power=specular_power)

    # Grid and slices
    grid = _make_uniform_grid_from_volume(vol_masked)
    _add_slices(plotter, grid, i_ras, j_ras, k_ras,
                vmin=vmin, vmax=vmax, cmap=cmap, slice_alpha=slice_alpha, plane_bias=plane_bias)

    # Camera framing
    if mask.any():
        idx = np.argwhere(mask)
        (xmin, ymin, zmin), (xmax, ymax, zmax) = idx.min(0), idx.max(0)
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    else:
        center = (nx / 2, ny / 2, nz / 2)
    cast(Any, plotter).set_focus(center)
    span = np.array([nx, ny, nz], dtype=float)
    dist = 2.2 * float(np.linalg.norm(span))
    cam_pos = (float(center[0] + dist), float(center[1] + dist), float(center[2] + 0.9 * dist))
    # Slight offset similar to original code
    cam_pos = (float(cam_pos[0] + 200), float(cam_pos[1]), float(cam_pos[2] - 300))
    try:
        cast(Any, plotter).set_position(cam_pos)
        cast(Any, plotter).set_viewup((0, 0, 1))
        plotter.camera.SetViewAngle(23)
    except Exception:
        pass

    # Screenshot
    try:
        img = plotter.screenshot(transparent_background=False, return_img=True)
    except TypeError:
        # Older PyVista signature fallback
        img = plotter.screenshot()
    finally:
        try:
            plotter.close()
        except Exception:
            pass

    # If RGBA, composite to black
    np_img = np.asarray(img)
    logger.debug("SR screenshot shape=%s dtype=%s", np_img.shape, np_img.dtype)
    if np_img.ndim == 3 and np_img.shape[-1] == 4:
        rgb = np_img[..., :3].astype(np.float32) / 255.0
        a = np_img[..., 3:4].astype(np.float32) / 255.0
        np_img = np.clip(rgb * a, 0.0, 1.0)
    elif np_img.dtype != np.float32:
        np_img = (np_img.astype(np.float32) / 255.0) if np_img.max() > 1.0 else np_img.astype(np.float32)
    return np_img


# -------------------- residual renderer (HR geometry only) -----------------

def render_residual_cutaway_octant_image(
    hr_LPS: np.ndarray,
    sr_LPS: np.ndarray,
    coords_kij: Tuple[int, int, int],
    *,
    window_size: Tuple[int, int] = (600, 600),
    plane_bias: float = 0.01,
    rmax: float = 100.0,
    cmap: Any = 'afmhot',
) -> np.ndarray:
    """
    Render |SR−HR| only on visible geometry: HR outer surface (octant-clipped)
    and the three wedge-clipped slice surfaces. Background is removed before
    sampling. Returns an RGB image for imshow.
    """
    # 1) Convert to RAS; indices LPS→RAS
    hr_ras = to_RAS_from_LPS(hr_LPS)
    sr_ras = to_RAS_from_LPS(sr_LPS)
    nx, ny, nz = hr_ras.shape
    k_lps, i_lps, j_lps = coords_kij
    i_ras, j_ras, k_ras = _lps_indices_to_ras(i_lps, j_lps, k_lps, (nx, ny, nz))

    # 2) Head mask from HR only; remove background on HR for sampling
    head_mask = compute_head_mask_from_hr(hr_ras)
    hr_masked = hr_ras.copy(); hr_masked[~head_mask] = np.nan
    # SR can remain unmasked; we will only sample on HR-derived geometry
    sr_unmasked = sr_ras
    mask_ratio = float(np.count_nonzero(head_mask)) / float(head_mask.size)
    logger.info("Residual render: hr shape=%s | sr shape=%s | head mask true=%.2f%%", hr_ras.shape, sr_ras.shape, 100.0 * mask_ratio)

    # 3) Uniform grids for probing
    hr_grid = _make_uniform_grid_from_volume(hr_masked)
    sr_grid = _make_uniform_grid_from_volume(sr_unmasked)

    # 4) HR surface → octant clip
    verts, faces = _marching_surface_from_mask(head_mask.astype(bool), 0.1)
    try:
        verts = laplacian_smooth(verts, faces, iterations=5, lam=0.45)
    except Exception:
        pass
    mesh = pv.PolyData(verts, _faces_to_pv(faces))
    bounds = (i_ras - 0.5, nx - 0.5, j_ras - 0.5, ny - 0.5, k_ras - 0.5, nz - 0.5)
    mesh = mesh.clip_box(bounds=bounds, invert=True, merge_points=True)

    # 5) Sample HR and SR on surface; color by |SR−HR|
    surf_hr = hr_grid.sample(mesh)
    logger.debug("Surface sampling: HR arrays=%s", list(surf_hr.point_data.keys()))
    if 'I' in surf_hr.point_data:
        try:
            surf_hr.point_data.rename('I', 'I_HR', deep=False)
        except Exception:
            arr = np.array(surf_hr['I'])
            surf_hr.point_data.remove('I')
            surf_hr['I_HR'] = arr
    surf_sr = sr_grid.sample(mesh)
    logger.debug("Surface sampling: SR arrays=%s", list(surf_sr.point_data.keys()))
    if 'I' in surf_sr.point_data:
        try:
            surf_sr.point_data.rename('I', 'I_SR', deep=False)
        except Exception:
            arr = np.array(surf_sr['I'])
            surf_sr.point_data.remove('I')
            surf_sr['I_SR'] = arr

    # Start from HR sampled surface (retains topology)
    mesh = surf_hr.copy(deep=False)
    if 'I_SR' in surf_sr.point_data:
        mesh['I_SR'] = surf_sr['I_SR']
    if ('I_SR' in mesh.point_data) and ('I_HR' in mesh.point_data):
        res_surf = np.abs(mesh['I_SR'] - mesh['I_HR'])
        invalid = ~np.isfinite(mesh['I_HR'])
        if invalid.any():
            res_surf = res_surf.astype(float)
            res_surf[invalid] = np.nan
        mesh.point_data.clear(); mesh['RES'] = res_surf
        # Log residual stats on surface
        finite = np.isfinite(res_surf)
        if finite.any():
            q1, q99 = np.percentile(res_surf[finite], [1, 99])
            logger.debug("Surface residual stats: min=%.6f q1=%.6f med=%.6f q99=%.6f max=%.6f",
                         float(res_surf[finite].min()), float(q1), float(np.median(res_surf[finite])), float(q99), float(res_surf[finite].max()))

    # 6) Create three slice geometries from HR grid, clip to wedge, sample both, color by |res|
    def _slice_and_color(normal, origin, clips: list[Tuple[Tuple[int,int,int], Tuple[float,float,float]]]):
        slc = hr_grid.slice(normal=normal, origin=origin)
        for nrm, org in clips:
            slc = slc.clip(normal=nrm, origin=org, invert=False)
        # Keep HR intensity
        if 'I' in slc.point_data:
            try:
                slc.point_data.rename('I', 'I_HR', deep=False)
            except Exception:
                arr = np.array(slc['I'])
                slc.point_data.remove('I')
                slc['I_HR'] = arr
        # Sample SR onto slice
        sampled = sr_grid.sample(slc)

        def _extract_point_I(ds: pv.DataSet) -> Optional[np.ndarray]:
            if 'I' in ds.point_data and ds.point_data['I'].size == slc.n_points:
                return np.asarray(ds.point_data['I'])
            if 'I' in ds.cell_data and ds.n_cells == slc.n_cells:
                ds2 = ds.cell_data_to_point_data(pass_cell_data=False)
                if 'I' in ds2.point_data and ds2.point_data['I'].size == slc.n_points:
                    return np.asarray(ds2.point_data['I'])
            return None

        vals = _extract_point_I(sampled)
        if vals is None:
            # Some PyVista builds invert semantics; sample with the slice as caller
            sampled2 = slc.sample(sr_grid)
            vals = _extract_point_I(sampled2)

        if vals is None:
            # Final fallback: trilinear interpolation via SciPy
            try:
                from scipy.ndimage import map_coordinates
                # slice points are already in voxel coords (origin -0.5, spacing 1)
                pts = slc.points
                coords = np.vstack([pts[:, 0], pts[:, 1], pts[:, 2]])
                vals = map_coordinates(sr_unmasked, coords, order=1, mode='nearest').astype(np.float32)
            except Exception as e:
                raise RuntimeError(f"SR sampling failed: {e}")

        slc['I_SR'] = vals
        if ('I_SR' in slc.point_data) and ('I_HR' in slc.point_data):
            res_slice = np.abs(slc['I_SR'] - slc['I_HR'])
            invalid = ~np.isfinite(slc['I_HR'])
            if invalid.any():
                res_slice = res_slice.astype(float)
                res_slice[invalid] = np.nan
            slc['RES'] = res_slice
        logger.debug("Slice residual added | points=%d | has RES=%s", slc.n_points, 'RES' in slc.point_data)
        return slc

    z0 = float(k_ras) + plane_bias
    x0 = float(i_ras) + plane_bias
    y0 = float(j_ras) + plane_bias
    axial = _slice_and_color((0,0,1), (0.0,0.0,z0), [((1,0,0),(float(i_ras),0.0,0.0)), ((0,1,0),(0.0,float(j_ras),0.0))])
    coronal = _slice_and_color((1,0,0), (x0,0.0,0.0), [((0,1,0),(0.0,float(j_ras),0.0)), ((0,0,1),(0.0,0.0,float(k_ras)))])
    sagittal = _slice_and_color((0,1,0), (0.0,y0,0.0), [((1,0,0),(float(i_ras),0.0,0.0)), ((0,0,1),(0.0,0.0,float(k_ras)))])

    # 7) Plotter and add geometries with |res| scalars
    pv.global_theme.background = "black"  # type: ignore[assignment]
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    try:
        cast(Any, plotter).enable_anti_aliasing("msaa")
    except Exception:
        pass
    try:
        edp = getattr(plotter, "enable_depth_peeling", None)
        if callable(edp):
            edp()
    except Exception:
        pass

    if 'RES' in mesh.point_data:
        plotter.add_mesh(mesh, scalars='RES', cmap=cmap, clim=(0.0, float(rmax)), nan_opacity=0.0,
                         opacity=1.0, show_scalar_bar=False, smooth_shading=True)
    for slc in (axial, coronal, sagittal):
        if 'RES' in slc.point_data:
            plotter.add_mesh(slc, scalars='RES', cmap=cmap, clim=(0.0, float(rmax)), nan_opacity=0.0,
                             opacity=0.95, show_scalar_bar=False)

    # Camera identical to SR view
    if head_mask.any():
        idx = np.argwhere(head_mask)
        (xmin, ymin, zmin), (xmax, ymax, zmax) = idx.min(0), idx.max(0)
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    else:
        center = (nx / 2, ny / 2, nz / 2)
    cast(Any, plotter).set_focus(center)
    span = np.array([nx, ny, nz], dtype=float)
    dist = 2.2 * float(np.linalg.norm(span))
    cam_pos = (float(center[0] + dist), float(center[1] + dist), float(center[2] + 0.9 * dist))
    cam_pos = (float(cam_pos[0] + 200), float(cam_pos[1]), float(cam_pos[2] - 300))
    try:
        cast(Any, plotter).set_position(cam_pos)
        cast(Any, plotter).set_viewup((0, 0, 1))
        plotter.camera.SetViewAngle(23)
    except Exception:
        pass

    try:
        img = plotter.screenshot(transparent_background=False, return_img=True)
    except TypeError:
        img = plotter.screenshot()
    finally:
        try:
            plotter.close()
        except Exception:
            pass

    np_img = np.asarray(img)
    logger.debug("Residual screenshot shape=%s dtype=%s", np_img.shape, np_img.dtype)
    if np_img.ndim == 3 and np_img.shape[-1] == 4:
        rgb = np_img[..., :3].astype(np.float32) / 255.0
        a = np_img[..., 3:4].astype(np.float32) / 255.0
        np_img = np.clip(rgb * a, 0.0, 1.0)
    elif np_img.dtype != np.float32:
        np_img = (np_img.astype(np.float32) / 255.0) if np_img.max() > 1.0 else np_img.astype(np.float32)
    return np_img

# -------------------- layout spec ------------------------------------------

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
    fig_width: float
    fig_height: float
    left_margin: float
    right_margin: float
    top_margin: float
    bottom_margin: float
    wspace: float
    hspace: float
    fmt: str
    log: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Per-pulse octant+surface grids: rows=3/5/7mm, cols=2xmodels (SR|Residual)")
    p.add_argument("--subject", required=True)
    p.add_argument("--highres_dir", required=True, type=Path)
    p.add_argument("--models_dir", required=True, type=Path)
    p.add_argument("--pulses", nargs="+", default=["t1c", "t1n", "t2f", "t2w"]) 
    p.add_argument("--models", nargs="+", default=None, help="Defaults to subdirs in models_dir")
    p.add_argument("--coords", nargs=3, type=int, default=None, metavar=("k", "i", "j"))
    p.add_argument("--coord_mode", choices=["auto", "bbox", "com"], default="auto")

    # Simple spacing (kept close to original defaults)
    p.add_argument('--fig_width', type=float, default=11.5)
    p.add_argument('--fig_height', type=float, default=6.9)
    p.add_argument('--left_margin', type=float, default=0.06)
    p.add_argument('--right_margin', type=float, default=0.995)
    p.add_argument('--top_margin', type=float, default=0.955)
    p.add_argument('--bottom_margin', type=float, default=0.08)
    p.add_argument('--wspace', type=float, default=0.02)
    p.add_argument('--hspace', type=float, default=0.02)

    p.add_argument("--out", type=Path, default=Path("figures"))
    p.add_argument("--fmt", choices=["pdf", "png"], default="pdf")
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
        fig_width=ns.fig_width,
        fig_height=ns.fig_height,
        left_margin=ns.left_margin,
        right_margin=ns.right_margin,
        top_margin=ns.top_margin,
        bottom_margin=ns.bottom_margin,
        wspace=ns.wspace,
        hspace=ns.hspace,
        fmt=ns.fmt,
        log=str(ns.log).upper(),
    )


# -------------------- main per-pulse figure --------------------------------

def make_pulse_figure(
    args: Args,
    pulse: str,
    coords_kij: Tuple[int, int, int],
    res_rows: Sequence[int] = RES_ROWS,
) -> None:
    # Load HR reference and segmentation
    logger.info("Start pulse figure | subject=%s | pulse=%s | coords(k,i,j)=%s", args.subject, pulse, coords_kij)
    hr_nii = load_nii(hr_pulse_path(args.highres_dir, args.subject, pulse))
    hr_LPS = data_LPS(hr_nii)
    seg_LPS: Optional[np.ndarray] = None
    seg_path = hr_seg_path(args.highres_dir, args.subject)
    if seg_path.exists():
        try:
            seg_LPS = data_LPS(load_nii(seg_path))
        except Exception:
            seg_LPS = None

    # Head mask from HR only; apply to HR and all SR/residuals
    head_mask = compute_head_mask_from_hr(hr_LPS)
    masked_hr = hr_LPS.copy(); masked_hr[~head_mask] = 0.0
    logger.info("HR volume shape=%s | mask true=%.2f%%", hr_LPS.shape, 100.0 * (np.count_nonzero(head_mask) / head_mask.size))

    # Cache SR (masked) and collect residuals for global symmetric window
    sr_cache: Dict[Tuple[str, int], np.ndarray] = {}
    all_residuals: List[np.ndarray] = []

    for m in args.models:
        for rmm in res_rows:
            sr_nii = load_nii(sr_model_path(args.models_dir, m, rmm, args.subject, pulse))
            # Resample if geometry mismatch
            if (sr_nii.shape != hr_nii.shape) or (not np.allclose(sr_nii.affine, hr_nii.affine)):
                try:
                    sr_nii = resample_like(sr_nii, hr_nii, order=1)
                except Exception as e:
                    raise RuntimeError(f"Resampling required but unavailable for {m} {rmm}mm") from e
            sr_LPS = data_LPS(sr_nii)
            masked_sr = sr_LPS.copy(); masked_sr[~head_mask] = 0.0
            sr_cache[(m, rmm)] = masked_sr
            logger.debug("Cached SR | model=%s | res=%dmm | shape=%s", m, rmm, masked_sr.shape)
            all_residuals.append(signed_residual(masked_sr, masked_hr))

    # Residual window
    rmax = max(symmetric_rmax(all_residuals, q=99.0), 1e-6)
    logger.info("Final rmax used for residuals: %.6f", rmax)
    div_cmap = build_black_center_diverging()

    # Layout
    n_rows = len(res_rows); n_cols = 2 * len(args.models)
    A4_W = 11.69
    fig_w = min(args.fig_width, A4_W); fig_h = args.fig_height
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(fig_w, fig_h), squeeze=True)
    fig.patch.set_facecolor('black')

    # Render panels
    for r_idx, rmm in enumerate(res_rows):
        for m_idx, m in enumerate(args.models):
            masked_sr = sr_cache[(m, rmm)]

            # SR octant+surface
            logger.debug("Render SR panel | row=%d res=%dmm | model=%s", r_idx, rmm, m)
            sr_img = render_cutaway_octant_image(
                masked_sr, coords_kij,
                cmap='gray', enforce_minmax=None, window_size=(600, 600),
            )
            axL = axes[r_idx, 2*m_idx]
            axL.imshow(sr_img, origin='upper', interpolation='nearest')
            axL.set_xticks([]); axL.set_yticks([])
            for s in axL.spines.values(): s.set_visible(False)
            axL.set_facecolor('black')

            # Residual octant+surface (absolute residual on HR geometry only)
            logger.debug("Render Residual panel | row=%d res=%dmm | model=%s", r_idx, rmm, m)
            res_img = render_residual_cutaway_octant_image(
                masked_hr, masked_sr, coords_kij,
                window_size=(600, 600), plane_bias=0.01, rmax=rmax, cmap='afmhot',
            )
            axR = axes[r_idx, 2*m_idx + 1]
            axR.imshow(res_img, origin='upper', interpolation='nearest')
            axR.set_xticks([]); axR.set_yticks([])
            for s in axR.spines.values(): s.set_visible(False)
            axR.set_facecolor('black')

    # Row labels (left: resolution)
    for r_idx, rmm in enumerate(res_rows):
        ax = axes[r_idx, 0]
        box = ax.get_position()
        y_mid = (box.y0 + box.y1) / 2.0
        fig.text(0.03, y_mid, f"{rmm}mm", color='white', ha='left', va='center')

    # Spacing
    plt.subplots_adjust(
        left=args.left_margin, right=args.right_margin,
        top=args.top_margin, bottom=args.bottom_margin,
        wspace=args.wspace, hspace=args.hspace,
    )

    # Model headers across each pair of columns
    top_y = 0.975
    for m_idx, m in enumerate(args.models):
        boxL = axes[0, 2*m_idx].get_position()
        boxR = axes[0, 2*m_idx + 1].get_position()
        x_center = 0.5 * ((boxL.x0 + boxL.x1) + (boxR.x0 + boxR.x1)) / 2.0
        fig.text(x_center, top_y, m, color='white', ha='center', va='top')

    # Colorbar for |SR−HR| normalized to global rmax
    sm = plt.cm.ScalarMappable(norm=Normalize(0.0, rmax), cmap=plt.get_cmap('afmhot'))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal', fraction=0.035, pad=0.05)
    cbar.set_label(r"Absolute Residual Error $\,|\mathrm{SR} - \mathrm{HR}|$", color='white')
    outline = getattr(cbar, 'outline', None)
    if outline is not None:
        try:
            outline.set_edgecolor('white'); outline.set_linewidth(0.8)
        except Exception:
            pass
    cbar.ax.tick_params(color='white', labelcolor='white')

    # Save
    out_dir = args.out_dir / "octants_by_model" / args.subject
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.subject}_{pulse}_octants_models_surface.{args.fmt}"
    logger.info("Saving figure to %s", out_file)
    if args.fmt == "pdf":
        with PdfPages(out_file) as pdf:
            pdf.savefig(fig, dpi=600, facecolor=fig.get_facecolor(), bbox_inches='tight')
    else:
        fig.savefig(out_file, dpi=600, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    logging.info("Saved: %s", out_file)


# -------------------- main --------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log, logging.DEBUG),
                        format="%(levelname)s: %(message)s")
    configure_matplotlib()

    coords = compute_coords(args.subject, args.highres_dir, args.pulses, args.coord_mode, args.coords)

    for pulse in args.pulses:
        make_pulse_figure(args, pulse, coords, RES_ROWS)


if __name__ == "__main__":
    main()

