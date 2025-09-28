#!/usr/bin/env python3
"""
octant_pyvista.py — PyVista renderer (no textures)

Create a 3-D “cut-away” visualisation: remove the first octant of the head
about (k_axial, i_coronal, j_sagittal) and show the three intersecting
slices restricted to that octant. Slices are true PyVista surfaces obtained
by sampling a UniformGrid and clipping with the other two half-spaces.

Axes follow RAS:
  +x ≡ anterior, +y ≡ right-lateral, +z ≡ cranial.

Dependencies
------------
- pyvista>=0.43
- numpy, matplotlib, scikit-image, nibabel (for NIfTI IO)
- mgmGrowth.tasks.superresolution.visualization.mask_brain:
    compute_head_mask_from_hr, laplacian_smooth

CLI example
-----------
python src/mgmGrowth/tasks/superresolution/visualization/octant_pyvista.py \
  /path/HR.nii.gz 70 100 110 \
  --save /path/octant.png --no_mask_extent --window 1600 1200
"""
from __future__ import annotations

import argparse
import logging
import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from skimage import measure  # marching_cubes

try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None

import pyvista as pv

from mgmGrowth.tasks.superresolution.visualization.mask_brain import (
    compute_head_mask_from_hr,
    laplacian_smooth,
)

# ──────────────────────────────────────────────────────────────────────────
# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ──────────────────────────────────────────────────────────────────────────
# IO

def load_volume(path: pathlib.Path) -> np.ndarray:
    """
    Load a 3-D scalar volume (.nii/.nii.gz or .npy) as float64 with shape (nx, ny, nz).
    Returns data in RAS orientation if NIfTI.
    """
    s = path.suffix.lower()
    if s == ".npy":
        return np.load(path).astype(np.float64, copy=False)
    if s in {".nii", ".gz", ".nii.gz"}:
        if nib is None:
            raise RuntimeError("Reading NIfTI requires nibabel.")
        img = nib.load(str(path))
        img = nib.as_closest_canonical(img)  # force RAS
        return np.asanyarray(img.get_fdata(), dtype=np.float64)
    raise ValueError(f"Unsupported file type: {path}")

# ──────────────────────────────────────────────────────────────────────────
# Config

@dataclass(frozen=True)
class CutawayConfig:
    """Rendering and geometry parameters."""
    iso_level: float = 0.1                     # marching cubes ISO for head mask
    mesh_alpha: float = 1.0
    mesh_color: Tuple[float, float, float] = (0.65, 0.65, 0.65)
    slice_alpha: float = 0.95
    cmap: str = "gray"
    min_margin: int = 1
    specular: float = 0.3
    specular_power: float = 20.0
    plane_bias: float = 0.05                   # shift slice plane into cavity along its normal

# ──────────────────────────────────────────────────────────────────────────
# Geometry helpers

def lps_indices_to_ras(i: int, j: int, k: int, shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convert LPS voxel indices to RAS index space (x,y flip; z unchanged)."""
    nx, ny, _ = shape
    return (nx - 1 - i, ny - 1 - j, k)

def marching_surface_from_mask(mask: np.ndarray, iso: float) -> tuple[np.ndarray, np.ndarray]:
    """Iso-surface from a binary head mask using marching cubes. Returns (verts, faces)."""
    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=iso)
    return verts, faces

def faces_to_pv(faces_tri: np.ndarray) -> np.ndarray:
    """Convert (F,3) triangle indices to PyVista 'faces' array [3,i,j,k,...]."""
    f = np.hstack([np.full((faces_tri.shape[0], 1), 3, dtype=np.int64), faces_tri.astype(np.int64)])
    return f.ravel()

def make_uniform_grid_from_volume(vol: np.ndarray) -> pv.ImageData():
    """
    Build a PyVista UniformGrid whose cell centers coincide with integer voxel indices.
    - origin = (-0.5,-0.5,-0.5)   so voxel centers are at (i,j,k)
    - dimensions = vol.shape + 1  so #cells == vol.shape
    Scalars are stored in cell_data["I"].
    """
    grid = pv.ImageData()
    grid.dimensions = np.array(vol.shape, dtype=int) + 1
    grid.spacing = (1.0, 1.0, 1.0)
    grid.origin = (-0.5, -0.5, -0.5)
    grid.cell_data["I"] = vol.ravel(order="F")
    return grid

# ──────────────────────────────────────────────────────────────────────────
# Rendering primitives

def _add_surface(plotter: pv.Plotter,
                 mask: np.ndarray,
                 cfg: CutawayConfig,
                 i0: int, j0: int, k0: int) -> pv.PolyData:
    """
    Add the outer head mesh after cutting away the first octant Ω={x≥i0, y≥j0, z≥k0}.
    Returns the clipped mesh (PolyData).
    """
    logger.info(f"[surface] mask shape={mask.shape}, iso={cfg.iso_level}")
    nx, ny, nz = mask.shape

    verts, faces = marching_surface_from_mask(mask, cfg.iso_level)
    verts = laplacian_smooth(verts, faces, iterations=5, lam=0.45)
    mesh_full = pv.PolyData(verts, faces_to_pv(faces))
    logger.debug(f"[surface] full verts={mesh_full.n_points}, faces={mesh_full.n_cells}")

    # Robust box clip to remove Ω (note the -0.5 bounds match marching_cubes coordinates)
    bounds = (i0 - 0.5, nx - 0.5, j0 - 0.5, ny - 0.5, k0 - 0.5, nz - 0.5)
    logger.info(f"[surface] clip_box bounds={bounds}")
    mesh_clip = mesh_full.clip_box(bounds=bounds, invert=True, merge_points=True)
    logger.debug(f"[surface] clipped verts={mesh_clip.n_points}, faces={mesh_clip.faces}")

    plotter.add_mesh(
        mesh_clip,
        color=cfg.mesh_color,
        opacity=cfg.mesh_alpha,
        smooth_shading=True,
        specular=float(np.clip(cfg.specular, 0.0, 1.0)),
        specular_power=float(cfg.specular_power),
        show_edges=False,
    )
    logger.info("[surface] actor added")
    return mesh_clip

def _add_axial_slice(plotter: pv.Plotter, grid: pv.ImageData(),
                     i: int, j: int, k: int, vmin: float, vmax: float,
                     cfg: CutawayConfig) -> None:
    """
    XY slice at z=k, then clip to x>=i and y>=j to keep the wedge. Bias into +z.
    """
    z0 = float(k) + cfg.plane_bias
    slc = grid.slice(normal=(0, 0, 1), origin=(0.0, 0.0, z0))
    slc = slc.clip(normal=(1, 0, 0), origin=(float(i), 0.0, 0.0), invert=False)   # keep x>=i
    slc = slc.clip(normal=(0, 1, 0), origin=(0.0, float(j), 0.0), invert=False)   # keep y>=j
    plotter.add_mesh(
        slc,
        scalars="I",
        cmap=cfg.cmap,
        clim=(vmin, vmax),
        opacity=cfg.slice_alpha,
        nan_opacity=0.0,
        show_scalar_bar=False,
    )
    logger.info("[axial] slice added at k=%d", k)

def _add_coronal_slice(plotter: pv.Plotter, grid: pv.ImageData(),
                       i: int, j: int, k: int, vmin: float, vmax: float,
                       cfg: CutawayConfig) -> None:
    """
    YZ slice at x=i, then clip to y>=j and z>=k to keep the wedge. Bias into +x.
    """
    x0 = float(i) + cfg.plane_bias
    slc = grid.slice(normal=(1, 0, 0), origin=(x0, 0.0, 0.0))
    slc = slc.clip(normal=(0, 1, 0), origin=(0.0, float(j), 0.0), invert=False)   # keep y>=j
    slc = slc.clip(normal=(0, 0, 1), origin=(0.0, 0.0, float(k)), invert=False)   # keep z>=k
    plotter.add_mesh(
        slc,
        scalars="I",
        cmap=cfg.cmap,
        clim=(vmin, vmax),
        opacity=cfg.slice_alpha,
        nan_opacity=0.0,
        show_scalar_bar=False,
    )
    logger.info("[coronal] slice added at i=%d", i)

def _add_sagittal_slice(plotter: pv.Plotter, grid: pv.ImageData(),
                        i: int, j: int, k: int, vmin: float, vmax: float,
                        cfg: CutawayConfig) -> None:
    """
    XZ slice at y=j, then clip to x>=i and z>=k to keep the wedge. Bias into +y.
    """
    y0 = float(j) + cfg.plane_bias
    slc = grid.slice(normal=(0, 1, 0), origin=(0.0, y0, 0.0))
    slc = slc.clip(normal=(1, 0, 0), origin=(float(i), 0.0, 0.0), invert=False)   # keep x>=i
    slc = slc.clip(normal=(0, 0, 1), origin=(0.0, 0.0, float(k)), invert=False)   # keep z>=k
    plotter.add_mesh(
        slc,
        scalars="I",
        cmap=cfg.cmap,
        clim=(vmin, vmax),
        opacity=cfg.slice_alpha,
        nan_opacity=0.0,
        show_scalar_bar=False,
    )
    logger.info("[sagittal] slice added at j=%d", j)

# ──────────────────────────────────────────────────────────────────────────
# Scene assembly

def plot_cutaway_octant_pv(
    vol: np.ndarray,
    k_axial: int, i_coronal: int, j_sagittal: int,
    *,
    cfg: CutawayConfig = CutawayConfig(),
    use_mask_extent: bool = True,
    window_size: Tuple[int, int] = (1400, 1000),
    off_screen: bool = False,
) -> pv.Plotter:
    """
    Build the scene:
      1) compute head mask and outer surface,
      2) create UniformGrid with NaN outside the mask,
      3) add three clipped slice surfaces.
    """
    nx, ny, nz = vol.shape
    k = int(np.clip(k_axial, cfg.min_margin, nz - 1 - cfg.min_margin))
    i = int(np.clip(i_coronal, cfg.min_margin, nx - 1 - cfg.min_margin))
    j = int(np.clip(j_sagittal, cfg.min_margin, ny - 1 - cfg.min_margin))

    # Brain mask and intensity range
    mask = compute_head_mask_from_hr(vol)
    vol_masked = vol.copy()
    vol_masked[~mask] = np.nan  # outside brain → transparent via nan_opacity=0

    inside = vol_masked[np.isfinite(vol_masked)]
    if inside.size:
        vmin, vmax = np.percentile(inside, [1.0, 99.0])
    else:
        vmin, vmax = (float(np.nanmin(vol_masked)), float(np.nanmax(vol_masked)))

    # Plotter
    pv.global_theme.background = "white"
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    plotter.enable_anti_aliasing("msaa")
    plotter.enable_depth_peeling()

    # Outer surface
    _ = _add_surface(plotter, mask, cfg, i, j, k)

    # Grid for slice sampling
    grid = make_uniform_grid_from_volume(vol_masked)

    # Add the three slices
    _add_axial_slice(plotter, grid, i, j, k, vmin, vmax, cfg)
    _add_coronal_slice(plotter, grid, i, j, k, vmin, vmax, cfg)
    _add_sagittal_slice(plotter, grid, i, j, k, vmin, vmax, cfg)

    # Camera framing
    if mask.any() and use_mask_extent:
        idx = np.argwhere(mask)
        (xmin, ymin, zmin), (xmax, ymax, zmax) = idx.min(0), idx.max(0)
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2)
    else:
        center = (nx / 2, ny / 2, nz / 2)

    plotter.set_focus(center)
    span = np.array([nx, ny, nz], dtype=float)
    dist = 2.2 * np.linalg.norm(span)
    cam_pos = (center[0] + dist, center[1] + dist, center[2] + 0.9 * dist)
    plotter.set_position(cam_pos)
    plotter.set_viewup((0, 0, 1))
    plotter.camera.SetViewAngle(30)
    return plotter

# ──────────────────────────────────────────────────────────────────────────
# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cut-away octant 3-D visualisation (PyVista, no textures).")
    p.add_argument("volume", type=pathlib.Path, help="Path to .nii/.nii.gz or .npy")
    p.add_argument("axial_k", type=int, help="Axial index k (z)")
    p.add_argument("coronal_i", type=int, help="Coronal index i (x)")
    p.add_argument("sagittal_j", type=int, help="Sagittal index j (y)")
    p.add_argument("--alpha_mesh", type=float, default=CutawayConfig.mesh_alpha)
    p.add_argument("--alpha_slice", type=float, default=CutawayConfig.slice_alpha)
    p.add_argument("--cmap", type=str, default=CutawayConfig.cmap)
    p.add_argument("--mesh_color", type=str, default=None,
                   help="Mesh color; hex like '#888888' or gray float 0–1.")
    p.add_argument("--index_space", choices=["ras", "lps"], default="ras",
                   help="Coordinate system of provided indices.")
    p.add_argument("--no_mask_extent", action="store_true",
                   help="Disable camera framing to mask bbox.")
    p.add_argument("--save", type=pathlib.Path, help="Output image path (PNG)")
    p.add_argument("--window", nargs=2, type=int, default=(1400, 1000),
                   help="Render window size: width height")
    p.add_argument("--specular", type=float, default=CutawayConfig.specular)
    p.add_argument("--specular_power", type=float, default=CutawayConfig.specular_power)
    p.add_argument("--plane_bias", type=float, default=CutawayConfig.plane_bias)
    p.add_argument("--loglevel", default="INFO", help="Logging level")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO),
                        format="[%(levelname)s] %(message)s")

    vol = load_volume(args.volume)
    cfg = CutawayConfig(
        mesh_alpha=args.alpha_mesh,
        slice_alpha=args.alpha_slice,
        cmap=args.cmap,
        specular=args.specular,
        specular_power=args.specular_power,
        plane_bias=args.plane_bias,
    )

    # Optional mesh color parsing
    if args.mesh_color is not None:
        import matplotlib.colors as mcolors
        if args.mesh_color.startswith("#"):
            mesh_rgb = mcolors.to_rgb(args.mesh_color)
        else:
            g = float(args.mesh_color)
            mesh_rgb = (g, g, g)
        cfg = CutawayConfig(
            mesh_alpha=cfg.mesh_alpha, slice_alpha=cfg.slice_alpha, cmap=cfg.cmap,
            mesh_color=mesh_rgb, specular=cfg.specular, specular_power=cfg.specular_power,
            plane_bias=cfg.plane_bias,
        )

    i, j, k = args.coronal_i, args.sagittal_j, args.axial_k
    if args.index_space.lower() == "lps":
        i, j, k = lps_indices_to_ras(i, j, k, vol.shape)

    off = bool(args.save)  # off-screen if we will screenshot
    plotter = plot_cutaway_octant_pv(
        vol, k, i, j,
        cfg=cfg,
        use_mask_extent=not args.no_mask_extent,
        window_size=(args.window[0], args.window[1]),
        off_screen=off,
    )

    if args.save:
        plotter.show(screenshot=str(args.save))
    else:
        plotter.show()

if __name__ == "__main__":
    main()
