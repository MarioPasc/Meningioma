#!/usr/bin/env python3
"""
octant_surface.py  —  PyVista renderer

Create a 3-D “cut-away” visualisation: remove the first octant of the head
about (k_axial, i_coronal, j_sagittal) and show the three intersecting slices
restricted to that octant, with air/background zeroed.

Axes follow A–R–S:
  +x ≡ anterior, +y ≡ right-lateral, +z ≡ cranial.

Dependencies
------------
- pyvista>=0.43
- numpy, matplotlib, scikit-image, nibabel (for NIfTI IO)
- mgmGrowth.tasks.superresolution.visualization.mask_brain:
    compute_head_mask_from_hr, laplacian_smooth

CLI example
-----------
python src/mgmGrowth/tasks/superresolution/visualization/octant_surface.py \
  /path/HR.nii.gz 70 100 110 \
  --save /path/octant_surface.png --no_mask_extent --window 1600 1200
"""
from __future__ import annotations

import argparse
import logging
import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure  # marching_cubes
from skimage import morphology
from skimage.draw import polygon as sk_polygon
from skimage.draw import polygon2mask

try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None

import pyvista as pv

from mgmGrowth.tasks.superresolution.visualization.mask_brain import (
    compute_head_mask_from_hr,
    laplacian_smooth,
)

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# ──────────────────────────────────────────────────────────────────────────
# Data + config

def load_volume(path: pathlib.Path) -> np.ndarray:
    """Load 3-D scalar volume (.nii/.nii.gz or .npy) as float64, shape (nx,ny,nz)."""
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

@dataclass(frozen=True)
class CutawayConfig:
    iso_level: float = 0.1
    mesh_alpha: float = 1.0
    mesh_color: Tuple[float, float, float] = (0.65, 0.65, 0.65)
    slice_alpha: float = 0.95
    cmap: str = "gray"
    min_margin: int = 1
    specular: float = 0.3
    specular_power: float = 20.0
    plane_bias: float = 0.05           # shift plane into cavity along its normal
    plane_inset: float = 0.02          # shrink plane edges to avoid touching mesh

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

def drop_faces_in_first_octant(verts: np.ndarray, faces: np.ndarray,
                               i0: int, j0: int, k0: int) -> np.ndarray:
    """Remove triangles with any vertex inside the first octant Ω={x≥i0,y≥j0,z≥k0}."""
    tri = verts[faces]  # (F,3,3)
    inside = (tri[..., 0] >= i0) & (tri[..., 1] >= j0) & (tri[..., 2] >= k0)
    keep = ~inside.any(axis=1)
    return faces[keep]

def faces_to_pv(faces_tri: np.ndarray) -> np.ndarray:
    """Convert (F,3) triangle indices to PyVista faces array [3,i,j,k,...]."""
    f = np.hstack([np.full((faces_tri.shape[0], 1), 3, dtype=np.int64), faces_tri.astype(np.int64)])
    return f.ravel()

def _largest_cc_mask(mask2d: np.ndarray, min_area: int = 256) -> np.ndarray:
    """
    Keep the largest connected component of a 2-D boolean mask and fill small holes.
    Connectivity=2 (8-neighborhood). Removes tiny speckles and isolates the brain
    cross-section on the slice.  """
    lab = measure.label(mask2d.astype(bool), connectivity=2)
    if lab.max() == 0:
        return mask2d.astype(bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = counts.argmax()
    cc = (lab == keep)
    # remove tiny islands that could remain and fill small holes
    cc = morphology.remove_small_objects(cc, min_size=min_area)
    cc = morphology.remove_small_holes(cc, area_threshold=min_area)
    return cc

def _inset(a0: float, a1: float, eps: float) -> tuple[float, float]:
    """Inset [a0,a1] by eps on each side, clamped to preserve order."""
    lo, hi = (a0 + eps, a1 - eps)
    if hi <= lo:
        mid = 0.5 * (a0 + a1)
        lo, hi = mid, mid  # degenerate but stable
    return lo, hi

def _rim_mask_from_mesh(mesh_closed: pv.PolyData,
                        origin: Tuple[float, float, float],
                        normal: Tuple[float, float, float],
                        u_axis: Tuple[float, float, float],
                        v_axis: Tuple[float, float, float],
                        u0w: float, u1w: float,
                        v0w: float, v1w: float,
                        nu: int, nv: int) -> np.ndarray:
    logger.info(f"[rim] origin={origin}, normal={normal}, "
                f"U-range=({u0w:.2f},{u1w:.2f}), V-range=({v0w:.2f},{v1w:.2f}), size={nv}x{nu}")

    # Slice the CLOSED mesh, then join segments into polylines
    rim = mesh_closed.slice(origin=origin, normal=normal).strip(join=True).clean()
    logger.info(f"[rim] slice points={rim.n_points}, lines_size={rim.lines.size}, n_lines={rim.n_lines}")
    if rim.n_points == 0 or rim.lines.size == 0:
        logger.warning("[rim] empty slice; returning empty mask")
        return np.zeros((nv, nu), dtype=bool)

    # Project points to (U,V) and rasterize each polyline
    P = rim.points - np.asarray(origin)[None, :]
    U = P @ np.asarray(u_axis)
    V = P @ np.asarray(v_axis)

    lines = rim.lines
    k = 0
    mask = np.zeros((nv, nu), dtype=bool)
    polys_used = 0
    while k < len(lines):
        n = int(lines[k]); k += 1
        ids = lines[k:k + n]; k += n
        if n < 3:
            logger.debug(f"[rim] skip short polyline n={n}")
            continue
        uu, vv = U[ids], V[ids]

        # Build polygon in pixel space (row=V, col=U), keep float precision
        # pixel centers are at integer indices; lower-left pixel center is (0,0)
        poly_rc = np.column_stack([vv - v0w - 0.5,  # rows = V
                                uu - u0w - 0.5]) # cols = U
        poly_mask = polygon2mask((nv, nu), poly_rc)
        mask |= poly_mask
        polys_used += 1

    logger.info(f"[rim] polygons used={polys_used}, mask true={int(mask.sum())}")
    mask = morphology.binary_closing(mask, morphology.disk(1))
    mask = morphology.remove_small_holes(mask, area_threshold=64)
    return mask

# ──────────────────────────────────────────────────────────────────────────
# Slice → RGBA texture

def _rgba_masked(img2d: np.ndarray,
                 mask2d: np.ndarray,
                 vmin: float, vmax: float,
                 cmap: str,
                 alpha_inside: float) -> np.ndarray:
    """Map intensities to uint8 RGBA; alpha=0 where mask is False."""
    eps = np.finfo(float).eps
    norm = (img2d - vmin) / max(vmax - vmin, eps)
    rgba = plt.get_cmap(cmap)(np.clip(norm, 0, 1))  # float RGBA in [0,1]
    a = np.where(mask2d, alpha_inside, 0.0)
    rgba[..., 3] = a
    return (rgba * 255).astype(np.uint8)

def _texture_from_slice(patch: np.ndarray,
                        mask2d: np.ndarray,
                        vmin: float, vmax: float,
                        cfg: CutawayConfig) -> pv.Texture:
    logger.debug(f"[texture] patch shape={patch.shape}, mask true={int(mask2d.sum())}, "
                 f"vmin={vmin:.3f}, vmax={vmax:.3f}")
    rgba = _rgba_masked(patch, mask2d, vmin, vmax, cfg.cmap, cfg.slice_alpha)
    rgba = rgba[::-1, :, :]  # align with plane UV
    tex = pv.numpy_to_texture(rgba)
    tex.repeat = False
    tex.interpolate = False
    logger.debug(f"[texture] built RGBA {rgba.shape}, repeat={tex.repeat}, interp={tex.interpolate}")
    return tex
# ──────────────────────────────────────────────────────────────────────────
# Scene assembly (PyVista)

def _add_surface(plotter: pv.Plotter,
                 mask: np.ndarray,
                 cfg: CutawayConfig,
                 i0: int, j0: int, k0: int) -> tuple[pv.PolyData, pv.PolyData]:
    logger.info(f"[surface] mask shape={mask.shape}, iso={cfg.iso_level}")
    nx, ny, nz = mask.shape

    # closed head mesh (for slicing)
    verts, faces = marching_surface_from_mask(mask, cfg.iso_level)
    verts = laplacian_smooth(verts, faces, iterations=5, lam=0.45)
    mesh_full = pv.PolyData(verts, faces_to_pv(faces))
    logger.debug(f"[surface] full verts={mesh_full.n_points}, faces={mesh_full.n_cells}")

    # visual mesh: remove first octant with a box clip
    bounds = (i0 - 0.5, nx - 0.5, j0 - 0.5, ny - 0.5, k0 - 0.5, nz - 0.5)
    logger.info(f"[surface] clip_box bounds={bounds}")
    mesh_clip = mesh_full.clip_box(bounds=bounds, invert=True, merge_points=True)  # robust removal
    logger.debug(f"[surface] clipped verts={mesh_clip.n_points}, faces={mesh_clip.faces}")

    plotter.add_mesh(mesh_clip,
                     color=cfg.mesh_color,
                     opacity=cfg.mesh_alpha,
                     smooth_shading=True,
                     specular=float(np.clip(cfg.specular, 0.0, 1.0)),
                     specular_power=float(cfg.specular_power),
                     show_edges=False)
    logger.info(f"[surface] actor added")
    return mesh_full, mesh_clip

# --- axial plane (XY @ z=k): add bias on +Z and inset rectangle
def _add_axial_plane(plotter, vol, mask3d, mesh_full, i0, j0, k, vmin, vmax, cfg):
    nx, ny, nz = vol.shape
    if not (0 <= k < nz): return
    patch = vol[i0:nx, j0:ny, k].T
    x0w, x1w = (i0 - 0.5), (nx - 0.5)
    y0w, y1w = (j0 - 0.5), (ny - 0.5)
    nu, nv = int(round(x1w - x0w)), int(round(y1w - y0w))

    rim_mask = _rim_mask_from_mesh(mesh_full,
                                   origin=(0.0, 0.0, float(k)),
                                   normal=(0.0, 0.0, 1.0),
                                   u_axis=(1.0, 0.0, 0.0),
                                   v_axis=(0.0, 1.0, 0.0),
                                   u0w=x0w, u1w=x1w, v0w=y0w, v1w=y1w,
                                   nu=nu, nv=nv)
    if not rim_mask.any():
        logger.warning("[axial] empty rim mask after strip; skip axial plane")
        return

    ys, xs = np.where(rim_mask)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    tex = _texture_from_slice(patch[r0:r1+1, c0:c1+1],
                              rim_mask[r0:r1+1, c0:c1+1], vmin, vmax, cfg)

    x0, x1 = x0w + c0, x0w + c1 + 1
    y0, y1 = y0w + r0, y0w + r1 + 1
    x0, x1 = _inset(x0, x1, cfg.plane_inset); y0, y1 = _inset(y0, y1, cfg.plane_inset)
    zw = float(k) + cfg.plane_bias

    plane = pv.Plane(center=((x0+x1)/2.0, (y0+y1)/2.0, zw),
                     direction=(0, 0, 1), i_size=(x1-x0), j_size=(y1-y0))
    plane.texture_map_to_plane(origin=(x0, y0, zw),
                               point_u=(x1, y0, zw),
                               point_v=(x0, y1, zw), inplace=True)
    plotter.add_mesh(plane, texture=tex)



# --- coronal plane (YZ @ x=i): add bias on +X and inset rectangle
def _add_coronal_plane(plotter, vol, mask3d, mesh_full, i, j0, k0, vmin, vmax, cfg):
    nx, ny, nz = vol.shape
    if not (0 <= i < nx): return
    patch = vol[i, j0:ny, k0:nz].T
    y0w, y1w = (j0 - 0.5), (ny - 0.5)
    z0w, z1w = (k0 - 0.5), (nz - 0.5)
    nu, nv = int(round(y1w - y0w)), int(round(z1w - z0w))

    rim_mask = _rim_mask_from_mesh(mesh_full,
                                   origin=(float(i), 0.0, 0.0),
                                   normal=(1.0, 0.0, 0.0),
                                   u_axis=(0.0, 1.0, 0.0),
                                   v_axis=(0.0, 0.0, 1.0),
                                   u0w=y0w, u1w=y1w, v0w=z0w, v1w=z1w,
                                   nu=nu, nv=nv)
    if not rim_mask.any():
        logger.warning("[coronal] empty rim mask after strip; skip coronal plane")
        return

    ys, xs = np.where(rim_mask)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    tex = _texture_from_slice(patch[r0:r1+1, c0:c1+1],
                              rim_mask[r0:r1+1, c0:c1+1], vmin, vmax, cfg)

    y0, y1 = y0w + c0, y0w + c1 + 1
    z0, z1 = z0w + r0, z0w + r1 + 1
    y0, y1 = _inset(y0, y1, cfg.plane_inset); z0, z1 = _inset(z0, z1, cfg.plane_inset)
    xw = float(i) + cfg.plane_bias

    plane = pv.Plane(center=(xw, (y0+y1)/2.0, (z0+z1)/2.0),
                     direction=(1, 0, 0), i_size=(y1-y0), j_size=(z1-z0))
    plane.texture_map_to_plane(origin=(xw, y0, z0),
                               point_u=(xw, y1, z0),
                               point_v=(xw, y0, z1), inplace=True)
    plotter.add_mesh(plane, texture=tex)


# --- sagittal plane (XZ @ y=j): add bias on +Y and inset rectangle
def _add_sagittal_plane(plotter, vol, mask3d, mesh_full, j, i0, k0, vmin, vmax, cfg):
    nx, ny, nz = vol.shape
    if not (0 <= j < ny): return
    patch = vol[i0:nx, j, k0:nz].T
    x0w, x1w = (i0 - 0.5), (nx - 0.5)
    z0w, z1w = (k0 - 0.5), (nz - 0.5)
    nu, nv = int(round(x1w - x0w)), int(round(z1w - z0w))

    rim_mask = _rim_mask_from_mesh(mesh_full,
                                   origin=(0.0, float(j), 0.0),
                                   normal=(0.0, 1.0, 0.0),
                                   u_axis=(1.0, 0.0, 0.0),
                                   v_axis=(0.0, 0.0, 1.0),
                                   u0w=x0w, u1w=x1w, v0w=z0w, v1w=z1w,
                                   nu=nu, nv=nv)
    if not rim_mask.any():
        logger.warning("[sagittal] empty rim mask after strip; skip sagittal plane")
        return

    ys, xs = np.where(rim_mask)
    r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
    tex = _texture_from_slice(patch[r0:r1+1, c0:c1+1],
                              rim_mask[r0:r1+1, c0:c1+1], vmin, vmax, cfg)

    x0, x1 = x0w + c0, x0w + c1 + 1
    z0, z1 = z0w + r0, z0w + r1 + 1
    x0, x1 = _inset(x0, x1, cfg.plane_inset); z0, z1 = _inset(z0, z1, cfg.plane_inset)
    yw = float(j) + cfg.plane_bias

    plane = pv.Plane(center=((x0+x1)/2.0, yw, (z0+z1)/2.0),
                     direction=(0, 1, 0), i_size=(x1-x0), j_size=(z1-z0))
    plane.texture_map_to_plane(origin=(x0, yw, z0),
                               point_u=(x1, yw, z0),
                               point_v=(x0, yw, z1), inplace=True)
    plotter.add_mesh(plane, texture=tex)



def plot_cutaway_octant_pv(
    vol: np.ndarray,
    k_axial: int, i_coronal: int, j_sagittal: int,
    *,
    cfg: CutawayConfig = CutawayConfig(),
    use_mask_extent: bool = True,
    window_size: Tuple[int, int] = (1400, 1000),
    off_screen: bool = False,
) -> pv.Plotter:
    nx, ny, nz = vol.shape
    k = int(np.clip(k_axial, cfg.min_margin, nz - 1 - cfg.min_margin))
    i = int(np.clip(i_coronal, cfg.min_margin, nx - 1 - cfg.min_margin))
    j = int(np.clip(j_sagittal, cfg.min_margin, ny - 1 - cfg.min_margin))

    mask = compute_head_mask_from_hr(vol)
    vol = vol.copy()
    vol[~mask] = 0.0
    inside = vol[mask]
    vmin, vmax = (np.percentile(inside, [1.0, 99.0]) if inside.size
                  else (float(vol.min()), float(vol.max())))

    pv.global_theme.background = "white"
    plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
    plotter.enable_anti_aliasing("msaa")
    plotter.enable_depth_peeling()  # correct translucent ordering

    mesh_full, mesh_clip = _add_surface(plotter, mask, cfg, i, j, k)
    _add_axial_plane(plotter,   vol, mask, mesh_full, i, j, k, vmin, vmax, cfg)
    _add_coronal_plane(plotter, vol, mask, mesh_full, i, j, k, vmin, vmax, cfg)
    _add_sagittal_plane(plotter, vol, mask, mesh_full, j, i, k, vmin, vmax, cfg)


    # Camera similar to mpl target view
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
    p = argparse.ArgumentParser(description="Cut-away octant 3-D visualisation (PyVista).")
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
    p.add_argument("--plane_inset", type=float, default=CutawayConfig.plane_inset)
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
        plane_inset=args.plane_inset,
    )


    # Optional mesh color parsing
    if args.mesh_color is not None:
        c = args.mesh_color
        if c.startswith("#"):
            import matplotlib.colors as mcolors
            cfg = CutawayConfig(mesh_alpha=cfg.mesh_alpha,
                                slice_alpha=cfg.slice_alpha,
                                cmap=cfg.cmap,
                                mesh_color=mcolors.to_rgb(c))
        else:
            g = float(c)
            cfg = CutawayConfig(mesh_alpha=cfg.mesh_alpha,
                                slice_alpha=cfg.slice_alpha,
                                cmap=cfg.cmap,
                                mesh_color=(g, g, g))

    i, j, k = args.coronal_i, args.sagittal_j, args.axial_k
    if args.index_space.lower() == "lps":
        i, j, k = lps_indices_to_ras(i, j, k, vol.shape)

    off = bool(args.save)  # off-screen if we will screenshot
    plotter = plot_cutaway_octant_pv(
        vol, k, i, j,
        cfg=cfg,
        use_mask_extent=not args.no_mask_extent,
        window_size=tuple(args.window),
        off_screen=off,
    )

    if args.save:
        plotter.show(screenshot=str(args.save))
    else:
        plotter.show()

if __name__ == "__main__":
    main()
