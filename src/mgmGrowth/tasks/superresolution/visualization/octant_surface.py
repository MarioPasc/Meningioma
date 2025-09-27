#!/usr/bin/env python3
"""
cutaway_octant_vis.py
Create a 3-D “cut-away” visualisation: remove the first octant of the head
about (k_axial, i_coronal, j_sagittal) and show the three intersecting slices
restricted to that octant, with air/background zeroed.

Axes follow your A–R–S convention:
  +x ≡ anterior, +y ≡ right-lateral, +z ≡ cranial.

References
----------
Lorensen WE, Cline HE. Marching Cubes: A High Resolution 3D Surface
Construction Algorithm. SIGGRAPH '87.

python src/mgmGrowth/tasks/superresolution/visualization/octant_surface.py /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/high_resolution/BraTS-MEN-00231-000/BraTS-MEN-00231-000-t1c.nii.gz 70 100 100 --save /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/result
s/figures
"""
from __future__ import annotations

import argparse
import logging
import pathlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import nibabel as nib
except ModuleNotFoundError:
    nib = None

# skimage is used only for surface extraction; fail clearly if missing
from skimage import measure  # marching_cubes

from mgmGrowth.tasks.superresolution.visualization.mask_brain import compute_head_mask_from_hr, laplacian_smooth

# ──────────────────────────────────────────────────────────────────────────

def apply_background_black(vol: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out everything outside the head mask."""
    out = vol.copy()
    out[~mask] = 0.0
    return out

def lps_indices_to_ras(i: int, j: int, k: int, shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convert LPS voxel indices to RAS index space (x,y flip; z unchanged)."""
    nx, ny, _ = shape
    return (nx - 1 - i, ny - 1 - j, k)
# ──────────────────────────────────────────────────────────────────────────
# IO and config
def load_volume(path: pathlib.Path) -> np.ndarray:
    """
    Load a 3-D scalar volume (.nii/.nii.gz or .npy) as float64, shape (nx,ny,nz).
    """
    s = path.suffix.lower()
    if s == ".npy":
        return np.load(path).astype(np.float64, copy=False)
    if s in {".nii", ".gz", ".nii.gz"}:
        if nib is None:
            raise RuntimeError("Reading NIfTI requires nibabel.")
        img = nib.load(str(path))
        img = nib.as_closest_canonical(img)  # force RAS axis order and polarity
        return np.asanyarray(img.get_fdata(), dtype=np.float64)

    raise ValueError(f"Unsupported file type: {path}")

@dataclass(frozen=True)
class CutawayConfig:
    """Parameters for cut-away rendering."""
    iso_level: float = 0.1         # iso for marching cubes on binary mask
    mesh_alpha: float = 1.0       # translucency for surface mesh
    mesh_color: Tuple[float, float, float] = (0.65, 0.65, 0.65) 
    slice_alpha: float = 0.95      # opacity for textured slice patches
    cmap: str = "gray"             # grayscale for slices
    min_margin: int = 1            # avoid first/last fully black slices

# ──────────────────────────────────────────────────────────────────────────
# Core geometry
def marching_surface_from_mask(mask: np.ndarray, iso: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Iso-surface from a binary head mask using marching cubes.

    Returns
    -------
    verts : (V,3) float array in voxel coordinates
    faces : (F,3) int array of triangle indices
    """
    verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=iso)
    return verts, faces

def drop_faces_in_first_octant(verts: np.ndarray, faces: np.ndarray,
                               i0: int, j0: int, k0: int) -> np.ndarray:
    tri = verts[faces]  # (F,3,3)
    # vertex-wise test
    v_in = (tri[..., 0] >= i0) & (tri[..., 1] >= j0) & (tri[..., 2] >= k0)
    # remove triangles with ANY vertex inside Ω
    keep = ~v_in.any(axis=1)
    return faces[keep]


# ──────────────────────────────────────────────────────────────────────────
# Slice patch helpers (restricted to Ω)
def _rgba_masked(img2d: np.ndarray,
                 mask2d: np.ndarray,
                 vmin: float, vmax: float,
                 cmap: str,
                 alpha_inside: float) -> np.ndarray:
    """
    Map intensities to RGBA and set alpha=0 where mask is False.
    """
    eps = np.finfo(float).eps
    norm = (img2d - vmin) / (max(vmax - vmin, eps))
    fc = plt.get_cmap(cmap)(np.clip(norm, 0, 1))     # shape (H,W,4)
    a = np.where(mask2d, alpha_inside, 0.0)          # per-pixel alpha
    fc[..., 3] = a
    return fc

def add_axial_patch(ax, vol, k, i0, j0, vmin, vmax, cfg, mask_3d):
    nx, ny, _ = vol.shape
    if not (0 <= k < vol.shape[2]): return
    patch = vol[i0:nx, j0:ny, k]
    m2d   = mask_3d[i0:nx, j0:ny, k]
    if not m2d.any(): return
    x_edges = np.arange(i0, nx + 1); y_edges = np.arange(j0, ny + 1)
    X, Y = np.meshgrid(x_edges, y_edges, indexing="ij")
    Z = np.full_like(X, k, dtype=float)
    fc = _rgba_masked(patch, m2d, vmin, vmax, cfg.cmap, cfg.slice_alpha)
    ax.plot_surface(X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1],
                    facecolors=fc, shade=False, rstride=1, cstride=1)

def add_coronal_patch(ax, vol, i, j0, k0, vmin, vmax, cfg, mask_3d):
    _, ny, nz = vol.shape
    if not (0 <= i < vol.shape[0]): return
    patch = vol[i, j0:ny, k0:nz]
    m2d   = mask_3d[i, j0:ny, k0:nz]
    if not m2d.any(): return
    y_edges = np.arange(j0, ny + 1); z_edges = np.arange(k0, nz + 1)
    Y, Z = np.meshgrid(y_edges, z_edges, indexing="ij")
    X = np.full_like(Y, i, dtype=float)
    fc = _rgba_masked(patch, m2d, vmin, vmax, cfg.cmap, cfg.slice_alpha)
    ax.plot_surface(X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1],
                    facecolors=fc, shade=False, rstride=1, cstride=1)

def add_sagittal_patch(ax, vol, j, i0, k0, vmin, vmax, cfg, mask_3d):
    nx, _, nz = vol.shape
    if not (0 <= j < vol.shape[1]): return
    patch = vol[i0:nx, j, k0:nz]
    m2d   = mask_3d[i0:nx, j, k0:nz]
    if not m2d.any(): return
    x_edges = np.arange(i0, nx + 1); z_edges = np.arange(k0, nz + 1)
    X, Z = np.meshgrid(x_edges, z_edges, indexing="ij")
    Y = np.full_like(X, j, dtype=float)
    fc = _rgba_masked(patch, m2d, vmin, vmax, cfg.cmap, cfg.slice_alpha)
    ax.plot_surface(X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1],
                    facecolors=fc, shade=False, rstride=1, cstride=1)


# ──────────────────────────────────────────────────────────────────────────
# Visualisation
def plot_cutaway_octant(
    vol: np.ndarray,
    k_axial: int, i_coronal: int, j_sagittal: int,
    *,
    cfg: CutawayConfig = CutawayConfig(),
    use_mask_extent: bool = True,
) -> plt.Figure:
    """
    Render the head surface with the first octant removed and draw the three
    intersecting slice patches restricted to that octant.

    Parameters
    ----------
    vol
        3-D HR magnitude volume, shape (nx,ny,nz).
    k_axial, i_coronal, j_sagittal
        Indices of the axial, coronal, sagittal planes.
    cfg
        Rendering parameters.
    use_mask_extent
        If True, crop the octant side length to the brain extent from the
        head mask to avoid empty borders.
    """
    log = logging.getLogger("cutaway")
    nx, ny, nz = vol.shape
    # guard against first/last empty slices
    k = int(np.clip(k_axial, cfg.min_margin, nz - 1 - cfg.min_margin))
    i = int(np.clip(i_coronal, cfg.min_margin, nx - 1 - cfg.min_margin))
    j = int(np.clip(j_sagittal, cfg.min_margin, ny - 1 - cfg.min_margin))


    # figure and equal aspect
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((nx, ny, nz))


    # head mask and background zeroing
    mask = compute_head_mask_from_hr(vol)
    vol = apply_background_black(vol, mask)

    # surface from mask  → smooth it
    verts, faces = marching_surface_from_mask(mask, cfg.iso_level)
    verts = laplacian_smooth(verts, faces, iterations=5, lam=0.45)

    # remove faces inside the octant
    faces_kept = drop_faces_in_first_octant(verts, faces, i, j, k)
    mesh = Poly3DCollection(verts[faces_kept], alpha=cfg.mesh_alpha, linewidths=0.2)
    mesh.set_zsort('min')
    mesh.set_facecolor(cfg.mesh_color)
    mesh.set_edgecolor((0, 0, 0))
    ax.add_collection3d(mesh)

    # percentile window from in-mask voxels
    inside = vol[mask]
    vmin, vmax = (np.percentile(inside, [1.0, 99.0]) if inside.size
                  else (float(vol.min()), float(vol.max())))

    add_axial_patch(ax, vol, k, i, j, vmin, vmax, cfg, mask)
    add_coronal_patch(ax, vol, i, j, k, vmin, vmax, cfg, mask)
    add_sagittal_patch(ax, vol, j, i, k, vmin, vmax, cfg, mask)


    # axes: crop to mask bbox if requested
    if use_mask_extent and mask.any():
        idx = np.argwhere(mask)
        (xmin, ymin, zmin), (xmax, ymax, zmax) = idx.min(0), idx.max(0)
        pad = 2
        ax.set_xlim(max(0, xmin - pad), min(vol.shape[0], xmax + pad))
        ax.set_ylim(max(0, ymin - pad), min(vol.shape[1], ymax + pad))
        ax.set_zlim(max(0, zmin - pad), min(vol.shape[2], zmax + pad))
    else:
        ax.set_xlim(0, nx); ax.set_ylim(0, ny); ax.set_zlim(0, nz)

    ax.set_xlabel("Δx anterior"); ax.set_ylabel("Δy right"); ax.set_zlabel("Δz cranial")
    ax.view_init(elev=22, azim=45)
    ax.set_axis_off()
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────────────────────────────────
# CLI
def parse_args() -> argparse.Namespace:
    """
    python src/mgmGrowth/tasks/superresolution/visualization/octant_surface.py /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/high_resolution/BraTS-MEN-00231-000/BraTS-MEN-00231-000-t1c.nii.gz 70 100 110 --save /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/octant_surface.png --no_mask_extent
    """
    p = argparse.ArgumentParser(description="Cut-away octant 3-D visualisation.")
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
                   help="Disable extent cropping by mask (keeps full cube).")
    p.add_argument("--save", type=pathlib.Path, help="Output image path")
    p.add_argument("--loglevel", default="INFO", help="Logging level")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.loglevel.upper(), logging.INFO),
                        format="[%(levelname)s] %(message)s")
    vol = load_volume(args.volume)
    cfg = CutawayConfig(mesh_alpha=args.alpha_mesh,
                        slice_alpha=args.alpha_slice,
                        cmap=args.cmap)
    if args.mesh_color is not None:
        c = args.mesh_color
        if c.startswith("#"):
            import matplotlib.colors as mcolors
            cfg = CutawayConfig(mesh_alpha=args.alpha_mesh,
                                slice_alpha=args.alpha_slice,
                                cmap=args.cmap,
                                mesh_color=mcolors.to_rgb(c))
        else:
            g = float(c); cfg = CutawayConfig(mesh_alpha=args.alpha_mesh,
                                            slice_alpha=args.alpha_slice,
                                            cmap=args.cmap,
                                            mesh_color=(g, g, g))
    # in main(), after vol = load_volume(...)
    i, j, k = args.coronal_i, args.sagittal_j, args.axial_k
    if args.index_space.lower() == "lps":
        i, j, k = lps_indices_to_ras(i, j, k, vol.shape)
    fig = plot_cutaway_octant(vol, k, i, j, cfg=cfg, use_mask_extent=not args.no_mask_extent)

    if args.save:
        fig.savefig(args.save, dpi=300, bbox_inches="tight")
    else:
        plt.show()

if __name__ == "__main__":
    main()