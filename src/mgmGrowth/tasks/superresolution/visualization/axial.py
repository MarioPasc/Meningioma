#!/usr/bin/env python3
"""
3D axial-slice triplet with background stripped.

Reads NIfTI, reorients to canonical RAS, builds a VTK ImageData with
point-data scalars, computes a head mask using
`src.mgmGrowth.tasks.superresolution.mask_brain.compute_head_mask_from_hr`,
slices both intensity and mask at z = k, k+Xmm, k+Ymm, thresholds by the mask,
and renders the three planes in 3D.

CLI:
  python axial_triplet.py vol.nii.gz --k 120 --x_mm 3 --y_mm 12 --save out.png
"""
from __future__ import annotations
import argparse
import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import nibabel as nib
import pyvista as pv

# import the provided mask function (project path first, fallbacks allowed)
try:
    from src.mgmGrowth.tasks.superresolution.mask_brain import compute_head_mask_from_hr
except Exception:
    try:
        from src.mgmgGrowth.tasks.superresolution.mask_brain import compute_head_mask_from_hr  # user typo path
    except Exception:
        from mask_brain import compute_head_mask_from_hr  # local fallback

LOG = logging.getLogger("axial_triplet")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass(frozen=True)
class Vol3D:
    """RAS-oriented volume container."""
    data: np.ndarray                      # float32, shape (X,Y,Z)
    spacing: Tuple[float, float, float]   # (dx,dy,dz) mm
    extent_mm: Tuple[float, float, float] # physical size in mm


def load_as_ras(path: str) -> Vol3D:
    """Load NIfTI, convert to canonical RAS, return data and spacing."""
    img = nib.load(path)
    img = nib.as_closest_canonical(img)
    vol = img.get_fdata(dtype=np.float32)
    dx, dy, dz = img.header.get_zooms()[:3]
    sx, sy, sz = vol.shape[0]*dx, vol.shape[1]*dy, vol.shape[2]*dz
    LOG.info("Loaded %s | shape=%s | spacing=(%.3f,%.3f,%.3f)mm", path, vol.shape, dx, dy, dz)
    return Vol3D(vol, (dx, dy, dz), (sx, sy, sz))


def compute_head_mask_ras(vol_ras: np.ndarray) -> np.ndarray:
    """
    Run the provided head/air mask, which expects LPS. Convert RAS↔LPS by flipping x,y.
    Returns a boolean mask in RAS orientation.
    """
    v_lps = vol_ras[::-1, ::-1, :]  # RAS → LPS
    m_lps = compute_head_mask_from_hr(v_lps)
    m_ras = m_lps[::-1, ::-1, :]    # LPS → RAS
    return m_ras.astype(bool)


def make_image_grid_point(vol: Vol3D, mask: np.ndarray) -> pv.ImageData:
    """
    Build a pyvista ImageData with point-data scalars 'intensity' and 'mask'.
    Using point-data ensures the slice inherits both arrays.
    """
    nx, ny, nz = vol.data.shape
    dx, dy, dz = vol.spacing

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)   # points grid
    grid.spacing    = (dx, dy, dz)
    grid.origin     = (0.0, 0.0, 0.0)

    grid.point_data["intensity"] = vol.data.ravel(order="F")
    grid.point_data["mask"]      = mask.astype(np.float32).ravel(order="F")
    return grid


def axial_slice_thresholded(grid: pv.ImageData, z_mm: float, name: str) -> pv.PolyData | None:
    """
    Slice at physical Z and keep only cells with mask >= 0.5.
    Returns PolyData or None if empty after threshold.
    """
    xmin, xmax, ymin, ymax, _, _ = grid.bounds
    cx = 0.5*(xmin+xmax); cy = 0.5*(ymin+ymax)
    sl = grid.slice(normal=(0,0,1), origin=(cx, cy, z_mm))
    if "mask" not in sl.point_data:
        return None
    sl = sl.threshold(value=0.5, scalars="mask", invert=False)
    return sl if sl.n_points > 3 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("nii", type=str)
    ap.add_argument("--k", type=int, required=True, help="Axial slice index (0-based) in canonical RAS")
    ap.add_argument("--x_mm", type=float, required=True, help="Offset in mm from k for 2nd slice")
    ap.add_argument("--y_mm", type=float, required=True, help="Offset in mm from k for 3rd slice")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--lw", type=float, default=0.0, help="outline line width")
    ap.add_argument("--alpha", type=float, default=1.0, help="slice opacity")
    args = ap.parse_args()

    vol = load_as_ras(args.nii)
    nx, ny, nz = vol.data.shape
    dx, dy, dz = vol.spacing
    sx, sy, sz = vol.extent_mm

    if not (0 <= args.k < nz):
        raise ValueError(f"k out of range [0,{nz-1}]")

    # z positions at point grid: z = k*dz
    z0 = float(args.k * dz)
    z1 = float(np.clip(z0 + max(args.x_mm, 0.0), 0.0, (nz-1)*dz))
    z2 = float(np.clip(z0 + args.y_mm, 0.0, (nz-1)*dz))

    # background stripping
    mask = compute_head_mask_ras(vol.data)
    grid = make_image_grid_point(vol, mask)

    # intensity display range from masked voxels
    masked_vals = vol.data[mask]
    vmin, vmax = np.percentile(masked_vals, (2.0, 98.0)) if masked_vals.size else (np.min(vol.data), np.max(vol.data))

    p = pv.Plotter(window_size=(1000, 800), off_screen=bool(args.save and args.no_show))
    p.set_background("white")
    #p.add_mesh(grid.outline(), color="black", line_width=args.lw)

    LOG.info(f"Slices selected: z0={z0:.1f}mm (k={args.k}), z1={z1:.1f}mm, z2={z2:.1f}mm")
    for z_mm, label in [(z0, "k"), (z1, "k+Xmm"), (z2, "k+Ymm")]:
        sl = axial_slice_thresholded(grid, z_mm, label)
        if sl is None:
            LOG.warning("Empty slice at z=%.3f mm after masking", z_mm)
            continue
        p.add_mesh(
            sl, scalars="intensity", cmap="gray", clim=(vmin, vmax),
            opacity=args.alpha, show_scalar_bar=False, name=label
        )

    # midpoint "..." text to suggest stacking
    text = pv.Text3D("...", depth=0.1*dz)
    scale = 0.08 * max(sx, sy)
    text.scale([scale, scale, scale])
    text.translate([0.5*sx - 0.02*sx, 0.5*sy, 0.5*(z0+z2)])
    p.add_mesh(text, color="black")

    # camera close to the reference orientation
    p.camera_position = [
        (1.6*sx,  1.6*sy, 1.2*sz-30),   # position: +X, +Y, +Z
        (0.5*sx,  0.5*sy, 0.5*sz),   # focal point: volume center
        (0, 0, 1),                   # view-up: +Z keeps Superior up
    ]
    LOG.info("Camera pos=%s | focal=%s", p.camera.position, p.camera.focal_point)
    p.camera.SetViewAngle(25)
    if args.save:
        p.save_graphic(str(args.save).replace(".png", ".pdf"))
        LOG.info("Saved -> %s", args.save)
    if not args.no_show:
        p.show()


if __name__ == "__main__":
    main()
