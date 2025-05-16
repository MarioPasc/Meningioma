#!/usr/bin/env python3
"""
octant.py – zoom-in on the anterior–right-cranial octant of three
orthogonal slices, with optional label-map overlay.

Label map semantics
-------------------
    0 : background        → no overlay
    1 : enhancing tumour  → red     (RGB = 1,0,0)
    2 : oedema            → green   (0,1,0)
    3 : tumour core rim   → blue    (0,0,1)
"""

from __future__ import annotations
import pathlib
from typing import Tuple, Final, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as _Axes3D   # noqa: F401 – 3-D proj


# ───────────────────────────────── helpers ─────────────────────────────── #
DEBUG_SHAPES = False          # set True only while debugging


def _rgba(data2d: np.ndarray,
          vmin: float, vmax: float,
          cmap: str, alpha: float) -> np.ndarray:
    """Greyscale → RGBA with global *alpha*."""
    normed = (data2d - vmin) / (vmax - vmin + np.finfo(float).eps)
    fc = plt.get_cmap(cmap)(normed)
    fc[..., 3] = alpha
    return fc


# ------------------------------------------------------------------ NEW helper
def _tint_segmentation(
        fc: np.ndarray,
        seg_patch: np.ndarray | None,
        *,
        seg_alpha: float,
        lut: dict[int, tuple[float, float, float]],
) -> np.ndarray:
    """
    Blend a translucent colour tint over the greyscale faces.

    Parameters
    ----------
    fc
        Base RGBA face-colour array (H, W, 4).
    seg_patch
        2-D label map aligned with *fc*; may be ``None``.
    seg_alpha
        Opacity of the tint (0 = invisible, 1 = fully opaque).
    lut
        Mapping {label: (r, g, b)} in the 0-1 range.

    Returns
    -------
    np.ndarray
        New face-colour array with the tint blended in.
    """
    if seg_patch is None:
        return fc

    out = fc.copy()
    for lbl, rgb in lut.items():
        if lbl == 0:          # skip background (air / brain tissue)
            continue
        mask = seg_patch == lbl
        if not mask.any():
            continue
        # α-blend only where mask is true
        out[mask, :3] = (
            (1.0 - seg_alpha) * out[mask, :3] +
            seg_alpha * np.asarray(rgb, dtype=float)
        )
        # out[mask, 3]   – leave existing alpha (greyscale) unchanged
    return out

def _brain_extent(octant: np.ndarray) -> int:
    """
    Length (voxels) of the smallest dimension that still contains
    **brain** voxels (`octant != 0`).

    Returns
    -------
    int
        L = min(ext_x, ext_y, ext_z); guaranteed > 0.
    """
    if not np.any(octant):
        raise ValueError("Selected octant contains no non-zero voxels.")
    coords = np.array(np.where(octant <= 20))
    ext_x = coords[0].max() + 1
    ext_y = coords[1].max() + 1
    ext_z = coords[2].max() + 1
    return int(min(ext_x, ext_y, ext_z))

# ───────────────────────────────────────────────────────────────────────── #


def _plot_single_patch(ax: plt.Axes,
                       patch: np.ndarray,
                       X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
                       vmin: float, vmax: float,
                       cmap: str, alpha: float,
                       *,
                       plane: str,
                       seg_patch: Optional[np.ndarray] = None,   # NEW
                       seg_alpha: float = 0.5):                 # NEW
    """Plot one rectangle at full resolution (stride = 1)."""
    fc = _rgba(patch, vmin, vmax, cmap, alpha)

    # NEW – blend segmentation overlay (if any)
    fc = _tint_segmentation(
            fc,
            seg_patch,
            seg_alpha=seg_alpha,
            lut={0: (0,0,0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1)},
    )

    Xf, Yf, Zf = X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1]

    if DEBUG_SHAPES:
        print(f"[{plane.upper()}]  patch {patch.shape}  grid→{Xf.shape}  "
              f"FC {fc.shape}")
        if Xf.shape != fc.shape[:2]:
            print("!! shape mismatch !!")

    ax.plot_surface(Xf, Yf, Zf,
                    facecolors=fc,
                    shade=False, rstride=1, cstride=1)


# ─────────────────────────────── main API ──────────────────────────────── #
def plot_octant(volume: np.ndarray,
                slice_indices: Tuple[int, int, int],
                *,
                cmap: str = "gray",
                alpha: float = 0.95,
                segmentation: Optional[np.ndarray] = None,  # NEW
                seg_alpha: float = 0.5,                    # NEW
                save: pathlib.Path | None = None) -> plt.Figure:
    """
    Zoom visualisation of the octant x≥i_c, y≥j_s, z≥k_a with optional labels.

    Parameters
    ----------
    volume
        Main image array (nx, ny, nz).
    slice_indices
        (k_axial, i_coronal, j_sagittal) — intersection voxel.
    cmap, alpha
        Greyscale colormap and transparency for the base image.
    segmentation
        Optional label map **aligned** with *volume*.  Values:
        0 background · 1 red · 2 green · 3 blue.
    seg_alpha
        Alpha value used for the coloured overlay (default 0.5).
    save
        If given, write figure to file; else show interactively.
    """
    if volume.ndim != 3:
        raise ValueError("`volume` must be 3-D.")
    if (segmentation is not None) and (segmentation.shape != volume.shape):
        raise ValueError("`segmentation` shape must match `volume`.")

    nx, ny, nz = volume.shape
    k_a, i_c, j_s = slice_indices
    if not (0 <= k_a < nz and 0 <= i_c < nx and 0 <= j_s < ny):
        raise ValueError("slice indices out of bounds")

    dx, dy, dz = nx - i_c, ny - j_s, nz - k_a
    if min(dx, dy, dz) == 0:
        raise ValueError("Chosen voxel lies on a boundary – octant empty.")

    vmin, vmax = float(volume.min()), float(volume.max())

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((dx, dy, dz))

    # ------------------ compute isotropic edge length ----------------------
    vol_oct = volume[i_c:nx, j_s:ny, k_a:nz]
    L = _brain_extent(vol_oct)          # the cube edge we’ll display
    

    # crop the octant data cube to L×L×L
    dx = dy = dz = L                    # replaces previous dx,dy,dz


    # ------------------ axial (xy @ z=0) -----------------------------------
    patch_ax = vol_oct[:dx, :dy, 0]                 # (dx, dy)
    seg_ax   = (segmentation[i_c:i_c+dx, j_s:j_s+dy, k_a]
                if segmentation is not None else None)

    x_edges = np.arange(dx + 1)
    y_edges = np.arange(dy + 1)
    X_ax, Y_ax = np.meshgrid(x_edges, y_edges, indexing="ij")
    Z_ax = np.zeros_like(X_ax)

    _plot_single_patch(ax, patch_ax,
                       X_ax, Y_ax, Z_ax,
                       vmin, vmax, cmap, alpha,
                       plane="axial",
                       seg_patch=seg_ax, seg_alpha=seg_alpha)

    # ------------------ coronal (yz @ x=0) ---------------------------------
    patch_co = vol_oct[0, :dy, :dz]               # (dz, dy)
    seg_co   = (segmentation[i_c, j_s:j_s+dy, k_a:k_a+dz]
                if segmentation is not None else None)

    y_edges = np.arange(dy + 1)
    z_edges = np.arange(dz + 1)
    Y_co, Z_co = np.meshgrid(y_edges, z_edges, indexing="ij")
    X_co = np.zeros_like(Y_co)

    _plot_single_patch(ax, patch_co,
                       X_co, Y_co, Z_co,
                       vmin, vmax, cmap, alpha,
                       plane="coronal",
                       seg_patch=seg_co, seg_alpha=seg_alpha)

    # ------------------ sagittal (xz @ y=0) --------------------------------
    patch_sa = vol_oct[:dx, 0, :dz]                 # (dx, dz)
    seg_sa   = (segmentation[i_c:i_c+dx, j_s, k_a:k_a+dz]
                if segmentation is not None else None)

    x_edges = np.arange(dx + 1)
    z_edges = np.arange(dz + 1)
    X_sa, Z_sa = np.meshgrid(x_edges, z_edges, indexing="ij")
    Y_sa = np.zeros_like(X_sa)

    _plot_single_patch(ax, patch_sa,
                       X_sa, Y_sa, Z_sa,
                       vmin, vmax, cmap, alpha,
                       plane="sagittal",
                       seg_patch=seg_sa, seg_alpha=seg_alpha)

    # ------------------ isotropic limits & cube ----------------------------
    _cube_wireframe(ax, dx, dy, dz)     # still draws the box

    ax.set_xlabel("Δx (vox anterior)")
    ax.set_ylabel("Δy (vox right-lat.)")
    ax.set_zlabel("Δz (vox cranial)")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    ax.set_box_aspect((1, 1, 1))        # isotropic
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=300)
    else:
        plt.show()
    return fig

def _cube_wireframe(ax: plt.Axes, dx: int, dy: int, dz: int) -> None:
    """Thin wireframe around the octant."""
    edges: Final = [
        ([0, 0, 0], [dx, 0, 0]),
        ([0, 0, 0], [0, dy, 0]),
        ([0, 0, 0], [0, 0, dz]),
        ([dx, dy, 0], [0, dy, 0]),
        ([dx, dy, 0], [dx, 0, 0]),
        ([dx, dy, 0], [dx, dy, dz]),
        ([dx, 0, dz], [0, 0, dz]),
        ([dx, 0, dz], [dx, dy, dz]),
        ([0, dy, dz], [0, 0, dz]),
        ([0, dy, dz], [dx, dy, dz]),
        ([0, 0, dz], [dx, 0, dz]),
        ([0, 0, 0], [0, dy, dz]),
    ]
    for s, e in edges:
        ax.plot3D(*zip(s, e), color="k", linewidth=0.4)


# test-block
if __name__ == "__main__":
    vol = np.random.randn(128, 128, 100)
    seg = np.zeros_like(vol, dtype=int)
    seg[90:, 90:, 60:] = 1
    seg[90:, 50:80, 60:] = 2
    seg[60:80, 90:, 60:] = 3

    plot_octant(vol, (70, 80, 90),
                segmentation=seg,
                cmap="gray",
                seg_alpha=0.4)
