#!/usr/bin/env python3
"""
orthoslice_vis: Visualise three orthogonal slices of a 3-D medical image.

The module can be **imported** (use :pyfunc:`plot_orthogonal_slices`) or run as
a **script**::

    python orthoslice_vis.py volume.nii.gz 64 100 80 --cmap gray --alpha 0.9

All coordinates follow the RAS convention:
  +x → anterior   |  +y → right-lateral   |  +z → cranial
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Tuple, Final

import numpy as np
import matplotlib.pyplot as plt

# Registers the 3-D projection; the import name is unused afterwards.
from mpl_toolkits.mplot3d import Axes3D as _Axes3D  # noqa: F401

try:
    import nibabel as nib  # optional but recommended
except ModuleNotFoundError:  # pragma: no cover
    nib = None


# --------------------------------------------------------------------------- #
#                              Data handling                                  #
# --------------------------------------------------------------------------- #
def load_volume(path: pathlib.Path) -> np.ndarray:
    """
    Load a 3-D scalar volume from *path*.

    Parameters
    ----------
    path
        File path pointing to either a NIfTI (`.nii`/`.nii.gz`) or a NumPy
        array file (`.npy`).

    Returns
    -------
    ndarray
        Array of shape ``(nx, ny, nz)`` with ``float64`` dtype.

    Raises
    ------
    ValueError
        If `nibabel` is unavailable for NIfTI input or the file extension
        is unsupported.
    """
    suffix: str = path.suffix.lower()
    if suffix in {".npy"}:
        return np.load(path).astype(np.float64, copy=False)

    if suffix in {".nii", ".gz", ".nii.gz"}:
        if nib is None:
            raise ValueError("Reading NIfTI requires nibabel to be installed.")
        return np.asanyarray(nib.load(str(path)).get_fdata())

    raise ValueError(f"Unsupported volume type: {path}")


# --------------------------------------------------------------------------- #
#                             Plotting helpers                                #
# --------------------------------------------------------------------------- #
def _rgba(data2d: np.ndarray, vmin: float, vmax: float,
          cmap: str, alpha: float) -> np.ndarray:
    """Return an RGBA array with global alpha."""
    normed: np.ndarray = (data2d - vmin) / (vmax - vmin + np.finfo(float).eps)
    fc: np.ndarray = plt.get_cmap(cmap)(normed)
    fc[..., 3] = alpha
    return fc


def _plot_quadrants(ax: plt.Axes, plane_data: np.ndarray, plane: str,
                    fixed: int, split1: int, split2: int,
                    dims: Tuple[int, int, int],
                    vmin: float, vmax: float,
                    cmap: str, alpha: float) -> None:
    """
    Draw the four non-overlapping rectangles that make up one anatomical plane.

    Notes
    -----
    * `plane` selects which coordinate is held constant:
      ``'axial'`` = z, ``'coronal'`` = x, ``'sagittal'`` = y.
    * `split1` and `split2` are the indices where the other two planes cut
      through this one.
    """
    nx, ny, nz = dims

    if plane == "axial":                     # xy-plane at z = fixed
        data = plane_data                    # (nx, ny)
        for xs, xe in ((0, split1), (split1, nx)):
            if xe == xs:
                continue
            for ys, ye in ((0, split2), (split2, ny)):
                if ye == ys:
                    continue
                _patch_axial(
                    ax, data[xs:xe, ys:ye], xs, xe, ys, ye, fixed,
                    vmin, vmax, cmap, alpha
                )

    elif plane == "coronal":                 # yz-plane at x = fixed
        data = plane_data                    # (ny, nz)
        for ys, ye in ((0, split1), (split1, ny)):
            if ye == ys:
                continue
            for zs, ze in ((0, split2), (split2, nz)):
                if ze == zs:
                    continue
                _patch_coronal(
                    ax, data[ys:ye, zs:ze], ys, ye, zs, ze, fixed,
                    vmin, vmax, cmap, alpha
                )

    else:                                    # sagittal – xz-plane at y = fixed
        data = plane_data                    # (nx, nz)
        for xs, xe in ((0, split1), (split1, nx)):
            if xe == xs:
                continue
            for zs, ze in ((0, split2), (split2, nz)):
                if ze == zs:
                    continue
                _patch_sagittal(
                    ax, data[xs:xe, zs:ze], xs, xe, zs, ze, fixed,
                    vmin, vmax, cmap, alpha
                )

def _patch_axial(ax: plt.Axes, patch: np.ndarray,
                 xs: int, xe: int, ys: int, ye: int, z: int,
                 vmin: float, vmax: float, cmap: str, alpha: float) -> None:
    """Plot one axial sub-rectangle."""
    x_edges = np.arange(xs, xe + 1)
    y_edges = np.arange(ys, ye + 1)
    X, Y = np.meshgrid(x_edges, y_edges, indexing="ij")
    Z = np.full_like(X, z, dtype=float)
    fc = _rgba(patch, vmin, vmax, cmap, alpha)
    ax.plot_surface(X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1],
                    facecolors=fc, shade=False, rstride=1, cstride=1)


def _patch_coronal(ax: plt.Axes, patch: np.ndarray,
                   ys: int, ye: int, zs: int, ze: int, x: int,
                   vmin: float, vmax: float, cmap: str, alpha: float) -> None:
    """Plot one coronal sub-rectangle."""
    y_edges = np.arange(ys, ye + 1)
    z_edges = np.arange(zs, ze + 1)
    Y, Z = np.meshgrid(y_edges, z_edges, indexing="ij")
    X = np.full_like(Y, x, dtype=float)
    fc = _rgba(patch, vmin, vmax, cmap, alpha)
    ax.plot_surface(X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1],
                    facecolors=fc, shade=False, rstride=1, cstride=1)


def _patch_sagittal(ax: plt.Axes, patch: np.ndarray,
                    xs: int, xe: int, zs: int, ze: int, y: int,
                    vmin: float, vmax: float, cmap: str, alpha: float) -> None:
    """Plot one sagittal sub-rectangle."""
    x_edges = np.arange(xs, xe + 1)
    z_edges = np.arange(zs, ze + 1)
    X, Z = np.meshgrid(x_edges, z_edges, indexing="ij")
    Y = np.full_like(X, y, dtype=float)
    fc = _rgba(patch, vmin, vmax, cmap, alpha)
    ax.plot_surface(X[:-1, :-1], Y[:-1, :-1], Z[:-1, :-1],
                    facecolors=fc, shade=False, rstride=1, cstride=1)


# --------------------------------------------------------------------------- #
#                              Public plotting                                #
# --------------------------------------------------------------------------- #
def plot_orthogonal_slices(volume: np.ndarray,
                           slice_indices: Tuple[int, int, int],
                           *,
                           cmap: str = "gray",
                           alpha: float = 0.95,
                           save: pathlib.Path | None = None
                           ) -> plt.Figure:
    """
    Visualise the three orthogonal slices that intersect at *slice_indices*.

    Parameters
    ----------
    volume
        3-D image with shape ``(nx, ny, nz)``.
    slice_indices
        Triple ``(axial_k, coronal_i, sagittal_j)``.
    cmap
        Matplotlib colormap name shared by all slices.
    alpha
        Surface transparency in the range [0, 1].
    save
        If given, write the figure to this path; the format is
        deduced from the suffix.  If *None*, an interactive window pops up.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the 3-D axes.

    Notes
    -----
    * The function partitions each anatomical plane into four quadrants to
      prevent mutual occlusion at the shared lines of intersection.
    """
    if volume.ndim != 3:
        raise ValueError("`volume` must be 3-D.")
    nx, ny, nz = volume.shape
    k_ax, i_cor, j_sag = slice_indices
    if not (0 <= k_ax < nz and 0 <= i_cor < nx and 0 <= j_sag < ny):
        raise ValueError("Slice indices out of bounds.")

    vmin, vmax = float(volume.min()), float(volume.max())

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((nx, ny, nz))

    # axial
    _plot_quadrants(ax, volume[:, :, k_ax], "axial", k_ax,
                    i_cor, j_sag, (nx, ny, nz),
                    vmin, vmax, cmap, alpha)
    # coronal
    _plot_quadrants(ax, volume[i_cor, :, :], "coronal", i_cor,
                    j_sag, k_ax, (nx, ny, nz),
                    vmin, vmax, cmap, alpha)
    # sagittal
    _plot_quadrants(ax, volume[:, j_sag, :], "sagittal", j_sag,
                    i_cor, k_ax, (nx, ny, nz),
                    vmin, vmax, cmap, alpha)

    # bounding cube
    _draw_bounding_box(ax, nx, ny, nz)

    ax.set_xlabel("Anterior (+x)")
    ax.set_ylabel("Right-lateral (+y)")
    ax.set_zlabel("Cranial (+z)")
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.view_init(elev=30, azim=-45)
    plt.tight_layout()

    if save is not None:
        fig.savefig(save, dpi=300)
    else:
        plt.show()

    return fig


def _draw_bounding_box(ax: plt.Axes, nx: int, ny: int, nz: int) -> None:
    """Draw a thin wireframe around the data cube."""
    edges: Final = [
        ([0, 0, 0], [nx, 0, 0]),
        ([0, 0, 0], [0, ny, 0]),
        ([0, 0, 0], [0, 0, nz]),
        ([nx, ny, 0], [0, ny, 0]),
        ([nx, ny, 0], [nx, 0, 0]),
        ([nx, ny, 0], [nx, ny, nz]),
        ([nx, 0, nz], [0, 0, nz]),
        ([nx, 0, nz], [nx, ny, nz]),
        ([0, ny, nz], [0, 0, nz]),
        ([0, ny, nz], [nx, ny, nz]),
        ([0, 0, nz], [nx, 0, nz]),
        ([0, 0, 0], [0, ny, nz]),
    ]
    for s, e in edges:
        ax.plot3D(*zip(s, e), color="k", linewidth=0.4)


# --------------------------------------------------------------------------- #
#                                CLI                                          #
# --------------------------------------------------------------------------- #
def _parse_cli() -> argparse.Namespace:
    """Set up command-line interface."""
    p = argparse.ArgumentParser(
        description="Visualise axial, coronal & sagittal slices of a "
                    "3-D medical image."
    )
    p.add_argument("volume",
                   type=pathlib.Path,
                   help="Path to .nii/.nii.gz or .npy 3-D volume.")
    p.add_argument("axial_k", type=int,
                   help="Axial (z) slice index.")
    p.add_argument("coronal_i", type=int,
                   help="Coronal (x) slice index.")
    p.add_argument("sagittal_j", type=int,
                   help="Sagittal (y) slice index.")
    p.add_argument("--cmap", default="gray",
                   help="Matplotlib colormap (default: gray).")
    p.add_argument("--alpha", default=0.95, type=float,
                   help="Slice transparency in [0,1] (default: 0.95).")
    p.add_argument("--save", type=pathlib.Path,
                   help="If given, write the figure to this file.")
    return p.parse_args()


def main() -> None:  # pragma: no cover
    """Entry point for ``python orthoslice_vis.py ...``."""
    args = _parse_cli()
    vol = load_volume(args.volume)
    plot_orthogonal_slices(
        vol,
        (args.axial_k, args.coronal_i, args.sagittal_j),
        cmap=args.cmap,
        alpha=args.alpha,
        save=args.save,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
