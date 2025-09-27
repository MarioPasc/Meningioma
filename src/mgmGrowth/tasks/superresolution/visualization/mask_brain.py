"""
Lorensen WE, Cline HE. “Marching Cubes: A High Resolution 3D Surface Construction Algorithm,” SIGGRAPH 1987.

Vincent L, Soille P. “Watersheds in Digital Spaces,” IEEE TPAMI 1991.

Desbrun M et al. “Implicit Fairing of Irregular Meshes using Diffusion and Curvature Flow,” SIGGRAPH 1999.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

# new imports
from scipy.ndimage import (binary_fill_holes, binary_dilation, binary_erosion,
                           label, generate_binary_structure, binary_propagation)
from skimage.filters import sobel
from skimage.segmentation import watershed
from typing import Iterable
import numpy as np

def laplacian_smooth(verts: np.ndarray, faces: np.ndarray,
                     iterations: int = 5, lam: float = 0.5) -> np.ndarray:
    """
    Umbrella-operator Laplacian smoothing on a manifold triangle mesh.

    Parameters
    ----------
    verts : (V,3) float
    faces : (F,3) int
    iterations : number of smoothing passes
    lam : step size in [0,1]; small values preserve detail

    Returns
    -------
    (V,3) float : smoothed vertices (does not modify input)
    """
    V = verts.copy()
    # build adjacency list
    neighbors: list[set[int]] = [set() for _ in range(len(V))]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        neighbors[a].update((b, c)); neighbors[b].update((a, c)); neighbors[c].update((a, b))
    for _ in range(max(0, iterations)):
        Vn = V.copy()
        for i, nb in enumerate(neighbors):
            if not nb: 
                continue
            mean_nb = V[np.fromiter(nb, dtype=int)].mean(axis=0)
            Vn[i] = V[i] + lam * (mean_nb - V[i])
        V = Vn
    return V


@dataclass(frozen=True)
class _MaskParams:
    """Internal parameters for robust head–air separation."""
    air_p_low: float = 1.0       # seed air at very dark intensities
    air_p_high: float = 20.0     # permit flood-fill through 'dark-enough' voxels
    erode_vox: int = 1           # erode head seed before watershed
    close_iters: int = 1         # final morphological smoothing
    connectivity: int = 2        # 6/18/26-connectivity selector

def _normalize01(v: np.ndarray) -> np.ndarray:
    """Percentile-based robust [0,1] rescale to reduce bias-field effects."""
    lo, hi = np.percentile(v[np.isfinite(v)], [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        v = v - np.nanmin(v); hi = np.nanmax(v); lo = 0.0
    return np.clip((v - lo) / (hi - lo + np.finfo(float).eps), 0.0, 1.0)

def compute_head_mask_from_hr(hr_vol_LPS: np.ndarray) -> np.ndarray:
    """
    Robust head/background mask.

    Pipeline:
      1) Robustly normalize intensities to [0,1].
      2) Seed outside air on the volume border at very low intensities.
      3) Flood-fill air through 'dark-enough' voxels to label exterior.
      4) Head = complement of exterior; keep largest 3D component.
      5) Edge refinement: watershed on the gradient with air vs. eroded-head markers.
      6) Fill holes and lightly close gaps.

    Returns
    -------
    mask : np.ndarray of bool
        True for head/skull/brain, False for air/background.
    """
    p = _MaskParams()
    v = np.asanyarray(hr_vol_LPS, dtype=np.float32)
    v[~np.isfinite(v)] = 0.0
    v = _normalize01(v)

    # 2) border air seeds
    air_thr_seed = np.percentile(v, p.air_p_low)
    air_thr_pass = np.percentile(v, p.air_p_high)
    border = np.zeros_like(v, dtype=bool)
    border[[0, -1], :, :] = True
    border[:, [0, -1], :] = True
    border[:, :, [0, -1]] = True
    air_seed = border & (v <= air_thr_seed)

    # 3) flood-fill exterior through dark voxels
    st = generate_binary_structure(3, p.connectivity)
    dark = v <= air_thr_pass
    exterior = binary_propagation(air_seed, mask=dark, structure=st)

    # 4) raw head and largest CC
    head0 = ~exterior
    labels, nlab = label(head0.astype(np.uint8), structure=st)
    if nlab == 0:
        return np.zeros_like(head0, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    head_lcc = labels == int(counts.argmax())

    # 5) watershed edge snap
    if p.erode_vox > 0:
        head_marker = binary_erosion(head_lcc, structure=st, iterations=p.erode_vox)
    else:
        head_marker = head_lcc
    markers = np.zeros_like(labels, dtype=np.int32)
    markers[exterior] = 1
    markers[head_marker] = 2
    grad = sobel(v)  # cheap isotropic gradient magnitude
    w = watershed(grad, markers=markers, connectivity=1, mask=(exterior | head_lcc))
    mask = w == 2

    # 6) fill and light closing
    mask = binary_fill_holes(mask)
    if p.close_iters > 0:
        mask = binary_dilation(mask, structure=st, iterations=p.close_iters)
        mask = binary_erosion(mask,  structure=st, iterations=p.close_iters)
    return mask
