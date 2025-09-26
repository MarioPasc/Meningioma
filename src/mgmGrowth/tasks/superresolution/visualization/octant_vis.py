#!/usr/bin/env python3
"""
make_qualitative_grid.py  —  HR vs SR octant views
"""

from __future__ import annotations
import pathlib, nibabel as nib
import mgmGrowth.tasks.superresolution.visualization.octant as oc
from mgmGrowth.tasks.superresolution.utils.imio import load_lps
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
# ───────────────────── user configuration ──────────────────────────────
PATIENT   = "BraTS-MEN-00231-000"
COORDS    = (65, 120, 135)                 # (x, y, z) slice indices
PULSES    = ("t1c", "t1n", "t2w", "t2f")
RES_MM    = (3, 5, 7)
MODELS    = ("BSPLINE", "SMORE", "UNIRES", "ECLARE")           # add more if available
# Where to save screenshots (same tree as metric plots)
OUT_ROOT  = pathlib.Path(
    "/media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/figures/octant"
)
FORMAT = "pdf"

# roots for data
HR_ROOT   = pathlib.Path(
    "/media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/high_resolution")
MODEL_ROOT = pathlib.Path(
    "/media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/models")

# ───────────────────── path helpers ────────────────────────────────────
def hr_path(pulse: str) -> pathlib.Path:
    return (HR_ROOT / PATIENT / f"{PATIENT}-{pulse}.nii.gz")

def sr_path(model: str, res: int, pulse: str) -> pathlib.Path:
    return (MODEL_ROOT / model / f"{res}mm" / "output_volumes" /
            f"{PATIENT}-{pulse}.nii.gz")

def seg_path() -> pathlib.Path:
    return HR_ROOT / PATIENT / f"{PATIENT}-seg.nii.gz"



def pad_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (a′, b′) where both arrays have identical shape.
    The smaller one is padded with zeros *at the trailing end*
    of each mismatched axis.

    Assumes size difference ≤ 2 voxels per axis.
    """
    if a.shape == b.shape:
        return a, b

    target = tuple(max(sa, sb) for sa, sb in zip(a.shape, b.shape))

    def _pad(arr, name):
        pad_width = []
        for s, t in zip(arr.shape, target):
            if s == t:
                pad_width.append((0, 0))
            elif s < t:
                pad_width.append((0, t - s))    # pad after last slice
            else:
                raise ValueError(f"{name} is larger than target along an axis")
        return np.pad(arr, pad_width, mode="constant", constant_values=0)

    warnings.warn("Volume/seg mismatch — zero-padding the smaller array")
    return _pad(a, "a"), _pad(b, "b")


def save_residual_octant(hr_path: pathlib.Path,
                         sr_path: pathlib.Path,
                         dst: pathlib.Path) -> None:
    hr_img = nib.load(str(hr_path))                 # keep nib image for geometry
    hr = load_lps(hr_path)                          # LPS data
    sr = load_lps(sr_path, like=hr_img, order=1)    # align & interpolate

    hr, sr = pad_to_match(hr, sr)      # ← keep *both* padded arrays
    res    = np.abs(sr - hr)           # now shapes match

    oc.plot_octant(
        res, COORDS,
        segmentation=None,
        cmap="magma",
        only_line=False,
        xticks=[], yticks=[], zticks=[], grid=False,
        figsize=(7, 7),
        save=Path(dst),
    )
    

# ─────────────────── patch save_octant ────────────────
def save_octant(src: pathlib.Path, dst: pathlib.Path) -> None:
    hr_img = nib.load(str(hr_path(PULSES[0])))      # any HR pulse as reference
    vol = load_lps(src,  like=hr_img, order=1)
    seg = load_lps(seg_path(), like=hr_img, order=0)

    vol, seg = pad_to_match(vol, seg)      # ← NEW line

    oc.plot_octant(
        vol, COORDS,
        segmentation=seg,
        only_line=True, seg_alpha=0.35,
        cmap="gray", figsize=(7, 7),
        xticks=[], yticks=[], zticks=[], grid=False,
        xlabel="Anterior (+x)", ylabel="Right (+y)", zlabel="Cranial (+z)",
        save=Path(dst)
    )

# ───────────────────── generate screenshots ────────────────────────────
for pulse in PULSES:
    # ground-truth once per pulse (use 3-mm folder for convenience)
    gt_png = OUT_ROOT / pulse / "3mm" / f"{pulse}_3mm_HR.{FORMAT}"
    gt_png.parent.mkdir(parents=True, exist_ok=True)
    save_octant(hr_path(pulse), gt_png)

    for res in RES_MM:
        # copy/rename the GT screenshot for other rows so LaTeX paths exist
        for other_png in (OUT_ROOT / pulse / f"{res}mm" /
                          f"{pulse}_{res}mm_HR.{FORMAT}",):
            other_png.parent.mkdir(parents=True, exist_ok=True)
            if not other_png.exists():
                other_png.symlink_to(gt_png)   # or shutil.copy if FS disallows links

        for model in MODELS:
            src_nii = sr_path(model, res, pulse)
            dst_png = (OUT_ROOT / pulse / f"{res}mm" /
                       f"{pulse}_{res}mm_{model}.{FORMAT}")
            dst_png.parent.mkdir(parents=True, exist_ok=True)
            save_octant(src_nii, dst_png)
            print("saved", dst_png)
            # save residual octant
            res_png = (OUT_ROOT / pulse / f"{res}mm" /
                    f"{pulse}_{res}mm_{model}_RES.{FORMAT}")
            save_residual_octant(hr_path(pulse), src_nii, res_png)
            plt.close("all")          # close all figures
            print("saved", res_png)


# create a stand-alone horizontal colour-bar 0‒max residual
cmap_name = "magma"          # or any perceptually uniform map you prefer
max_val   = 100               # adjust if needed

fig_cb, ax_cb = plt.subplots(figsize=(6, 0.5))
norm = mpl.colors.Normalize(vmin=0, vmax=max_val)
mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap_name, norm=norm,
                          orientation="horizontal")
ax_cb.set_xlabel("|SR − HR| (intensity)")
fig_cb.savefig(OUT_ROOT / f"residual_cbar.{FORMAT}",
               dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig_cb)
print("saved residual colour-bar → residual_cbar.{FORMAT}")

roi_display = {
    "#EE6677": "Enhancing Tumor",
    "#228833": "Edema",
    "#4477AA": "Non-Enhancing Tumor",
}

handles = [mpl.patches.Patch(facecolor=hexc, edgecolor="black", label=lab)
           for hexc, lab in roi_display.items()]

fig_leg, ax_leg = plt.subplots(figsize=(6, 0.8))
ax_leg.legend(handles=handles, loc="center", ncol=len(handles),
              frameon=False, handlelength=1.5, fontsize=18)
plt.axis("off")
fig_leg.savefig(OUT_ROOT / f"roi_legend_octant.{FORMAT}",
                dpi=300, bbox_inches="tight", transparent=True)
plt.close(fig_leg)
print(f"saved ROI legend → roi_legend_octant.{FORMAT}")
