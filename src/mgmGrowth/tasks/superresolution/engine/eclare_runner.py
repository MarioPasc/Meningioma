#!/usr/bin/env python3
"""
eclare_runner.py
================

Light-weight subprocess wrappers around the **ECLARE** command-line tools.

∙ Training is implicit in *run-eclare* (ECLARE is self-supervised) but separate
  wrappers for ``eclare-train``/``eclare-test`` are provided so that external
  pipelines that expected the SMORE API continue to work unchanged.

Author  :  Mario Pascual González
Created : 2025-07-06
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional, Protocol, Sequence

from mgmGrowth.tasks.superresolution import LOGGER as _L  # global project logger
from src.mgmGrowth.tasks.superresolution.tools.paths import ensure_dir



class _CfgProtocol(Protocol):
    """Duck-type protocol for the parts of a config object we need."""

    gpu_id: int          # Which GPU to use
    patch_size: int      # Network receptive field (ECLARE ignores but kept)
    batch_size: int      # Mini-batch size            ( 〃 )
    n_blocks: int        # Number of RRDB blocks      ( 〃 )
    n_channels: int      # Number of feature maps     ( 〃 )
    n_patches: int       # #Training patches          ( 〃 )
    n_rots: int          # #Rotations in inference    ( 〃 )



def _run(cmd: Sequence[str]) -> None:
    """Utility: run *cmd* with logging & error bubbling."""
    _L.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)  # ⇐ raises CalledProcessError on failure


def train_volume(
    lr_path: Path,
    cfg: _CfgProtocol,
    weight_dir: Path,
    *,
    relative_slice_thickness: float | None = None,
    blur_kernel: Path | None = None,
) -> Path:
    """
    Dummy wrapper around **eclare-train**.

    ECLARE performs self-supervised training inside *run-eclare*.  This helper
    is provided only for API compatibility; it merely shells out to *eclare-train*
    so that you can still store intermediate weights for inspection.
    """
    ensure_dir(weight_dir)
    _run(
        [
            "eclare-train",
            "--in-fpath",
            str(lr_path),
            "--weight-dir",
            str(weight_dir),
            "--gpu-id",
            str(cfg.gpu_id),
            *(
                ["--relative-slice-thickness", str(relative_slice_thickness)]
                if relative_slice_thickness is not None
                else []
            ),
            *(
                ["--relative-slice-profile-fpath", str(blur_kernel)]
                if blur_kernel
                else []
            ),
        ]
    )
    return weight_dir


def infer_volume(
    lr_path: Path,
    weights: Path,
    cfg: _CfgProtocol,
    out_path: Path,
    *,
    n_rots: int | None = None,
) -> None:
    """Run **eclare-test** given a weight checkpoint produced above."""
    ensure_dir(out_path.parent)
    _run(
        [
            "eclare-test",
            "--in-fpath",
            str(lr_path),
            "--out-fpath",
            str(out_path),
            "--gpu-id",
            str(cfg.gpu_id),
            "--weight-dir",
            str(weights),
            *(
                ["--n-rots", str(n_rots if n_rots is not None else cfg.n_rots)]
            ),
        ]
    )



def run_eclare(
    lr_path: Path,
    out_dir: Path,
    *,
    cfg: Optional[_CfgProtocol] = None,
    relative_slice_thickness: float | None = None,
    blur_kernel: Optional[Path] = None,
    gpu_id: int | None = None,
    suffix: str = "_eclare",
    slice_profile_type: str | None = None,
) -> tuple[Path, Path]:
    """
    End-to-end *single-volume* super-resolution using **run-eclare**.

    Returns
    -------
    model_state_dir : Path
        Directory containing the learned model state (for repro/debug).
    sr_path : Path
        Path to the generated super-resolved NIfTI volume.
    """
    ensure_dir(out_dir)
    gid = gpu_id if gpu_id is not None else (cfg.gpu_id if cfg else 0)

    cmd = [
        "run-eclare",
        "--in-fpath",
        str(lr_path),
        "--out-dir",
        str(out_dir),
        "--gpu-id",
        str(gid),
    ]

    if relative_slice_thickness is not None:
        cmd += ["--relative-slice-thickness", str(relative_slice_thickness)]

    if blur_kernel:
        cmd += ["--relative-slice-profile-fpath", str(blur_kernel)]

    if slice_profile_type:
        cmd += ["--relative-slice-profile-type", slice_profile_type]

    if suffix:
        cmd += ["--suffix", suffix]

    _run(cmd)

    # ---------------------------------------------------------------------#
    # Heuristic: ECLARE writes                                             #
    #   out_dir/weights/        (best_weights.pt)                          #
    #   out_dir/<basename>/<basename>_eclare.nii.gz (default suffix)       #
    # ---------------------------------------------------------------------#
    weights_dir = out_dir / "weights"
    sr_path = out_dir / f"{lr_path.stem}{suffix}.nii.gz"

    return weights_dir, sr_path
