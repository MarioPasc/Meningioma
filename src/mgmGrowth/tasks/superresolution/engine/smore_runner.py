# file: src/mgmGrowth/tasks/superresolution/engine/smore_runner.py
"""Lightweight subprocess helpers around the SMORE command-line tools."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from mgmGrowth.tasks.superresolution import LOGGER
from src.mgmGrowth.tasks.superresolution import LOGGER as _L
from src.mgmGrowth.tasks.superresolution.config import SmoreConfig
from src.mgmGrowth.tasks.superresolution.tools.paths import ensure_dir


def _run(cmd: list[str]) -> None:
    _L.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_volume(
    lr_path: Path,
    cfg: SmoreConfig,
    weight_dir: Path,
    slice_thick_mm: float,
) -> Path:
    LOGGER.info("=== Training %s ===", lr_path.stem)
    LOGGER.info(f"Slice thickness: {slice_thick_mm} mm")
    LOGGER.info(f"Weight dir: {weight_dir}")
    LOGGER.info(f"GPU ID: {cfg.gpu_id}")
    LOGGER.info(f"Patch size: {cfg.patch_size}")
    LOGGER.info(f"Batch size: {cfg.batch_size}")
    LOGGER.info(f"Num blocks: {cfg.n_blocks}")
    LOGGER.info(f"Num channels: {cfg.n_channels}")
    LOGGER.info(f"Num patches: {cfg.n_patches}")
    LOGGER.info(f"Num rotations: {cfg.n_rots}")
    LOGGER.info("============================================")

    _run(
        [
            "smore-train",
            "--in-fpath",
            str(lr_path),
            "--weight-dir",
            str(weight_dir),
            "--gpu-id",
            str(cfg.gpu_id),
            "--slice-thickness",
            str(slice_thick_mm),
            "--patch-size",
            str(cfg.patch_size),
            "--num-blocks",
            str(cfg.n_blocks),
            "--num-channels",
            str(cfg.n_channels),
            "--batch-size",
            str(cfg.batch_size),
            "--n-patches",
            str(cfg.n_patches),
        ]
    )
    return weight_dir


def infer_volume(
    lr_path: Path,
    weights: Path,
    cfg: SmoreConfig,
    out_path: Path,
) -> None:
    ensure_dir(out_path.parent)
    _run(
        [
            "smore-test",
            "--in-fpath",
            str(lr_path),
            "--out-fpath",
            str(out_path),
            "--gpu-id",
            str(cfg.gpu_id),
            "--weight-dir",
            str(weights),
            "--n-rots",
            str(cfg.n_rots),
        ]
    )

def run_smore(
    lr_path: Path,
    out_dir: Path,
    *,
    cfg: Optional[SmoreConfig] = None,
    slice_thickness: Optional[float] = None,
    blur_kernel: Optional[Path] = None,
    gpu_id: int = 0,
    suffix: str = "",
) -> tuple[Path, Path]:
    """
    Call `run-smore` on a **single** LR volume.

    Returns
    -------
    weights_dir, sr_path
    """
    ensure_dir(out_dir)
    cmd = [
        "run-smore",
        "--in-fpath", str(lr_path),
        "--out-dir", str(out_dir),
        "--gpu-id", str(gpu_id),
    ]
    if slice_thickness is not None:
        cmd += ["--slice-thickness", str(slice_thickness)]
    if blur_kernel:
        cmd += ["--blur-kernel-file", str(blur_kernel)]
    if suffix:
        cmd += ["--suffix", suffix]

    _L.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    weights = out_dir / "weights"
    sr = out_dir / f"{lr_path.stem}{suffix}_SR.nii.gz"
    return weights, sr
