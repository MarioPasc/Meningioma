# file: src/mgmGrowth/tasks/superresolution/engine/smore_runner.py
"""Lightweight subprocess helpers around the SMORE command-line tools."""
from __future__ import annotations

import subprocess
from pathlib import Path

from src.mgmGrowth.tasks.superresolution import LOGGER as _L
from src.mgmGrowth.tasks.superresolution.config import SmoreConfig
from src.mgmGrowth.tasks.superresolution.tools.paths import ensure_dir


def _run(cmd: list[str]) -> None:
    _L.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_volume(
    lr_path: Path,
    cfg: SmoreConfig,
    out_root: Path,
    slice_thick_mm: float,
) -> Path:
    weight_dir = ensure_dir(out_root / lr_path.stem / "weights")
    _run(
        [
            str(cfg.smore_root / "smore-train"),
            "--in-fpath",
            str(lr_path),
            "--weight-dir",
            str(weight_dir.parent),
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
            str(cfg.smore_root / "smore-test"),
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
