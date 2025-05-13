# file: src/mgmGrowth/tasks/superresolution/config.py
"""Typed configuration objects."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class SmoreConfig:
    """Hyper-parameters and paths for the SMORE engine."""
    smore_root: Path
    gpu_id: int = 0
    patch_size: int = 48
    n_blocks: int = 16
    n_channels: int = 32
    batch_size: int = 32
    n_patches: int = 832_000
    n_rots: int = 2
