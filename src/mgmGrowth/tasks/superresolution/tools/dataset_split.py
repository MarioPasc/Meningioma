# file: src/mgmGrowth/tasks/superresolution/tools/dataset_split.py
from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Iterable, Tuple

from src.mgmGrowth.tasks.superresolution import LOGGER as _L
from .paths import ensure_dir


def _iter_patient_dirs(root: Path) -> Iterable[Path]:
    yield from (p for p in root.iterdir() if p.is_dir() and any(p.glob("*.nii.gz")))


def train_test_split(
    root: Path,
    out_root: Path,
    test_ratio: float = 0.2,
    seed: int | None = None,
    *,
    copy: bool = False,
) -> Tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    patients = sorted(_iter_patient_dirs(root))
    rng.shuffle(patients)

    n_test = int(round(len(patients) * test_ratio))
    test_set = set(patients[:n_test])

    train_dirs, test_dirs = [], []
    for src in patients:
        subset = "test" if src in test_set else "train"
        dst = ensure_dir(out_root / subset / src.name)
        if copy:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            for f in src.iterdir():
                (dst / f.name).symlink_to(f.resolve())
        (test_dirs if subset == "test" else train_dirs).append(dst)

    _L.info("Split => %d train / %d test", len(train_dirs), len(test_dirs))
    return train_dirs, test_dirs
