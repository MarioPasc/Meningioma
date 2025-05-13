# file: src/mgmGrowth/tasks/superresolution/cli/train_smore.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.mgmGrowth.tasks.superresolution import LOGGER
from src.mgmGrowth.tasks.superresolution.config import SmoreConfig
from src.mgmGrowth.tasks.superresolution.engine.smore_runner import train_volume
from src.mgmGrowth.tasks.superresolution.tools.paths import ensure_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--slice-dz", type=float, required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    cfg = SmoreConfig(gpu_id=args.gpu)
    weights_root = ensure_dir(args.train_root / "_smore_weights")
    LOGGER.info("Weights will be saved to %s", weights_root)
    for vol in args.train_root.rglob("*t2w.nii.gz"):  # extend pattern if needed
        train_volume(vol, cfg, weights_root, args.slice_dz)
        LOGGER.info("Trained SMORE on %s", vol.name)


if __name__ == "__main__":
    main()
