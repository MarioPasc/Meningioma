# file: src/mgmGrowth/tasks/superresolution/cli/infer_smore.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.mgmGrowth.tasks.superresolution import LOGGER
from src.mgmGrowth.tasks.superresolution.config import SmoreConfig
from src.mgmGrowth.tasks.superresolution.engine.smore_runner import infer_volume
from src.mgmGrowth.tasks.superresolution.tools.paths import ensure_dir


def _weights(weights_root: Path, lr_path: Path) -> Path:
    return weights_root / lr_path.stem / "weights"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--test-root", type=Path, required=True)
    p.add_argument("--weights-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    cfg = SmoreConfig(gpu_id=args.gpu)
    ensure_dir(args.out_root)

    for lr_vol in args.test_root.rglob("*t2w.nii.gz"):
        w_dir = _weights(args.weights_root, lr_vol)
        out = ensure_dir(args.out_root / lr_vol.parents[0].name) / (
            lr_vol.stem + "_SR.nii.gz"
        )
        infer_volume(lr_vol, w_dir, cfg, out)
        LOGGER.info("Inference ✓ %s", out)


if __name__ == "__main__":
    main()
