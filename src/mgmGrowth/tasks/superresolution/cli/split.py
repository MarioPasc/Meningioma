# file: src/mgmGrowth/tasks/superresolution/cli/split.py
from __future__ import annotations

import argparse
from pathlib import Path

from src.mgmGrowth.tasks.superresolution import LOGGER
from src.mgmGrowth.tasks.superresolution.tools import train_test_split


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ds-root", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--copy", action="store_true")
    args = p.parse_args()

    train_test_split(
        args.ds_root,
        args.out_root,
        test_ratio=args.test_ratio,
        seed=args.seed,
        copy=args.copy,
    )
    LOGGER.info("Split saved at %s", args.out_root)


if __name__ == "__main__":
    main()
