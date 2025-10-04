# -*- coding: utf-8 -*-
"""
Train SAGNEF per (spacing, pulse) across folds in kfolds_manifest.json.
"""
from __future__ import annotations
import argparse, os, yaml
from types import SimpleNamespace
from typing import List
from ..engine.trainer import train_one_fold, TrainUnitConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True, help="YAML config")
    ap.add_argument("--spacings", type=str, default=None, help="Comma list to override")
    ap.add_argument("--pulses", type=str, default=None, help="Comma list to override")
    ap.add_argument("--folds", type=str, default=None, help="Comma list to override")
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        raw = yaml.safe_load(f)
    # SimpleNamespace for dot access
    cfg = SimpleNamespace(**raw)

    if args.spacings: cfg.data["spacings"] = [s.strip() for s in args.spacings.split(",")]
    if args.pulses:   cfg.data["pulses"]   = [p.strip() for p in args.pulses.split(",")]
    folds = None
    if args.folds:
        folds = [x.strip() for x in args.folds.split(",")]

    # discover folds from manifest
    import json
    with open(cfg.data["cv_json"], "r") as f:
        mani = json.load(f)
    all_folds = list(mani.keys())
    run_folds = folds if folds else (cfg.train["fold_subset"] if cfg.train["fold_subset"] else all_folds)

    for spacing in cfg.data["spacings"]:
        for pulse in cfg.data["pulses"]:
            unit = TrainUnitConfig(spacing=spacing, pulse=pulse, experts=cfg.data["experts"])
            for fold in run_folds:
                train_one_fold(cfg, unit, fold)

if __name__ == "__main__":
    main()
