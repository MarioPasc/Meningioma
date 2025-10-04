# -*- coding: utf-8 -*-
"""
Predict with a trained SAGNEF checkpoint for a given (spacing, pulse).
"""
from __future__ import annotations
import argparse, os, yaml, json
from types import SimpleNamespace
import torch
import numpy as np
from ..engine.model import SAGNEF
from ..utils.io import load_nii, to_np, normalize, save_like
from ..engine.val import predict_volume

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--spacing", type=str, required=True)
    ap.add_argument("--pulse", type=str, required=True)
    ap.add_argument("--experts", type=str, required=True, help="Comma-ordered expert files to blend")
    ap.add_argument("--lr", type=str, default=None)
    ap.add_argument("--ref", type=str, required=True, help="Reference HR-like NIfTI for affine/shape")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        raw = yaml.safe_load(f)
    cfg = SimpleNamespace(**raw)
    device = torch.device(cfg.device)

    # build input stack
    chans = []
    if cfg.data["include_lr"] and args.lr:
        chans.append(normalize(to_np(load_nii(args.lr)), cfg.data["normalization"],
                               cfg.data["norm_percentiles"][0], cfg.data["norm_percentiles"][1])[None])
    exp_paths = [s.strip() for s in args.experts.split(",")]
    for p in exp_paths:
        chans.append(normalize(to_np(load_nii(p)), cfg.data["normalization"],
                               cfg.data["norm_percentiles"][0], cfg.data["norm_percentiles"][1])[None])
    X = np.concatenate(chans, axis=0)

    # load model
    ck = torch.load(args.ckpt, map_location=device)
    n_exp = len(exp_paths)
    in_ch = n_exp + (1 if (cfg.data["include_lr"] and args.lr) else 0)
    model = SAGNEF(in_ch=in_ch, n_experts=n_exp, wf=24).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()

    y_hat = predict_volume(model, X, tuple(cfg.patch["patch_size"]),
                           tuple(cfg.patch["stride"]), cfg.patch["hann_blend"], device)
    ref = load_nii(args.ref)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_like(y_hat[0], ref, args.out)

if __name__ == "__main__":
    main()
