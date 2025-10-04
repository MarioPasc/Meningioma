# -*- coding: utf-8 -*-
"""
Training loop per (spacing, pulse, fold) with TQDM and CSV logging.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os, csv, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.mgmGrowth.tasks.superresolution.sagnef.utils.io import SubjectEntry, entries_for_split, load_manifest, dataset_sanity_fix
from src.mgmGrowth.tasks.superresolution.sagnef.utils.dataset import PatchDataset
from src.mgmGrowth.tasks.superresolution.sagnef.engine.model import SAGNEF
from src.mgmGrowth.tasks.superresolution.sagnef.engine.losses import SAGNEFLoss, LossConfig
from src.mgmGrowth.tasks.superresolution.sagnef.engine.val import evaluate_fold

@dataclass
class TrainUnitConfig:
    spacing: str
    pulse: str
    experts: List[str]

def set_seed(seed: int) -> None:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def make_dataloaders(train_entries: List[SubjectEntry],
                     val_entries: List[SubjectEntry],
                     cfg) -> Tuple[DataLoader, DataLoader]:
    ds_tr = PatchDataset(train_entries, tuple(cfg.patch["patch_size"]),
                         cfg.patch["train_patches_per_volume"],
                         cfg.data["include_lr"], cfg.data["normalization"],
                         tuple(cfg.data["norm_percentiles"]))
    ds_va = PatchDataset(val_entries, tuple(cfg.patch["patch_size"]),
                         cfg.patch["val_patches_per_volume"],
                         cfg.data["include_lr"], cfg.data["normalization"],
                         tuple(cfg.data["norm_percentiles"]))
    dl_tr = DataLoader(ds_tr, batch_size=cfg.train["batch_size"], shuffle=True,
                       num_workers=cfg.train["num_workers"], pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=cfg.train["batch_size"], shuffle=False,
                       num_workers=cfg.train["num_workers"], pin_memory=True)
    return dl_tr, dl_va

def train_one_fold(cfg, unit: TrainUnitConfig, fold: str) -> None:
    set_seed(cfg.seed)
    manifest = load_manifest(cfg.data["cv_json"])
    # entries
    tr_entries = entries_for_split(manifest, fold, "train", unit.spacing, unit.pulse, unit.experts)
    va_entries = entries_for_split(manifest, fold, "test",  unit.spacing, unit.pulse, unit.experts)
    if cfg.data["sanity_fix"]:
        dataset_sanity_fix(tr_entries + va_entries, cfg.data["sanity_max_vox_diff"])

    include_lr = cfg.data["include_lr"]
    n_exp = len(unit.experts)
    in_ch = n_exp + (1 if include_lr else 0)

    # model + optim
    model = SAGNEF(in_ch=in_ch, n_experts=n_exp, wf=24).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train["lr"], weight_decay=cfg.train["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device=="cuda" and cfg.amp))
    # loss
    lcfg = LossConfig(cfg.loss["w_mse"], cfg.loss["w_ssim"], cfg.loss["w_tv"], cfg.loss["w_entropy"],
                      cfg.loss["ssim"]["window"], cfg.loss["ssim"]["sigma"],
                      cfg.loss["ssim"]["K1"], cfg.loss["ssim"]["K2"])
    loss_fn = SAGNEFLoss(lcfg)

    # IO
    base = os.path.join(cfg.out_root, unit.spacing)
    out_pred = os.path.join(base, "output_volumes")
    out_mdat = os.path.join(base, "model_data", unit.pulse, fold)
    os.makedirs(out_pred, exist_ok=True); os.makedirs(out_mdat, exist_ok=True)

    dl_tr, dl_va = make_dataloaders(tr_entries, va_entries, cfg)

    best_val = float("inf")
    no_imp = 0
    # CSV log
    csv_path = os.path.join(out_mdat, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","split","loss","mse","ssim","tv","entropy","time_s"])
        writer.writeheader()

        for epoch in range(1, cfg.train["epochs"] + 1):
            t0 = time.time()
            model.train()
            total = {"loss":0.0,"mse":0.0,"ssim":0.0,"tv":0.0,"entropy":0.0}
            for X, Y in tqdm(dl_tr, desc=f"{unit.spacing}-{unit.pulse} | {fold} | train", ncols=88):
                X = X.to(cfg.device, non_blocking=True).float()
                Y = Y.to(cfg.device, non_blocking=True).float()
                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(cfg.device=="cuda" and cfg.amp)):
                    y_hat, W = model(X)
                    comp = loss_fn(y_hat, Y, W)
                    loss = comp["total"]
                scaler.scale(loss).backward()
                if cfg.train["grad_clip_norm"]>0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train["grad_clip_norm"])
                scaler.step(opt); scaler.update()
                for k in total.keys():
                    if k=="loss": total[k]+=comp["total"].item()
                    else: total[k]+=comp[k].item()
            ntr = len(dl_tr)
            writer.writerow(dict(epoch=epoch, split="train", loss=total["loss"]/ntr,
                                 mse=total["mse"]/ntr, ssim=total["ssim"]/ntr,
                                 tv=total["tv"]/ntr, entropy=total["entropy"]/ntr,
                                 time_s=round(time.time()-t0,3)))

            # ---- validation ----
            model.eval()
            vtotal = {"loss":0.0,"mse":0.0,"ssim":0.0,"tv":0.0,"entropy":0.0}
            with torch.no_grad():
                for X, Y in tqdm(dl_va, desc=f"{unit.spacing}-{unit.pulse} | {fold} | val  ", ncols=88):
                    X = X.to(cfg.device).float(); Y = Y.to(cfg.device).float()
                    with torch.cuda.amp.autocast(enabled=(cfg.device=="cuda" and cfg.amp)):
                        y_hat, W = model(X)
                        comp = loss_fn(y_hat, Y, W)
                    vtotal["loss"] += comp["total"].item()
                    vtotal["mse"]  += comp["mse"].item()
                    vtotal["ssim"] += comp["ssim"].item()
                    vtotal["tv"]   += comp["tv"].item()
                    vtotal["entropy"] += comp["entropy"].item()
            nva = len(dl_va)
            vloss = vtotal["loss"]/max(nva,1)
            writer.writerow(dict(epoch=epoch, split="val", loss=vloss,
                                 mse=vtotal["mse"]/max(nva,1), ssim=vtotal["ssim"]/max(nva,1),
                                 tv=vtotal["tv"]/max(nva,1), entropy=vtotal["entropy"]/max(nva,1),
                                 time_s=round(time.time()-t0,3)))
            # checkpoints
            if (epoch % cfg.train["save_every"] == 0) or (vloss < best_val):
                ckpt = {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "cfg": dict(cfg),
                    "val_loss": vloss
                }
                torch.save(ckpt, os.path.join(out_mdat, f"model_epoch{epoch:03d}_val{vloss:.6f}.pt"))

            # early stop
            if vloss < best_val - 1e-6:
                best_val = vloss; no_imp = 0
            else:
                no_imp += 1
                if no_imp >= cfg.train["patience"]:
                    break

    # final: full-volume evaluation on held-out test (manifest 'test')
    test_dir = os.path.join(cfg.out_root, unit.spacing, "output_volumes")
    rep = evaluate_fold(model, va_entries, cfg, test_dir, cfg.data["include_lr"], sorted(unit.experts), torch.device(cfg.device))
    with open(os.path.join(out_mdat, "test_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
