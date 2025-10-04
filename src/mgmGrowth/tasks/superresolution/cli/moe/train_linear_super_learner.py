#!/usr/bin/env python3
# train_linear_super_learner.py
# Author: Mario collaborator
# License: MIT

"""
Global Linear Super Learner (stacked regression) with:
- One-shot dataset sanitation: resample mis-shaped SR (and optional LR) volumes to HR grid and overwrite them.
- Patient-level K-fold training from a JSON manifest.
- TQDM progress bars for long loops; logging restricted to fold/test summaries.
- Modular "loss" API (now: convex weighted MSE + optional gradient-domain L2).
- Intensity standardization per patient to stabilize pooling.
- Edge-aware weighting of voxel errors to fight smoothing.
- Streaming accumulation of sufficient statistics for a simplex-constrained QP.

Justification:
- NiBabel resampling for alignment. :contentReference[oaicite:5]{index=5}
- MRI intensity standardization improves comparability across scans. :contentReference[oaicite:6]{index=6}
- SSIM/PSNR diagnostics; SSIM better aligns with perceived quality. :contentReference[oaicite:7]{index=7}
- Edge/gradient emphasis reduces blur and improves structure fidelity while staying convex in our linear setting. :contentReference[oaicite:8]{index=8}
- SLSQP handles simplex constraints efficiently. :contentReference[oaicite:9]{index=9}
"""

from __future__ import annotations
import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.optimize import minimize, Bounds, LinearConstraint
from tqdm.auto import tqdm

# --------------------------- logging ---------------------------

def setup_logging(verbosity: int) -> None:
    level = logging.INFO if verbosity == 0 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")

# --------------------------- config ---------------------------

@dataclass(frozen=True)
class TrainConfig:
    cv_json: Path
    out_root: Path
    spacings: Tuple[str, ...]
    pulses: Tuple[str, ...]
    models: Tuple[str, ...]
    save_train_outputs: bool
    # preproc
    resample_order: int
    affine_atol: float
    sanitize: bool
    sanitize_targets: Tuple[str, ...]  # e.g., ("SR",) or ("SR","LR")
    norm: str                          # 'none' or 'zscore'
    use_seg_mask: bool                 # try HR segmentation if present
    # loss knobs
    lambda_grad: float                 # gradient-domain weight
    lambda_edge: float                 # edge-aware weighting strength
    edge_power: float                  # nonlinearity for edge weights
    lambda_ridge: float                # ridge on weights for conditioning
    dtype: str = "float32"

@dataclass(frozen=True)
class LossSpec:
    """Convex loss spec. Training uses weighted MSE + λ_grad * gradient-MSE."""
    name: str = "mse"  # reserved for future; training path uses MSE family

# --------------------------- IO & alignment ---------------------------

def load_manifest(p: Path) -> Dict:
    with open(p, "r") as f:
        return json.load(f)

def load_nii(path: Path) -> nib.Nifti1Image:
    return nib.load(str(path))

def get_fdata32(img: nib.Nifti1Image) -> np.ndarray:
    return np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)

def resample_like(src: nib.Nifti1Image, like: nib.Nifti1Image, order: int) -> nib.Nifti1Image:
    try:
        from nibabel.processing import resample_from_to
    except Exception as e:
        raise RuntimeError("Resampling requires nibabel[scipy].") from e
    return resample_from_to(src, (like.shape, like.affine), order=order)

def patient_paths(entry: Dict, models: Tuple[str, ...], spacing: str, pulse: str) -> Tuple[List[Path], Path]:
    sr_paths = [Path(entry[m][spacing][pulse]) for m in models]
    hr_path = Path(entry["HR"][spacing][pulse])
    return sr_paths, hr_path

# --------------------------- dataset sanitation (NEW) ---------------------------

def sanitize_dataset(
    manifest: Dict,
    models: Tuple[str, ...],
    spacings: Tuple[str, ...],
    pulses: Tuple[str, ...],
    resample_order: int,
    affine_atol: float,
    targets: Tuple[str, ...] = ("SR",),
) -> None:
    """
    One-shot pass: align mis-shaped volumes to the HR grid and overwrite.
    targets: ('SR',) or ('SR','LR')
    SR: per-model outputs; LR: low-resolution inputs.
    """
    seen: set[Path] = set()
    # Build a flat list of (src_path, hr_img) to inspect only once per file path
    tasks: List[Tuple[Path, nib.Nifti1Image]] = []

    for fkey in manifest.keys():
        for split in ("train", "test"):
            for patient, entry in manifest[fkey][split].items():
                for spacing in spacings:
                    # HR reference per (patient, spacing, *any* pulse). Use t1c if present.
                    ref_pulse = "t1c" if "t1c" in pulses else pulses[0]
                    _, hr_path = patient_paths(entry, models, spacing, ref_pulse)
                    hr_img = load_nii(hr_path)
                    if "SR" in targets:
                        for m in models:
                            for pulse in pulses:
                                p = Path(entry[m][spacing][pulse])
                                if p not in seen:
                                    tasks.append((p, hr_img))
                                    seen.add(p)
                    if "LR" in targets:
                        for pulse in pulses:
                            p = Path(entry["LR"][spacing][pulse])
                            if p not in seen:
                                tasks.append((p, hr_img))
                                seen.add(p)

    for src_path, hr_img in tqdm(tasks, desc="Sanity-align volumes", unit="vol"):
        try:
            src_img = load_nii(src_path)
        except Exception:
            continue  # skip unreadable
        if (src_img.shape != hr_img.shape) or (not np.allclose(src_img.affine, hr_img.affine, atol=affine_atol)):
            # Resample and overwrite in place
            out = resample_like(src_img, hr_img, order=resample_order)
            nib.save(out, src_path)

# --------------------------- masks & normalization (NEW) ---------------------------

def guess_seg_path(hr_path: Path) -> Optional[Path]:
    # Replace trailing -<pulse>.nii.gz with -seg.nii.gz
    stem = hr_path.name
    # e.g., BraTS-MEN-XXXX-000-t1c.nii.gz -> BraTS-MEN-XXXX-000-seg.nii.gz
    if "-t" in stem:
        seg_name = stem.rsplit("-", 1)[0] + "-seg.nii.gz"
        seg_path = hr_path.parent / seg_name
        return seg_path if seg_path.is_file() else None
    return None

def make_training_mask(hr: np.ndarray, hr_path: Path, use_seg: bool) -> np.ndarray:
    """
    Prefer segmentation if available; otherwise foreground mask hr>p5.
    """
    mask = None
    if use_seg:
        seg_path = guess_seg_path(hr_path)
        if seg_path and seg_path.is_file():
            try:
                seg = get_fdata32(load_nii(seg_path))
                if seg.shape == hr.shape:
                    mask = seg > 0
            except Exception:
                mask = None
    if mask is None:
        pos = hr[hr > 0]
        if pos.size == 0:
            return np.ones_like(hr, dtype=bool)
        thr = np.percentile(pos, 5.0)
        mask = hr > float(thr)
    return mask.astype(bool)

def normalize_intensity(arr: np.ndarray, mask: np.ndarray, method: str) -> np.ndarray:
    if method.lower() == "none":
        return arr.astype(np.float32, copy=True)
    if method.lower() == "zscore":
        vals = arr[mask]
        mu = float(vals.mean()) if vals.size else 0.0
        sd = float(vals.std()) if vals.size else 1.0
        sd = sd if sd > 1e-8 else 1.0
        out = (arr - mu) / sd
        return out.astype(np.float32, copy=False)
    raise ValueError(f"Unknown normalization '{method}'.")

# --------------------------- sufficient statistics (ENHANCED) ---------------------------

@dataclass
class QuadStats:
    Q: np.ndarray    # (M,M) = sum weighted X^T X (+ grad term)
    b: np.ndarray    # (M,)   = sum weighted X^T y (+ grad term)
    c: float         # y^T y (diag), for diagnostics only

def accumulate_stats_weighted(
    entries: Iterable[Dict],
    models: Tuple[str, ...],
    spacing: str,
    pulse: str,
    dtype: np.dtype,
    resample_order: int,
    affine_atol: float,
    norm: str,
    use_seg_mask: bool,
    lambda_grad: float,
    lambda_edge: float,
    edge_power: float
) -> QuadStats:
    """
    Build weighted normal equations with optional gradient-domain term.

    Weighted image term:
        w_img = 1 + lambda_edge * (||∇HR|| / max)^edge_power
    Gradient-domain term (if lambda_grad>0):
        Adds sum_c ||∇c(X w) - ∇c(y)||^2 = w^T Q_grad w - 2 b_grad^T w + const.

    All operations respect a foreground mask to reduce background dominance.
    """
    M = len(models)
    Q = np.zeros((M, M), dtype=dtype)
    B = np.zeros((M,), dtype=dtype)
    csum = np.array(0.0, dtype=dtype)

    for entry in tqdm(entries, desc=f"Accumulate stats [{spacing} {pulse}]", unit="pt"):
        sr_paths, hr_path = patient_paths(entry, models, spacing, pulse)
        hr_img = load_nii(hr_path)
        hr = get_fdata32(hr_img)

        # Foreground mask
        mask = make_training_mask(hr, hr_path, use_seg_mask)

        # Edge weights from HR gradients
        if lambda_edge > 0:
            gx, gy, gz = np.gradient(hr)
            gm = np.sqrt(gx*gx + gy*gy + gz*gz)
            gmax = float(gm.max()) if gm.size else 0.0
            w_img = 1.0 + lambda_edge * ((gm / (gmax + 1e-8)) ** edge_power)
        else:
            w_img = 1.0

        # Normalize HR in-mask
        hr_n = normalize_intensity(hr, mask, norm)
        y = hr_n[mask].ravel()

        # Stack experts
        X_rows: List[np.ndarray] = []
        grads: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = [] if lambda_grad > 0 else None

        for p in sr_paths:
            x = get_fdata32(load_nii(p))  # files are sanitized already; shapes should match
            x = normalize_intensity(x, mask, norm)
            X_rows.append(x[mask].ravel())
            if grads is not None:
                gx, gy, gz = np.gradient(x)
                grads.append((gx[mask].ravel(), gy[mask].ravel(), gz[mask].ravel()))

        X = np.vstack(X_rows)  # (M, N)
        if np.isscalar(w_img):
            wv = np.full_like(y, float(w_img))
        else:
            wv = (w_img[mask].ravel()).astype(np.float32)

        # Weighted image term via X_tilde = X * sqrt(w)
        sw = np.sqrt(wv)
        Xw = X * sw[np.newaxis, :]
        yw = y * sw
        Q += Xw @ Xw.T
        B += Xw @ yw
        csum += float(yw @ yw)

        # Gradient-domain term
        if grads is not None:
            # HR gradients
            gyx, gyy, gyz = np.gradient(hr_n)
            gyx = gyx[mask].ravel(); gyy = gyy[mask].ravel(); gyz = gyz[mask].ravel()
            # reuse the same weights wv for gradients
            swg = sw  # same diag weight
            # Precompute expert gradient dot-products
            for i in range(M):
                gix, giy, giz = grads[i]
                # b_grad_i = <gi, gy> (sum over 3 components), weighted
                B[i] += lambda_grad * ( (gix*swg) @ (gyx*swg) + (giy*swg) @ (gyy*swg) + (giz*swg) @ (gyz*swg) )
                for j in range(i, M):
                    gjx, gjy, gjz = grads[j]
                    val = (gix*swg) @ (gjx*swg) + (giy*swg) @ (gjy*swg) + (giz*swg) @ (gjz*swg)
                    Q[i, j] += lambda_grad * val
                    if j != i:
                        Q[j, i] += lambda_grad * val

    return QuadStats(Q=Q, b=B, c=float(csum))

# --------------------------- solver ---------------------------

def solve_simplex_qp(stats: QuadStats, w0: np.ndarray, lambda_ridge: float = 0.0) -> np.ndarray:
    """
    Solve min_w w^T (Q+λI) w - 2 b^T w  s.t. w>=0, sum(w)=1 (λ small ridge for conditioning).
    """
    Q = stats.Q.copy()
    b = stats.b.copy()
    if lambda_ridge > 0:
        M = Q.shape[0]
        Q += lambda_ridge * np.eye(M, dtype=Q.dtype)

    def fun(w: np.ndarray) -> float:
        return float(w @ Q @ w - 2.0 * b @ w)

    def jac(w: np.ndarray) -> np.ndarray:
        return 2.0 * (Q @ w - b)

    M = b.size
    lin_con = LinearConstraint(np.ones((1, M)), np.array([1.0]), np.array([1.0]))
    bounds = Bounds(lb=np.zeros(M), ub=np.ones(M))
    res = minimize(fun, w0, method="SLSQP", jac=jac, bounds=bounds, constraints=[lin_con],
                   options=dict(maxiter=500, ftol=1e-12))
    if not res.success:
        logging.warning("SLSQP did not converge: %s", res.message)
    w = res.x
    w[w < 0] = 0
    s = w.sum()
    return (np.ones_like(w)/M) if s <= 0 else (w / s)

# --------------------------- evaluation & writing ---------------------------

def blend_for_entry(entry: Dict, models: Tuple[str, ...], spacing: str, pulse: str, w: np.ndarray) -> Tuple[np.ndarray, nib.Nifti1Image, np.ndarray]:
    sr_paths, hr_path = patient_paths(entry, models, spacing, pulse)
    hr_img = load_nii(hr_path)
    hr = get_fdata32(hr_img)
    blended = np.zeros_like(hr, dtype=np.float32)
    for wm, p in zip(w, sr_paths):
        xi = get_fdata32(load_nii(p))
        blended += wm * xi
    return blended, hr_img, hr

def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.mean(d*d))

def write_blend(blended: np.ndarray, ref_img: nib.Nifti1Image, out_dir: Path, out_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(blended.astype(np.float32), affine=ref_img.affine, header=ref_img.header), out_dir / out_name)

# --------------------------- training loop ---------------------------

def run_for_spacing_pulse(
    cfg: TrainConfig,
    manifest: Dict,
    spacing: str,
    pulse: str,
    loss_spec: LossSpec,
    metrics_rows: List[Dict]
) -> None:
    folds = sorted(manifest.keys())
    out_spacing_root = cfg.out_root / "LINEAR_SUPER_LEARNER" / spacing
    out_vols = out_spacing_root / "output_volumes"
    out_model = out_spacing_root / "model_data" / pulse
    out_vols.mkdir(parents=True, exist_ok=True)
    out_model.mkdir(parents=True, exist_ok=True)

    for fkey in folds:
        train_entries = list(manifest[fkey]["train"].values())
        test_entries = list(manifest[fkey]["test"].values())

        # --- accumulate statistics with TQDM bars ---
        stats = accumulate_stats_weighted(
            entries=train_entries,
            models=cfg.models,
            spacing=spacing,
            pulse=pulse,
            dtype=np.dtype(cfg.dtype),
            resample_order=cfg.resample_order,
            affine_atol=cfg.affine_atol,
            norm=cfg.norm,
            use_seg_mask=cfg.use_seg_mask,
            lambda_grad=cfg.lambda_grad,
            lambda_edge=cfg.lambda_edge,
            edge_power=cfg.edge_power
        )

        # Solve for weights
        M = len(cfg.models)
        w0 = np.ones(M, dtype=np.float64) / M
        w = solve_simplex_qp(stats, w0, lambda_ridge=cfg.lambda_ridge)

        # --- validation and logging only here ---
        train_losses = []
        for e in tqdm(train_entries, desc=f"Blend train [{spacing} {pulse} {fkey}]", unit="pt"):
            blended, hr_img, hr = blend_for_entry(e, cfg.models, spacing, pulse, w)
            train_losses.append(compute_mse(blended, hr))
        test_losses = []
        for e in tqdm(test_entries, desc=f"Blend test [{spacing} {pulse} {fkey}]", unit="pt"):
            blended, hr_img, hr = blend_for_entry(e, cfg.models, spacing, pulse, w)
            test_losses.append(compute_mse(blended, hr))
            # Always write test outputs
            out_name = Path(patient_paths(e, cfg.models, spacing, pulse)[1]).name
            write_blend(blended, hr_img, out_vols, out_name)
        if cfg.save_train_outputs:
            for e in tqdm(train_entries, desc=f"Write train [{spacing} {pulse} {fkey}]", unit="pt"):
                blended, hr_img, _ = blend_for_entry(e, cfg.models, spacing, pulse, w)
                out_name = Path(patient_paths(e, cfg.models, spacing, pulse)[1]).name
                write_blend(blended, hr_img, out_vols, out_name)

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        test_loss = float(np.mean(test_losses)) if test_losses else float("nan")
        logging.info("Fold %s | spacing=%s pulse=%s | weights=%s | train loss=%.6f | test loss=%.6f",
                     fkey, spacing, pulse, np.round(w, 6).tolist(), train_loss, test_loss)

        # Store weights and metrics
        fold_dir = out_model / fkey
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir / "weights.json", "w") as f:
            json.dump({m: float(w[i]) for i, m in enumerate(cfg.models)}, f, indent=2)

        row = {
            "spacing": spacing, "pulse": pulse, "fold": fkey,
            "n_train": len(train_entries), "n_test": len(test_entries),
            "train_loss": train_loss, "test_loss": test_loss,
        }
        for i, m in enumerate(cfg.models):
            row[f"w_{m}"] = float(w[i])
        metrics_rows.append(row)

    # Save metrics CSV per pulse
    df = pd.DataFrame(metrics_rows).sort_values(["spacing", "pulse", "fold"])
    (out_model / "metrics.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_model / "metrics.csv", index=False)

# --------------------------- CLI ---------------------------

def parse_args() -> argparse.Namespace:
    """
    python src/mgmGrowth/tasks/superresolution/cli/moe/train_linear_super_learner.py \
        --cv-json /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/kfolds_manifest.json \
        --out-root /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution \
        --spacings 3mm,5mm,7mm --pulses t1c,t1n,t2w,t2f \
        --models BSPLINE,ECLARE,SMORE,UNIRES \
        --sanitize --sanitize-targets SR \
        --norm zscore --use-seg-mask \
        --lambda-grad 0.2 --lambda-edge 0.5 --edge-power 1.0 --lambda-ridge 1e-6

    """
    p = argparse.ArgumentParser(description="Linear Super Learner with dataset sanitation, TQDM, and convex edge/gradient-augmented loss.")
    p.add_argument("--cv-json", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--spacings", type=str, default="3mm,5mm,7mm")
    p.add_argument("--pulses", type=str, default="t1c,t1n,t2w,t2f")
    p.add_argument("--models", type=str, default="BSPLINE,ECLARE,SMORE,UNIRES")
    p.add_argument("--save-train-outputs", action="store_true")

    # preproc
    p.add_argument("--resample-order", type=int, default=1)
    p.add_argument("--affine-atol", type=float, default=1e-3)
    p.add_argument("--sanitize", action="store_true", help="Pre-align mis-shaped SR/LR files in-place before training.")
    p.add_argument("--sanitize-targets", type=str, default="SR", help="Comma list among {SR,LR}.")
    p.add_argument("--norm", type=str, default="zscore", choices=["none","zscore"])
    p.add_argument("--use-seg-mask", action="store_true", help="Use HR segmentation if available.")

    # loss knobs
    p.add_argument("--lambda-grad", type=float, default=0.2, help="Weight of gradient-domain L2 term.")
    p.add_argument("--lambda-edge", type=float, default=0.5, help="Edge-aware voxel weighting strength.")
    p.add_argument("--edge-power", type=float, default=1.0, help="Exponent for edge weighting nonlinearity.")
    p.add_argument("--lambda-ridge", type=float, default=1e-6, help="Small ridge to improve conditioning.")

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    setup_logging(verbosity=int(args.verbose))
    cfg = TrainConfig(
        cv_json=args.cv_json.expanduser().resolve(),
        out_root=args.out_root.expanduser().resolve(),
        spacings=tuple(s.strip() for s in args.spacings.split(",")),
        pulses=tuple(s.strip() for s in args.pulses.split(",")),
        models=tuple(s.strip() for s in args.models.split(",")),
        save_train_outputs=bool(args.save_train_outputs),
        resample_order=int(args.resample_order),
        affine_atol=float(args.affine_atol),
        sanitize=bool(args.sanitize),
        sanitize_targets=tuple(s.strip() for s in args.sanitize_targets.split(",")) if args.sanitize_targets else ("SR",),
        norm=args.norm,
        use_seg_mask=bool(args.use_seg_mask),
        lambda_grad=float(args.lambda_grad),
        lambda_edge=float(args.lambda_edge),
        edge_power=float(args.edge_power),
        lambda_ridge=float(args.lambda_ridge),
    )
    manifest = load_manifest(cfg.cv_json)

    # One-shot sanitation before any training
    if cfg.sanitize:
        sanitize_dataset(manifest, cfg.models, cfg.spacings, cfg.pulses, cfg.resample_order, cfg.affine_atol, cfg.sanitize_targets)

    loss_spec = LossSpec(name="mse")

    metrics_rows: List[Dict] = []
    for spacing in cfg.spacings:
        for pulse in cfg.pulses:
            run_for_spacing_pulse(cfg, manifest, spacing, pulse, loss_spec, metrics_rows)

    logging.info("Done")

if __name__ == "__main__":
    main()
