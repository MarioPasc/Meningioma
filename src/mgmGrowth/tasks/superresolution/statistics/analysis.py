#!/usr/bin/env python3
# sr_stats_pipeline.py
"""
Statistical pipeline for SR benchmarking.

Inputs
------
1) metrics.npz produced by metrics.py
   shape: (P, 3, 3, M, 4, 4, 2) for mean/std over slices per patient & ROI
2) Volume roots (for radiomics only):
   HR_ROOT / <pid>/<pid>-{t1c,t2w,t2f}.nii.gz and <pid>-seg.nii.gz
   RESULTS_ROOT / <MODEL>/<3mm|5mm|7mm>/output_volumes/<pid>-<pulse>.nii.gz

Outputs
-------
sr_stats_summary.npz  containing:
  - meta: pulses, resolutions_mm, models, metric_names, roi_labels
  - lmm_emm:  dict[pulse]['emm_table'] ndarray with columns:
              ['pulse','resolution','model','n_subjects','mean','se','lcl','ucl']
  - lmm_contrasts: dict[pulse]['contrasts'] list of dicts per (resolution, model)
                   keys: estimate, se, z, p_raw, p_holm, hedges_g, g_ci
  - friedman: dict[pulse]['by_res_roi'] list of dicts {res, roi, chi2, p}
  - wilcoxon: dict[pulse]['pairs'] list of dicts per (res, roi, model_vs_baseline)
              keys: n, W, z, p_raw, p_holm, dz, dz_ci
  - icc: dict[pulse]['by_res_roi_model'] list of dicts with per-feature ICC(2,1)

Usage
-----

python src/mgmGrowth/tasks/superresolution/statistics/analysis.py \
    --metrics_npz /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/metrics/metrics.npz \ 
    --hr_root /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/high_resolution \
    --results_root /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/ \
    --out_npz /media/mpascual/PortableSSD/Meningiomas/tasks/superresolution/results/stats.npz

Notes
-----
- Primary endpoints by pulse: {'t1c':'SSIM','t2w':'SSIM','t2f':'PSNR'}
- MixedLM: value ~ C(model)*C(resolution) + C(roi); random intercepts:
           subject (group) and subjectxroi (variance component)
- All p-value adjustments per family use Holm.
"""

from __future__ import annotations
import argparse, json, logging, math, pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import nibabel as nib
from skimage.feature import graycomatrix, graycoprops
import statsmodels.formula.api as smf
import patsy
from statsmodels.stats.multitest import multipletests

# ----------------------------- logging ------------------------------------
# Reuse the super-resolution logger so messages integrate with the rest
# of the SR pipelines/tools.
from mgmGrowth.tasks.superresolution import LOGGER as LOG
import sys

def _configure_sr_logging():
    """Ensure our logger emits to stdout with a clear format."""
    if not LOG.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[SR:analysis] %(levelname)s | %(message)s"))
        LOG.addHandler(handler)
    # Be verbose for diagnostics
    LOG.setLevel(logging.DEBUG)
    LOG.propagate = False

# ----------------------------- dataclass ----------------------------------
@dataclass(frozen=True)
class Paths:
    metrics_npz: pathlib.Path
    hr_root: pathlib.Path
    results_root: pathlib.Path
    out_npz: pathlib.Path

# ----------------------------- helpers ------------------------------------
PRIMARY_BY_PULSE = {"t1c": "SSIM", "t2w": "SSIM", "t2f": "PSNR"}
ROI_LABELS = ("all", "core", "edema", "surround")

def load_metrics_npz(path: pathlib.Path) -> Dict:
    """Load metrics.npz from metrics.py with allow_pickle for lists."""
    d = np.load(path, allow_pickle=True)
    out = {k: d[k].tolist() if d[k].dtype == object else d[k] for k in d.files}
    return out

def to_long_df(metrics: Dict) -> pd.DataFrame:
    """
    Convert metrics arrays to a tidy DataFrame at patient level
    using the 'mean' statistic across slices.
    """
    arr = metrics["metrics"]  # (P,3,3,M,4,4,2)
    patient_ids = metrics["patient_ids"]
    pulses = list(metrics["pulses"])
    resolutions = list(metrics["resolutions_mm"])
    models = list(metrics["models"])
    metric_names = list(metrics["metric_names"])
    roi_labels = list(metrics["roi_labels"])
    stat_names = list(metrics["stat_names"])
    mean_idx = stat_names.index("mean")

    rows = []
    P, n_pulses, n_res, n_models, n_metrics, n_rois, _ = arr.shape
    for p in range(P):
        for ip in range(n_pulses):
            for ir in range(n_res):
                for im in range(n_models):
                    for imet in range(n_metrics):
                        for iro in range(n_rois):
                            val = arr[p, ip, ir, im, imet, iro, mean_idx]
                            if np.isnan(val):
                                continue
                            rows.append({
                                "patient": patient_ids[p],
                                "pulse": pulses[ip],
                                "resolution": resolutions[ir],
                                "model": models[im],
                                "metric": metric_names[imet],
                                "roi": roi_labels[iro],
                                "value": float(val)
                            })
    return pd.DataFrame(rows)

def to_jsonable(obj):
    """
    Recursively convert NumPy/Pandas objects to built-in Python
    so json.dumps never sees np.* dtypes.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Series,)):
        return obj.tolist()
    if isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj

def holm_adjust(pvals: List[float]) -> List[float]:
    """Holm step-down adjustment."""
    if not pvals:
        return []
    return multipletests(pvals, alpha=0.05, method="holm")[1].tolist()

def fe_series(fit) -> pd.Series:
    """
    Return fixed-effect parameters as a named Series.
    Works for MixedLM and OLS-robust fallback.
    """
    if hasattr(fit, "fe_params"):  # MixedLM
        b = fit.fe_params
    else:  # OLS robust wrapper in your fallback
        b = fit.params
    if isinstance(b, pd.Series):
        return b
    # last resort: build from model exog_names
    names = fit.model.exog_names
    return pd.Series(np.asarray(b).ravel(), index=names, dtype=float)

def fe_cov(fit, names: List[str]) -> pd.DataFrame:
    """
    Return FE covariance aligned to 'names'.
    If the returned matrix has a different shape, reindex or truncate safely.
    """
    V = fit.cov_params()
    if isinstance(V, pd.DataFrame):
        # some statsmodels builds include extra rows/cols (e.g., scale terms)
        V = V.reindex(index=names, columns=names)
    else:
        V = np.asarray(V)
        p = len(names)
        if V.shape[0] != p:
            V = V[:p, :p]  # truncate to FE dimension
        V = pd.DataFrame(V, index=names, columns=names)
    return V

def ensure_psd(V_df: pd.DataFrame, eps: float = 1e-9) -> np.ndarray:
    """Return a symmetric positive semidefinite matrix from V_df.

    - Symmetrize
    - Eigen-decompose and clip negative eigenvalues to 0
    - Add tiny jitter on the diagonal if everything is too close to 0
    """
    V = V_df.to_numpy(dtype=float)
    V = 0.5 * (V + V.T)
    try:
        w, Q = np.linalg.eigh(V)
    except np.linalg.LinAlgError:
        # Strongly ill-conditioned – add jitter and retry
        d = V.shape[0]
        V = V + eps * np.eye(d)
        w, Q = np.linalg.eigh(V)
    w_clipped = np.clip(w, 0.0, None)
    V_psd = (Q @ np.diag(w_clipped) @ Q.T)
    # Enforce symmetry numerically
    V_psd = 0.5 * (V_psd + V_psd.T)
    if w_clipped.min(initial=0.0) < eps:
        V_psd = V_psd + eps * np.eye(V_psd.shape[0])
    return V_psd

def build_fe_X(fit, synth: pd.DataFrame, rhs: str, names: List[str]) -> np.ndarray:
    """
    Build a FE design matrix for 'synth' that exactly matches FE parameter names.
    Drops unknown columns and adds missing zero-columns.
    """
    X = patsy.dmatrix("1 + " + rhs, synth, return_type="dataframe")
    # align columns to FE names
    for col in names:
        if col not in X.columns:
            X[col] = 0.0
    X = X[names]  # drop extras
    return X.to_numpy(dtype=float)

def align_X_with_cov(fit, synth: pd.DataFrame, formula_rhs: str):
    """
    Build FE design X for 'synth' and align its columns to the covariance
    matrix order. Returns (X: ndarray, V: ndarray, cols: list[str]).
    """
    # Raw covariance of FE params (DataFrame preferred)
    V_raw = fit.cov_params()
    if hasattr(V_raw, "index") and hasattr(V_raw, "columns"):
        V_cols = list(V_raw.columns)
        V = V_raw.to_numpy()
    else:
        # Fallback: use model exog names
        V_cols = list(getattr(fit.model, "exog_names", []))
        V = np.asarray(V_raw)

    # Build FE design matrix and align to V_cols
    try:
        X_df = patsy.dmatrix("1 + " + formula_rhs, synth, return_type="dataframe")
    except Exception as e:
        LOG.error("patsy.dmatrix failed for synth columns=%s → %s", list(synth.columns), e)
        raise
    missing = [c for c in V_cols if c not in X_df.columns]
    for c in missing:
        X_df[c] = 0.0
    extra = [c for c in X_df.columns if c not in V_cols]
    if extra:
        LOG.debug("Design has extra columns not in cov: %s", extra)
    X_df = X_df[V_cols]

    X = X_df.to_numpy()
    # Final sanity log
    LOG.debug("Design/Cov alignment: X%s vs V%s | missing=%d extra=%d",
              X.shape, V.shape, len(missing), len(extra))
    return X, V, V_cols

def pack_table(df: pd.DataFrame, label: str) -> np.recarray:
    """
    Convert a DataFrame with mixed dtypes to a NumPy recarray for npz storage.
    Logs schema. Returns empty recarray if df is empty.
    """
    if df.empty:
        LOG.warning("pack_table: %s is empty; storing empty recarray.", label)
        return np.recarray(0, dtype=[])
    rec = df.to_records(index=False)
    LOG.debug("pack_table: %s schema=%s", label, rec.dtype)
    return rec

def ensure_psd(V_df: pd.DataFrame, eps: float = 1e-9) -> np.ndarray:
    """
    Project a symmetric covariance to the nearest PSD by eigenvalue clipping.
    Returns numpy array. Logs when clipping/NaNs occur.
    """
    V_sym = 0.5 * (V_df.values + V_df.values.T)
    if np.isnan(V_sym).any():
        LOG.warning("FE covariance contains NaNs; replacing NaNs with zero before PSD projection.")
        V_sym = np.nan_to_num(V_sym, copy=False)

    w, Q = np.linalg.eigh(V_sym)
    n_neg = int((w < 0).sum())
    if n_neg > 0:
        LOG.warning("FE covariance not PSD: %d negative eigenvalues (min=%.2e). Clipping to ≥ %.1e.",
                    n_neg, w.min(), eps)
    w_clipped = np.clip(w, a_min=eps, a_max=None)
    V_psd = (Q * w_clipped) @ Q.T
    return V_psd


# ----------------------------- LMM ----------------------------------------
def fit_lmm_primary(df: pd.DataFrame, pulse: str):
    primary = PRIMARY_BY_PULSE[pulse]
    sub = df.query("pulse == @pulse and metric == @primary").copy()
    if sub.empty:
        return pd.DataFrame(), []

    sub["subject"]       = sub["patient"].astype("category")
    sub["roi_c"]         = sub["roi"].astype("category")
    sub["resolution_c"]  = sub["resolution"].astype("category")
    sub["model_c"]       = sub["model"].astype("category")

    # keep complete panels (patient, roi) across all model×resolution cells
    full_n = sub["model_c"].nunique() * sub["resolution_c"].nunique()
    cnt = (sub.assign(cell=sub["model_c"].astype(str) + "|" + sub["resolution_c"].astype(str))
              .groupby(["patient","roi"])["cell"].nunique().reset_index(name="n"))
    sub = sub.merge(cnt.loc[cnt["n"] == full_n, ["patient","roi"]], on=["patient","roi"], how="inner")
    if sub.empty:
        raise RuntimeError("No complete (patient, roi) panels. Fill or drop missing cells.")

    levels_roi = sub["roi_c"].cat.categories
    levels_mod = sub["model_c"].cat.categories
    levels_res = sub["resolution_c"].cat.categories
    baseline   = "BSPLINE" if "BSPLINE" in levels_mod else levels_mod[0]
    fe_rhs     = "C(model_c)*C(resolution_c) + C(roi_c)"  # reuse everywhere

    # try MixedLM with subject random intercept, then without interaction; else OLS-CR
    for formula in [f"value ~ {fe_rhs}", "value ~ C(model_c) + C(resolution_c) + C(roi_c)"]:
        try:
            m = smf.mixedlm(formula, data=sub, groups=sub["subject"])
            fit = m.fit(method="lbfgs", reml=True)
            break
        except Exception as e:
            LOG.error("MixedLM failed for '%s' → %s", formula, type(e).__name__)
            fit = None
    if fit is None:
        import statsmodels.api as sm
        ols = smf.ols(f"value ~ {fe_rhs}", data=sub).fit()
        fit = ols.get_robustcov_results(cov_type="cluster", groups=sub["subject"])
        # wrap to present .model and .cov_params like MixedLM
        class _Wrap:
            fe_params = fit.params
            def cov_params(self): return fit.cov_params()
            class _M:
                exog_names = fit.model.exog_names
            model = _M()
        fit = _Wrap()
        LOG.info("Falling back to OLS with cluster-robust SEs.")

    # --- in fit_lmm_primary(), after you have 'fit' and before computing EMM/contrasts ---
    # b = fe_series(fit)                 # named FE vector
    # names = list(b.index)              # FE names
    # V = fe_cov(fit, names)             # FE covariance aligned to names
    # --- in fit_lmm_primary(), after you build b (fe_series) and V (fe_cov) ---
    b = fe_series(fit)
    names = list(b.index)
    V_df = fe_cov(fit, names)
    V_psd = ensure_psd(V_df)  # numpy array, PSD by construction

    # EMM over ROI and pairwise contrasts vs baseline
    emm_rows, contrasts, pvals, tmp = [], [], [], []
    for r in levels_res:
        for m_ in levels_mod:
            # when you build 'synth' for EMMs and contrasts, keep categories explicit:
            # synth = pd.DataFrame({
            #     "model_c":      pd.Categorical([m]*len(levels_roi), categories=levels_mod),
            #     "resolution_c": pd.Categorical([r]*len(levels_roi), categories=levels_res),
            #     "roi_c":        pd.Categorical(levels_roi,          categories=levels_roi),
            # })
            synth = pd.DataFrame({
                "model_c":      pd.Categorical([m_]*len(levels_roi), categories=levels_mod),
                "resolution_c": pd.Categorical([r]*len(levels_roi),  categories=levels_res),
                "roi_c":        pd.Categorical(levels_roi,            categories=levels_roi),
            })
            # then build X with aligned columns and use b,V
            # X = build_fe_X(fit, synth, fe_rhs, names)
            # pred = X @ b.values
            # se = float(np.sqrt(np.mean(np.diag(X @ V.values @ X.T))))
            # --- EMM block: replace SE computation and add debug logging ---
            X = build_fe_X(fit, synth, fe_rhs, names)
            pred = X @ b.values
            mean = float(np.mean(pred))
            row_vars = np.einsum('ij,jk,ik->i', X, V_psd, X)  # safe even if nearly singular
            if (not np.all(np.isfinite(row_vars))) or (np.nanmin(row_vars) < 0):
                LOG.error("Row variances invalid for EMM (finite=%s, min=%.3e). Forcing nonneg.",
                          np.all(np.isfinite(row_vars)), np.nanmin(row_vars))
            row_vars = np.clip(row_vars, 0.0, None)
            se = float(np.sqrt(np.mean(row_vars)))
            zc = stats.norm.ppf(0.975)
            emm_rows.append({
                "pulse": pulse, "resolution": str(r), "model": str(m_),
                "n_subjects": sub["subject"].nunique(),
                "mean": mean, "se": se, "lcl": mean - zc*se, "ucl": mean + zc*se
            })

    for r in levels_res:
        for m_ in levels_mod:
            if m_ == baseline: 
                continue
            synth = pd.DataFrame({
                "model_c":      pd.Categorical([m_, baseline], categories=levels_mod),
                "resolution_c": pd.Categorical([r, r],         categories=levels_res),
                "roi_c":        pd.Categorical([levels_roi[0], levels_roi[0]], categories=levels_roi)
            })
            LOG.debug("CONTRAST: building design for r=%s, m=%s vs baseline=%s", r, m_, baseline)
            # for pairwise contrasts:
            # synth2 = pd.DataFrame({
            #     "model_c":      pd.Categorical([m_, baseline], categories=levels_mod),
            #     "resolution_c": pd.Categorical([r, r],         categories=levels_res),
            #     "roi_c":        pd.Categorical([levels_roi[0], levels_roi[0]], categories=levels_roi),
            # })
            # X2 = build_fe_X(fit, synth2, fe_rhs, names)
            # c = (X2[0] - X2[1]).reshape(-1, 1)
            # est = float((c.T @ b.values)[0])
            # se  = float(np.sqrt((c.T @ V.values @ c)[0, 0]))
            # --- CONTRAST block: replace SE computation and add guards ---
            X2 = build_fe_X(fit, synth, fe_rhs, names)
            cvec = (X2[0] - X2[1]).reshape(-1, 1)
            est = float((cvec.T @ b.values)[0])
            var = float((cvec.T @ V_psd @ cvec)[0, 0])
            if (not np.isfinite(var)) or (var < 0):
                LOG.warning("Contrast variance invalid (var=%.3e). Forcing nonneg.", var)
                var = max(var, 0.0)
            se  = math.sqrt(var) if var > 0 else float("nan")
            z   = est / se if (se and np.isfinite(se)) else float("nan")
            p   = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else float("nan")
            tmp.append((r, m_, est, se, z, p)); pvals.append(p)

    padj = multipletests(pvals, alpha=0.05, method="holm")[1].tolist() if pvals else []
    for (r, m_, est, se, z, p), p_h in zip(tmp, padj):
        # --- CONTRAST pairing: replace .query(...) with boolean indexing (no int() inside query) ---
        r_val = int(r)  # r is a Categorical level; cast in Python, not inside query
        paired = sub[(sub["resolution"] == r_val) & (sub["model"] == m_)]
        base   = sub[(sub["resolution"] == r_val) & (sub["model"] == baseline)]
        if paired.empty or base.empty:
            LOG.warning("No paired rows for contrast pulse=%s res=%s model=%s vs %s; skipping effect size.",
                        pulse, r_val, m_, baseline)
            n = 0; g = np.nan; g_ci = (np.nan, np.nan)
        else:
            merged = pd.merge(
                paired[["patient","roi","value"]].rename(columns={"value":"v1"}),
                base  [["patient","roi","value"]].rename(columns={"value":"v0"}),
                on=["patient","roi"], how="inner"
            )
            diffs = (merged["v1"] - merged["v0"]).to_numpy()
            n  = diffs.size
            sd = np.std(diffs, ddof=1) if n > 1 else np.nan
            dz = np.mean(diffs) / sd if sd and sd > 0 else np.nan
            J  = 1 - 3/(4*n - 1) if n > 1 else np.nan
            g  = dz * J if np.isfinite(dz) else np.nan
            if n > 1 and np.isfinite(dz):
                se_dz = math.sqrt((1/n) + (dz**2)/(2*(n-1)))
                zcrit = stats.norm.ppf(0.975)
                g_ci = (g - zcrit*se_dz*J, g + zcrit*se_dz*J)
            else:
                g_ci = (np.nan, np.nan)
        contrasts.append({
            "pulse": pulse, "resolution": str(r),
            "model": str(m_), "baseline": baseline,
            "estimate": est, "se": se, "z": z,
            "p_raw": p, "p_holm": p_h, "hedges_g": g, "g_ci": g_ci, "n_pairs": int(n)
        })

    return pd.DataFrame(emm_rows), contrasts



# ----------------------- Nonparametric confirmation -----------------------
def friedman_wilcoxon(df: pd.DataFrame, pulse: str) -> Tuple[List[dict], List[dict]]:
    """
    Friedman omnibus and Wilcoxon signed-rank confirmations per (resolution, roi)
    for the pulse-specific primary endpoint. Uses boolean indexing to avoid
    pandas.query/numexpr pitfalls.
    """
    primary = PRIMARY_BY_PULSE[pulse]
    # filter once for pulse + primary metric
    sub = df[(df["pulse"] == pulse) & (df["metric"] == primary)].copy()
    if sub.empty:
        LOG.warning("NP tests[%s]: empty after filtering to primary=%s", pulse, primary)
        return [], []

    models = sorted(sub["model"].unique().tolist())
    baseline = "BSPLINE" if "BSPLINE" in models else models[0]
    friedman_out, wilcoxon_out = [], []

    LOG.debug(
        "NP tests[%s]: rows=%d subjects=%d models=%s baseline=%s",
        pulse, len(sub), sub["patient"].nunique(), models, baseline
    )

    for res in sorted(sub["resolution"].unique()):
        for roi in ROI_LABELS:
            mask = (sub["resolution"] == res) & (sub["roi"] == roi)
            piv = (
                sub.loc[mask]
                   .pivot_table(index="patient", columns="model", values="value", aggfunc="mean")
                   .dropna(axis=0, how="any")
            )
            LOG.debug("Pivot[%s]: res=%s roi=%s shape=%s", pulse, res, roi, tuple(piv.shape))
            if piv.shape[0] < 3 or piv.shape[1] < 2:
                continue

            # Friedman omnibus across available models in this pivot
            cols = [m for m in models if m in piv.columns]
            try:
                chi2, p = stats.friedmanchisquare(*[piv[m].to_numpy(dtype=float) for m in cols])
                friedman_out.append({
                    "pulse": pulse, "resolution": int(res), "roi": roi,
                    "n": int(piv.shape[0]), "chi2": float(chi2), "p_raw": float(p)
                })
            except Exception as e:
                LOG.warning("Friedman[%s]: res=%s roi=%s failed → %s", pulse, res, roi, type(e).__name__)
                continue

            # Wilcoxon vs baseline, per non-baseline model
            pairs = []
            for m in cols:
                if m == baseline:
                    continue
                x = piv[m].to_numpy(dtype=float)
                y = piv[baseline].to_numpy(dtype=float)
                if x.size < 5:
                    continue
                W, p = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", method="approx")
                diffs = x - y
                n = diffs.size
                sd = np.std(diffs, ddof=1) if n > 1 else np.nan
                dz = np.mean(diffs) / sd if sd and sd > 0 else np.nan
                pairs.append((m, W, p, n, dz))

            if not pairs:
                continue
            adj = holm_adjust([p for (_, _, p, _, _) in pairs])

            for (m, W, p, n, dz), p_h in zip(pairs, adj):
                sign = np.sign(np.nanmean(piv[m] - piv[baseline]))
                z = stats.norm.ppf(1 - p/2) * sign if np.isfinite(p) and p > 0 else np.nan
                if n > 1 and np.isfinite(dz):
                    se_dz = math.sqrt((1/n) + (dz**2)/(2*(n-1)))
                    zcrit = stats.norm.ppf(0.975)
                    dz_ci = (dz - zcrit*se_dz, dz + zcrit*se_dz)
                else:
                    dz_ci = (np.nan, np.nan)
                wilcoxon_out.append({
                    "pulse": pulse, "resolution": int(res), "roi": roi,
                    "model": m, "baseline": baseline,
                    "n": int(n), "W": float(W), "z": float(z),
                    "p_raw": float(p), "p_holm": float(p_h),
                    "dz": float(dz) if np.isfinite(dz) else np.nan,
                    "dz_ci": dz_ci
                })

    return friedman_out, wilcoxon_out


# ----------------------- Radiomic stability (ICC) -------------------------
def load_vol(path: pathlib.Path, like_img: nib.Nifti1Image | None = None, order: int = 1) -> np.ndarray:
    """Load NIfTI as float32 array in voxel space of 'like_img' if provided."""
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return np.asarray(data)

def roi_mask_from_seg(seg: np.ndarray, label: int | None) -> np.ndarray:
    return np.ones_like(seg, dtype=bool) if label is None else (seg == label)

def first_order_features(a: np.ndarray) -> Dict[str, float]:
    """Simple, robust first-order features inside a mask."""
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {k: np.nan for k in ["mean","var","skew","kurt","entropy"]}
    hist, edges = np.histogram(a, bins=256, density=True)
    p = hist/np.sum(hist)
    p = p[p>0]
    entr = -np.sum(p*np.log2(p))
    return {
        "mean": float(np.mean(a)),
        "var": float(np.var(a, ddof=1)) if a.size>1 else np.nan,
        "skew": float(stats.skew(a, bias=False)) if a.size>2 else np.nan,
        "kurt": float(stats.kurtosis(a, fisher=True, bias=False)) if a.size>3 else np.nan,
        "entropy": float(entr)
    }

def glcm_features(slice2d: np.ndarray, mask2d: np.ndarray) -> Dict[str, float]:
    """
    Few texture features from GLCM on a single axial slice.
    Intensities are quantized to 32 levels.
    """
    if not mask2d.any():
        return {"contr": np.nan, "homog": np.nan, "corr": np.nan}
    vals = slice2d[mask2d]
    vmin, vmax = np.percentile(vals, [1, 99])
    if vmax <= vmin:
        return {"contr": np.nan, "homog": np.nan, "corr": np.nan}
    q = np.clip(((slice2d - vmin) / (vmax - vmin) * 31).astype(np.uint8), 0, 31)
    # 1-pixel distance, 0 deg; average over directions could be added if desired
    glcm = graycomatrix(q, distances=[1], angles=[0], levels=32, symmetric=True, normed=True)
    return {
        "contr": float(graycoprops(glcm, "contrast")[0,0]),
        "homog": float(graycoprops(glcm, "homogeneity")[0,0]),
        "corr":  float(graycoprops(glcm, "correlation")[0,0]),
    }

def extract_radiomics(hr_vol: np.ndarray, sr_vol: np.ndarray, seg_vol: np.ndarray, roi_label: int | None) -> Dict[str, float]:
    """Compute slice-averaged first-order + GLCM features for HR and SR."""
    feats = {}
    mask = roi_mask_from_seg(seg_vol, roi_label)
    if not mask.any():
        return {k: np.nan for k in ["mean","var","skew","kurt","entropy","contr","homog","corr"]}

    # first-order on all voxels inside ROI
    feats.update(first_order_features(hr_vol[mask]))
    # rename with suffixes later

    # texture: average per-slice GLCM within ROI
    contr, homog, corr = [], [], []
    Z = hr_vol.shape[0]
    for z in range(Z):
        roi2d = mask[z]
        if not roi2d.any():
            continue
        f_hr = glcm_features(hr_vol[z], roi2d)
        f_sr = glcm_features(sr_vol[z], roi2d)
        contr.append((f_hr["contr"], f_sr["contr"]))
        homog.append((f_hr["homog"], f_sr["homog"]))
        corr.append((f_hr["corr"],  f_sr["corr"]))
    def avg(pairs, idx):
        arr = np.array([p[idx] for p in pairs if np.all(np.isfinite(p))])
        return float(np.mean(arr)) if arr.size else np.nan
    # Return paired as dict with HR/SR suffix to align ICC input downstream
    out = {}
    fo_hr = first_order_features(hr_vol[mask]); fo_sr = first_order_features(sr_vol[mask])
    for k in ["mean","var","skew","kurt","entropy"]:
        out[f"{k}_HR"] = fo_hr[k]; out[f"{k}_SR"] = fo_sr[k]
    out["contr_HR"] = avg(contr, 0); out["contr_SR"] = avg(contr, 1)
    out["homog_HR"] = avg(homog, 0); out["homog_SR"] = avg(homog, 1)
    out["corr_HR"]  = avg(corr,  0); out["corr_SR"]  = avg(corr,  1)
    return out

def icc2_1(values: np.ndarray) -> float:
    """
    ICC(2,1): two-way random, absolute agreement, single measurement.
    values shape: (n_subjects, k_raters)
    """
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return np.nan
    n, k = values.shape
    x = values
    mpt = np.nanmean(x, axis=1, keepdims=True)
    mtr = np.nanmean(x, axis=0, keepdims=True)
    m = np.nanmean(x)
    # sums of squares with nan-handling
    ss_total = np.nansum((x - m)**2)
    ss_rows = k * np.nansum((mpt - m)**2)
    ss_cols = n * np.nansum((mtr - m)**2)
    ss_err  = ss_total - ss_rows - ss_cols
    df_rows, df_cols, df_err = n-1, k-1, (n-1)*(k-1)
    ms_rows = ss_rows/df_rows if df_rows>0 else np.nan
    ms_cols = ss_cols/df_cols if df_cols>0 else np.nan
    ms_err  = ss_err/df_err  if df_err>0  else np.nan
    if any(np.isnan([ms_rows, ms_cols, ms_err])):
        return np.nan
    return (ms_rows - ms_err) / (ms_rows + (k-1)*ms_err + k*(ms_cols - ms_err)/n)

def collect_paths(hr_root: pathlib.Path, results_root: pathlib.Path, pulse: str, model: str, res_mm: int) -> List[Tuple[pathlib.Path, pathlib.Path, pathlib.Path]]:
    """Return list of (hr, seg, sr) paths for patients available in both roots."""
    out = []
    sr_dir = results_root / model / f"{res_mm}mm" / "output_volumes"
    for patient_dir in sorted(hr_root.iterdir()):
        pid = patient_dir.name
        hr_p = patient_dir / f"{pid}-{pulse}.nii.gz"
        seg_p = patient_dir / f"{pid}-seg.nii.gz"
        sr_p = sr_dir / f"{pid}-{pulse}.nii.gz"
        if hr_p.exists() and seg_p.exists() and sr_p.exists():
            out.append((hr_p, seg_p, sr_p))
    return out

def run_radiomics(paths: Paths, pulses: List[str], resolutions: List[int], models: List[str]) -> Dict[str, List[dict]]:
    """
    Compute ICC(2,1) for HR vs SR per feature, resolution, model, and ROI.
    Features: mean,var,skew,kurt,entropy,contr,homog,corr
    """
    icc_out: Dict[str, List[dict]] = {p: [] for p in pulses}
    roi_map = {"all": None, "core": 1, "edema": 2, "surround": 3}
    for pulse in pulses:
        for res in resolutions:
            for model in models:
                pairs = collect_paths(paths.hr_root, paths.results_root, pulse, model, res)
                if not pairs:
                    continue
                # accumulate per-subject features
                per_roi_feats: Dict[str, Dict[str, Dict[str, float]]] = {roi: {} for roi in ROI_LABELS}
                subj_ids = []
                for hr_p, seg_p, sr_p in pairs:
                    try:
                        hr = load_vol(hr_p); sr = load_vol(sr_p); seg = load_vol(seg_p)
                    except Exception:
                        continue
                    pid = hr_p.parent.name
                    subj_ids.append(pid)
                    for roi in ROI_LABELS:
                        feats = extract_radiomics(hr, sr, seg, roi_map[roi])
                        per_roi_feats[roi][pid] = feats
                # compute ICC per feature within each ROI
                for roi in ROI_LABELS:
                    if not per_roi_feats[roi]:
                        continue
                    # build arrays HR vs SR across subjects
                    keys = sorted(per_roi_feats[roi].keys())
                    feat_names = [k[:-3] for k in per_roi_feats[roi][keys[0]].keys() if k.endswith("_HR")]
                    for feat in feat_names:
                        hr_vals = np.array([per_roi_feats[roi][k][f"{feat}_HR"] for k in keys], float)
                        sr_vals = np.array([per_roi_feats[roi][k][f"{feat}_SR"] for k in keys], float)
                        X = np.vstack([hr_vals, sr_vals]).T
                        icc = float(icc2_1(X))
                        icc_out[pulse].append({
                            "pulse": pulse, "resolution": int(res), "model": model,
                            "roi": roi, "feature": feat, "n": int(len(keys)), "ICC2_1": icc
                        })
    return icc_out

# ----------------------------- main orchestration -------------------------
def main():
    _configure_sr_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_npz", type=pathlib.Path, required=True)
    ap.add_argument("--hr_root", type=pathlib.Path, required=True)
    ap.add_argument("--results_root", type=pathlib.Path, required=True)
    ap.add_argument("--out_npz", type=pathlib.Path, required=True)
    args = ap.parse_args()
    paths = Paths(args.metrics_npz, args.hr_root, args.results_root, args.out_npz)

    LOG.info("Starting SR statistics analysis …")
    # 1) Load and tidy metrics
    metr = load_metrics_npz(paths.metrics_npz)
    df = to_long_df(metr)
    pulses = list(metr["pulses"]); resolutions = list(metr["resolutions_mm"]); models = list(metr["models"])
    LOG.info("Loaded metrics: rows=%d | pulses=%s | resolutions=%s | models=%s",
             len(df), pulses, resolutions, models)

    # 2) LMM per pulse
    lmm_emm_all = {}
    lmm_contr_all = {}
    for p in pulses:
        emm_table, contrasts = fit_lmm_primary(df, p)
        lmm_emm_all[p] = {"emm_table": pack_table(emm_table, f"EMM[{p}]")}
        lmm_contr_all[p] = {"contrasts": contrasts}

    # 3) Nonparametric confirmations
    friedman_all, wilcoxon_all = {}, {}
    for p in pulses:
        fr, wi = friedman_wilcoxon(df, p)
        # Holm across pairs grouped within each (p,res,roi)
        # Already applied inside; store raw dicts
        friedman_all[p] = {"by_res_roi": fr}
        wilcoxon_all[p] = {"pairs": wi}

    # 4) Radiomic stability ICC
    icc_all = run_radiomics(paths, pulses, resolutions, models)

    # 5) Persist everything needed for plotting
    meta_dict = {
        "pulses": [str(x) for x in pulses],
        "resolutions_mm": [int(x) for x in resolutions],
        "models": [str(x) for x in models],
        "metric_names": [str(x) for x in metr["metric_names"]],
        "roi_labels": [str(x) for x in metr["roi_labels"]],
        "primary_by_pulse": {str(k): str(v) for k, v in PRIMARY_BY_PULSE.items()},
    }

    try:
        meta_json = json.dumps(meta_dict, ensure_ascii=False)
    except TypeError as e:
        LOG.warning("meta_dict not JSON-serializable (%s). Coercing to built-ins.", e)
        meta_json = json.dumps(to_jsonable(meta_dict), ensure_ascii=False)

    np.savez_compressed(
        paths.out_npz,
        meta=meta_json,
        lmm_emm=np.array(lmm_emm_all, dtype=object),
        lmm_contrasts=np.array(lmm_contr_all, dtype=object),
        friedman=np.array(friedman_all, dtype=object),
        wilcoxon=np.array(wilcoxon_all, dtype=object),
        icc=np.array(icc_all, dtype=object),
    )
    LOG.info("Saved: %s", paths.out_npz)
        

if __name__ == "__main__":
    main()
