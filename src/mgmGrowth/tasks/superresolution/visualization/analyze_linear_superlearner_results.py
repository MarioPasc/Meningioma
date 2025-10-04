#!/usr/bin/env python3
# analyze_linear_superlearner_results.py

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ------------------------------------------------------------------
# Matplotlib config (from your snippet, with safe fallbacks)
# ------------------------------------------------------------------
def configure_matplotlib() -> None:
    try:
        import scienceplots  # noqa: F401
        plt.style.use(['science'])
    except Exception as e:
        logging.warning("scienceplots not available: %s", e)
    plt.rcParams.update({
        'figure.dpi': 600,
        'font.size': 10,
        'font.family': 'serif',
        'font.serif': ['Times'],
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'savefig.bbox': 'tight',
    })
    try:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
    except Exception as e:
        logging.warning("LaTeX not available: %s", e)
        plt.rcParams['text.usetex'] = False

# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------
def find_metrics_csvs(out_root: Path) -> List[Path]:
    base = out_root / "LINEAR_SUPER_LEARNER"
    return sorted(base.glob("*/model_data/*/metrics.csv"))

def read_all_metrics(out_root: Path) -> pd.DataFrame:
    files = find_metrics_csvs(out_root)
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Some runs write a global metrics.csv into each pulse dir.
            # Keep only rows that match this pulse directory if present.
            pulse_dir = f.parent.name
            spacing_dir = f.parent.parent.parent.name  # .../<spacing>/model_data/<pulse>/
            if {"spacing","pulse","fold"}.issubset(df.columns):
                df = df[df["pulse"].astype(str) == pulse_dir]
            # Inject path (helps if you later filter by spacing/pulse)
            df["__metrics_path"] = str(f)
            dfs.append(df)
        except Exception as e:
            logging.warning("Failed reading %s: %s", f, e)
    if not dfs:
        return pd.DataFrame()
    allm = pd.concat(dfs, ignore_index=True)
    # Deduplicate by (spacing,pulse,fold)
    allm = allm.drop_duplicates(subset=["spacing","pulse","fold"], keep="last")
    # Normalize types
    allm["spacing"] = allm["spacing"].astype(str)
    allm["pulse"] = allm["pulse"].astype(str)
    allm["fold"] = allm["fold"].astype(str)
    return allm

def backfill_weights_from_json(row: pd.Series, models: List[str]) -> pd.Series:
    """
    If some w_* columns are missing in metrics.csv, try reading the fold's weights.json.
    """
    metrics_path = Path(row["__metrics_path"])
    pulse_dir = metrics_path.parent
    fold_dir = pulse_dir / row["fold"]
    wjson = fold_dir / "weights.json"
    if not wjson.is_file():
        return row
    try:
        data = json.loads(wjson.read_text(encoding="utf-8"))
        for m in models:
            col = f"w_{m}"
            if col not in row or pd.isna(row[col]):
                if m in data:
                    row[col] = float(data[m])
    except Exception:
        pass
    return row

# ------------------------------------------------------------------
# Prep
# ------------------------------------------------------------------
def get_models_from_columns(df: pd.DataFrame) -> List[str]:
    return [c[len("w_"):] for c in df.columns if c.startswith("w_")]

def add_weight_checks(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    wcols = [f"w_{m}" for m in models]
    tmp = df[wcols].astype(float)
    df["w_sum"] = tmp.sum(axis=1)
    df["w_min"] = tmp.min(axis=1)
    df["simplex_violation"] = (df["w_sum"] - 1.0).abs()
    df["neg_violation"] = (tmp.min(axis=1) < -1e-8).astype(int)
    return df

# ------------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------------
def bar_with_scatter(ax, labels, means, stds, points_dict):
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3)
    for i, lab in enumerate(labels):
        ys = points_dict.get(lab, [])
        if len(ys):
            jitter = (np.random.rand(len(ys)) - 0.5) * 0.15
            ax.scatter(np.full(len(ys), x[i]) + jitter, ys, s=10, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.margins(x=0.05)

def stacked_bars_per_fold(ax, weights_per_fold: pd.DataFrame, models: List[str]):
    # weights_per_fold index = fold, columns = models
    folds = weights_per_fold.index.tolist()
    x = np.arange(len(folds))
    bottom = np.zeros(len(folds))
    for m in models:
        vals = weights_per_fold[m].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=m)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(folds, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Weight")
    ax.legend(ncol=min(4, len(models)))

# ------------------------------------------------------------------
# Panels per (spacing, pulse)
# ------------------------------------------------------------------
def plot_panel_per_condition(df: pd.DataFrame, models: List[str], outdir: Path) -> None:
    if df.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    wcols = [f"w_{m}" for m in models]

    for (s, p), g in df.groupby(["spacing","pulse"], dropna=False):
        # ---- Loss summary
        losses = {
            "Train MSE": g["train_loss"].astype(float).tolist(),
            "Test MSE":  g["test_loss"].astype(float).tolist()
        }
        loss_means = [np.nanmean(losses["Train MSE"]), np.nanmean(losses["Test MSE"])]
        loss_stds  = [np.nanstd(losses["Train MSE"]),  np.nanstd(losses["Test MSE"])]

        # ---- Weights stats
        wstats = g[wcols].astype(float)
        w_mean = wstats.mean(axis=0).rename(lambda c: c[len("w_"):])
        w_std  = wstats.std(axis=0).rename(lambda c: c[len("w_"):])

        # ---- Per-fold stacked weights
        per_fold = g.set_index("fold")[wcols].astype(float)
        per_fold.columns = [c[len("w_"):] for c in per_fold.columns]

        configure_matplotlib()
        fig = plt.figure(figsize=(6.4, 5.0))
        gs = GridSpec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.0, 1.0], hspace=0.45, wspace=0.35)

        # A: Train/Test loss (mean±SD + fold dots)
        axA = fig.add_subplot(gs[0, 0])
        bar_with_scatter(axA, ["Train MSE", "Test MSE"], loss_means, loss_stds, losses)
        axA.set_ylabel("MSE")
        axA.set_title(rf"Loss summary")

        # B: Final weights (mean±SD across folds)
        axB = fig.add_subplot(gs[0, 1])
        labs = list(w_mean.index)
        bar_with_scatter(axB, labs, w_mean.values, w_std.values,
                         {m: per_fold[m].values.tolist() for m in labs})
        axB.set_ylabel("Weight")
        axB.set_ylim(0, 1)
        axB.set_title("Ensemble weights")

        # C: Per-fold stacked composition
        axC = fig.add_subplot(gs[1, :])
        stacked_bars_per_fold(axC, per_fold, models=labs)
        axC.set_title("Weights by fold")

        fig.suptitle(rf"spacing={s}, pulse={p}", y=0.99)
        fout = outdir / f"PANEL_spacing-{s}_pulse-{p}.png"
        fig.savefig(fout)
        plt.close(fig)
        logging.info("Saved panel to %s", fout)

# ------------------------------------------------------------------
# Global heatmaps
# ------------------------------------------------------------------
def plot_weights_heatmap(df: pd.DataFrame, models: List[str], outdir: Path) -> None:
    if df.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    pivot = (
        df.groupby(["spacing","pulse"], as_index=False)[[f"w_{m}" for m in models]]
          .mean(numeric_only=True)
    )
    pivot["condition"] = pivot["spacing"].astype(str) + " | " + pivot["pulse"].astype(str)
    mat = pivot.set_index("condition")[[f"w_{m}" for m in models]]
    mat.columns = [c[len("w_"):] for c in mat.columns]

    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(max(4.2, 0.3*mat.shape[1] + 2), max(3.2, 0.28*mat.shape[0] + 2)))
    im = ax.imshow(mat.to_numpy(), aspect="auto")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist())
    ax.set_xlabel("Expert")
    ax.set_ylabel("Condition (spacing | pulse)")
    ax.set_title("Mean final weights")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_ylabel("Weight", rotation=90)
    fout = outdir / "WEIGHTS_heatmap.png"
    fig.savefig(fout)
    plt.close(fig)
    logging.info("Saved weights heatmap to %s", fout)

def plot_loss_overview(df: pd.DataFrame, outdir: Path) -> None:
    if df.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    agg = (
        df.groupby(["spacing","pulse"], as_index=False)[["train_loss","test_loss"]]
          .agg(['mean','std','count'])
    )
    # Flatten columns
    agg.columns = ['_'.join(filter(None, c)).strip('_') for c in agg.columns.values]
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    # Bar pairs per condition
    idx = np.arange(len(agg))
    width = 0.4
    ax.bar(idx - width/2, agg["train_loss_mean"], yerr=agg["train_loss_std"], width=width, capsize=3, label="Train")
    ax.bar(idx + width/2, agg["test_loss_mean"],  yerr=agg["test_loss_std"],  width=width, capsize=3, label="Test")
    labels = [f"{r.spacing}|{r.pulse}" for r in agg.itertuples()]
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title("Loss overview across conditions")
    ax.legend()
    fig.tight_layout()
    fout = outdir / "LOSS_overview.png"
    fig.savefig(fout)
    plt.close(fig)
    logging.info("Saved loss overview to %s", fout)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Analyze results from train_linear_super_learner.py")
    ap.add_argument("--out-root", type=Path, required=True, help="Same --out-root used in training.")
    ap.add_argument("--outdir", type=Path, default=None, help="Output dir for figures. Default: <out-root>/LINEAR_SUPER_LEARNER/analysis")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    allm = read_all_metrics(args.out_root)
    if allm.empty:
        logging.error("No metrics.csv files found under %s/LINEAR_SUPER_LEARNER", args.out_root)
        return

    models = get_models_from_columns(allm)
    if not models:
        # Try to backfill from any weights.json in first pulse folder
        logging.warning("No w_* columns found in metrics.csv. Attempting backfill from weights.json")
        # Guess models from a weights.json
        any_metrics = Path(allm.iloc[0]["__metrics_path"])
        for wj in any_metrics.parent.glob("*/weights.json"):
            d = json.loads(wj.read_text(encoding="utf-8"))
            models = list(d.keys())
            break
        if not models:
            logging.error("Models not found. Aborting.")
            return
        # Backfill each row
        allm = allm.apply(lambda r: backfill_weights_from_json(r, models), axis=1)

    # Checks
    allm = add_weight_checks(allm, models)
    violations = allm[(allm["simplex_violation"] > 1e-5) | (allm["neg_violation"] > 0)]
    if not violations.empty:
        vpath = (args.out_root / "LINEAR_SUPER_LEARNER" / "analysis" / "constraint_violations.csv")
        vpath.parent.mkdir(parents=True, exist_ok=True)
        violations.to_csv(vpath, index=False)
        logging.warning("Constraint violations saved to %s", vpath)

    # Save aggregated table
    agg = allm.copy()
    agg_path = (args.out_root / "LINEAR_SUPER_LEARNER" / "analysis" / "aggregated_metrics.csv")
    if agg_path.is_file():
        logging.warning("Using existing %s", agg_path)
        agg = pd.read_csv(agg_path)
    else:
        logging.info("Saving aggregated metrics to %s", agg_path)
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg.to_csv(agg_path, index=False)

    # Plots
    base_fig_dir = args.outdir or (args.out_root / "LINEAR_SUPER_LEARNER" / "analysis")
    logging.info("Generating plots in %s", base_fig_dir)
    plot_panel_per_condition(allm, models, base_fig_dir / "panels")
    plot_weights_heatmap(allm, models, base_fig_dir)
    plot_loss_overview(allm, base_fig_dir)

    logging.info("Done. All plots saved to %s", base_fig_dir)

if __name__ == "__main__":
    main()
