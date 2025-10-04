#!/usr/bin/env python3
# analyze_linear_superlearner_results.py

from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

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
def bar_with_scatter(ax, labels, means, stds, points_dict, ylabel="", title=""):
    """Enhanced bar plot with individual data points."""
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, alpha=0.7, edgecolor='black', linewidth=0.5)
    for i, lab in enumerate(labels):
        ys = points_dict.get(lab, [])
        if len(ys):
            jitter = (np.random.rand(len(ys)) - 0.5) * 0.15
            ax.scatter(np.full(len(ys), x[i]) + jitter, ys, s=15, alpha=0.6, 
                      color='darkred', zorder=3, edgecolors='black', linewidths=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.margins(x=0.05)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3, linestyle='--')

def stacked_bars_per_fold(ax, weights_per_fold: pd.DataFrame, models: List[str]):
    # weights_per_fold index = fold, columns = models
    folds = weights_per_fold.index.tolist()
    x = np.arange(len(folds))
    bottom = np.zeros(len(folds))
    
    # Use a clean color palette
    colors = sns.color_palette("Set3", len(models))
    
    for i, m in enumerate(models):
        vals = weights_per_fold[m].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=m, color=colors[i], 
               edgecolor='black', linewidth=0.5)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(folds, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Weight")
    ax.legend(ncol=min(4, len(models)), fontsize=8)
    ax.grid(alpha=0.3, linestyle='--', axis='y')

# ------------------------------------------------------------------
# New comprehensive plotting functions
# ------------------------------------------------------------------
def plot_optimization_diagnostics(df: pd.DataFrame, outdir: Path) -> None:
    """Plot optimization convergence and diagnostics."""
    if df.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    
    configure_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Weight entropy vs diversity score
    ax = axes[0, 0]
    scatter = ax.scatter(df["avg_pairwise_correlation"], df["weight_entropy"], 
                        c=df["diversity_score"], s=50, alpha=0.6, 
                        cmap='viridis', edgecolors='black', linewidths=0.5)
    ax.set_xlabel("Avg Pairwise Model Correlation")
    ax.set_ylabel("Weight Entropy")
    ax.set_title("Model Diversity vs Weight Distribution")
    plt.colorbar(scatter, ax=ax, label="Diversity Score")
    ax.grid(alpha=0.3, linestyle='--')
    
    # Condition number distribution
    ax = axes[0, 1]
    cond_nums = df["Q_condition_number"].dropna()
    ax.hist(np.log10(cond_nums), bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel(r"$\log_{10}$(Condition Number)")
    ax.set_ylabel("Frequency")
    ax.set_title("Optimization Matrix Conditioning")
    ax.axvline(np.log10(cond_nums.median()), color='red', linestyle='--', 
               label=f'Median: {cond_nums.median():.1e}')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    
    # Optimization iterations
    ax = axes[1, 0]
    iter_data = df.groupby(["spacing", "pulse"])["opt_iterations"].agg(['mean', 'std', 'count'])
    labels = [f"{idx[0]}|{idx[1]}" for idx in iter_data.index]
    x = np.arange(len(labels))
    ax.bar(x, iter_data['mean'], yerr=iter_data['std'], capsize=3, 
           alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Iterations")
    ax.set_title("Optimization Convergence Speed")
    ax.grid(alpha=0.3, linestyle='--')
    
    # Success rate
    ax = axes[1, 1]
    success_rate = df.groupby(["spacing", "pulse"])["opt_success"].mean() * 100
    labels = [f"{idx[0]}|{idx[1]}" for idx in success_rate.index]
    x = np.arange(len(labels))
    bars = ax.bar(x, success_rate.values, alpha=0.7, edgecolor='black', linewidth=0.5)
    # Color code: green if 100%, yellow if >90%, red otherwise
    for i, (bar, val) in enumerate(zip(bars, success_rate.values)):
        if val >= 100:
            bar.set_color('green')
        elif val >= 90:
            bar.set_color('yellow')
        else:
            bar.set_color('red')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Success Rate (\%)")
    ax.set_title("Optimization Success Rate")
    ax.axhline(100, color='green', linestyle='--', alpha=0.5)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    fout = outdir / "OPTIMIZATION_diagnostics.png"
    fig.savefig(fout)
    plt.close(fig)
    logging.info("Saved optimization diagnostics to %s", fout)

def plot_performance_metrics(df: pd.DataFrame, outdir: Path) -> None:
    """Plot comprehensive performance metrics (MAE, PSNR, correlation, etc.)."""
    if df.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Check which metrics are available
    metric_cols = [c for c in df.columns if c.startswith(('train_', 'test_'))]
    metric_names = set([c.split('_', 1)[1] for c in metric_cols if '_' in c])
    
    if not metric_names:
        logging.warning("No detailed metrics found for performance plotting")
        return
    
    configure_matplotlib()
    
    # Create subplots for each metric type
    n_metrics = len(metric_names)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for idx, metric in enumerate(sorted(metric_names)):
        ax = axes[idx]
        train_col = f"train_{metric}"
        test_col = f"test_{metric}"
        
        if train_col not in df.columns or test_col not in df.columns:
            continue
        
        # Group by condition
        grouped = df.groupby(["spacing", "pulse"])[[train_col, test_col]].agg(['mean', 'std'])
        labels = [f"{idx[0]}|{idx[1]}" for idx in grouped.index]
        x = np.arange(len(labels))
        width = 0.35
        
        # Train bars
        train_mean = grouped[(train_col, 'mean')].values
        train_std = grouped[(train_col, 'std')].values
        ax.bar(x - width/2, train_mean, width, yerr=train_std, 
               label='Train', capsize=3, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Test bars
        test_mean = grouped[(test_col, 'mean')].values
        test_std = grouped[(test_col, 'std')].values
        ax.bar(x + width/2, test_mean, width, yerr=test_std,
               label='Test', capsize=3, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.legend()
        ax.grid(alpha=0.3, linestyle='--')
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    fig.tight_layout()
    fout = outdir / "PERFORMANCE_metrics.png"
    fig.savefig(fout)
    plt.close(fig)
    logging.info("Saved performance metrics to %s", fout)

def plot_model_diversity_analysis(df: pd.DataFrame, models: List[str], outdir: Path) -> None:
    """Plot model diversity and correlation patterns."""
    if df.empty or "diversity_score" not in df.columns:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Diversity score by condition
    ax = axes[0]
    grouped = df.groupby(["spacing", "pulse"])["diversity_score"].agg(['mean', 'std', 'count'])
    labels = [f"{idx[0]}|{idx[1]}" for idx in grouped.index]
    x = np.arange(len(labels))
    
    bars = ax.bar(x, grouped['mean'], yerr=grouped['std'], capsize=3,
                  alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Color bars by diversity level
    for bar, val in zip(bars, grouped['mean']):
        if val > 0.5:
            bar.set_color('green')
        elif val > 0.3:
            bar.set_color('yellow')
        else:
            bar.set_color('red')
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Diversity Score")
    ax.set_title("Model Diversity by Condition")
    ax.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='High diversity')
    ax.axhline(0.3, color='yellow', linestyle='--', alpha=0.5, label='Moderate diversity')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    
    # Correlation vs weight entropy
    ax = axes[1]
    for (s, p), g in df.groupby(["spacing", "pulse"]):
        ax.scatter(g["avg_pairwise_correlation"], g["weight_entropy"],
                  label=f"{s}|{p}", s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
    ax.set_xlabel("Avg Pairwise Model Correlation")
    ax.set_ylabel("Weight Entropy")
    ax.set_title("Model Similarity vs Weight Distribution")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Add reference lines
    max_entropy = np.log(len(models))  # Maximum entropy for uniform distribution
    ax.axhline(max_entropy, color='green', linestyle='--', alpha=0.5, 
               label=f'Max entropy: {max_entropy:.2f}')
    
    fig.tight_layout()
    fout = outdir / "MODEL_diversity.png"
    fig.savefig(fout)
    plt.close(fig)
    logging.info("Saved model diversity analysis to %s", fout)

def plot_loss_decomposition(df: pd.DataFrame, outdir: Path) -> None:
    """Plot image loss vs gradient loss contributions."""
    if df.empty:
        return
    
    # Check if loss decomposition metrics exist
    has_decomp = all(c in df.columns for c in ['train_image_loss', 'train_grad_loss', 
                                                 'test_image_loss', 'test_grad_loss'])
    if not has_decomp:
        logging.warning("Loss decomposition metrics not found, skipping plot")
        return
    
    outdir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Train loss decomposition
    ax = axes[0]
    grouped = df.groupby(["spacing", "pulse"])[['train_image_loss', 'train_grad_loss']].mean()
    labels = [f"{idx[0]}|{idx[1]}" for idx in grouped.index]
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, grouped['train_image_loss'], width, label='Image Loss',
           alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, grouped['train_grad_loss'], width, label='Gradient Loss',
           alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Loss")
    ax.set_title("Train Loss Decomposition")
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    
    # Test loss decomposition
    ax = axes[1]
    grouped = df.groupby(["spacing", "pulse"])[['test_image_loss', 'test_grad_loss']].mean()
    
    ax.bar(x - width/2, grouped['test_image_loss'], width, label='Image Loss',
           alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, grouped['test_grad_loss'], width, label='Gradient Loss',
           alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Loss")
    ax.set_title("Test Loss Decomposition")
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    fout = outdir / "LOSS_decomposition.png"
    fig.savefig(fout)
    plt.close(fig)
    logging.info("Saved loss decomposition to %s", fout)

# ------------------------------------------------------------------
# Panels per (spacing, pulse)
# ------------------------------------------------------------------
def plot_panel_per_condition(df: pd.DataFrame, models: List[str], outdir: Path) -> None:
    """Enhanced comprehensive panel for each (spacing, pulse) condition."""
    if df.empty:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    wcols = [f"w_{m}" for m in models]

    for (s, p), g in df.groupby(["spacing","pulse"], dropna=False):
        configure_matplotlib()
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 3, height_ratios=[1.0, 1.0, 1.0], width_ratios=[1.0, 1.0, 1.0], 
                      hspace=0.40, wspace=0.40)

        # A: Loss summary (Train/Test MSE)
        axA = fig.add_subplot(gs[0, 0])
        losses = {
            "Train": g["train_loss"].astype(float).tolist(),
            "Test":  g["test_loss"].astype(float).tolist()
        }
        loss_means = [np.nanmean(losses["Train"]), np.nanmean(losses["Test"])]
        loss_stds  = [np.nanstd(losses["Train"]),  np.nanstd(losses["Test"])]
        bar_with_scatter(axA, ["Train", "Test"], loss_means, loss_stds, losses, 
                        ylabel="MSE", title="Overall Loss")

        # B: Performance metrics (MAE, PSNR, Correlation)
        axB = fig.add_subplot(gs[0, 1])
        if "test_mae" in g.columns and "test_psnr" in g.columns and "test_correlation" in g.columns:
            metrics_data = {
                "MAE": g["test_mae"].astype(float).tolist(),
                "PSNR": g["test_psnr"].astype(float).tolist(),
                "Corr": g["test_correlation"].astype(float).tolist(),
            }
            # Normalize to [0,1] for visualization
            mae_norm = (1 - np.array(metrics_data["MAE"]) / np.max(metrics_data["MAE"])).tolist()
            psnr_norm = (np.array(metrics_data["PSNR"]) / np.max(metrics_data["PSNR"])).tolist()
            
            metric_means = [
                np.nanmean(mae_norm),
                np.nanmean(psnr_norm),
                np.nanmean(metrics_data["Corr"])
            ]
            metric_stds = [
                np.nanstd(mae_norm),
                np.nanstd(psnr_norm),
                np.nanstd(metrics_data["Corr"])
            ]
            bar_with_scatter(axB, ["MAE↓", "PSNR↑", "Corr↑"], metric_means, metric_stds,
                           {"MAE↓": mae_norm, "PSNR↑": psnr_norm, "Corr↑": metrics_data["Corr"]},
                           ylabel="Normalized Score", title="Test Performance")
        else:
            axB.text(0.5, 0.5, "Metrics N/A", ha='center', va='center', transform=axB.transAxes)
            axB.set_title("Test Performance")

        # C: Ensemble weights (mean ± std)
        axC = fig.add_subplot(gs[0, 2])
        wstats = g[wcols].astype(float)
        w_mean = wstats.mean(axis=0).rename(lambda c: c[len("w_"):])
        w_std  = wstats.std(axis=0).rename(lambda c: c[len("w_"):])
        per_fold = g.set_index("fold")[wcols].astype(float)
        per_fold.columns = [c[len("w_"):] for c in per_fold.columns]
        
        labs = list(w_mean.index)
        bar_with_scatter(axC, labs, w_mean.values, w_std.values,
                        {m: per_fold[m].values.tolist() for m in labs},
                        ylabel="Weight", title="Ensemble Weights")
        axC.set_ylim(0, 1)

        # D: Optimization diagnostics
        axD = fig.add_subplot(gs[1, 0])
        if "opt_iterations" in g.columns and "weight_entropy" in g.columns:
            opt_data = {
                "Iterations": g["opt_iterations"].astype(float).tolist(),
                "Entropy": g["weight_entropy"].astype(float).tolist(),
            }
            # Normalize
            iter_norm = (np.array(opt_data["Iterations"]) / np.max(opt_data["Iterations"])).tolist()
            max_entropy = np.log(len(models))
            entropy_norm = (np.array(opt_data["Entropy"]) / max_entropy).tolist()
            
            means = [np.nanmean(iter_norm), np.nanmean(entropy_norm)]
            stds = [np.nanstd(iter_norm), np.nanstd(entropy_norm)]
            bar_with_scatter(axD, ["Iter (norm)", "Entropy (norm)"], means, stds,
                           {"Iter (norm)": iter_norm, "Entropy (norm)": entropy_norm},
                           ylabel="Normalized Value", title="Optimization")
        else:
            axD.text(0.5, 0.5, "Opt Stats N/A", ha='center', va='center', transform=axD.transAxes)
            axD.set_title("Optimization")

        # E: Model diversity
        axE = fig.add_subplot(gs[1, 1])
        if "diversity_score" in g.columns and "avg_pairwise_correlation" in g.columns:
            div_data = {
                "Diversity": g["diversity_score"].astype(float).tolist(),
                "1-Corr": g["diversity_score"].astype(float).tolist(),  # Same as diversity
            }
            means = [np.nanmean(div_data["Diversity"]), 
                    1 - np.nanmean(g["avg_pairwise_correlation"].astype(float))]
            stds = [np.nanstd(div_data["Diversity"]),
                   np.nanstd(g["avg_pairwise_correlation"].astype(float))]
            bar_with_scatter(axE, ["Diversity", "1-Corr"], means, stds,
                           {"Diversity": div_data["Diversity"], 
                            "1-Corr": (1 - g["avg_pairwise_correlation"]).tolist()},
                           ylabel="Score", title="Model Diversity")
            axE.set_ylim(0, 1)
        else:
            axE.text(0.5, 0.5, "Diversity N/A", ha='center', va='center', transform=axE.transAxes)
            axE.set_title("Model Diversity")

        # F: Loss decomposition
        axF = fig.add_subplot(gs[1, 2])
        if "test_image_loss" in g.columns and "test_grad_loss" in g.columns:
            img_losses = g["test_image_loss"].astype(float).tolist()
            grad_losses = g["test_grad_loss"].astype(float).tolist()
            means = [np.nanmean(img_losses), np.nanmean(grad_losses)]
            stds = [np.nanstd(img_losses), np.nanstd(grad_losses)]
            bar_with_scatter(axF, ["Image", "Gradient"], means, stds,
                           {"Image": img_losses, "Gradient": grad_losses},
                           ylabel="Loss", title="Loss Components")
        else:
            axF.text(0.5, 0.5, "Loss Decomp N/A", ha='center', va='center', transform=axF.transAxes)
            axF.set_title("Loss Components")

        # G: Per-fold stacked weights (full width)
        axG = fig.add_subplot(gs[2, :])
        stacked_bars_per_fold(axG, per_fold, models=labs)
        axG.set_title("Weight Composition by Fold")

        fig.suptitle(rf"Comprehensive Analysis: spacing={s}, pulse={p}", y=0.995, fontsize=12)
        fout = outdir / f"PANEL_spacing-{s}_pulse-{p}.png"
        fig.savefig(fout, dpi=300)
        plt.close(fig)
        logging.info("Saved panel to %s", fout)

# ------------------------------------------------------------------
# Global heatmaps
# ------------------------------------------------------------------
def plot_weights_heatmap(df: pd.DataFrame, models: List[str], outdir: Path) -> None:
    """Enhanced weights heatmap with better styling."""
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
    fig, ax = plt.subplots(figsize=(max(5.0, 0.4*mat.shape[1] + 3), max(4.0, 0.35*mat.shape[0] + 2)))
    
    # Use seaborn for better heatmap
    im = sns.heatmap(mat.to_numpy(), annot=True, fmt='.3f', cmap='RdYlGn', 
                     vmin=0, vmax=1, cbar_kws={'label': 'Weight'},
                     linewidths=0.5, linecolor='gray', ax=ax)
    
    ax.set_xticks(np.arange(mat.shape[1]) + 0.5)
    ax.set_xticklabels(list(mat.columns), rotation=45, ha="right")
    ax.set_yticks(np.arange(mat.shape[0]) + 0.5)
    ax.set_yticklabels(list(mat.index), rotation=0)
    ax.set_xlabel("Expert Model")
    ax.set_ylabel("Condition (spacing | pulse)")
    ax.set_title("Mean Ensemble Weights Across Folds")
    
    fig.tight_layout()
    fout = outdir / "WEIGHTS_heatmap.png"
    fig.savefig(fout, dpi=300)
    plt.close(fig)
    logging.info("Saved weights heatmap to %s", fout)

def plot_loss_overview(df: pd.DataFrame, outdir: Path) -> None:
    """Enhanced loss overview with train/test comparison."""
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Bar chart comparison
    ax = axes[0]
    idx = np.arange(len(agg))
    width = 0.35
    ax.bar(idx - width/2, agg["train_loss_mean"], width, yerr=agg["train_loss_std"], 
           capsize=3, label="Train", alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.bar(idx + width/2, agg["test_loss_mean"], width, yerr=agg["test_loss_std"],  
           capsize=3, label="Test", alpha=0.7, edgecolor='black', linewidth=0.5)
    labels = [f"{r.spacing}|{r.pulse}" for r in agg.itertuples()]
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MSE")
    ax.set_title("Loss Overview Across Conditions")
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    
    # Right: Generalization gap (test - train)
    ax = axes[1]
    gap = agg["test_loss_mean"] - agg["train_loss_mean"]
    colors = ['green' if g < 0.01 else 'yellow' if g < 0.05 else 'red' for g in gap]
    bars = ax.bar(idx, gap, alpha=0.7, edgecolor='black', linewidth=0.5)
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Generalization Gap (Test - Train)")
    ax.set_title("Overfitting Analysis")
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
    ax.axhline(0.01, color='green', linestyle='--', alpha=0.5, label='Small gap')
    ax.axhline(0.05, color='yellow', linestyle='--', alpha=0.5, label='Moderate gap')
    ax.legend()
    ax.grid(alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    fout = outdir / "LOSS_overview.png"
    fig.savefig(fout, dpi=300)
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
    
    # Per-condition comprehensive panels
    plot_panel_per_condition(allm, models, base_fig_dir / "panels")
    
    # Global summary plots
    plot_weights_heatmap(allm, models, base_fig_dir)
    plot_loss_overview(allm, base_fig_dir)
    
    # New diagnostic plots
    plot_optimization_diagnostics(allm, base_fig_dir)
    plot_performance_metrics(allm, base_fig_dir)
    plot_model_diversity_analysis(allm, models, base_fig_dir)
    plot_loss_decomposition(allm, base_fig_dir)

    logging.info("Done. All plots saved to %s", base_fig_dir)

if __name__ == "__main__":
    main()
