#!/usr/bin/env python3
"""
Script to provide insights from the experiment JSON file.
It produces visualizations on the preferred covariance models (general,
per pulse type, per variogram type) and prints insights about the best
performing patients.
"""

import os
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scienceplots  # type: ignore

# Apply the requested scienceplots style.
plt.style.use(["science", "ieee", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def load_experiment_data(json_path: str) -> Dict[str, Any]:
    """
    Load the experiment JSON file.

    Args:
        json_path: Path to the JSON file.

    Returns:
        A dictionary with the JSON contents.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def parse_experiment_data(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Parse the experiment JSON data into a pandas DataFrame.
    Each row corresponds to one covariance model fit and includes:
      - 'pulse'     : Pulse type (e.g., "SUSC", "T1", etc.)
      - 'patient'   : Patient identifier (as string)
      - 'variogram' : Variogram type (e.g., "Isotropic", "Anisotropic_0_degree", etc.)
      - 'model'     : Covariance model name (e.g., "Matern", "Gaussian", etc.)
      - 'r2'        : The r2 value (float)

    Args:
        data: Dictionary loaded from the JSON experiment file.

    Returns:
        A pandas DataFrame containing one row per covariance model fit.
    """
    rows: List[Dict[str, Any]] = []
    experiment = data.get("Variogram_Experiment", {})
    for pulse, pulse_dict in experiment.items():
        for patient, patient_dict in pulse_dict.items():
            for variogram, variogram_dict in patient_dict.items():
                models = variogram_dict.get("models", {})
                for model_name, model_dict in models.items():
                    r2_value = model_dict.get("r2", np.nan)
                    rows.append(
                        {
                            "pulse": pulse,
                            "patient": patient,
                            "variogram": variogram,
                            "model": model_name,
                            "r2": r2_value,
                        }
                    )
    df = pd.DataFrame(rows)
    df["r2"] = pd.to_numeric(df["r2"], errors="coerce")
    return df


def get_model_colors(models: List[str]) -> Dict[str, Any]:
    """
    Generate a dictionary mapping each model name to a color sampled
    from a continuous non-cyclic colormap ("viridis") partitioned into
    len(models) evenly spaced segments.

    Args:
        models: List of covariance model names.

    Returns:
        A dictionary mapping model names to RGBA color tuples.
    """
    cmap = plt.get_cmap("viridis")
    n_models = len(models)
    if n_models > 1:
        colors = [cmap(i / (n_models - 1)) for i in range(n_models)]
    else:
        colors = [cmap(0.5)]
    return {model: color for model, color in zip(models, colors)}


def plot_preferred_covariance_model_general(
    df: pd.DataFrame, output_folder: str, image_format: str
) -> None:
    """
    Generate a figure with two subplots that share the same y-axis (covariance model names):
      - Left: A horizontal bar plot showing counts per model.
      - Right: A horizontal boxplot showing the distribution of r$^2$ values per model.
    Both subplots use explicit y-positions to ensure alignment.

    Args:
        df: DataFrame with columns ['model', 'r2', ...].
        output_folder: Folder to save the output image.
        image_format: Image file format (e.g., "png", "pdf").
    """
    # Count occurrences per model and sort by decreasing count.
    model_counts = df.groupby("model").size()
    sorted_models = model_counts.sort_values(ascending=False).index.tolist()

    # Create a color dictionary using a continuous colormap.
    model_colors = get_model_colors(sorted_models)

    # Prepare data for the boxplot.
    boxplot_data = [
        df.loc[df["model"] == model, "r2"].dropna() for model in sorted_models
    ]

    # Create explicit y positions for each model.
    y_positions = np.arange(len(sorted_models))

    # Create a figure with two subplots sharing the y-axis.
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 5))

    # Left subplot: Horizontal bar plot.
    axs[0].barh(
        y_positions,
        [model_counts[model] for model in sorted_models],
        color=[model_colors[model] for model in sorted_models],
        align="center",
    )
    axs[0].set_xlabel("Count")
    axs[0].set_title("Covariance Model Counts (General)")
    axs[0].set_yticks(y_positions)
    axs[0].set_yticklabels(sorted_models)
    axs[0].tick_params(top=False, right=False)

    # Right subplot: Horizontal boxplot with explicit y positions.
    bp = axs[1].boxplot(
        boxplot_data,
        vert=False,
        patch_artist=True,
        positions=y_positions,  # use the same y positions as the bar plot
    )
    for patch, model in zip(bp["boxes"], sorted_models):
        patch.set_facecolor(model_colors[model])
    axs[1].set_xlabel(r"$r^2$ Value")
    axs[1].set_title(r"$r^2$ Distribution per Covariance Model (General)")
    axs[1].set_yticks(y_positions)
    axs[1].set_yticklabels(sorted_models)
    axs[1].tick_params(top=False, right=False)
    plt.setp(axs[1].get_yticklabels(), rotation=0)

    for ax in axs:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    out_path = os.path.join(
        output_folder, f"preferred_covariance_general.{image_format}"
    )
    plt.savefig(out_path)
    plt.close()
    print(f"Saved general covariance model figure to {out_path}")


def plot_preferred_covariance_model_per_pulse(
    df: pd.DataFrame, output_folder: str, image_format: str
) -> None:
    """
    Generate visualizations for the preferred covariance model per pulse type.
    The figure contains two subplots:
      1. A grouped bar chart (per pulse type) showing counts of covariance models.
      2. A boxplot showing the distribution of r$^2$ values for each pulse type.
    A legend mapping each covariance model to its color is added outside (below)
    the subplots.

    Args:
        df: DataFrame with columns ['pulse', 'model', 'r2', ...].
        output_folder: Folder to save the output image.
        image_format: Image file format (e.g., "png", "pdf").
    """
    # Determine overall ordering of models by total count.
    overall_counts = df.groupby("model").size().sort_values(ascending=False)
    ordered_models = overall_counts.index.tolist()
    model_colors = get_model_colors(ordered_models)

    # Pivot table: rows = pulse type, columns = model counts.
    pivot_counts = df.groupby(["pulse", "model"]).size().unstack(fill_value=0)
    pivot_counts = pivot_counts.reindex(columns=ordered_models, fill_value=0)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    pulse_types = pivot_counts.index.tolist()
    x = np.arange(len(pulse_types))
    total_width = 0.8
    n_models = len(ordered_models)
    bar_width = total_width / n_models

    # Left subplot: Grouped bar chart.
    for i, model in enumerate(ordered_models):
        counts = pivot_counts[model].values
        axs[0].bar(
            x + i * bar_width,
            counts,
            width=bar_width,
            color=model_colors[model],
        )
    axs[0].set_xticks(x + total_width / 2 - bar_width / 2)
    axs[0].set_xticklabels(pulse_types, rotation=0)
    axs[0].set_xlabel("Pulse Type")
    axs[0].set_ylabel("Count")
    axs[0].set_title("Covariance Model Counts per Pulse Type")
    axs[0].tick_params(top=False, right=False)

    # Right subplot: Boxplot of r$^2$ values per pulse type.
    pulse_box_data = [
        df.loc[df["pulse"] == pulse, "r2"].dropna() for pulse in pulse_types
    ]
    bp = axs[1].boxplot(pulse_box_data, patch_artist=True, labels=pulse_types)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgray")
    axs[1].set_xlabel("Pulse Type")
    axs[1].set_ylabel(r"$r^2$ Value")
    axs[1].set_title(r"$r^2$ Distribution per Pulse Type")
    axs[1].tick_params(top=False, right=False)
    plt.setp(axs[1].get_xticklabels(), rotation=0)

    for ax in axs:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # Create legend handles for covariance models.
    handles = [
        mpatches.Patch(color=model_colors[model], label=model)
        for model in ordered_models
    ]
    # Place the legend below the subplots.
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.15),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(
        output_folder, f"preferred_covariance_per_pulse.{image_format}"
    )
    plt.savefig(out_path)
    plt.close()
    print(f"Saved covariance per pulse figure to {out_path}")


def plot_preferred_covariance_model_per_variogram(
    df: pd.DataFrame, output_folder: str, image_format: str
) -> None:
    """
    Generate visualizations for the preferred covariance model per variogram type.
    The figure contains two subplots:
      1. A grouped bar chart (per variogram type) showing counts of covariance models.
      2. A boxplot showing the distribution of r$^2$ values for each variogram type.
    The variogram type names are shortened as follows:
      - "Isotropic" remains "Isotropic"
      - "Anisotropic_0_degree" becomes "Anisotropic 0º"
      - "Anisotropic_45_degree" becomes "Anisotropic 45º"
      - "Anisotropic_90_degree" becomes "Anisotropic 90º"
      - "Anisotropic_135_degree" becomes "Anisotropic 135º"
    A legend mapping each covariance model to its color is added outside (below)
    the subplots.

    Args:
        df: DataFrame with columns ['variogram', 'model', 'r2', ...].
        output_folder: Folder to save the output image.
        image_format: Image file format (e.g., "png", "pdf").
    """
    overall_counts = df.groupby("model").size().sort_values(ascending=False)
    ordered_models = overall_counts.index.tolist()
    model_colors = get_model_colors(ordered_models)

    # Pivot table: rows = variogram type, columns = model counts.
    pivot_counts = df.groupby(["variogram", "model"]).size().unstack(fill_value=0)
    pivot_counts = pivot_counts.reindex(columns=ordered_models, fill_value=0)

    # Mapping to shorten variogram names.
    variogram_mapping = {
        "Isotropic": "Isotropic",
        "Anisotropic_0_degree": "Anisotropic 0º",
        "Anisotropic_45_degree": "Anisotropic 45º",
        "Anisotropic_90_degree": "Anisotropic 90º",
        "Anisotropic_135_degree": "Anisotropic 135º",
    }

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    variogram_types = pivot_counts.index.tolist()
    variogram_names = [variogram_mapping.get(v, v) for v in variogram_types]
    x = np.arange(len(variogram_types))
    total_width = 0.8
    n_models = len(ordered_models)
    bar_width = total_width / n_models

    # Left subplot: Grouped bar chart.
    for i, model in enumerate(ordered_models):
        counts = pivot_counts[model].values
        axs[0].bar(
            x + i * bar_width,
            counts,
            width=bar_width,
            color=model_colors[model],
        )
    axs[0].set_xticks(x + total_width / 2 - bar_width / 2)
    axs[0].set_xticklabels(variogram_names, rotation=0)
    axs[0].set_xlabel("Variogram Type")
    axs[0].set_ylabel("Count")
    axs[0].set_title("Covariance Model Counts per Variogram Type")
    axs[0].tick_params(top=False, right=False)

    # Right subplot: Boxplot of r$^2$ values per variogram type.
    vario_box_data = [
        df.loc[df["variogram"] == v, "r2"].dropna() for v in variogram_types
    ]
    bp = axs[1].boxplot(vario_box_data, patch_artist=True, labels=variogram_names)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgray")
    axs[1].set_xlabel("Variogram Type")
    axs[1].set_ylabel(r"$r^2$ Value")
    axs[1].set_title(r"$r^2$ Distribution per Variogram Type")
    axs[1].tick_params(top=False, right=False)
    plt.setp(axs[1].get_xticklabels(), rotation=0)

    for ax in axs:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # Create legend handles.
    handles = [
        mpatches.Patch(color=model_colors[model], label=model)
        for model in ordered_models
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, -0.15),
    )
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(
        output_folder, f"preferred_covariance_per_variogram.{image_format}"
    )
    plt.savefig(out_path)
    plt.close()
    print(f"Saved covariance per variogram figure to {out_path}")


def print_best_pulse_patient_per_pulse(df: pd.DataFrame) -> None:
    """
    For each pulse type, compute the mean r$^2$ across all covariance models
    (and variogram types) for each patient, and print the pulse/patient couple with
    the highest mean r$^2$.

    Args:
        df: DataFrame with columns ['pulse', 'patient', 'r2', ...].
    """
    df_valid = df.dropna(subset=["r2"])
    group = df_valid.groupby(["pulse", "patient"])["r2"].mean().reset_index()
    print("Best pulse/patient couple (per pulse) based on mean r$^2$:")
    for pulse, sub_df in group.groupby("pulse"):
        best = sub_df.loc[sub_df["r2"].idxmax()]
        print(
            f"  Pulse: {pulse} | Patient: {best['patient']} | Mean r$^2$: {best['r2']:.4f}"
        )


def print_top5_patients_with_all_pulses(df: pd.DataFrame) -> None:
    """
    Compute the overall mean r$^2$ across all covariance model fits for each patient,
    but only consider patients that have all 4 pulse types available (T1, T1SIN, T2, SUSC).
    Then print the top 5 best performing patients (sorted descending by mean r$^2$).

    Args:
        df: DataFrame with columns ['patient', 'pulse', 'r2', ...].
    """
    df_valid = df.dropna(subset=["r2"])
    # Count unique pulses per patient.
    patient_pulses = df_valid.groupby("patient")["pulse"].nunique()
    # Only patients with at least 4 pulse types.
    eligible_patients = patient_pulses[patient_pulses >= 4].index
    df_eligible = df_valid[df_valid["patient"].isin(eligible_patients)]
    overall = df_eligible.groupby("patient")["r2"].mean().reset_index()
    top5 = overall.sort_values(by="r2", ascending=False).head(5)
    print(
        "Top-5 best performing patients (with all 4 pulses available) based on overall mean r$^2$:"
    )
    for _, row in top5.iterrows():
        print(f"  Patient: {row['patient']} | Overall Mean r$^2$: {row['r2']:.4f}")


def main() -> None:
    """
    Main function to load the experiment data, generate visualizations,
    and print insights.
    """
    # ----- PARAMETERS -----
    # Path to the input JSON file
    input_json = "/home/mariopasc/Python/Datasets/Meningiomas/variogram_fitting_experiment/variogram_experiment_results.json"
    # Output folder for images (will be created if it does not exist)
    output_folder = "scripts/noise_modelling/gaussian_random_markov_fields/variogram_estimation_experiment/results"
    # Output image format: e.g., "png" or "pdf"
    image_format = "svg"
    # -----------------------

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data = load_experiment_data(input_json)
    df = parse_experiment_data(data)

    # --- Question 1: Preferred covariance model (general) ---
    plot_preferred_covariance_model_general(df, output_folder, image_format)

    # --- Question 2: Preferred covariance model per pulse type ---
    plot_preferred_covariance_model_per_pulse(df, output_folder, image_format)

    # --- Question 3: Preferred covariance model per variogram type ---
    plot_preferred_covariance_model_per_variogram(df, output_folder, image_format)

    # --- Question 4: Pulse/Patient couple (per pulse) with highest mean r$^2$ ---
    print_best_pulse_patient_per_pulse(df)

    # --- Question 5: Top-5 best performing patients (only if all 4 pulses are available) ---
    print_top5_patients_with_all_pulses(df)


if __name__ == "__main__":
    main()
