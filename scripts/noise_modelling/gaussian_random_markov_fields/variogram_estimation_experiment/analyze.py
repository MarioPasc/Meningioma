#!/usr/bin/env python3
"""
Script to provide insights from the experiment JSON file.
It produces visualizations on the preferred covariance models (general,
per pulse type, per variogram type) and prints the best pulse/patient couple
(s) according to the mean r2 across all covariance models.
"""

import os
import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # type: ignore

# Apply the requested scienceplots style
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
    The resulting DataFrame has the columns:
      - 'pulse'       : Pulse type (e.g., "SUSC", "T1", etc.)
      - 'patient'     : Patient identifier (as string)
      - 'variogram'   : Variogram type (e.g., "Isotropic", "Anisotropic_0_degree", etc.)
      - 'model'       : Covariance model name (e.g., "Matern", "Gaussian", etc.)
      - 'r2'          : The r2 value (float)

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


def plot_preferred_covariance_model_general(
    df: pd.DataFrame, output_folder: str, image_format: str
) -> None:
    """
    Generate a figure with two subplots:
      1. A barplot showing counts of covariance models (x: model, y: count).
      2. A boxplot showing the distribution of r2 values per covariance model.
         The boxplots are displayed in the same order as the barplot (models ordered
         by decreasing counts).

    A unique color is assigned per model and used consistently in both plots.

    Args:
        df: DataFrame with columns ['model', 'r2', ...].
        output_folder: Folder to save the output image.
        image_format: Image file format (e.g., "png", "pdf").
    """
    # Count the number of occurrences per model and sort in decreasing order.
    model_counts = df.groupby("model").size()
    sorted_models = model_counts.sort_values(ascending=False).index.tolist()

    # Create a color dictionary for the models using a colormap.
    cmap = plt.get_cmap("tab10", len(sorted_models))
    model_colors = {model: cmap(i) for i, model in enumerate(sorted_models)}

    # Prepare data for the boxplot (list of r2 arrays in the order of sorted_models)
    boxplot_data = [
        df.loc[df["model"] == model, "r2"].dropna() for model in sorted_models
    ]

    # Create a figure with two subplots (side by side).
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left subplot: Barplot of model counts
    axs[0].bar(
        sorted_models,
        [model_counts[model] for model in sorted_models],
        color=[model_colors[model] for model in sorted_models],
    )
    axs[0].set_xlabel("Covariance Model")
    axs[0].set_ylabel("Count")
    axs[0].set_title("Covariance Model Counts (General)")

    # --- Right subplot: Boxplot of r2 values
    bp = axs[1].boxplot(
        boxplot_data,
        patch_artist=True,
        labels=sorted_models,
    )
    # Set the color of each box
    for patch, model in zip(bp["boxes"], sorted_models):
        patch.set_facecolor(model_colors[model])
    axs[1].set_xlabel("Covariance Model")
    axs[1].set_ylabel("r2 Value")
    axs[1].set_title("r2 Distribution per Covariance Model (General)")

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
      2. A boxplot showing the distribution of r2 values for each pulse type.

    Args:
        df: DataFrame with columns ['pulse', 'model', 'r2', ...].
        output_folder: Folder to save the output image.
        image_format: Image file format (e.g., "png", "pdf").
    """
    # Determine the overall ordering of models (by total count) to be used consistently.
    overall_counts = df.groupby("model").size().sort_values(ascending=False)
    ordered_models = overall_counts.index.tolist()

    # Pivot table: rows = pulse type, columns = model counts.
    pivot_counts = df.groupby(["pulse", "model"]).size().unstack(fill_value=0)
    # Reindex columns to follow the overall ordering (if present)
    pivot_counts = pivot_counts.reindex(columns=ordered_models, fill_value=0)

    # Create a color dictionary for the models.
    cmap = plt.get_cmap("tab10", len(ordered_models))
    model_colors = {model: cmap(i) for i, model in enumerate(ordered_models)}

    # Create figure with two subplots.
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left subplot: Grouped Bar Chart (counts per pulse type)
    pulse_types = pivot_counts.index.tolist()
    x = np.arange(len(pulse_types))
    total_width = 0.8
    n_models = len(ordered_models)
    bar_width = total_width / n_models

    for i, model in enumerate(ordered_models):
        # Get counts for this model (ensure order of pulse types)
        counts = pivot_counts[model].values
        # Compute positions with an offset for each model.
        axs[0].bar(
            x + i * bar_width,
            counts,
            width=bar_width,
            color=model_colors[model],
            label=model,
        )
    axs[0].set_xticks(x + total_width / 2 - bar_width / 2)
    axs[0].set_xticklabels(pulse_types)
    axs[0].set_xlabel("Pulse Type")
    axs[0].set_ylabel("Count")
    axs[0].set_title("Covariance Model Counts per Pulse Type")
    # Do not add a legend as requested.

    # --- Right subplot: Boxplot of r2 values per pulse type
    pulse_box_data = [
        df.loc[df["pulse"] == pulse, "r2"].dropna() for pulse in pulse_types
    ]
    bp = axs[1].boxplot(pulse_box_data, patch_artist=True, labels=pulse_types)
    # Optionally, color the boxes with a uniform color or by pulse type if desired.
    for patch, pulse in zip(bp["boxes"], pulse_types):
        patch.set_facecolor("lightgray")
    axs[1].set_xlabel("Pulse Type")
    axs[1].set_ylabel("r2 Value")
    axs[1].set_title("r2 Distribution per Pulse Type")

    plt.tight_layout()
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
      2. A boxplot showing the distribution of r2 values for each variogram type.

    Args:
        df: DataFrame with columns ['variogram', 'model', 'r2', ...].
        output_folder: Folder to save the output image.
        image_format: Image file format (e.g., "png", "pdf").
    """
    # Determine the overall ordering of models by total count.
    overall_counts = df.groupby("model").size().sort_values(ascending=False)
    ordered_models = overall_counts.index.tolist()

    # Pivot table: rows = variogram type, columns = model counts.
    pivot_counts = df.groupby(["variogram", "model"]).size().unstack(fill_value=0)
    pivot_counts = pivot_counts.reindex(columns=ordered_models, fill_value=0)

    # Create a color dictionary for models.
    cmap = plt.get_cmap("tab10", len(ordered_models))
    model_colors = {model: cmap(i) for i, model in enumerate(ordered_models)}

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    variogram_types = pivot_counts.index.tolist()
    x = np.arange(len(variogram_types))
    total_width = 0.8
    n_models = len(ordered_models)
    bar_width = total_width / n_models

    # --- Left subplot: Grouped Bar Chart (counts per variogram type)
    for i, model in enumerate(ordered_models):
        counts = pivot_counts[model].values
        axs[0].bar(
            x + i * bar_width,
            counts,
            width=bar_width,
            color=model_colors[model],
            label=model,
        )
    axs[0].set_xticks(x + total_width / 2 - bar_width / 2)
    axs[0].set_xticklabels(variogram_types, rotation=45, ha="right")
    axs[0].set_xlabel("Variogram Type")
    axs[0].set_ylabel("Count")
    axs[0].set_title("Covariance Model Counts per Variogram Type")
    # No legend

    # --- Right subplot: Boxplot of r2 values per variogram type
    vario_box_data = [
        df.loc[df["variogram"] == vario, "r2"].dropna() for vario in variogram_types
    ]
    bp = axs[1].boxplot(vario_box_data, patch_artist=True, labels=variogram_types)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgray")
    axs[1].set_xlabel("Variogram Type")
    axs[1].set_ylabel("r2 Value")
    axs[1].set_title("r2 Distribution per Variogram Type")

    plt.tight_layout()
    out_path = os.path.join(
        output_folder, f"preferred_covariance_per_variogram.{image_format}"
    )
    plt.savefig(out_path)
    plt.close()
    print(f"Saved covariance per variogram figure to {out_path}")


def print_best_pulse_patient_per_pulse(df: pd.DataFrame) -> None:
    """
    For each pulse type, compute the mean r2 across all covariance models (and variogram types)
    for each patient, and print the pulse/patient couple with the highest mean r2.

    Args:
        df: DataFrame with columns ['pulse', 'patient', 'r2', ...].
    """
    # Exclude NaN r2 values.
    df_valid = df.dropna(subset=["r2"])
    # Group by pulse and patient and compute the mean r2.
    group = df_valid.groupby(["pulse", "patient"])["r2"].mean().reset_index()
    print("Best pulse/patient couple (per pulse) based on mean r2:")
    # For each pulse, find the patient with maximum mean r2.
    for pulse, sub_df in group.groupby("pulse"):
        best = sub_df.loc[sub_df["r2"].idxmax()]
        print(
            f"  Pulse: {pulse} | Patient: {best['patient']} | Mean r2: {best['r2']:.4f}"
        )


def print_best_patient_overall(df: pd.DataFrame) -> None:
    """
    Compute the overall mean r2 across all pulses and covariance models for each patient
    and print the patient with the highest overall mean r2.

    Args:
        df: DataFrame with columns ['patient', 'r2', ...].
    """
    df_valid = df.dropna(subset=["r2"])
    overall = df_valid.groupby("patient")["r2"].mean().reset_index()
    best = overall.loc[overall["r2"].idxmax()]
    print("Best patient overall (averaged across all pulses):")
    print(f"  Patient: {best['patient']} | Overall Mean r2: {best['r2']:.4f}")


def main() -> None:
    """
    Main function to load the experiment data, generate visualizations,
    and print insights.
    """
    # ----- PARAMETERS -----
    # Path to the input JSON file (adjust as needed)
    input_json = "/home/mariopasc/Python/Datasets/Meningiomas/variogram_fitting_experiment/variogram_experiment_results.json"
    # Output folder for images (will be created if it does not exist)
    output_folder = "scripts/noise_modelling/gaussian_random_markov_fields/variogram_estimation_experiment/results"
    # Output image format: e.g., "png" or "pdf"
    image_format = "svg"
    # -----------------------

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load and parse the experiment JSON data.
    data = load_experiment_data(input_json)
    df = parse_experiment_data(data)

    # --- Question 1: Preferred covariance model (general)
    plot_preferred_covariance_model_general(df, output_folder, image_format)

    # --- Question 2: Preferred covariance model per pulse type
    plot_preferred_covariance_model_per_pulse(df, output_folder, image_format)

    # --- Question 3: Preferred covariance model per variogram type
    plot_preferred_covariance_model_per_variogram(df, output_folder, image_format)

    # --- Question 4: Pulse/Patient couple (per pulse) with highest mean r2
    print_best_pulse_patient_per_pulse(df)

    # --- Question 5: Patient with highest overall mean r2 (across all pulses)
    print_best_patient_overall(df)


if __name__ == "__main__":
    main()
