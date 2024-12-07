import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter
import os
import nrrd
from typing import Dict
from tqdm import tqdm
from natsort import natsorted
from Meningioma.image_processing.nrrd_processing import (
    transversal_axis,
    open_nrrd,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "ieee", "grid", "std-colors"])
plt.rcParams["font.size"] = 10
plt.rcParams.update({"figure.dpi": "100"})


class AdquisitionStats:

    def __init__(self, transformed_dir: str, target_dir: str) -> None:
        """
        Class to generate statistics and visualizations related to the amount of patients in the
        adquisition-based dataset of neuroimaging meningioma data.

        Args:
            transformed_dir (str): Path to the transformed dataset.
            target_dir (str): Path where the statistics and visualizations will be saved.
        """
        self.transformed_dir = transformed_dir
        self.target_dir = target_dir

        # Define RM pulses and TC
        self.rm_pulses = ["T1", "T1SIN", "SUSC", "T2"]
        self.adquisition_types = ["RM", "TC"]

    def _count_patients_by_category(self) -> pd.DataFrame:
        """
        Count the number of patients per category (Total, Control, Meningioma).

        Returns:
            pd.DataFrame: DataFrame containing the counts for each pulse and patient category.
        """
        data = defaultdict(
            lambda: {
                "Total Patients": 0,
                "Control Patients": 0,
                "Meningioma Patients": 0,
            }
        )

        # Count patients in RM pulses
        rm_root = os.path.join(self.transformed_dir, "RM")
        for pulse in self.rm_pulses:
            pulse_path = os.path.join(rm_root, pulse)
            if os.path.exists(pulse_path):
                for patient_folder in natsorted(os.listdir(pulse_path)):
                    patient_path = os.path.join(pulse_path, patient_folder)
                    if os.path.isdir(patient_path):
                        patient_id = patient_folder.replace("P", "")
                        image_file = f"{pulse}_P{patient_id}.nrrd"
                        segmentation_file = f"{pulse}_P{patient_id}_seg.nrrd"

                        # Check if the image file exists
                        if os.path.exists(
                            os.path.join(patient_path, image_file)
                        ):
                            data[f"RM/{pulse}"]["Total Patients"] += 1
                            if os.path.exists(
                                os.path.join(patient_path, segmentation_file)
                            ):
                                data[f"RM/{pulse}"]["Meningioma Patients"] += 1
                            else:
                                data[f"RM/{pulse}"]["Control Patients"] += 1

        # Count patients in TC
        tc_path = os.path.join(self.transformed_dir, "TC")
        if os.path.exists(tc_path):
            for patient_folder in natsorted(os.listdir(tc_path)):
                patient_path = os.path.join(tc_path, patient_folder)
                if os.path.isdir(patient_path):
                    patient_id = patient_folder.replace("P", "")
                    image_file = f"TC_P{patient_id}.nrrd"
                    segmentation_file = f"TC_P{patient_id}_seg.nrrd"

                    # Check if the image file exists
                    if os.path.exists(os.path.join(patient_path, image_file)):
                        data["TC"]["Total Patients"] += 1
                        if os.path.exists(
                            os.path.join(patient_path, segmentation_file)
                        ):
                            data["TC"]["Meningioma Patients"] += 1
                        else:
                            data["TC"]["Control Patients"] += 1

        # Convert to DataFrame
        df = pd.DataFrame(data).T
        return df

    def plot_patient_distribution(self, df: pd.DataFrame) -> None:
        """
        Generate a barplot for patient distribution.

        Args:
            df (pd.DataFrame): DataFrame containing the counts for each pulse and patient category.
        """
        # Define the order of bars
        keys = ["TC", "RM/T1", "RM/T1SIN", "RM/T2", "RM/SUSC"]
        categories = [
            "Total Patients",
            "Meningioma Patients",
            "Control Patients",
        ]

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the bars
        bar_width = 0.2
        for i, category in enumerate(categories):
            ax.bar(
                [x + i * bar_width for x in range(len(keys))],
                df[category].reindex(keys),
                width=bar_width,
                label=category,
                alpha=0.6,
            )

        # Set the x-axis labels and ticks
        ax.set_xticks([x + bar_width for x in range(len(keys))])
        ax.set_xticklabels(keys)

        # Set labels and title
        ax.set_xlabel("Pulses/Acquisitions")
        ax.set_ylabel("Number of Patients")
        ax.set_title("Patient Distribution by Pulse/Acquisition and Category")

        # Add legend
        ax.legend()

        # Adjust layout and save the plot
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.target_dir, "patient_distribution_barplot.png")
        )
        plt.show()

    def generate_stats(self) -> None:
        """
        Generate and save statistics and visualizations for the dataset.
        """
        # Generate patient counts
        df = self._count_patients_by_category()

        # Plot the distribution
        self.plot_patient_distribution(df)

        # Save the DataFrame as a CSV for reference
        df.to_csv(
            os.path.join(self.target_dir, "patient_distribution_stats.csv")
        )


class SizeStats:
    def __init__(
        self, source: str, target: str, verbose: bool = False
    ) -> None:
        """
        Class to generate statistics and visualizations related to the image sizes of the neuroimaging slices.

        Args:
            source (str): Dataset path.
            target (str): Target folder to save the results.
            verbose (bool): If set to True, display detailed information during processing.
        """
        self.source = source
        self.target = target
        self.verbose = verbose
        self.images_df = pd.DataFrame(
            columns=["Pulse", "Patient", "Height", "Width", "Slices"]
        )
        self.segmentations_df = pd.DataFrame(
            columns=["Pulse", "Patient", "Height", "Width", "Slices"]
        )
        self.images_csv = os.path.join(self.target, "images_sizes.csv")
        self.segmentations_csv = os.path.join(
            self.target, "segmentations_sizes.csv"
        )

    def _log(self, message: str) -> None:
        """Log messages if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _process_file(
        self,
        file_path: str,
        pulse: str,
        patient: str,
        acquisition: str,
        is_segmentation: bool,
    ) -> None:
        """Process a single NRRD file (either segmentation or image) and append its size details to the respective dataframe."""
        try:
            _, header = open_nrrd(file_path, return_header=True)
        except Exception:
            return

        axis = transversal_axis(file_path)
        sizes = header["sizes"]
        row = {
            "Pulse": f"{acquisition}/{pulse}" if pulse else acquisition,
            "Patient": patient,
            "Height": sizes[(axis + 1) % 3],
            "Width": sizes[(axis + 2) % 3],
            "Slices": sizes[axis],
        }

        if is_segmentation:
            self.segmentations_df = pd.concat(
                [self.segmentations_df, pd.DataFrame([row])],
                ignore_index=True,
            )
        else:
            self.images_df = pd.concat(
                [self.images_df, pd.DataFrame([row])], ignore_index=True
            )

    def _process_patient_folder(
        self, patient_path: str, pulse: str, acquisition: str
    ) -> None:
        """Process all NRRD files (images and segmentations) within a patient's folder."""
        patient = os.path.basename(patient_path)

        for file in os.listdir(patient_path):
            file_path = os.path.join(patient_path, file)

            if file.endswith("_seg.nrrd"):
                self._process_file(
                    file_path,
                    pulse,
                    patient,
                    acquisition,
                    is_segmentation=True,
                )
            elif file.endswith(".nrrd"):
                self._process_file(
                    file_path,
                    pulse,
                    patient,
                    acquisition,
                    is_segmentation=False,
                )

    def _process_acquisition(
        self, acquisition: str, acquisition_path: str
    ) -> None:
        """Process an acquisition folder which could be 'RM' (with pulses) or 'TC' (without pulses)."""
        if acquisition == "RM":
            for pulse in os.listdir(acquisition_path):
                pulse_path = os.path.join(acquisition_path, pulse)

                if not os.path.isdir(pulse_path):
                    continue

                self._log(f"  Pulse: {pulse}")
                for patient in os.listdir(pulse_path):
                    patient_path = os.path.join(pulse_path, patient)

                    if not os.path.isdir(patient_path):
                        continue

                    self._log(f"    Patient: {patient}")
                    self._process_patient_folder(
                        patient_path, pulse, acquisition
                    )
        elif acquisition == "TC":
            for patient in os.listdir(acquisition_path):
                patient_path = os.path.join(acquisition_path, patient)

                if not os.path.isdir(patient_path):
                    continue

                self._log(f"  Patient: {patient}")
                self._process_patient_folder(
                    patient_path, pulse=None, acquisition=acquisition
                )

    def _analyze_image_sizes(self) -> Dict[str, pd.DataFrame]:
        """Main function to traverse the dataset and collect image and segmentation sizes."""
        for acquisition in tqdm(
            os.listdir(self.source), desc="Analyzing Acquisitions"
        ):
            acquisition_path = os.path.join(self.source, acquisition)

            if not os.path.isdir(acquisition_path):
                continue

            self._log(f"Analyzing Acquisition: {acquisition}")
            self._process_acquisition(acquisition, acquisition_path)

        return {
            "images": self.images_df,
            "segmentations": self.segmentations_df,
        }

    def _scatter_plot_height_vs_width(self, csv_path: str) -> None:
        """
        Generates a scatter plot of height vs. width, colored and shaped by the Pulse attribute.

        Args:
            csv_path (str): Path to the images_sizes.csv file.

        Saves:
            A scatter plot image as 'scatter_height_vs_width.png' in the target directory.
        """
        # Load the data
        df = pd.read_csv(csv_path)

        # Define unique pulses for coloring and shaping
        pulses = df["Pulse"].unique()

        # Temporarily apply the scatter style from scienceplots
        with plt.style.context(["science", "scatter"]):
            # Create the scatter plot
            plt.figure(figsize=(10, 6))

            # Get the current marker cycle from the scatter style
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            markers_and_colors = prop_cycle.by_key()
            marker_cycle = markers_and_colors["marker"]  # List of markers
            color_cycle = markers_and_colors["color"]  # List of colors

            # Iterate through pulses and plot each with a different marker and color
            for i, pulse in enumerate(pulses):
                pulse_data = df[df["Pulse"] == pulse]
                marker = marker_cycle[
                    i % len(marker_cycle)
                ]  # Cycle through markers
                color = color_cycle[
                    i % len(color_cycle)
                ]  # Cycle through colors
                plt.scatter(
                    pulse_data["Width"],
                    pulse_data["Height"],
                    label=pulse,
                    marker=marker,
                    color=color,
                    alpha=0.8,
                )

            # Add labels, title, and legend
            plt.title("Image Height vs. Width")
            plt.xlabel("Width")
            plt.ylabel("Height")
            plt.legend(title="Pulse", loc="lower right")
            plt.ylim(0, None)
            plt.tight_layout()

            # Save the figure
            output_path = os.path.join(
                self.target, "scatter_height_vs_width.png"
            )
            plt.savefig(output_path)
            plt.show()

    def _plot_violin_height_width(self, csv_path: str) -> None:
        """
        Generates a plot with two violin plots. One shows the distribution of heights, and the other shows
        the distribution of widths for each acquisition type (Pulse).

        Args:
            csv_path (str): Path to the images_sizes.csv file.

        Saves:
            A plot image as 'violin_height_width.png' in the target directory.
        """
        # Load the data
        df = pd.read_csv(csv_path)

        # Set up the figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

        # Prepare data for the violin plots
        pulses = df["Pulse"].unique()
        height_data = [df[df["Pulse"] == pulse]["Height"] for pulse in pulses]
        width_data = [df[df["Pulse"] == pulse]["Width"] for pulse in pulses]

        # Violin plot for Height
        axes[0].violinplot(
            height_data, showmeans=True, showmedians=True, showextrema=True
        )
        axes[0].set_title("Height Distribution")
        axes[0].set_xlabel("Acquisition Type")
        axes[0].set_ylabel("Height")
        axes[0].set_ylim(0, None)
        axes[0].set_xticks(range(1, len(pulses) + 1))
        axes[0].set_xticklabels(pulses)

        # Violin plot for Width
        axes[1].violinplot(
            width_data, showmeans=True, showmedians=True, showextrema=True
        )
        axes[1].set_title("Width Distribution")
        axes[1].set_xlabel("Acquisition Type")
        axes[1].set_ylabel("Width")
        axes[1].set_ylim(0, None)
        axes[1].set_xticks(range(1, len(pulses) + 1))
        axes[1].set_xticklabels(pulses)

        # Adjust layout and save the figure
        plt.tight_layout()
        output_path = os.path.join(self.target, "violin_height_width.png")
        plt.savefig(output_path)

    def _barplot_most_frequent_sizes(self, csv_path: str) -> None:
        """
        Generates a bar plot of the 4 most frequent (height, width) pairs, with an "Other" category
        that aggregates all other pairs.

        Args:
            csv_path (str): Path to the images_sizes.csv file.

        Saves:
            A bar plot image as 'barplot_frequent_sizes.png' in the target directory.
        """
        # Load the data
        df = pd.read_csv(csv_path)

        # Group by (Height, Width) pairs and count occurrences
        size_pairs = list(zip(df["Height"], df["Width"]))
        size_counts = Counter(size_pairs)

        # Get the 4 most common size pairs
        most_common = size_counts.most_common(4)

        # Prepare data for the bar plot
        labels = [
            f"{height}x{width}" for (height, width), _ in most_common
        ] + ["Other"]
        counts = [count for _, count in most_common] + [
            sum(
                count
                for pair, count in size_counts.items()
                if pair not in [pair for pair, _ in most_common]
            )
        ]

        # Generate the bar plot using matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))

        # Get the default colors from the std-colors palette
        bar_positions = range(len(labels))
        ax.bar(
            bar_positions,
            counts,
            color=plt.rcParams["axes.prop_cycle"].by_key()["color"],
            alpha=0.8,
        )

        # Set the x-ticks and labels
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(labels)

        # Set labels and title
        ax.set_xlabel("Size (Height, Width)")
        ax.set_ylabel("Number of Occurrences")
        ax.set_title("Most Frequent (Height, Width) Pairs")

        # Adjust layout and save the figure
        plt.tight_layout()
        output_path = os.path.join(self.target, "barplot_frequent_sizes.png")
        plt.savefig(output_path)

    def _heatmap_size_frequency(self, csv_path: str) -> None:
        """
        Generates a heatmap where the x-axis represents the width and the y-axis represents the height.
        The color intensity represents the frequency of that (height, width) pair.

        Args:
            csv_path (str): Path to the images_sizes.csv file.

        Saves:
            A heatmap image as 'heatmap_size_frequency.png' in the target directory.
        """
        # Load the data
        df = pd.read_csv(csv_path)

        # Create a pivot table to aggregate frequencies for each (height, width) pair
        pivot_table = df.pivot_table(
            index="Height", columns="Width", aggfunc="size", fill_value=0
        )

        # Sort the pivot table's index (Height) and columns (Width) in natural order
        pivot_table = pivot_table.sort_index(
            axis=0, ascending=True
        )  # Sort rows by Height
        pivot_table = pivot_table.sort_index(
            axis=1, ascending=True
        )  # Sort columns by Width

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_table, cmap="viridis", cbar_kws={"label": "Frequency"}
        )
        plt.title("Frequency of Image (Height, Width) Pairs")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.tight_layout()

        # Save the heatmap
        output_path = os.path.join(self.target, "heatmap_size_frequency.png")
        os.makedirs(self.target, exist_ok=True)
        plt.savefig(output_path)
        plt.show()

    def analyze_dimensionality(self) -> None:
        results = self._analyze_image_sizes()

        # Save results to CSV files
        os.makedirs(self.target, exist_ok=True)
        results.get("images").to_csv(self.images_csv, index=False)
        results.get("segmentations").to_csv(
            self.segmentations_csv, index=False
        )

    def show_plots(self) -> None:
        self._scatter_plot_height_vs_width(self.images_csv)
        self._barplot_most_frequent_sizes(self.images_csv)
        self._heatmap_size_frequency(self.images_csv)
        self._plot_violin_height_width(self.images_csv)


def main() -> int:
    figures_root = "./docs/figures/"
    source_transformed = os.path.join(
        "/home/mariopasc/Python/Datasets/Meningiomas",
        "Meningioma_Adquisition",
    )

    stats_folder = os.path.join(figures_root, "data_stats")
    stats_generator = AdquisitionStats(
        transformed_dir=source_transformed, target_dir=stats_folder
    )
    stats_generator.generate_stats()

    size_stats_folder = os.path.join(figures_root, "size_stats")
    stats_generator = SizeStats(
        source=source_transformed, target=size_stats_folder, verbose=True
    )
    stats_generator.analyze_dimensionality()
    stats_generator.show_plots()


if __name__ == "__main__":
    main()
