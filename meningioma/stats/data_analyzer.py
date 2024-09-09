import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter
import os
import nrrd
from typing import Dict
from tqdm import tqdm
from natsort import natsorted
import scienceplots
plt.style.use(['science', 'ieee', 'grid', 'std-colors']) 
plt.rcParams['font.size'] = 10
plt.rcParams.update({'figure.dpi': '100'})
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
        self.rm_pulses = ['T1', 'T1SIN', 'SUSC', 'T2']
        self.adquisition_types = ['RM', 'TC']

    def _count_patients_by_category(self) -> pd.DataFrame:
        """
        Count the number of patients per category (Total, Control, Meningioma).
        
        Returns:
            pd.DataFrame: DataFrame containing the counts for each pulse and patient category.
        """
        data = defaultdict(lambda: {'Total Patients': 0, 'Control Patients': 0, 'Meningioma Patients': 0})

        # Count patients in RM pulses
        rm_root = os.path.join(self.transformed_dir, 'RM')
        for pulse in self.rm_pulses:
            pulse_path = os.path.join(rm_root, pulse)
            if os.path.exists(pulse_path):
                for patient_folder in os.listdir(pulse_path):
                    patient_path = os.path.join(pulse_path, patient_folder)
                    if os.path.isdir(patient_path):
                        patient_id = patient_folder.replace('P', '')
                        image_file = f'{pulse}_P{patient_id}.nrrd'
                        segmentation_file = f'{pulse}_P{patient_id}_seg.nrrd'

                        # Check if the image file exists
                        if os.path.exists(os.path.join(patient_path, image_file)):
                            data[f'RM/{pulse}']['Total Patients'] += 1
                            if os.path.exists(os.path.join(patient_path, segmentation_file)):
                                data[f'RM/{pulse}']['Meningioma Patients'] += 1
                            else:
                                data[f'RM/{pulse}']['Control Patients'] += 1

        # Count patients in TC
        tc_path = os.path.join(self.transformed_dir, 'TC')
        if os.path.exists(tc_path):
            for patient_folder in os.listdir(tc_path):
                patient_path = os.path.join(tc_path, patient_folder)
                if os.path.isdir(patient_path):
                    patient_id = patient_folder.replace('P', '')
                    image_file = f'TC_P{patient_id}.nrrd'
                    segmentation_file = f'TC_P{patient_id}_seg.nrrd'

                    # Check if the image file exists
                    if os.path.exists(os.path.join(patient_path, image_file)):
                        data['TC']['Total Patients'] += 1
                        if os.path.exists(os.path.join(patient_path, segmentation_file)):
                            data['TC']['Meningioma Patients'] += 1
                        else:
                            data['TC']['Control Patients'] += 1

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
        keys = ['TC', 'RM/T1', 'RM/T1SIN', 'RM/T2', 'RM/SUSC']
        categories = ['Total Patients', 'Meningioma Patients', 'Control Patients']

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the bars
        bar_width = 0.2
        for i, category in enumerate(categories):
            ax.bar(
                [x + i * bar_width for x in range(len(keys))],
                df[category].reindex(keys),
                width=bar_width,
                label=category, alpha=0.6
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
        plt.savefig(os.path.join(self.target_dir, "patient_distribution_barplot.png"))
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
        df.to_csv(os.path.join(self.target_dir, "patient_distribution_stats.csv"))

class SizeStats:
    def __init__(self, source: str, target: str, verbose: bool = False) -> None:
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

        # CSV target files
        self.images_csv = os.path.join(self.target, 'images_sizes.csv')
        self.segmentations_csv = os.path.join(self.target, 'segmentations_sizes.csv')


    def _analyze_image_sizes(self) -> Dict[str, pd.DataFrame]:
        """
        Save and organize dimensionality data into a DataFrame for posterior visualizations and analysis.
        The DataFrame structure is as follows: 
        | Pulse | Patient | Height | Width | Slices |

        This DataFrame structure will be used for images and segmentations for each patient, generating 2 `.csv` files.

        Returns:
            Dict[str, pd.DataFrame]: Contains two DataFrames for images and segmentations.
        """
        images_df = pd.DataFrame(columns=['Pulse', 'Patient', 'Height', 'Width', 'Slices'])
        segmentations_df = pd.DataFrame(columns=['Pulse', 'Patient', 'Height', 'Width', 'Slices'])

        for adquisition in tqdm(os.listdir(self.source), desc="Analyzing Acquisitions"):  # First level is the acquisition format: RM or TC
            adquisition_path = os.path.join(self.source, adquisition)

            if not os.path.isdir(adquisition_path):
                continue  # Skip non-directory files

            if self.verbose:
                print(f"Analyzing Acquisition: {adquisition}")

            if adquisition == 'RM':  # RM folder has pulses within itself
                for pulse in os.listdir(adquisition_path):
                    pulse_path = os.path.join(adquisition_path, pulse)

                    if not os.path.isdir(pulse_path):
                        continue  # Skip non-directory files

                    if self.verbose:
                        print(f"  Pulse: {pulse}")

                    for patient in natsorted(os.listdir(pulse_path)):  # List the patients within the current pulse
                        patient_path = os.path.join(pulse_path, patient)

                        if not os.path.isdir(patient_path):
                            continue  # Skip non-directory files

                        if self.verbose:
                            print(f"    Patient: {patient}")

                        for file in os.listdir(patient_path):  # List the image files
                            file_path = os.path.join(patient_path, file)

                            if file.endswith('_seg.nrrd'):  # Segmentation file
                                segmentation_data, segmentation_header = nrrd.read(file_path)
                                segmentations_df = pd.concat([segmentations_df, pd.DataFrame({
                                    'Pulse': [f'{adquisition}/{pulse}'],
                                    'Patient': [patient],
                                    'Height': [segmentation_header['sizes'][0]],
                                    'Width': [segmentation_header['sizes'][1]],
                                    'Slices': [segmentation_header['sizes'][2]]
                                })], ignore_index=True)

                            elif file.endswith('.nrrd'):  # Image file
                                try:
                                    image_data, image_header = nrrd.read(file_path)
                                except Exception as e:
                                    if self.verbose: 
                                        print(f"    Error reading file {file} for Patient: {patient}. Error: {str(e)}")                      
                                    continue

                                images_df = pd.concat([images_df, pd.DataFrame({
                                    'Pulse': [f'{adquisition}/{pulse}'],
                                    'Patient': [patient],
                                    'Height': [image_header['sizes'][0]],
                                    'Width': [image_header['sizes'][1]],
                                    'Slices': [image_header['sizes'][2]]
                                })], ignore_index=True)

            elif adquisition == 'TC':  # TC folder does not have any pulses or image formats
                for patient in natsorted(os.listdir(adquisition_path)):
                    patient_path = os.path.join(adquisition_path, patient)

                    if not os.path.isdir(patient_path):
                        continue  # Skip non-directory files

                    if self.verbose:
                        print(f"  Patient: {patient}")

                    for file in os.listdir(patient_path):  # List the image files
                        file_path = os.path.join(patient_path, file)

                        if file.endswith('_seg.nrrd'):  # Segmentation file
                            segmentation_data, segmentation_header = nrrd.read(file_path)
                            segmentations_df = pd.concat([segmentations_df, pd.DataFrame({
                                'Pulse': [adquisition],
                                'Patient': [patient],
                                'Height': [segmentation_header['sizes'][0]],
                                'Width': [segmentation_header['sizes'][1]],
                                'Slices': [segmentation_header['sizes'][2]]
                            })], ignore_index=True)

                        elif file.endswith('.nrrd'):  # Image file
                            try:
                                image_data, image_header = nrrd.read(file_path)
                            except Exception as e:
                                if self.verbose: 
                                    print(f"    Error reading file {file} for Patient: {patient}. Error: {str(e)}")                      
                                continue
                            images_df = pd.concat([images_df, pd.DataFrame({
                                'Pulse': [adquisition],
                                'Patient': [patient],
                                'Height': [image_header['sizes'][0]],
                                'Width': [image_header['sizes'][1]],
                                'Slices': [image_header['sizes'][2]]
                            })], ignore_index=True)

        return {'images': images_df, 'segmentations': segmentations_df}

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
        pulses = df['Pulse'].unique()

        # Temporarily apply the scatter style from scienceplots
        with plt.style.context(['science', 'scatter']):
            # Create the scatter plot
            plt.figure(figsize=(10, 6))

            # Get the current marker cycle from the scatter style
            prop_cycle = plt.rcParams['axes.prop_cycle']
            markers_and_colors = prop_cycle.by_key()
            marker_cycle = markers_and_colors['marker']  # List of markers
            color_cycle = markers_and_colors['color']    # List of colors

            # Iterate through pulses and plot each with a different marker and color
            for i, pulse in enumerate(pulses):
                pulse_data = df[df['Pulse'] == pulse]
                marker = marker_cycle[i % len(marker_cycle)]  # Cycle through markers
                color = color_cycle[i % len(color_cycle)]     # Cycle through colors
                plt.scatter(pulse_data['Width'], pulse_data['Height'], 
                            label=pulse, marker=marker, color=color, alpha=0.8)

            # Add labels, title, and legend
            plt.title('Image Height vs. Width')
            plt.xlabel('Width')
            plt.ylabel('Height')
            plt.legend(title='Pulse', loc='lower right')
            plt.ylim(0, None)
            plt.tight_layout()

            # Save the figure
            output_path = os.path.join(self.target, 'scatter_height_vs_width.png')
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
        pulses = df['Pulse'].unique()
        height_data = [df[df['Pulse'] == pulse]['Height'] for pulse in pulses]
        width_data = [df[df['Pulse'] == pulse]['Width'] for pulse in pulses]

        # Violin plot for Height
        axes[0].violinplot(height_data, showmeans=True, showmedians=True, showextrema=True)
        axes[0].set_title('Height Distribution')
        axes[0].set_xlabel('Acquisition Type')
        axes[0].set_ylabel('Height')
        axes[0].set_ylim(0, None)
        axes[0].set_xticks(range(1, len(pulses) + 1))
        axes[0].set_xticklabels(pulses)

        # Violin plot for Width
        axes[1].violinplot(width_data, showmeans=True, showmedians=True, showextrema=True)
        axes[1].set_title('Width Distribution')
        axes[1].set_xlabel('Acquisition Type')
        axes[1].set_ylabel('Width')
        axes[1].set_ylim(0, None)
        axes[1].set_xticks(range(1, len(pulses) + 1))
        axes[1].set_xticklabels(pulses)

        # Adjust layout and save the figure
        plt.tight_layout()
        output_path = os.path.join(self.target, 'violin_height_width.png')
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
        size_pairs = list(zip(df['Height'], df['Width']))
        size_counts = Counter(size_pairs)

        # Get the 4 most common size pairs
        most_common = size_counts.most_common(4)

        # Prepare data for the bar plot
        labels = [f'{height}x{width}' for (height, width), _ in most_common] + ['Other']
        counts = [count for _, count in most_common] + [sum(count for pair, count in size_counts.items() if pair not in [pair for pair, _ in most_common])]

        # Generate the bar plot using matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get the default colors from the std-colors palette
        bar_positions = range(len(labels))
        ax.bar(bar_positions, counts, color=plt.rcParams['axes.prop_cycle'].by_key()['color'], alpha=0.8)

        # Set the x-ticks and labels
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(labels)

        # Set labels and title
        ax.set_xlabel('Size (Height, Width)')
        ax.set_ylabel('Number of Occurrences')
        ax.set_title('Most Frequent (Height, Width) Pairs')

        # Adjust layout and save the figure
        plt.tight_layout()
        output_path = os.path.join(self.target, 'barplot_frequent_sizes.png')
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
        pivot_table = df.pivot_table(index='Height', columns='Width', aggfunc='size', fill_value=0)

        # Sort the pivot table's index (Height) and columns (Width) in natural order
        pivot_table = pivot_table.sort_index(axis=0, ascending=True)  # Sort rows by Height
        pivot_table = pivot_table.sort_index(axis=1, ascending=True)  # Sort columns by Width

        # Plot the heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, cmap='viridis', cbar_kws={'label': 'Frequency'})
        plt.title('Frequency of Image (Height, Width) Pairs')
        plt.xlabel('Width')
        plt.ylabel('Height')
        plt.tight_layout()

        # Save the heatmap
        output_path = os.path.join(self.target, 'heatmap_size_frequency.png')
        os.makedirs(self.target, exist_ok=True)
        plt.savefig(output_path)
        plt.show()

    def analyze_dimensionality(self) -> None:
        results = self._analyze_image_sizes()

        # Save results to CSV files
        os.makedirs(self.target, exist_ok=True)
        results.get('images').to_csv(self.images_csv, index=False)
        results.get('segmentations').to_csv(self.segmentations_csv, index=False)

    def show_plots(self) -> None:
        self._scatter_plot_height_vs_width(self.images_csv)
        self._barplot_most_frequent_sizes(self.images_csv)
        self._heatmap_size_frequency(self.images_csv)
        self._plot_violin_height_width(self.images_csv)


def main() -> int:
    figures_root = './docs/figures/'
    source_transformed = os.path.join('/home/mariopasc/Python/Datasets/Meningiomas', 'Meningioma_Adquisition')

    stats_folder = os.path.join(figures_root, 'data_stats')
    stats_generator = AdquisitionStats(transformed_dir=source_transformed, target_dir=stats_folder)
    stats_generator.generate_stats()

    size_stats_folder = os.path.join(figures_root, 'size_stats')
    stats_generator = SizeStats(source=source_transformed, 
                                target=size_stats_folder, verbose=True)
    #stats_generator.analyze_dimensionality()
    stats_generator.show_plots()


if __name__ == '__main__':
    main()