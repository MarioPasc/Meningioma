# This script file tunes the sigma hyperparameter used in the Parzen-Rosenblatt window method (https://en.wikipedia.org/wiki/Kernel_density_estimation).
# The tuning of this hyperparameter is performed using both the Kullback–Leibler divergence (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
# and the Bhattacharyya distance (https://en.wikipedia.org/wiki/Bhattacharyya_distance). 
# The results of the experiment are stored in a csv file, which includes the optimal value of sigma to approximate the PDF of the background RM images' noise and 
# the image that it is related to.
# Several plots are generated to ensure a comprehensive analysis of the results of the algorithm, as well as log files containing the execution details.
# The user may set some global environments for the script, such as OUTPUT_FILE, which contains the path to the results of the script. 

##############
#   SETUP    #
##############

# Install the Meningioma package

import subprocess
import sys
import importlib.util
import os

# Path to the directory where the built package is located
MENINGIOMA_PACKAGE_PATH = "./src/Meningioma/dist/Meningioma-0.1.0-py3-none-any.whl"

def is_package_installed(package_name: str) -> bool:
    """Check if a package is installed."""
    package_spec = importlib.util.find_spec(package_name)
    return package_spec is not None

# Package name as specified in your setup.py
PACKAGE_NAME = "meningioma"

# Check if the package is already installed
if is_package_installed(PACKAGE_NAME):
    print(f"{PACKAGE_NAME} is already installed.")
else:
    # Install the package if not installed
    print(f"{PACKAGE_NAME} is not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", MENINGIOMA_PACKAGE_PATH])

# Test importing to ensure installation
try:
    # Import the module you need to test from your package
    from image_processing import ImageProcessing
    from metrics import Metrics
    import pandas as pd
    import numpy as np
    from typing import List
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import scienceplots
    plt.style.use(['science', 'ieee', 'std-colors'])
    plt.rcParams['font.size'] = 10
    plt.rcParams.update({'figure.dpi': '100'})
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    print("Package installed and imported successfully.")
except ImportError as e:
    print("Failed to import the package. Check the structure and path.")
    print(e)

# Configure global variables

# Input paths

DATASET_FOLDER = "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition"
OUTPUT_FOLDER = "/home/mariopasc/Python/Results/Meningioma/dataset_crafting"
CSV_PATH = os.path.join(OUTPUT_FOLDER, 'results.csv')

# Main parameters
H_VALUES = [0.00005, 0.0001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]        # Bandwidths for KDE models
SIGMA_VALUES = [9, 9.5, 10, 10.5, 11, 11.5]    # Sigma values for Rician distribution
N_SLICES = 6                       # Number of slices around the middle slice
PULSE_TYPES = ['T1']
TRAIN_TEST_SPLIT = 0.67            # Percentage for train set
RANDOM_SEED = 42                   # Seed for reproducibility

METADATA_PATH = os.path.join(DATASET_FOLDER, "metadata.xlsx")

##############
#   SCRIPT   #
##############

# Overview:
# For each patient:
#   1. Take the middle transversal slice +- N slices around this slice (2N+1 slices in total).
#   2. Extract all the background pixels and relate them to the slice they came from. 
#   3. Divide this set of pixel data into train and test.
#   4. Train models:
#       4.1. Train a Parzen-Rosenblatt KDE model with each h_i \epsilon [h_1, h_2, ..., h_n] given by the user. We will have N models
#            trained with the same training set for the specific image. 
#       4.2. Compute the Bhattacharyya's Distance and Kullback-Leibler's Divergence using the validation set for each Parze-Rosenblatt model. 
#       4.3. For each \sigma_i \epsilon [\sigma_1, \sigma_2, ..., \sigma_K] given by the user, compute the Rice distribution for each \sigma value. 
#       4.4. Compute the Bhattacharyya's Distance and Kullback-Leibler's Divergence using the validation set for each Rician distribution obtained.
#       4.5. Throughout all this steps, a `.csv` file will be created. This csv file will contain the following columns:
#            | Pulse | Patient | Model | Hyperparameter | Bhattacharyya | Kullback-Leibler | 
#               - Pulse will be [T1, T2, T1SIN, SUSC]
#               - Patient will contain the patient number
#               - Model will be either "ParzenRosenblatt" or "Rician"
#               - Hyperparameter will be the values used for the Model. If its Rician, it should be interpreted as the sigma value, if its ParzenRosenblatt, as h
#               - Bhattacharyya will contain the Bhattacharyya distance between between the model and the discrete distribution of the validation set.
#               - Kullback-Leibler will contain the Kullback-Leibler divergence between the model and the discrete distribution of the validation set.
#   5. Plotting stage: Using the .csv file, we will plot an specific Pareto front for a given pair of Pulse-Patient unique ID. This Pareto front will have the 
#      Bhattacharyya in the X-axis and the Kullback-Leibler in the Y-axis; We will have 2 data series, one for each Model value; We will plot as many points as 
#      samples are in the Hyperparameter column; The Pareto front will be drawn for each data series, showing the minimum front for each model.


# Initialize an empty dataframe to store results
results_df = pd.DataFrame(columns=['Pulse', 'Patient', 'Model', 'Hyperparameter', 'Bhattacharyya', 'Kullback-Leibler'])

def plot_distributions(patient_ids: List[int], target_pulse: str):
    for pulse in PULSE_TYPES:
        for patient_id in patient_ids:
            if pulse != target_pulse:
                continue  # Skip pulses not specified for plotting

            nrrd_path = f"{DATASET_FOLDER}/RM/{pulse}/P{patient_id}/{pulse}_P{patient_id}.nrrd"
            
            # 1. Load MRI Image
            image_data = ImageProcessing.open_nrrd_file(nrrd_path)
            transversal_axis = ImageProcessing.get_transversal_axis(nrrd_path)
            mid_slice_idx = image_data.shape[transversal_axis] // 2
            
            # 2. Extract 2N+1 slices around the middle slice
            slices = [ImageProcessing.extract_transversal_slice(image_data, transversal_axis, slice_index=mid_slice_idx + offset)
                      for offset in range(-N_SLICES, N_SLICES + 1)]

            # Split slices into train and test sets at the image level
            train_slices, test_slices = train_test_split(
                slices, train_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
            )

            # Training Phase: Collect noise data from all training slices
            train_noise_data = []
            for img in train_slices:
                mask = ImageProcessing.segment_intracraneal_region(img)
                filled_mask = ImageProcessing.fill_mask(mask)
                bbox = ImageProcessing.find_largest_bbox(filled_mask)
                noise_values = ImageProcessing.extract_noise_outside_bbox(img, bbox, filled_mask)
                train_noise_data.extend(noise_values)
            train_noise_data = np.array(train_noise_data)

            # Train Parzen-Rosenblatt KDE models with different h values
            kde_models = {h: ImageProcessing.kde(train_noise_data, h=h, return_x_values=False) for h in H_VALUES}

            # Testing Phase: Collect noise data from all test slices
            test_noise_data = []
            for img in test_slices:
                mask = ImageProcessing.segment_intracraneal_region(img)
                filled_mask = ImageProcessing.fill_mask(mask)
                bbox = ImageProcessing.find_largest_bbox(filled_mask)
                noise_values = ImageProcessing.extract_noise_outside_bbox(img, bbox, filled_mask)
                test_noise_data.extend(noise_values)
            test_noise_data = np.array(test_noise_data)

            # Generate x_values for model evaluations
            x_values = np.linspace(np.min(test_noise_data), np.max(test_noise_data), 1000)

            # Create Rician models with different sigma values
            rician_models = {sigma: ImageProcessing.rician(x_values, sigma) for sigma in SIGMA_VALUES}

            # Plotting
            def truncate_colormap(cmap, min_val=0.2, max_val=1.0, n=100):
                new_cmap = mcolors.LinearSegmentedColormap.from_list(
                    f"trunc({cmap.name},{min_val:.2f},{max_val:.2f})",
                    cmap(np.linspace(min_val, max_val, n))
                )
                return new_cmap

            # Apply truncated colormaps
            truncated_reds = truncate_colormap(plt.cm.Reds, min_val=np.min(H_VALUES) * 0.5, max_val=np.max(H_VALUES))
            truncated_blues = truncate_colormap(plt.cm.Blues, min_val=np.min(H_VALUES) * 0.5, max_val=np.max(SIGMA_VALUES))


            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot KDE models with gradient red (using truncated colormap)
            norm_h = mcolors.Normalize(vmin=min(H_VALUES), vmax=max(H_VALUES))
            for h, kde_pdf in kde_models.items():
                ax.plot(x_values, kde_pdf, color=truncated_reds(norm_h(h)))

            # Plot Rician models with gradient blue (using truncated colormap)
            norm_sigma = mcolors.Normalize(vmin=min(SIGMA_VALUES), vmax=max(SIGMA_VALUES))
            for sigma, rician_pdf in rician_models.items():
                ax.plot(x_values, rician_pdf, color=truncated_blues(norm_sigma(sigma)))

            # Create colorbars for h and sigma values with the truncated colormaps
            sm_h = plt.cm.ScalarMappable(cmap=truncated_reds, norm=norm_h)
            sm_sigma = plt.cm.ScalarMappable(cmap=truncated_blues, norm=norm_sigma)
            sm_h.set_array([])
            sm_sigma.set_array([])
            fig.colorbar(sm_h, ax=ax, label='KDE Bandwidth (h)', orientation='vertical', pad=0.1)
            fig.colorbar(sm_sigma, ax=ax, label='Rician $\sigma$', orientation='vertical', pad=0.1)

            # Plot the test distribution last, with a wider line width for prominence
            test_hist_counts, test_hist_edges = np.histogram(test_noise_data, bins=121, density=True)
            test_hist_centers = (test_hist_edges[:-1] + test_hist_edges[1:]) / 2
            ax.plot(test_hist_centers, test_hist_counts, color='grey', label="Test", linewidth=2.5, zorder=10)

            # Legend configuration with custom entries
            custom_lines = [
                plt.Line2D([0], [0], color='#808080', lw=2),
                plt.Line2D([0], [0], color='#70020e', lw=2),
                plt.Line2D([0], [0], color='#08326d', lw=2),
            ]
            ax.legend(custom_lines, ['Test', 'Parzen-Rosenblatt KDE', 'Rician'], loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False)

            # Final plot adjustments
            ax.set_title(f"Noise Distribution Comparison for Pulse {pulse} (Patient {patient_id})")
            ax.set_xlabel("Noise Intensity")
            ax.set_ylabel("Density")
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.get_xaxis().tick_bottom()
            ax.get_yaxis().tick_left()
            ax.set_xscale('log')
            plt.tight_layout()

            # Save or show the plot
            plot_path = os.path.join(OUTPUT_FOLDER, f"{pulse}_distribution_plot_patient_{patient_id}.png")
            plot_path = os.path.join(OUTPUT_FOLDER, f"{pulse}_distribution_plot_patient_{patient_id}.pdf")
            plt.savefig(plot_path, bbox_inches='tight')
            #plt.show()

            # End plotting for this pulse
            break

def perform_test(patient_ids: List[int]):
    for pulse in PULSE_TYPES:
        for patient_id in patient_ids:
            nrrd_path = f"{DATASET_FOLDER}/RM/{pulse}/P{patient_id}/{pulse}_P{patient_id}.nrrd"
            
            # 1. Load MRI Image
            image_data = ImageProcessing.open_nrrd_file(nrrd_path)
            transversal_axis = ImageProcessing.get_transversal_axis(nrrd_path)
            mid_slice_idx = image_data.shape[transversal_axis] // 2
            
            # 2. Extract 2N+1 slices around the middle slice
            slices = [ImageProcessing.extract_transversal_slice(image_data, transversal_axis, slice_index=mid_slice_idx + offset)
                      for offset in range(-N_SLICES, N_SLICES + 1)]

            # Split slices into train and test sets at the image level
            train_slices, test_slices = train_test_split(
                slices, train_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
            )

            # Training Phase: Collect noise data from all training slices
            train_noise_data = []
            for img in train_slices:
                mask = ImageProcessing.segment_intracraneal_region(img)
                filled_mask = ImageProcessing.fill_mask(mask)
                bbox = ImageProcessing.find_largest_bbox(filled_mask)
                noise_values = ImageProcessing.extract_noise_outside_bbox(img, bbox, filled_mask)
                train_noise_data.extend(noise_values)
            train_noise_data = np.array(train_noise_data)

            # Train Parzen-Rosenblatt KDE models with different h values
            kde_models = {h: ImageProcessing.kde(train_noise_data, h=h, return_x_values=False) for h in H_VALUES}
            x_values = np.linspace(np.min(train_noise_data), np.max(train_noise_data), 1000)

            # Testing Phase: Collect noise data from all test slices
            test_noise_data = []
            for img in test_slices:
                mask = ImageProcessing.segment_intracraneal_region(img)
                filled_mask = ImageProcessing.fill_mask(mask)
                bbox = ImageProcessing.find_largest_bbox(filled_mask)
                noise_values = ImageProcessing.extract_noise_outside_bbox(img, bbox, filled_mask)
                test_noise_data.extend(noise_values)
            test_noise_data = np.array(test_noise_data)

            # Create Rician models with different sigma values
            rician_models = {sigma: ImageProcessing.rician(x_values, sigma) for sigma in SIGMA_VALUES}

            # Evaluate Parzen-Rosenblatt models with test data
            for h, kde_pdf in kde_models.items():
                # Compute Bhattacharyya's and KL Divergences
                bhattacharyya_distance = Metrics.compute_bhattacharyya_distance(noise_values=test_noise_data, reference_pdf=kde_pdf, x_values=x_values)
                kl_divergence = Metrics.compute_kl_divergence(noise_values=test_noise_data, reference_pdf=kde_pdf, x_values=x_values)

                # Record results for Parzen-Rosenblatt model
                results_df.loc[len(results_df)] = [pulse, patient_id, 'ParzenRosenblatt', h, bhattacharyya_distance, kl_divergence]

            # Evaluate Rician models with test data
            for sigma, rician_pdf in rician_models.items():
                # Compute Bhattacharyya's and KL Divergences
                bhattacharyya_distance = Metrics.compute_bhattacharyya_distance(noise_values=test_noise_data, reference_pdf=rician_pdf, x_values=x_values)
                kl_divergence = Metrics.compute_kl_divergence(noise_values=test_noise_data, reference_pdf=rician_pdf, x_values=x_values)

                # Record results for Rician model
                results_df.loc[len(results_df)] = [pulse, patient_id, 'Rician', sigma, bhattacharyya_distance, kl_divergence]

    # Save results to CSV
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Ensure output directory exists
    results_df.to_csv(CSV_PATH, index=False)
    print("Results saved to", CSV_PATH)


perform_test(patient_ids=[1])
plot_distributions(patient_ids=[1], target_pulse="T1")
