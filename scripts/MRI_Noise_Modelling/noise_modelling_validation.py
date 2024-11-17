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
    import logging
    import time
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
CSV_PATH = os.path.join(OUTPUT_FOLDER, 'noise_modelling_results.csv')

# Main parameters
H_VALUES = [0.00005, 0.0001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]        # Bandwidths for KDE models
SIGMA_VALUES = [8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5]    # Sigma values for Rician distribution
N_SLICES = 6                       # Number of slices around the middle slice
PULSE_TYPES = ['T1', 'T2', 'T1SIN', 'SUSC']
TRAIN_TEST_SPLIT = 0.67            # Percentage for train set
RANDOM_SEED = 42                   # Seed for reproducibility

METADATA_PATH = os.path.join(DATASET_FOLDER, "metadata.xlsx")

# Setup logging
os.makedirs("./logs", exist_ok=True)  
logging.basicConfig(
    filename="logs/script.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
results_df = pd.DataFrame(columns=['Pulse', 'Patient', 'Model', 'Hyperparameter', 'Bhattacharyya', 'Kullback-Leibler', 'ExecutionTime'])

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

def perform_test_normal(patient_ids: List[int]):
    total_start_time = time.perf_counter()
    for pulse in PULSE_TYPES:
        for patient_id in patient_ids:
            try:
                patient_start_time = time.perf_counter()
                # Construct the file path
                nrrd_path = f"{DATASET_FOLDER}/RM/{pulse}/P{patient_id}/{pulse}_P{patient_id}.nrrd"
                pulse_directory = os.path.dirname(nrrd_path)

                # Check if the directory for the pulse exists
                if not os.path.exists(pulse_directory):
                    logging.warning(f"Pulse {pulse} is not available for Patient {patient_id}. Skipping this patient.")
                    continue  # Skip to the next patient without saving any data

                # Step 1: Load MRI Image
                try:
                    start_time = time.perf_counter()
                    image_data = ImageProcessing.open_nrrd_file(nrrd_path)
                    transversal_axis = ImageProcessing.get_transversal_axis(nrrd_path)
                    mid_slice_idx = image_data.shape[transversal_axis] // 2
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    logging.info(f"Successfully loaded image for Patient {patient_id} and Pulse {pulse} in {elapsed_time:.4f} seconds")
                except Exception as e:
                    logging.error(f"Failed to load image for Patient {patient_id} and Pulse {pulse}: {str(e)}")
                    results_df.loc[len(results_df)] = [pulse, patient_id, 'ERROR', None, None, None, None]
                    continue

                # Step 2: Extract 2N+1 slices around the middle slice
                try:
                    start_time = time.perf_counter()
                    # Get the total number of slices along the transversal axis
                    total_slices = image_data.shape[transversal_axis]

                    # Compute the slice indices, ensuring they stay within the valid range
                    start_idx = max(mid_slice_idx - N_SLICES, 0)
                    end_idx = min(mid_slice_idx + N_SLICES + 1, total_slices)  # +1 because the end index is exclusive
                    slices = [
                        ImageProcessing.extract_transversal_slice(image_data, transversal_axis, slice_index=index)
                        for index in range(start_idx, end_idx)
                    ]

                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    logging.info(f"Extracted {len(slices)} slices for Patient {patient_id} in {elapsed_time:.4f} seconds")

                    # Log the number of slices taken and the total number of slices
                    logging.info(f"Taken {len(slices)} slices from {total_slices} total slices for Patient {patient_id}")

                    # If the extracted range is smaller than the requested 2N+1 slices, log a warning
                    if len(slices) < (2 * N_SLICES + 1):
                        logging.warning(
                            f"Requested {2 * N_SLICES + 1} slices, but only {len(slices)} slices were available for Patient {patient_id}"
                        )

                except Exception as e:
                    logging.error(f"Failed to extract slices for Patient {patient_id}: {str(e)}")
                    results_df.loc[len(results_df)] = [pulse, patient_id, 'ERROR', None, None, None, None]
                    continue

                # Step 3: Split slices into train and test sets
                try:
                    start_time = time.perf_counter()
                    train_slices, test_slices = train_test_split(
                        slices, train_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
                    )
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    logging.info(f"Train-test split successful for Patient {patient_id} in {elapsed_time:.4f} seconds")
                except Exception as e:
                    logging.error(f"Failed to split slices for Patient {patient_id}: {str(e)}")
                    results_df.loc[len(results_df)] = [pulse, patient_id, 'ERROR', None, None, None, None]
                    continue

                # Step 4: Create the Training set
                try:
                    start_time = time.perf_counter()
                    train_noise_data = []
                    for img in train_slices:
                        mask = ImageProcessing.segment_intracraneal_region(img)
                        filled_mask = ImageProcessing.fill_mask(mask)
                        bbox = ImageProcessing.find_largest_bbox(filled_mask)
                        noise_values = ImageProcessing.extract_noise_outside_bbox(img, bbox, filled_mask)
                        train_noise_data.extend(noise_values)
                    train_noise_data = np.array(train_noise_data)
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    logging.info(f"Training data collection successful for Patient {patient_id} in {elapsed_time:.4f} seconds")
                except Exception as e:
                    logging.error(f"Failed to collect training noise data for Patient {patient_id}: {str(e)}")
                    results_df.loc[len(results_df)] = [pulse, patient_id, 'ERROR', None, None, None, None]
                    continue

                # Step 5: Create the Test set
                try:
                    start_time = time.perf_counter()
                    test_noise_data = []
                    for img in test_slices:
                        mask = ImageProcessing.segment_intracraneal_region(img)
                        filled_mask = ImageProcessing.fill_mask(mask)
                        bbox = ImageProcessing.find_largest_bbox(filled_mask)
                        noise_values = ImageProcessing.extract_noise_outside_bbox(img, bbox, filled_mask)
                        test_noise_data.extend(noise_values)
                    test_noise_data = np.array(test_noise_data)
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    logging.info(f"Testing data collection successful for Patient {patient_id} in {elapsed_time:.4f} seconds")
                except Exception as e:
                    logging.error(f"Failed to collect testing noise data for Patient {patient_id}: {str(e)}")
                    results_df.loc[len(results_df)] = [pulse, patient_id, 'ERROR', None, None, None, None]
                    continue

                # Step 6: Train Parzen-Rosenblatt KDE models
                try:
                    start_time = time.perf_counter()
                    kde_models = {h: ImageProcessing.kde(train_noise_data, h=h, return_x_values=False) for h in H_VALUES}
                    x_values = np.linspace(np.min(train_noise_data), np.max(train_noise_data), 1000)
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    logging.info(f"KDE models trained successfully for Patient {patient_id} in {elapsed_time:.4f} seconds")
                except Exception as e:
                    logging.error(f"Failed to train KDE models for Patient {patient_id}: {str(e)}")
                    results_df.loc[len(results_df)] = [pulse, patient_id, 'ERROR', None, None, None, None]
                    continue

                # Step 7: Create Rician models
                try:
                    start_time = time.perf_counter()
                    rician_models = {sigma: ImageProcessing.rician(x_values, sigma) for sigma in SIGMA_VALUES}
                    end_time = time.perf_counter()
                    elapsed_time = end_time - start_time
                    logging.info(f"Rician models created successfully for Patient {patient_id} in {elapsed_time:.4f} seconds")
                except Exception as e:
                    logging.error(f"Failed to create Rician models for Patient {patient_id}: {str(e)}")
                    results_df.loc[len(results_df)] = [pulse, patient_id, 'ERROR', None, None, None, None]
                    continue

                # Evaluate Parzen-Rosenblatt KDE models
                try:
                    start_time = time.perf_counter()
                    for h, kde_pdf in kde_models.items():
                        bhattacharyya_distance = Metrics.compute_bhattacharyya_distance(
                            noise_values=test_noise_data, reference_pdf=kde_pdf, x_values=x_values
                        )
                        kl_divergence = Metrics.compute_kl_divergence(
                            noise_values=test_noise_data, reference_pdf=kde_pdf, x_values=x_values
                        )
                        elapsed_time = time.perf_counter() - start_time
                        results_df.loc[len(results_df)] = [
                            pulse, patient_id, 'ParzenRosenblatt', h,
                            bhattacharyya_distance, kl_divergence, elapsed_time
                        ]
                        logging.info(f"Evaluated KDE model with h={h} for Patient {patient_id} in {elapsed_time:.4f} seconds")
                        start_time = time.perf_counter()  # Reset start time for next iteration
                except Exception as e:
                    logging.error(f"Failed to evaluate Parzen-Rosenblatt models for Patient {patient_id}: {str(e)}")
                    results_df.loc[len(results_df)] = [pulse, patient_id, 'ERROR', None, None, None, None]
                    continue

                # Evaluate Rician models
                try:
                    start_time = time.perf_counter()
                    for sigma, rician_pdf in rician_models.items():
                        bhattacharyya_distance = Metrics.compute_bhattacharyya_distance(
                            noise_values=test_noise_data, reference_pdf=rician_pdf, x_values=x_values
                        )
                        kl_divergence = Metrics.compute_kl_divergence(
                            noise_values=test_noise_data, reference_pdf=rician_pdf, x_values=x_values
                        )
                        elapsed_time = time.perf_counter() - start_time
                        results_df.loc[len(results_df)] = [
                            pulse, patient_id, 'Rician', sigma,
                            bhattacharyya_distance, kl_divergence, elapsed_time
                        ]
                        logging.info(f"Evaluated Rician model with sigma={sigma} for Patient {patient_id} in {elapsed_time:.4f} seconds")
                        start_time = time.perf_counter()  # Reset start time for next iteration
                except Exception as e:
                    logging.error(f"Failed to evaluate Rician models for Patient {patient_id}: {str(e)}")
                    results_df.loc[len(results_df)] = [pulse, patient_id, 'ERROR', None, None, None, None]
                    continue

                patient_end_time = time.perf_counter()
                patient_elapsed_time = patient_end_time - patient_start_time
                logging.info(f"Total time for Patient {patient_id}: {patient_elapsed_time:.2f} seconds")

            except Exception as e:
                logging.critical(f"Unexpected error for Patient {patient_id} and Pulse {pulse}: {str(e)}")
                results_df.loc[len(results_df)] = [pulse, patient_id, 'ERROR', None, None, None, None]
                continue

    total_end_time = time.perf_counter()
    total_elapsed_time = total_end_time - total_start_time
    logging.info(f"Total execution time for perform_test: {total_elapsed_time:.2f} seconds")

    # Save results to CSV
    
    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        results_df.to_csv(CSV_PATH, index=False)
        logging.info(f"Results successfully saved to {CSV_PATH}")
    except Exception as e:
        logging.error(f"Failed to save results to CSV: {str(e)}")
    
import os
import numpy as np
import time
import logging
from typing import List
from sklearn.model_selection import train_test_split

# Assume all necessary imports and constants like PULSE_TYPES, DATASET_FOLDER, etc., are defined

def perform_test(patient_ids: List[int]):
    total_start_time = time.perf_counter()
    for pulse in PULSE_TYPES:
        for patient_id in patient_ids:
            try:
                patient_start_time = time.perf_counter()
                nrrd_path = f"{DATASET_FOLDER}/RM/{pulse}/P{patient_id}/{pulse}_P{patient_id}.nrrd"
                pulse_directory = os.path.dirname(nrrd_path)

                if not os.path.exists(pulse_directory):
                    logging.warning(f"Pulse {pulse} is not available for Patient {patient_id}. Skipping this patient.")
                    continue

                # Step 1: Load MRI Image
                try:
                    start_time = time.perf_counter()
                    image_data = ImageProcessing.open_nrrd_file(nrrd_path)
                    transversal_axis = ImageProcessing.get_transversal_axis(nrrd_path)
                    mid_slice_idx = image_data.shape[transversal_axis] // 2
                    elapsed_time = time.perf_counter() - start_time
                    logging.info(f"Successfully loaded image for Patient {patient_id} and Pulse {pulse} in {elapsed_time:.4f} seconds")
                except Exception as e:
                    logging.error(f"Failed to load image for Patient {patient_id} and Pulse {pulse}: {str(e)}")
                    continue

                # Step 2: Extract 2N+1 slices around the middle slice
                try:
                    start_time = time.perf_counter()
                    total_slices = image_data.shape[transversal_axis]
                    start_idx = max(mid_slice_idx - N_SLICES, 0)
                    end_idx = min(mid_slice_idx + N_SLICES + 1, total_slices)
                    slices = [
                        ImageProcessing.extract_transversal_slice(image_data, transversal_axis, index)
                        for index in range(start_idx, end_idx)
                    ]
                    elapsed_time = time.perf_counter() - start_time
                    logging.info(f"Extracted {len(slices)} slices for Patient {patient_id} in {elapsed_time:.4f} seconds")

                    if len(slices) < (2 * N_SLICES + 1):
                        logging.warning(
                            f"Requested {2 * N_SLICES + 1} slices, but only {len(slices)} slices were available for Patient {patient_id}"
                        )
                except Exception as e:
                    logging.error(f"Failed to extract slices for Patient {patient_id}: {str(e)}")
                    continue

                # Step 3: Split slices into train and test sets
                try:
                    start_time = time.perf_counter()
                    train_slices, test_slices = train_test_split(
                        slices, train_size=TRAIN_TEST_SPLIT, random_state=RANDOM_SEED
                    )
                    elapsed_time = time.perf_counter() - start_time
                    logging.info(f"Train-test split successful for Patient {patient_id} in {elapsed_time:.4f} seconds")
                except Exception as e:
                    logging.error(f"Failed to split slices for Patient {patient_id}: {str(e)}")
                    continue

                # Step 4: Create the Training set
                try:
                    start_time = time.perf_counter()
                    train_noise_data = []
                    for img in train_slices:
                        mask = ImageProcessing.segment_intracraneal_region(img)
                        filled_mask = ImageProcessing.fill_mask(mask)
                        bbox = ImageProcessing.find_largest_bbox(filled_mask)
                        noise_values = ImageProcessing.extract_noise_outside_bbox(img, bbox, filled_mask)
                        train_noise_data.extend(noise_values)
                    train_noise_data = np.array(train_noise_data)
                    elapsed_time = time.perf_counter() - start_time
                    logging.info(f"Training data collection successful for Patient {patient_id} in {elapsed_time:.4f} seconds")
                except Exception as e:
                    logging.error(f"Failed to collect training noise data for Patient {patient_id}: {str(e)}")
                    continue

                # Step 5: Create the Test set
                try:
                    start_time = time.perf_counter()
                    test_noise_data = []
                    for img in test_slices:
                        mask = ImageProcessing.segment_intracraneal_region(img)
                        filled_mask = ImageProcessing.fill_mask(mask)
                        bbox = ImageProcessing.find_largest_bbox(filled_mask)
                        noise_values = ImageProcessing.extract_noise_outside_bbox(img, bbox, filled_mask)
                        test_noise_data.extend(noise_values)
                    test_noise_data = np.array(test_noise_data)
                    elapsed_time = time.perf_counter() - start_time
                    logging.info(f"Testing data collection successful for Patient {patient_id} in {elapsed_time:.4f} seconds")
                except Exception as e:
                    logging.error(f"Failed to collect testing noise data for Patient {patient_id}: {str(e)}")
                    continue

                # Step 6: Train Parzen-Rosenblatt KDE models
                try:
                    start_time = time.perf_counter()
                    kde_models = {h: ImageProcessing.kde(train_noise_data, h=h, return_x_values=False) for h in H_VALUES}
                    x_values = np.linspace(np.min(train_noise_data), np.max(train_noise_data), 1000)
                    elapsed_time = time.perf_counter() - start_time
                    logging.info(f"KDE models trained successfully for Patient {patient_id} in {elapsed_time:.4f} seconds")
                except Exception as e:
                    logging.error(f"Failed to train KDE models for Patient {patient_id}: {str(e)}")
                    continue

                # Step 7: Create Rician models
                try:
                    start_time = time.perf_counter()
                    rician_models = {sigma: ImageProcessing.rician(x_values, sigma) for sigma in SIGMA_VALUES}
                    elapsed_time = time.perf_counter() - start_time
                    logging.info(f"Rician models created successfully for Patient {patient_id} in {elapsed_time:.4f} seconds")
                except Exception as e:
                    logging.error(f"Failed to create Rician models for Patient {patient_id}: {str(e)}")
                    continue

                # Save the data
                save_folder = f"saved_data/Patient_{patient_id}_{pulse}"
                os.makedirs(save_folder, exist_ok=True)

                try:
                    np.savez(f"{save_folder}/train_noise_data.npz", data=train_noise_data)
                    np.savez(f"{save_folder}/test_noise_data.npz", data=test_noise_data)
                    np.savez(f"{save_folder}/x_values.npz", data=x_values)
                    np.savez(f"{save_folder}/kde_models.npz", **{str(h): kde for h, kde in kde_models.items()})
                    np.savez(f"{save_folder}/rician_models.npz", **{str(sigma): model for sigma, model in rician_models.items()})
                    logging.info(f"Data successfully saved for Patient {patient_id} and Pulse {pulse}")
                except Exception as e:
                    logging.error(f"Failed to save data for Patient {patient_id} and Pulse {pulse}: {str(e)}")
                    continue

                patient_elapsed_time = time.perf_counter() - patient_start_time
                logging.info(f"Total time for Patient {patient_id}: {patient_elapsed_time:.2f} seconds")

            except Exception as e:
                logging.critical(f"Unexpected error for Patient {patient_id} and Pulse {pulse}: {str(e)}")
                continue

    total_elapsed_time = time.perf_counter() - total_start_time
    logging.info(f"Total execution time for perform_test: {total_elapsed_time:.2f} seconds")


H_VALUES = [0.0001, 0.001, 0.01, 0.1, 1]        # Bandwidths for KDE models
SIGMA_VALUES = [8]    # Sigma values for Rician distribution
N_SLICES = 6                       # Number of slices around the middle slice
PULSE_TYPES = ['T1']
TRAIN_TEST_SPLIT = 0.67            # Percentage for train set
RANDOM_SEED = 42                   # Seed for reproducibility
perform_test(patient_ids=[1])


# plot_distributions(patient_ids=[1], target_pulse="T1")

