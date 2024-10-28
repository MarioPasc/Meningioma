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
    from typing import List
    from sklearn.model_selection import train_test_split

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
H_VALUES = [0.5, 1.0, 1.5]        # Bandwidths for KDE models
SIGMA_VALUES = [0.5, 1.0, 1.5]    # Sigma values for Rician distribution
N_SLICES = 5                       # Number of slices around the middle slice
PULSE_TYPES = ['T1', 'T2', 'T1SIN', 'SUSC']
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

def main(patient_ids: List[int]):
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

            for slice_image, data_type in zip([train_slices, test_slices], ["train", "test"]):
                for img in slice_image:
                    # 3. Segment intracranial region and mask background
                    mask = ImageProcessing.segment_intracraneal_region(img)
                    filled_mask = ImageProcessing.fill_mask(mask)

                    # 4. Identify largest bounding box and extract noise outside it
                    bbox = ImageProcessing.find_largest_bbox(filled_mask)
                    noise_values = ImageProcessing.extract_noise_outside_bbox(img, bbox, filled_mask)

                    # Model training and evaluation based on noise extracted from `train` or `test` images

                    if data_type == "train":
                        # Parzen-Rosenblatt KDE for train slices
                        for h in H_VALUES:
                            kde_pdf, x_values = ImageProcessing.kde(noise_values, h=h, return_x_values=True)

                            # Compute Bhattacharyya's and KL Divergences using Metrics class
                            bhattacharyya_distance = Metrics.compute_bhattacharyya_distance(noise_values, kde_pdf)
                            kl_divergence = Metrics.compute_kl_divergence(noise_values, kde_pdf)

                            # Record results in DataFrame
                            results_df.loc[len(results_df)] = [pulse, patient_id, 'ParzenRosenblatt', h, bhattacharyya_distance, kl_divergence]

                    # Rician Distribution for train and test slices
                    for sigma in SIGMA_VALUES:
                        rician_pdf = ImageProcessing.rician(x_values, sigma)

                        # Compute Bhattacharyya's and KL Divergences using Metrics class
                        bhattacharyya_distance = Metrics.compute_bhattacharyya_distance(noise_values, rician_pdf)
                        kl_divergence = Metrics.compute_kl_divergence(noise_values, rician_pdf)

                        # Record results in DataFrame
                        results_df.loc[len(results_df)] = [pulse, patient_id, 'Rician', sigma, bhattacharyya_distance, kl_divergence]

    # Save results to CSV
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Ensure output directory exists
    results_df.to_csv(CSV_PATH, index=False)
    print("Results saved to", CSV_PATH)

main(patient_ids=[1])