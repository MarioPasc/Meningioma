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
    import image_processing
    print("Package installed and imported successfully.")
except ImportError as e:
    print("Failed to import the package. Check the structure and path.")
    print(e)

# Configure global variables

# Input paths
DATASET_FOLDER = "/home/mariopasc/Python/Datasets/Meningiomas/Meningioma_Adquisition"
METADATA_PATH = os.path.join(DATASET_FOLDER, "metadata.xlsx")

# Output paths
OUTPUT_FOLDER = "/home/mariopasc/Python/Results/Meningioma/dataset_crafting"

##############
#   SCRIPT   #
##############

# Overview:


import pandas as pd

metadata = pd.read_excel(io = METADATA_PATH, header = 1)

print(metadata)

