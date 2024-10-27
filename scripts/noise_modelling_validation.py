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

import subprocess
import sys

# Path to the directory where the building
MENINGIOMA_PACKAGE_PATH = "./src/Meningioma/dist/Meningioma-0.1.0-py3-none-any.whl"  
# Install the package
subprocess.check_call([sys.executable, "-m", "pip", "install", MENINGIOMA_PACKAGE_PATH])

# Test importing to ensure installation
try:
    import image_processing
    print("Package installed and imported successfully.")
except ImportError as e:
    print("Failed to import the package. Check the structure and path.")
    print(e)


##############
#   SCRIPT   #
##############

