#!/bin/bash

scriptspath="scripts/MRI_Noise_Modelling/"

# Set paths
export INPUT_IMAGE_PATH="$scriptspath/input_mri_image.png"
export OUTPUT_MATRIX_PATH="$scriptspath/responsibilities.mat"

# Run the MATLAB script
/usr/local/MATLAB/R2024b/bin/matlab -batch "addpath('src/Meningioma/external/HDIR/hdir'); ExecHDIR_MRI('$INPUT_IMAGE_PATH', '$OUTPUT_MATRIX_PATH');"

# Run the Python visualization script
python3 $scriptspath/visualize_mri.py --image $INPUT_IMAGE_PATH --matfile $OUTPUT_MATRIX_PATH --output $scriptspath/visualization.png

