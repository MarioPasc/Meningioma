#!/bin/bash

# Set paths
export INPUT_IMAGE_PATH="scripts/MRI_Noise_Modelling/SUSC_P29_MIDDLE_SLICE.png"
export OUTPUT_MATRIX_PATH="scripts/MRI_Noise_Modelling/responsibilities.mat"

# Run the MATLAB script
/usr/local/MATLAB/R2024b/bin/matlab -batch "addpath('src/Meningioma/external/HDIR/hdir'); ExecHDIR_MRI('$INPUT_IMAGE_PATH', '$OUTPUT_MATRIX_PATH');"

# Run the Python visualization script
python3 scripts/MRI_Noise_Modelling/visualize_mri.py --image "$INPUT_IMAGE_PATH" --matfile "$OUTPUT_MATRIX_PATH"
