#!/usr/bin/env bash
#SBATCH -J nnUNet_Training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8  # Adjust based on your needs
#SBATCH --gres=gpu:1       # Each array job requests one GPU
#SBATCH --mem=32gb         # Adjust based on your dataset size
#SBATCH --constraint=cal   # Use nodes with the 'cal' feature
#SBATCH --time=72:00:00    # Adjust based on expected training time
#SBATCH --error=nnunet_fold_%a_%j.err
#SBATCH --output=nnunet_fold_%a_%j.out
#SBATCH --array=0-4        # For 5-fold cross validation
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mpascual@uma.es

set -e
echo "Job started at $(date)"

# Parameters - customize these or pass as environment variables
DATASET_ID=${DATASET_ID:-"501"}  # Dataset ID or name
CONFIGURATION=${CONFIGURATION:-"3d_fullres"}  # nnUNet configuration
SAVE_NPZ=${SAVE_NPZ:-false}  # Whether to save softmax predictions

# Map array index to fold number (direct mapping for 5-fold CV)
FOLD=$SLURM_ARRAY_TASK_ID

echo "========================================="
echo "Processing nnUNet fold: $FOLD"
echo "Dataset: $DATASET_ID, Configuration: $CONFIGURATION"
echo "Save NPZ: $SAVE_NPZ"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Job ID: $SLURM_JOB_ID, Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "========================================="

# Load modules - adjust based on your cluster's configuration
module load cuda/11.8  # Change to match your cluster's available CUDA version
module load miniconda  # Or whatever module system you use for Python

# Activate conda environment
source activate nnunet_env  # Replace with your actual environment name

# Set nnUNet environment variables - IMPORTANT: use your actual paths
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed" 
export nnUNet_results="/path/to/nnUNet_results"

# Create temp dir in localscratch according to documentation
MYLOCALSCRATCH=$LOCALSCRATCH/$USER/$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID
mkdir -p $MYLOCALSCRATCH

# Print information about the temporary directory
echo "Using local scratch directory: $MYLOCALSCRATCH"
echo "Available space in $LOCALSCRATCH: $(df -h $LOCALSCRATCH | awk 'NR==2 {print $4}')"

# Use local scratch for temporary files
export TMPDIR="$MYLOCALSCRATCH"

# Print GPU information
nvidia-smi

# Build the nnUNet command
CMD="nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD"

# Add npz flag if requested
if [ "$SAVE_NPZ" = true ]; then
    CMD="$CMD --npz"
fi

echo "Executing command: $CMD"
echo "Starting training at $(date)"

# Execute the training with timing
cd $MYLOCALSCRATCH  # Change to local scratch directory for faster I/O
time $CMD

# Check exit status
TRAINING_STATUS=$?
if [ $TRAINING_STATUS -ne 0 ]; then
    echo "ERROR: nnUNet training failed with status $TRAINING_STATUS!"
    exit $TRAINING_STATUS
fi

echo "Training completed successfully at $(date)"

# Optionally create a marker file to indicate completion
touch "${nnUNet_results}/trained_model_${DATASET_ID}_${CONFIGURATION}_fold${FOLD}_$(date +%Y%m%d_%H%M%S).complete"

# Clean up localscratch properly
echo "Cleaning up localscratch directory..."
if cd $LOCALSCRATCH/$USER; then
    if [ ! -z "$MYLOCALSCRATCH" ]; then
        rm -rf --one-file-system $MYLOCALSCRATCH
    fi
fi

echo "Job completed at $(date)"