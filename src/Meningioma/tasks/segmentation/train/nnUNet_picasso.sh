#!/usr/bin/env bash
#SBATCH -J nnUNet_Training
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40  # Increased for multi-GPU job
#SBATCH --gres=gpu:5        # Request 5 GPUs in a single job
#SBATCH --mem=160gb         # Increased for multi-GPU job
##SBATCH --constraint=cal    # Use nodes with the 'cal' feature
#SBATCH --time=00:01:00     # Adjust based on expected training time
#SBATCH --error=nnunet_training_%j.err
#SBATCH --output=nnunet_training_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mpascual@uma.es

set -e
echo "Job started at $(date)"

# Parameters - customize these or pass as environment variables
DATASET_ID=${DATASET_ID:-"501"}  # Dataset ID or name
CONFIGURATION=${CONFIGURATION:-"3d_fullres"}  # nnUNet configuration
SAVE_NPZ=${SAVE_NPZ:-false}  # Whether to save softmax predictions
FOLDS=${FOLDS:-5}  # Number of folds to train

echo "========================================="
echo "Dataset: $DATASET_ID, Configuration: $CONFIGURATION"
echo "Save NPZ: $SAVE_NPZ"
echo "Folds: $FOLDS"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "========================================="

# Load modules 
module load miniconda 
source activate meningiomas

# Set nnUNet environment variables
ROOT="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/brats"
export nnUNet_raw="$ROOT/nnUNet_raw"
export nnUNet_preprocessed="$ROOT/nnUNet_preprocessed" 
export nnUNet_results="$ROOT/nnUNet_results"

# Print GPU information
echo "Available GPUs:"
nvidia-smi

# Determine GPU IDs from CUDA_VISIBLE_DEVICES or default to 0-4
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # Convert comma-separated string to space-separated for the Python script
    GPU_IDS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ')
    echo "Using GPUs from CUDA_VISIBLE_DEVICES: $GPU_IDS"
else
    # Default to first 5 GPUs if CUDA_VISIBLE_DEVICES not set
    GPU_IDS="0 1 2 3 4"
    echo "CUDA_VISIBLE_DEVICES not set, using default GPUs: $GPU_IDS"
fi

# Build command to run the Python script
SCRIPT_PATH="/home/mariopasc/Python/Projects/Meningioma/src/Meningioma/tasks/segmentation/train/nnUNet_trainer.py"

CMD="python $SCRIPT_PATH --dataset_id $DATASET_ID --configuration $CONFIGURATION --folds $FOLDS --gpus $GPU_IDS"

if [ "$SAVE_NPZ" = true ]; then
    CMD="$CMD --save_npz"
fi

echo "Executing command: $CMD"
echo "Starting multi-GPU training at $(date)"

# Execute the training with timing
time $CMD

# Check exit status
TRAINING_STATUS=$?
if [ $TRAINING_STATUS -ne 0 ]; then
    echo "ERROR: nnUNet training failed with status $TRAINING_STATUS!"
    exit $TRAINING_STATUS
fi

echo "Training completed successfully at $(date)"

# Create a marker file to indicate completion
touch "${nnUNet_results}/trained_model_${DATASET_ID}_${CONFIGURATION}_all_folds_$(date +%Y%m%d_%H%M%S).complete"

echo "Job completed at $(date)"