#!/usr/bin/env bash
#
#
# A convenience script for Ubuntu that creates/activates a
# conda environment and installs the Meningioma package in editable mode.
#
# Usage:
#   chmod +x setup_project.sh
#   ./setup_project.sh

set -e  # Exit immediately if a command exits with a non-zero status

# --------------------------------------------------
# Create/Activate conda environment, Install Meningioma
# --------------------------------------------------
CONDA_ENV_NAME="meningiomas"
PYTHON_VERSION="3.10"

# --------------------------------------------------
# Check for existing FSL
# --------------------------------------------------
echo "=== Checking for an existing FSL installation (looking for 'susan') ==="
if command -v susan &> /dev/null
then
    echo "[OK] FSL is already installed at: $(which susan)"
else
    echo "FSL not found, please, run ./setup_fsl.sh, and then re-run setup_project.sh"
    echo "Exiting"
    exit 1
fi

echo "=== Checking conda availability ==="

if ! command -v conda &> /dev/null
then
    echo "[ERROR] 'conda' not found. Please install Miniconda or Anaconda first."
    exit 1
fi

echo "=== Creating (or activating) conda environment: $CONDA_ENV_NAME ==="
if conda info --envs | grep "$CONDA_ENV_NAME" >/dev/null 2>&1
then
    echo "Conda environment '$CONDA_ENV_NAME' already exists. Activating..."
else
    echo "Conda environment '$CONDA_ENV_NAME' not found. Creating..."
    conda create -y -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION"
fi

echo "Activating environment..."
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

echo "=== Installing Meningioma in editable mode (pip install -e .) ==="
pip install -e .

echo "=== Installation successful! ==="
echo "Conda environment: $CONDA_ENV_NAME"
echo "You can now run: conda activate $CONDA_ENV_NAME"
echo "FSL is installed at: $FSLDIR"
