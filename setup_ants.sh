#!/usr/bin/env bash
#
# install_ants.sh
#
# A convenience script for Ubuntu that:
#   1) Checks if ANTs is already installed (looking for 'antsRegistration').
#   2) If not installed, downloads the precompiled ANTs binaries for Ubuntu 22.04,
#      and unzips them into ~/ants.
#   3) Temporarily updates PATH to include ~/ants/bin. Also advises the user on
#      how to make this change permanent (e.g. in ~/.bashrc).
#
# Usage:
#   chmod +x install_ants.sh
#   ./install_ants.sh
#
# Notes:
#   - This script uses wget and unzip. Make sure you have them installed:
#       sudo apt-get update && sudo apt-get install -y wget unzip
#   - The script will only add ANTs to PATH for the current shell session.
#     To make it permanent, append the export line to your ~/.bashrc or similar.

set -e  # Exit immediately if a command exits with a non-zero status.

echo "=== Checking for an existing ANTs installation (looking for 'antsRegistration') ==="

if command -v antsRegistration &> /dev/null
then
    echo "[OK] ANTs is already installed at: $(which antsRegistration)"
else
    # 1) Define the download URL for the ANTs 2.5.4 Ubuntu 22.04 binary release
    ANTS_URL="https://github.com/ANTsX/ANTs/releases/download/v2.5.4/ants-2.5.4-ubuntu-22.04-X64-gcc.zip"
    ANTS_ZIP="ants-2.5.4-ubuntu-22.04-X64-gcc.zip"
    
    echo "[INFO] ANTs not found. Attempting to download and install to ~/ants"
    echo "Downloading from: $ANTS_URL"
    
    # 2) Download zip
    wget -O "$ANTS_ZIP" "$ANTS_URL"
    
    # 3) Create the target directory and unzip
    ANTS_DIR="$HOME/ants"
    mkdir -p "$ANTS_DIR"
    echo "[INFO] Extracting $ANTS_ZIP into $ANTS_DIR ..."
    unzip -q "$ANTS_ZIP" -d "$ANTS_DIR"
    rm "$ANTS_ZIP"
    
    # Typically, the bin folder is immediately inside the unzipped directory, e.g. ~/ants/bin
    # If the structure changes in the future, adjust accordingly.
    
    # 4) Temporarily add to PATH (this session only)
    export PATH="$ANTS_DIR/bin:$PATH"
    
    echo "========================="
    if command -v antsRegistration &> /dev/null
    then
        echo "[OK] ANTs installed successfully at: $(which antsRegistration)"
    else
        echo "[ERROR] ANTs installation appears to have failed or PATH is not updated."
        echo "Check the contents of $ANTS_DIR, and verify 'antsRegistration' is in the bin folder."
        echo
        echo "For future sessions, add the following line to your ~/.bashrc:"
        echo "echo 'export PATH=/home/{user}/ants/ants-2.5.4/bin:$PATH' >> ~/.bashrc"
        echo "source ~/.bashrc"
        echo
        echo "Then log out and log back in, or source your ~/.bashrc, to finalize the installation."
    fi
fi
