#!/usr/bin/env bash
#
#
# A convenience script for Ubuntu that:
#   1) Checks if FSL is already installed (looking for 'susan').
#   2) If not installed, downloads fslinstaller.py, and pipes an empty line
#      to accept the default installation path (~/fsl) non-interactively.
#
# Usage:
#   chmod +x install_fsl_and_meningioma.sh
#   ./install_fsl_and_meningioma.sh
#
# Notes:
#   - The fslinstaller.py script may prompt for sudo if you install outside your home directory.
#   - This script assumes Python3, conda, and wget are installed on your Ubuntu system.

set -e  # Exit immediately if a command exits with a non-zero status

# --------------------------------------------------
# 1) Check for existing FSL
# --------------------------------------------------
echo "=== Checking for an existing FSL installation (looking for 'susan') ==="
if command -v susan &> /dev/null
then
    echo "[OK] FSL is already installed at: $(which susan)"
else
    # --------------------------------------------------
    # 2) Download and run official fslinstaller.py
    # --------------------------------------------------
    INSTALLER_URL="https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py"
    INSTALLER_FILE="fslinstaller.py"
    
    echo "[INFO] FSL not found. Attempting non-interactive install via fslinstaller.py"
    echo "Downloading from: $INSTALLER_URL"
    wget -O "$INSTALLER_FILE" "$INSTALLER_URL"
    
    echo "Running FSL installer (piping a newline to accept default '~/fsl' path)."
    # If you need to accept more prompts, you can chain multiple lines or use 'yes "" | ...'
    if command -v python3 &> /dev/null; then
        (echo "") | python3 "$INSTALLER_FILE"
    else
        (echo "") | python "$INSTALLER_FILE"
    fi
    
    echo "========================="
    echo "FSL Installation complete. Log out and log in for changes to take place."
    echo "afterwards, run ./setup_project.sh"
    rm "$INSTALLER_FILE"
fi