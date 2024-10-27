import os
import importlib.util

def install_conda_requirements_with_pip(requirements_file: str):
    """
    Reads a Conda-formatted requirements file and installs the listed packages with pip if not already installed.
    
    Parameters:
    - requirements_file (str): Path to the Conda requirements file.
    
    The function will ignore Conda-specific packages (indicated by prefixes like '_'),
    and will check if each package is already installed before attempting to install it.
    """
    with open(requirements_file, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip comments and Conda-specific packages
            if line.startswith('#') or '=' not in line or line.startswith('_'):
                continue
            
            # Split package name and version
            package, version = line.split('=')[:2]
            pip_format_package = f"{package}=={version}"
            
            # Check if the package is already installed
            if importlib.util.find_spec(package) is None:
                print(f"{package} not found. Installing...")
                
                # Install the package using pip
                result = os.system(f"pip install {pip_format_package}")
                if result == 0:
                    print(f"Successfully installed {pip_format_package}")
                else:
                    print(f"Failed to install {pip_format_package}")
            else:
                print(f"{package} is already installed.")

# Usage
install_conda_requirements_with_pip("conda_requirements.txt")
