import os
import urllib.request

def download_synthmorph_model(model_name="shapes-dice-vel-3-res-8-16-32-256f.h5", 
                             output_dir=None):
    """
    Downloads the SynthMorph model if not already present.
    
    Args:
        model_name: Name of the model file
        output_dir: Directory to save the model. If None, saves to the current directory.
    
    Returns:
        Path to the downloaded model
    """
    if output_dir is None:
        output_dir = os.getcwd()
    
    model_path = os.path.join(output_dir, model_name)
    
    if not os.path.exists(model_path):
        print(f"Downloading {model_name}...")
        url = f"https://surfer.nmr.mgh.harvard.edu/ftp/data/voxelmorph/synthmorph/{model_name}"
        urllib.request.urlretrieve(url, model_path)
        print(f"Downloaded to {model_path}")
    else:
        print(f"Model found at {model_path}")
    
    return model_path