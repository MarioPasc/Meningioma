#!/usr/bin/env python3
"""
run_smore.py
===========

Unified entry point for SMORE super-resolution, supporting both training and inference modes.
Processes each pulse type individually across each resolution.

Example usage:
    python -m src.mgmGrowth.tasks.superresolution.cli.run_smore \
        --config src/mgmGrowth/tasks/superresolution/cfg/smore_cfg.yaml

Output:

root/
└── SMORE/
    ├── 3mm/
    │   ├── weights/
    │   │   ├── BraTS-MEN-00018-000-t1c.pt
    │   │   ├── BraTS-MEN-00018-000-t2w.pt
    │   │   └── ...
    │   └── output_volumes/
    │       ├── BraTS-MEN-00018-000-t1c.nii.gz
    │       ├── BraTS-MEN-00018-000-t2w.nii.gz
    │       └── ...
    ├── 5mm/
    │   ├── weights/
    │   └── output_volumes/
    └── 7mm/
        ├── weights/
        └── output_volumes/
"""
from __future__ import annotations

import argparse
import sys
import shutil
from pathlib import Path

from src.mgmGrowth.tasks.superresolution import LOGGER
from src.mgmGrowth.tasks.superresolution.config import SmoreFullConfig, SmoreConfig
from src.mgmGrowth.tasks.superresolution.engine.smore_runner import run_smore, infer_volume
from src.mgmGrowth.tasks.superresolution.tools.paths import ensure_dir


def run_training_and_inference(config: SmoreFullConfig) -> None:
    """Run SMORE training and inference in one step using run_smore."""
    LOGGER.info("Running SMORE in training mode (with inference)")
    
    # Process each resolution
    for resolution in config.low_res_slices:
        resolution_dir = config.train_root / resolution
        if not resolution_dir.exists():
            LOGGER.warning(f"Directory for {resolution} not found at {resolution_dir}, skipping")
            continue
            
        LOGGER.info(f"Processing {resolution} data from {resolution_dir}")
        
        # Extract numerical slice thickness value
        slice_dz = float(resolution.replace("mm", ""))
        
        # Create output directories for this resolution
        smore_resolution_dir = ensure_dir(config.out_root / "SMORE" / resolution)
        weights_dir = ensure_dir(smore_resolution_dir / "weights")
        output_dir = ensure_dir(smore_resolution_dir / "output_volumes")
        
        # Process each pulse type
        for pulse in config.pulses:
            LOGGER.info(f"Processing {pulse} pulse type")
            
            # Find all volumes with this pulse type
            pulse_volumes = list(resolution_dir.rglob(f"*-{pulse}.nii.gz"))
            
            if not pulse_volumes:
                LOGGER.warning(f"No {pulse} volumes found in {resolution_dir}")
                continue
                
            LOGGER.info(f"Found {len(pulse_volumes)} {pulse} volumes to process")
            
            # Process each volume
            for vol in pulse_volumes:
                LOGGER.info(f"Training and inferring on volume: {vol}")
                
                # Get base name (without any extensions)
                base_name = vol.stem
                
                # Create temporary output directory
                temp_dir = ensure_dir(smore_resolution_dir / "temp")
                
                # Suffix for the output files
                suffix = "_smore"
                
                # Run SMORE (trains and infers in one step)
                run_smore(
                    vol, 
                    temp_dir,
                    cfg=config.network,
                    slice_thickness=slice_dz,
                    gpu_id=config.network.gpu_id,
                    suffix=suffix
                )
                
                # ======= CORRECT PATH HANDLING FROM OBSERVED STRUCTURE =======
                # ACTUAL SMORE directory tree:
                #   temp_dir/
                #   └── base_name/
                #       ├── base_name_suffix.nii.gz
                #       └── weights/
                #           └── best_weights.pt
                
                # Paths to generated files
                volume_dir = temp_dir / base_name
                best_weights = volume_dir / "weights" / "best_weights.pt"
                sr_volume = volume_dir / f"{base_name}{suffix}.nii.gz"
                
                LOGGER.info(f"Looking for weights at: {best_weights}")
                LOGGER.info(f"Looking for SR volume at: {sr_volume}")
                
                # Target paths
                target_weights = weights_dir / f"{base_name}.pt"
                target_sr = output_dir / f"{base_name}.nii.gz"
                
                # Move files to their final locations
                if best_weights.exists():
                    shutil.copy2(best_weights, target_weights)
                    LOGGER.info(f"Saved weights to {target_weights}")
                else:
                    LOGGER.error(f"Weights file not found at {best_weights}")
                
                if sr_volume.exists():
                    shutil.copy2(sr_volume, target_sr)
                    LOGGER.info(f"Saved SR volume to {target_sr}")
                else:
                    LOGGER.error(f"SR volume not found at {sr_volume}")
                
                # Clean up temporary directory
                try:
                    if volume_dir.exists():
                        shutil.rmtree(volume_dir, ignore_errors=True)
                        LOGGER.info(f"Cleaned up temporary directory {volume_dir}")
                except Exception as e:
                    LOGGER.warning(f"Failed to clean up temporary directory: {e}")
                
                LOGGER.info(f"Processing completed for {vol.name}")


def run_inference_only(config: SmoreFullConfig) -> None:
    """Run SMORE inference based on configuration."""
    LOGGER.info("Running SMORE in inference-only mode")
    
    # Process each resolution
    for resolution in config.low_res_slices:
        resolution_dir = config.test_root / resolution
        if not resolution_dir.exists():
            LOGGER.warning(f"Directory for {resolution} not found at {resolution_dir}, skipping")
            continue
            
        LOGGER.info(f"Processing {resolution} data from {resolution_dir}")
        
        # Create output directory for this resolution
        output_dir = ensure_dir(config.out_root / "SMORE" / resolution / "output_volumes")
        
        # Get weights directory for this resolution
        weights_dir = config.out_root / "SMORE" / resolution / "weights"
        if not weights_dir.exists():
            LOGGER.warning(f"Weights directory not found for {resolution} at {weights_dir}, skipping")
            continue
        
        # Process each pulse type
        for pulse in config.pulses:
            LOGGER.info(f"Processing {pulse} pulse type")
            
            # Find all volumes with this pulse type
            pulse_volumes = list(resolution_dir.rglob(f"*-{pulse}.nii.gz"))
            
            if not pulse_volumes:
                LOGGER.warning(f"No {pulse} volumes found in {resolution_dir}")
                continue
                
            LOGGER.info(f"Found {len(pulse_volumes)} {pulse} volumes to process")
            
            # Run inference on each volume
            for vol in pulse_volumes:
                LOGGER.info(f"Running inference on: {vol}")
                
                # Find corresponding weights file
                weights_file = weights_dir / f"{vol.stem}.pt"
                
                if not weights_file.exists():
                    LOGGER.warning(f"Weights file not found for {vol.name} at {weights_file}, skipping")
                    continue
                
                # Output path
                out_path = output_dir / f"{vol.stem}.nii.gz"
                
                # Run inference
                infer_volume(vol, weights_dir, config.network, out_path)
                LOGGER.info(f"Inference completed for {vol.name} → {out_path}")


def main() -> None:
    """Main entry point for SMORE processing."""
    parser = argparse.ArgumentParser(description="Run SMORE super-resolution training or inference")
    parser.add_argument("--config", type=Path, required=True, 
                        help="Path to SMORE configuration YAML file")
    args = parser.parse_args()
    
    # Validate config file exists
    if not args.config.exists():
        LOGGER.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Load and validate configuration
    try:
        config = SmoreFullConfig(args.config)
        errors = config.validate()
        if errors:
            for error in errors:
                LOGGER.error(error)
            LOGGER.error("Configuration validation failed")
            sys.exit(1)
    except Exception as e:
        LOGGER.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Run in appropriate mode
    if config.mode == "train":
        run_training_and_inference(config)
    elif config.mode == "inference":
        run_inference_only(config)
    else:
        LOGGER.error(f"Unknown mode: {config.mode}")
        sys.exit(1)
    
    LOGGER.info("SMORE processing completed successfully")


if __name__ == "__main__":
    main()