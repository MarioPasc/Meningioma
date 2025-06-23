#!/usr/bin/env python3
"""
run_smore.py
===========

Unified entry point for SMORE super-resolution, supporting both training and inference modes.
Processes each pulse type individually across each resolution.

Example usage:
    python -m src.mgmGrowth.tasks.superresolution.cli.run_smore \
        --config src/mgmGrowth/tasks/superresolution/cfg/smore_cfg.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.mgmGrowth.tasks.superresolution import LOGGER
from src.mgmGrowth.tasks.superresolution.config import SmoreFullConfig
from src.mgmGrowth.tasks.superresolution.engine.smore_runner import train_volume, infer_volume
from src.mgmGrowth.tasks.superresolution.tools.paths import ensure_dir


def get_weights_path(resolution: str, pulse: str, weights_root: Path) -> Path:
    """Determine weights directory for a specific resolution and pulse."""
    return weights_root / resolution / "weights" / pulse


def get_output_path(resolution: str, out_root: Path, volume_path: Path) -> Path:
    """Determine output path for a specific resolution and volume."""
    output_dir = ensure_dir(out_root / "SMORE" / resolution / "output_volumes")
    return output_dir / volume_path.name


def run_training(config: SmoreFullConfig) -> None:
    """Run SMORE training based on configuration."""
    LOGGER.info("Running SMORE in training mode")
    
    # Process each resolution
    for resolution in config.low_res_slices:
        resolution_dir = config.train_root / resolution
        if not resolution_dir.exists():
            LOGGER.warning(f"Directory for {resolution} not found at {resolution_dir}, skipping")
            continue
            
        LOGGER.info(f"Processing {resolution} data from {resolution_dir}")
        
        # Extract numerical slice thickness value
        slice_dz = float(resolution.replace("mm", ""))
        
        # Process each pulse type
        for pulse in config.pulses:
            LOGGER.info(f"Processing {pulse} pulse type")
            
            # Create weights directory for this resolution and pulse
            weights_dir = ensure_dir(config.weights_root / "SMORE" / resolution / "weights" / pulse)
            
            # Find all volumes with this pulse type
            pulse_volumes = list(resolution_dir.rglob(f"*-{pulse}.nii.gz"))
            
            if not pulse_volumes:
                LOGGER.warning(f"No {pulse} volumes found in {resolution_dir}")
                continue
                
            LOGGER.info(f"Found {len(pulse_volumes)} {pulse} volumes to process")
            
            # Train on each volume
            for vol in pulse_volumes:
                LOGGER.info(f"Training on volume: {vol}")
                train_volume(vol, config.network, weights_dir, slice_dz)
                LOGGER.info(f"Training completed for {vol.name}")


def run_inference(config: SmoreFullConfig) -> None:
    """Run SMORE inference based on configuration."""
    LOGGER.info("Running SMORE in inference mode")
    
    # Process each resolution
    for resolution in config.low_res_slices:
        resolution_dir = config.test_root / resolution
        if not resolution_dir.exists():
            LOGGER.warning(f"Directory for {resolution} not found at {resolution_dir}, skipping")
            continue
            
        LOGGER.info(f"Processing {resolution} data from {resolution_dir}")
        
        # Process each pulse type
        for pulse in config.pulses:
            LOGGER.info(f"Processing {pulse} pulse type")
            
            # Get weights directory for this resolution and pulse
            weights_dir = config.weights_root / "SMORE" / resolution / "weights" / pulse
            
            if not weights_dir.exists():
                LOGGER.warning(f"Weights directory not found for {resolution}/{pulse} at {weights_dir}, skipping")
                continue
                
            # Find all volumes with this pulse type
            pulse_volumes = list(resolution_dir.rglob(f"*-{pulse}.nii.gz"))
            
            if not pulse_volumes:
                LOGGER.warning(f"No {pulse} volumes found in {resolution_dir}")
                continue
                
            LOGGER.info(f"Found {len(pulse_volumes)} {pulse} volumes to process")
            
            # Create output directory for this resolution
            output_dir = ensure_dir(config.out_root / "SMORE" / resolution / "output_volumes")
            
            # Run inference on each volume
            for vol in pulse_volumes:
                LOGGER.info(f"Running inference on: {vol}")
                out_path = output_dir / vol.name
                
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
        run_training(config)
    elif config.mode == "inference":
        run_inference(config)
    else:
        LOGGER.error(f"Unknown mode: {config.mode}")
        sys.exit(1)
    
    LOGGER.info("SMORE processing completed successfully")


if __name__ == "__main__":
    main()