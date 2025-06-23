"""Typed configuration objects."""
from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass(slots=True, frozen=True)
class SmoreConfig:
    """Hyper-parameters and paths for the SMORE engine."""
    gpu_id: int = 0
    patch_size: int = 48
    n_blocks: int = 16
    n_channels: int = 32
    batch_size: int = 32
    n_patches: int = 832_000
    n_rots: int = 2

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> SmoreConfig:
        """Create a SmoreConfig from a dictionary (e.g., from YAML)."""
        network_config = config_dict.get("network", {})
        return cls(
            gpu_id=config_dict.get("processing", {}).get("gpu_id", 0),
            patch_size=network_config.get("patch_size", 48),
            n_blocks=network_config.get("n_blocks", 16),
            n_channels=network_config.get("n_channels", 32),
            batch_size=network_config.get("batch_size", 32),
            n_patches=network_config.get("n_patches", 832_000),
            n_rots=network_config.get("n_rots", 2),
        )


class SmoreFullConfig:
    """Full configuration for SMORE processing, including paths and mode."""
    
    def __init__(self, config_path: Path):
        """Load configuration from a YAML file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Extract basic configuration
        self.mode = self.config.get("mode", "train")
        
        # Create SmoreConfig for network parameters
        self.network = SmoreConfig.from_dict(self.config)
        
        # Extract paths
        data_config = self.config.get("data", {})
        self.train_root = Path(data_config.get("train_root", "")) if "train_root" in data_config else None
        self.test_root = Path(data_config.get("test_root", "")) if "test_root" in data_config else None
        self.weights_root = Path(data_config.get("weights_root", "")) if "weights_root" in data_config else None
        self.out_root = Path(data_config.get("out_root", "")) if "out_root" in data_config else None
        
        # Extract processing parameters
        proc_config = self.config.get("processing", {})
        self.low_res_slices = proc_config.get("low_res_slices", ["3mm", "5mm", "7mm"])
        self.pulses = proc_config.get("pulses", ["t1c", "t2w"])
    
    def validate(self) -> List[str]:
        """
        Validate the configuration based on the selected mode.
        
        Returns:
            List of error messages, empty if valid.
        """
        errors = []
        
        # Common validations
        if not hasattr(self, 'mode') or self.mode not in ["train", "inference"]:
            errors.append(f"Invalid mode: {self.mode}. Must be 'train' or 'inference'.")
        
        if not hasattr(self, 'pulses') or not self.pulses:
            errors.append("At least one pulse type must be specified.")
            
        if not hasattr(self, 'low_res_slices') or not self.low_res_slices:
            errors.append("At least one low-resolution slice thickness must be specified.")
        
        # Weights root is required for both modes
        if not self.weights_root:
            errors.append("weights_root is required for both training and inference modes.")
        
        # Mode-specific validations
        if self.mode == "train":
            if not self.train_root:
                errors.append("train_root is required for training mode.")
            elif not self.train_root.exists():
                errors.append(f"train_root path does not exist: {self.train_root}")

        elif self.mode == "inference":
            if not self.test_root:
                errors.append("test_root is required for inference mode.")
            elif not self.test_root.exists():
                errors.append(f"test_root path does not exist: {self.test_root}")
                
            if not self.weights_root.exists():
                errors.append(f"weights_root path does not exist: {self.weights_root}")
                
            if not self.out_root:
                errors.append("out_root is required for inference mode.")
        
        return errors