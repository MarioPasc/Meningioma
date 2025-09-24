#!/usr/bin/env python3
"""
eclare.py
=========

Unified entry point for **ECLARE** processing.

The interface mirrors *smore.py* so that existing pipelines or notebooks that
previously imported `run_smore` / `infer_volume` can switch by replacing only
the module name.

Author  :  Mario 
Created : 2025-07-06

@inproceedings{remedios2023self,
  title={Self-supervised super-resolution for anisotropic {MR} images with and without slice gap},
  author={Remedios, Samuel W and Han, Shuo and Zuo, Lianrui and Carass, Aaron and Pham, Dzung L and Prince, Jerry L and Dewey, Blake E},
  booktitle={Simulation and Synthesis in Medical Imaging (SASHIMI)},
  pages={118--128},
  year={2023},
  organization={Springer}
}

"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from mgmGrowth.tasks.superresolution import LOGGER
from src.mgmGrowth.tasks.superresolution.config import (
    SmoreFullConfig as EclareFullConfig,   # ← re-use existing config class
    SmoreConfig as EclareConfig,           #   to avoid touching rest of code
)
from src.mgmGrowth.tasks.superresolution.engine.eclare_runner import (
    run_eclare,
    infer_volume,
)
from src.mgmGrowth.tasks.superresolution.tools.paths import ensure_dir



def run_training_and_inference(config: EclareFullConfig) -> None:
    """Run ECLARE self-super-resolution on *train* data (like SMORE train)."""
    LOGGER.info("Running ECLARE in training-and-inference mode")

    for resolution in config.low_res_slices:
        resolution_dir = config.train_root / resolution
        if not resolution_dir.exists():
            LOGGER.warning("Directory for %s not found at %s, skipping",
                           resolution, resolution_dir)
            continue

        LOGGER.info("Processing %s data from %s", resolution, resolution_dir)

        # Extract Δz in mm from directory name like "3mm"
        relative_slice_thickness = float(resolution.replace("mm", ""))

        eclare_resolution_dir = ensure_dir(config.out_root / "ECLARE" / resolution)
        weights_dir = ensure_dir(eclare_resolution_dir / "weights")
        output_dir = ensure_dir(eclare_resolution_dir / "output_volumes")

        for pulse in config.pulses:
            LOGGER.info("Processing %s pulse type", pulse)
            pulse_volumes = list(resolution_dir.rglob(f"*-{pulse}.nii.gz"))
            if not pulse_volumes:
                LOGGER.warning("No %s volumes found in %s", pulse, resolution_dir)
                continue

            for vol in pulse_volumes:
                LOGGER.info("Running ECLARE on %s", vol.name)

                base_name = vol.stem.rstrip(".nii.gz")

                # NB: suffix must match eclare_runner default
                suffix = "_eclare"

                # Run ECLARE directly in the resolution directory (no temp dir)
                weights_src_dir, sr_generated = run_eclare(
                    vol,
                    eclare_resolution_dir,
                    cfg=config.network,
                    relative_slice_thickness=relative_slice_thickness,
                    gpu_id=config.network.gpu_id,
                    suffix=suffix,
                )

                # ~~~ Collect outputs ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                best_weights = weights_src_dir / "best_weights.pt"
                sr_volume = sr_generated

                target_weights = weights_dir / f"{base_name}.pt"
                target_sr = output_dir / f"{base_name}.nii.gz"

                if best_weights.exists():
                    LOGGER.info("Looking for best weights at %s", best_weights)
                    shutil.copy2(best_weights, target_weights)
                    LOGGER.info("Saved weights → %s", target_weights)
                else:
                    LOGGER.warning("Weights file not found: %s", best_weights)

                if sr_volume.exists():
                    LOGGER.info("Looking for SR volume at %s", sr_volume)
                    shutil.copy2(sr_volume, target_sr)
                    LOGGER.info("Saved SR volume → %s", target_sr)
                else:
                    LOGGER.error("SR volume missing: %s", sr_volume)



def run_inference_only(config: EclareFullConfig) -> None:
    """Run ECLARE in *inference-only* mode (assumes weights already exist)."""
    LOGGER.info("Running ECLARE in inference-only mode")

    for resolution in config.low_res_slices:
        resolution_dir = config.test_root / resolution
        if not resolution_dir.exists():
            LOGGER.warning("Directory for %s not found at %s, skipping",
                           resolution, resolution_dir)
            continue

        output_dir = ensure_dir(config.out_root / "ECLARE" / resolution
                                / "output_volumes")
        weights_dir = config.out_root / "ECLARE" / resolution / "weights"
        if not weights_dir.exists():
            LOGGER.warning("Weights directory missing for %s: %s",
                           resolution, weights_dir)
            continue

        for pulse in config.pulses:
            LOGGER.info("Pulse type: %s", pulse)
            pulse_volumes = list(resolution_dir.rglob(f"*-{pulse}.nii.gz"))
            if not pulse_volumes:
                LOGGER.warning("No %s volumes in %s", pulse, resolution_dir)
                continue

            for vol in pulse_volumes:
                weights_file = weights_dir / f"{vol.stem}.pt"
                if not weights_file.exists():
                    LOGGER.warning("Weights missing for %s", vol.name)
                    continue

                out_path = output_dir / f"{vol.stem}.nii.gz"
                infer_volume(vol, weights_dir, config.network, out_path)
                LOGGER.info("Inference done: %s → %s", vol.name, out_path)


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Run ECLARE self-super-resolution (train or inference)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config compatible with ECLAREFullConfig",
    )
    args = parser.parse_args()

    if not args.config.exists():
        LOGGER.error("Configuration file not found: %s", args.config)
        sys.exit(1)

    try:
        config = EclareFullConfig(args.config)  # reuse validation
        errors = config.validate()
        if errors:
            for err in errors:
                LOGGER.error(err)
            sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load configuration: %s", exc)
        sys.exit(1)

    if config.mode == "train":
        run_training_and_inference(config)
    elif config.mode == "inference":
        run_inference_only(config)
    else:
        LOGGER.error("Unknown mode %s (expected 'train' or 'inference')",
                     config.mode)
        sys.exit(1)

    LOGGER.info("ECLARE processing completed successfully")


if __name__ == "__main__":  # pragma: no cover
    main()
