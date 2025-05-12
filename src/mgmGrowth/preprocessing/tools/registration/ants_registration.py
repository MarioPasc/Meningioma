#!/usr/bin/env python3
"""
High-level utilities to register 3-D MR brain volumes to the SRI-24 template
with ANTs (via Nipype) in two modes:

* **full registration** Rigid ➜ Affine ➜ SyN – for reference T1 volumes  
* **secondary modality** Rigid ➜ Affine       – for T2 / FLAIR / SWI, etc.

The choice is controlled by the Boolean `full_registration` argument (default
*True*).  All stage-specific parameters are read from the corresponding
section (“full_registration” or “secondary_modality”) of the YAML file passed
to *register_image_to_sri24* or to the CLI.

The YAML file also contains a global `interpolation` key that sets the
resampling interpolation for **every** ApplyTransforms or antsRegistration
output warp (“Linear”, “BSpline”, “NearestNeighbor”, …).

The code is written for clarity, complete typing, and minimum repetition.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import SimpleITK as sitk
import yaml
from nipype.interfaces.ants import ApplyTransforms, Registration

# --------------------------------------------------------------------------------------
# Logging helper – replace with your project logger if you have one
# --------------------------------------------------------------------------------------
import logging

LOGGER = logging.getLogger("ants_registration")
if not LOGGER.handlers:  # avoid duplicate handlers in notebooks
    _h = logging.StreamHandler(stream=sys.stdout)
    _h.setFormatter(
        logging.Formatter("[%(levelname)s] %(asctime)s  %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    LOGGER.addHandler(_h)
LOGGER.setLevel(logging.INFO)

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _now() -> str:
    """Return current time as nice string (for profiling prints)."""
    return time.strftime("%H:%M:%S")


def _select_stage(cfg: Dict[str, Any], full: bool) -> Dict[str, Any]:
    """
    Extract stage-specific parameter block from YAML.

    Parameters
    ----------
    cfg   : parsed YAML dict
    full  : if True → use cfg["full_registration"], else cfg["secondary_modality"]

    Returns
    -------
    stage-dict  (empty dict if the key is missing)
    """
    key = "full_registration" if full else "secondary_modality"
    if key not in cfg:  # graceful degradation; caller will fall back to defaults
        LOGGER.warning(f"YAML missing section “{key}”.  Using hard-coded defaults.")
    return cast(Dict[str, Any], cfg.get(key, {}))


KEY_MAP = {
    "iterations": "number_of_iterations",
    "metrics": "metric",
    "metric_weights": "metric_weight",
}
def _remap_stage_keys(stage: Dict[str, Any]) -> Dict[str, Any]:
    """Return a *copy* of *stage* with plural keys renamed for Nipype."""
    return {KEY_MAP.get(k, k): v for k, v in stage.items()}


# --------------------------------------------------------------------------------------
# Core registration functions
# --------------------------------------------------------------------------------------
def register_to_sri24(
    moving_image: sitk.Image,
    atlas_path: str,
    output_dir: Union[str, Path],
    output_prefix: str,
    output_image_path: Union[str, Path],
    *,
    full_registration: bool = True,
    interpolation: str = "Linear",
    stage_params: Optional[Dict[str, Any]] = None,
    # global options …
    initial_transform: Optional[str] = None,
    dimension: int = 3,
    histogram_matching: bool = True,
    winsorize_lower_quantile: float = 0.005,
    winsorize_upper_quantile: float = 0.995,
    number_threads: int = 1,
    verbose: bool = False,
    cleanup: bool = True,
) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Run ANTs antsRegistration.

    When *stage_params* is not given the function constructs them internally
    from the hard-coded defaults **and** the `full_registration` flag.

    Returns
    -------
    (registered, params) where **params** is a dict with composite / individual
    transforms and helper file paths.
    """
    t0 = time.time()
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ defaults
    if stage_params is None:
        if full_registration:          # 3-stage R+A+SyN
            stage_params = dict(
                transforms           = ["Rigid", "Affine", "SyN"],
                transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)],
                number_of_iterations = [[1000, 500, 250, 100]] * 3,
                shrink_factors       = [[8, 4, 2, 1]] * 3,
                smoothing_sigmas     = [[3, 2, 1, 0]] * 3,
                metric               = ["MI", "MI", "MI"],
                metric_weight        = [1.0, 1.0, 1.0],
                sampling_strategy    = ["Regular", "Regular", "None"],
                sampling_percentage  = [0.25, 0.25, None],
                radius_or_number_of_bins = [32, 32, 4],
                convergence_threshold     = [1e-6] * 3,
                convergence_window_size   = [10] * 3,
                sigma_units               = ["vox"] * 3,
            )
        else:                          # 2-stage R+A
            stage_params = dict(
                transforms           = ["Rigid", "Affine"],
                transform_parameters = [(0.1,), (0.1,)],
                number_of_iterations = [[1000, 500, 250, 100]] * 2,
                shrink_factors       = [[8, 4, 2, 1]] * 2,
                smoothing_sigmas     = [[3, 2, 1, 0]] * 2,
                metric               = ["MI", "MI"],
                metric_weight        = [1.0, 1.0],
                sampling_strategy    = ["Regular", "Regular"],
                sampling_percentage  = [0.25, 0.25],
                radius_or_number_of_bins = [32, 32],
                convergence_threshold     = [1e-6, 1e-6],
                convergence_window_size   = [10, 10],
                sigma_units               = ["vox", "vox"],
            )

    # ‣ map any legacy plural keys coming from YAML / callers
    stage_params = _remap_stage_keys(stage_params)

    # ------------------------------------------------------------------ I/O
    moving_path = out_dir / "moving.nii.gz"
    sitk.WriteImage(moving_image, str(moving_path))

    reg = Registration()
    reg.inputs.fixed_image             = str(atlas_path)
    reg.inputs.moving_image            = str(moving_path)
    reg.inputs.dimension               = dimension
    reg.inputs.use_histogram_matching  = histogram_matching
    reg.inputs.winsorize_lower_quantile= winsorize_lower_quantile
    reg.inputs.winsorize_upper_quantile= winsorize_upper_quantile
    reg.inputs.num_threads             = number_threads
    reg.inputs.output_transform_prefix = str(out_dir / output_prefix)
    reg.inputs.output_warped_image     = str(output_image_path)
    reg.inputs.interpolation           = interpolation

    print(f"=== Registration Configuration Parameters ===")
    print(f"moving_image → {moving_path}")
    print(f"fixed_image  → {atlas_path}")
    print(f"output_warped_image → {output_image_path}")
    print(f"output_transform_prefix → {out_dir / output_prefix}")
    print(f"dimension   → {dimension}")
    print(f"interpolation → {interpolation}")
    print(f"histogram_matching → {histogram_matching}")
    for k, v in stage_params.items():
        if k == "convergence_threshold":
            # ANTs expects a list of floats, not a list of strings
            v = [float(x) for x in v]
        print(f"reg.inputs.{k} → {v}")
        setattr(reg.inputs, k, v)   # ← all keys now valid
    print(f"==============================================")

    if initial_transform:
        reg.inputs.initial_moving_transform = initial_transform
    if verbose: reg.terminal_output = "stream"

    run_result = reg.run()

    registered_img = sitk.ReadImage(run_result.outputs.warped_image)
    registered_img.CopyInformation(sitk.ReadImage(str(atlas_path)))
    atlas_sitk = sitk.ReadImage(str(atlas_path))
    registered_img.CopyInformation(atlas_sitk)

    params = dict(
        composite_transform=run_result.outputs.composite_transform,
        inverse_composite_transform=run_result.outputs.inverse_composite_transform,
        fixed_image_path=str(atlas_path),
        moving_image_path=str(moving_path),
    )
    # add individual transforms if they exist
    for suffix in ("0GenericAffine.mat", "1GenericAffine.mat", "2Warp.nii.gz",
                   "2InverseWarp.nii.gz"):
        f = f"{out_dir / output_prefix}{suffix}"
        if os.path.exists(f):
            params[Path(f).stem] = f

    json_path = out_dir / "transform_params.json"
    with open(json_path, "w") as fp:
        json.dump(params, fp, indent=2)

    if verbose:
        LOGGER.info(f"[{_now()}]   registration finished in "
                    f"{time.time()-t0:.1f} s; parameters → {json_path}")

    if cleanup:
        moving_path.unlink(missing_ok=True)

    return registered_img, params


def apply_composed_transforms(
    input_image: sitk.Image,
    t1_to_atlas: Dict[str, Any],
    modality_to_t1: Dict[str, Any],
    output_path: Union[str, Path],
    *,
    interpolation: str = "Linear",
    invert_modality_to_t1: bool = False,
    dimension: int = 3,
    number_threads: int = 1,
    verbose: bool = False,
    cleanup: bool = True,
) -> sitk.Image:
    """
    Cascade (modality→T1) ∘ (T1→atlas) in a **single** resampling step.
    """
    t0 = time.time()
    tmp_dir = Path(output_path).parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    moving_tmp = tmp_dir / "temp_input.nii.gz"
    sitk.WriteImage(input_image, str(moving_tmp))

    # pick proper transform filenames
    mod2t1 = (
        modality_to_t1["inverse_composite_transform"]
        if invert_modality_to_t1
        else modality_to_t1["composite_transform"]
    )
    t12atl = t1_to_atlas["composite_transform"]

    at = ApplyTransforms()
    at.inputs.dimension = dimension
    at.inputs.input_image = str(moving_tmp)
    at.inputs.reference_image = t1_to_atlas["fixed_image_path"]
    at.inputs.transforms = [t12atl, mod2t1]  # reversed order inside ANTs
    at.inputs.output_image = str(output_path)
    at.inputs.interpolation = interpolation
    at.inputs.num_threads = number_threads
    if verbose:
        at.terminal_output = "stream"
        LOGGER.info(f"[{_now()}] ApplyTransforms running…")
    at.run()

    resampled = sitk.ReadImage(str(output_path))
    atlas_ref = sitk.ReadImage(t1_to_atlas["fixed_image_path"])
    resampled.CopyInformation(atlas_ref)

    if cleanup:
        moving_tmp.unlink(missing_ok=True)

    if verbose:
        LOGGER.info(f"[{_now()}]   ApplyTransforms done in {time.time()-t0:.1f} s")
    return resampled


# --------------------------------------------------------------------------------------
# YAML-driven convenience wrapper
# --------------------------------------------------------------------------------------
def register_image_to_sri24(
    moving: sitk.Image,
    *,
    yaml_path: Union[str, Path],
    moving_mask: Optional[sitk.Image] = None,
    full_registration: bool = True,
    verbose: bool = False,
    cleanup: bool = True,
) -> Union[
    Tuple[sitk.Image, Dict[str, Any]],
    Tuple[sitk.Image, sitk.Image, Dict[str, Any]],
]:
    """
    One-liner that reads the YAML, chooses the correct parameter block,
    launches *register_to_sri24*, and (if a mask is provided) warps the mask.
    """
    cfg = cast(Dict[str, Any], yaml.safe_load(Path(yaml_path).read_text()))
    atlas_path = cfg["atlas_path"]
    out_dir = Path(cfg.get("output_dir", "./registration_output")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # pull global options
    interpolation = cfg.get("interpolation", "Linear")
    global_kwargs = dict(
        dimension=cfg.get("dimension", 3),
        histogram_matching=bool(cfg.get("histogram_matching", True)),
        winsorize_lower_quantile=float(cfg.get("winsorize_lower_quantile", 0.005)),
        winsorize_upper_quantile=float(cfg.get("winsorize_upper_quantile", 0.995)),
        number_threads=int(cfg.get("number_threads", 1)),
        initial_transform=cfg.get("initial_transform") or None,
    )

    # prepare stage-specific dict
    stage_cfg = _select_stage(cfg, full_registration)
    output_prefix = cfg.get("output_transform_prefix", "transform_")
    output_reg = out_dir / cfg.get("output_registered", "registered.nii.gz")

    registered, params = register_to_sri24(
        moving,
        atlas_path=atlas_path,
        output_dir=out_dir,
        output_prefix=output_prefix,
        output_image_path=output_reg,
        full_registration=full_registration,
        interpolation=interpolation,
        stage_params=stage_cfg,
        verbose=verbose,
        cleanup=cleanup,
        **global_kwargs,
    )

    # ------------------------------------------------------------------ mask handling
    if moving_mask is None:
        return registered, params

    mask_out = out_dir / cfg.get("output_mask", "registered_mask.nii.gz")
    at = ApplyTransforms()
    at.inputs.dimension = global_kwargs["dimension"]
    sitk.WriteImage(moving_mask, str(out_dir / "temp_mask.nii.gz"))
    at.inputs.input_image = str(out_dir / "temp_mask.nii.gz")
    at.inputs.reference_image = atlas_path
    at.inputs.transforms = [params["composite_transform"]]
    at.inputs.output_image = str(mask_out)
    at.inputs.interpolation = "NearestNeighbor"
    at.inputs.num_threads = global_kwargs["number_threads"]
    if verbose:
        at.terminal_output = "stream"
    at.run()
    warped_mask = sitk.ReadImage(str(mask_out))
    warped_mask.CopyInformation(sitk.ReadImage(atlas_path))
    return registered, warped_mask, params


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _cli() -> None:
    p = argparse.ArgumentParser(description="Register an MR volume to SRI-24.")
    p.add_argument("-i", "--input", required=True, help="moving image (NIfTI)")
    p.add_argument("-y", "--yaml", required=True, help="registration_sri24.yaml")
    p.add_argument("-s", "--secondary", action="store_true",
                   help="use the *secondary_modality* parameter block")
    p.add_argument("-m", "--mask", help="optional label mask (NIfTI)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    mov = sitk.ReadImage(args.input)
    mask = sitk.ReadImage(args.mask) if args.mask else None

    res = register_image_to_sri24(
        mov,
        yaml_path=args.yaml,
        moving_mask=mask,
        full_registration=not args.secondary,
        verbose=args.verbose,
    )

    LOGGER.info("✔ Registration finished.  See output directory specified in YAML.")


if __name__ == "__main__":
    _cli()
