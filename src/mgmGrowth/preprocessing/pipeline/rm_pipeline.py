#!/usr/bin/env python3
"""
MGM Growth – RM (MRI) preprocessing pipeline
-------------------------------------------

Changes in this revision
~~~~~~~~~~~~~~~~~~~~~~~~
1. **Composite-warp option**

   * `_process_registration(..., apply_composite: bool)`  
     • *If* ``apply_composite is True`` **and** ``pulse_name != "T1"``  
       → register the modality to **T1-native** with the *secondary_modality*
       block (Rigid + Affine only) *then* cascade that transform with the
       previously-saved **T1 → atlas** warp using
       ``apply_composed_transforms``.  
     • *Else* fall back to a full Rigid + Affine + SyN registration.

   * The first time T1 is processed the script stores  
      `T1_native.nii.gz`  *(pre-registration)* and  
      `transform_params_T1.json` *(T1 → atlas)*  
     in *patient_output_dir* for later composite use.

2. **Parameter plumbing**

   * Every step now receives ``apply_composite`` via `kwargs`
     (default ``False``).  Only the registration step uses it.

3. **Imports & typing**

   * Added imports: ``yaml`` and the two new helpers from
     `ants_registration.py` – `register_to_sri24`, `apply_composed_transforms`.

4. **Minor clean-ups**

   * Switched `config_path` → `yaml_path` to match the new API.
   * Extra logging for clarity, robust fall-backs if the T1 assets are missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import os

import SimpleITK as sitk
import yaml                                  

from mgmGrowth.preprocessing import LOGGER
from mgmGrowth.preprocessing.tools.remove_extra_channels import remove_first_channel
from mgmGrowth.preprocessing.tools.nrrd_to_nifti import nifti_write_3d
from mgmGrowth.preprocessing.tools.casting import cast_volume_and_optional_mask
from mgmGrowth.preprocessing.tools.denoise_susan import denoise_susan
from mgmGrowth.preprocessing.tools.bias_field_corr_n4 import (
    generate_brain_mask_sitk,
    n4_bias_field_correction,
)
from mgmGrowth.preprocessing.tools.skull_stripping.fsl_bet import (
    fsl_bet_brain_extraction,
)
from mgmGrowth.preprocessing.tools.registration.ants_registration import (  
    register_image_to_sri24,
    register_to_sri24,
    apply_composed_transforms,
)

# Global configuration, only has debudding pourposes
SAVE_INTERMEDIATE: bool = False


def rm_pipeline(
    pulse_data: Dict[str, Any],
    preprocessing_plan: Dict[str, Any],
    patient_output_dir: Path,
    patient_id: str,
    pulse_name: str,
    verbose: bool = False,
    t1_brain_mask_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete RM preprocessing pipeline for a single pulse sequence.
    (Docstring unchanged – see header for new composite-warp behaviour.)
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # state bookkeeping
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    processed_pulse = pulse_data.copy()
    state: Dict[str, Any] = {
        "current_image": None,
        "current_header": None,
        "current_mask": None,
        "brain_mask": None,
        "processed_pulse": processed_pulse,
    }

    # output folders
    patient_output_dir.mkdir(exist_ok=True)
    other_dir = patient_output_dir / "other"
    other_dir.mkdir(exist_ok=True)

    if verbose:
        LOGGER.info(f"\n[RM / {pulse_name}] ▶  starting preprocessing …")

    # ordered list of (step-name, function)
    pipeline_steps = [
        ("remove_channel", _process_remove_channel),
        ("export_nifti", _process_export_nifti),
        ("load_segmentation_mask", _process_load_mask),
        ("cast_volume", _process_cast_volume),
        ("denoise", _process_denoise),
        ("brain_mask", _process_brain_mask),
        ("bias_field_correction", _process_bias_correction),
        ("registration", _process_registration),
    ]

    # choose brain-extraction flavour
    if pulse_name == "T1" or t1_brain_mask_path is None:
        pipeline_steps.append(("brain_extraction", _process_brain_extraction))
    else:
        pipeline_steps.append(
            ("brain_extraction", _process_brain_extraction_with_t1_mask)
        )

    # ----------------------------------------------------------------------
    # main loop
    # ----------------------------------------------------------------------
    for step_name, step_func in pipeline_steps:
        # step-specific plan (∅ for load_mask which always runs)
        step_plan = preprocessing_plan.get(step_name, {})

        # build kwargs
        kwargs: Dict[str, Any] = {
            **state,  # current_image, mask, …
            "pulse_data": pulse_data,
            "preprocessing_plan": step_plan,
            "patient_output_dir": patient_output_dir,
            "patient_id": patient_id,
            "pulse_name": pulse_name,
            "verbose": verbose,
            "save_intermediate": SAVE_INTERMEDIATE,
            "others_dir": other_dir,
            "t1_brain_mask_path": t1_brain_mask_path,
            "apply_composite": preprocessing_plan.get("registration", {}).get(
                "apply_composite", True
            ),
        }

        # skip disabled steps (except mask loading)
        if step_name not in preprocessing_plan and step_name != "load_segmentation_mask":
            continue

        if verbose and step_name != "load_segmentation_mask":
            LOGGER.info(f"[RM / {step_name.upper()}]: Starting step")

        try:
            step_result: Dict[str, Any] = step_func(**kwargs)
            # merge new state
            for k, v in step_result.items():
                state[k] = v
            # propagate paths for caller
            if "processed_data" in step_result:
                processed_pulse.update(step_result["processed_data"])
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.error(f"[RM / {step_name.upper()}] ❌  {exc}")

    # ----------------------------------------------------------------------
    # final outputs (always atlas-space after registration)
    # ----------------------------------------------------------------------
    if state["current_image"] is not None:
        out_img = patient_output_dir / f"{pulse_name}_{patient_id}.nii.gz"
        sitk.WriteImage(state["current_image"], str(out_img))
        processed_pulse["final_image_path"] = str(out_img)
    if state["current_mask"] is not None:
        out_msk = patient_output_dir / f"{pulse_name}_{patient_id}_seg.nii.gz"
        sitk.WriteImage(state["current_mask"], str(out_msk))
        processed_pulse["final_mask_path"] = str(out_msk)

    return processed_pulse

def _process_remove_channel(
    current_image: Optional[sitk.Image],
    current_header: Optional[Dict[str, Any]],
    pulse_data: Dict[str, Any],
    preprocessing_plan: Dict[str, Any],
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Process the remove_channel step in the RM pipeline.
    
    Args:
        current_image: Current SimpleITK image object or None
        current_header: Current image header or None
        pulse_data: Dictionary containing pulse data and metadata
        preprocessing_plan: Dictionary with step-specific configuration
        verbose: Whether to print detailed processing information
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with updated processing state
    """
    result = {
        "current_image": current_image,
        "current_header": current_header
    }
    
    if verbose:
        LOGGER.info(f"[RM / CHANNEL REMOVAL] Processing {Path(pulse_data['volume_path']).name}")
    
    channel = preprocessing_plan.get("channel", 0)
    if verbose:
        LOGGER.info(f"[RM / CHANNEL REMOVAL] Extracting channel {channel}")
    
    # Apply remove_channel function to the volume
    current_image, current_header = remove_first_channel(
        pulse_data["volume_path"], 
        channel, 
        verbose=verbose
    )
    
    if verbose:
        LOGGER.info(f"[RM / CHANNEL REMOVAL] Completed: size={current_image.GetSize()}, "
              f"dimensions={current_image.GetDimension()}, "
              f"spacing={current_image.GetSpacing()}")
    
    result["current_image"] = current_image
    result["current_header"] = current_header
    
    return result


def _process_export_nifti(
    current_image: Optional[sitk.Image],
    current_header: Optional[Dict[str, Any]],
    pulse_data: Dict[str, Any],
    preprocessing_plan: Dict[str, Any],
    patient_output_dir: Path,
    patient_id: str,
    pulse_name: str,
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Process the export_nifti step in the RM pipeline.
    
    Args:
        current_image: Current SimpleITK image object or None
        current_header: Current image header or None
        pulse_data: Dictionary containing pulse data and metadata
        preprocessing_plan: Dictionary with step-specific configuration
        patient_output_dir: Output directory for the patient's processed files
        patient_id: Patient identifier (e.g., 'P1')
        pulse_name: Name of the pulse sequence (e.g., 'T1', 'T2', 'SUSC')
        verbose: Whether to print detailed processing information
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with updated processing state and processed data
    """
    result = {
        "current_image": current_image,
        "current_header": current_header,
        "processed_data": {}
    }
    
    if verbose:
        LOGGER.info(f"[RM / NIFTI EXPORT] Converting to NIfTI format")
    
    # Output path for the NIfTI file - This is a main output file, not an intermediate
    nifti_filename = f"{pulse_name}_{patient_id}.nii.gz"
    nifti_path = patient_output_dir / nifti_filename
    
    # If we have a current_image from a previous step, use it
    # Otherwise, use the original volume path
    input_data = (current_image, current_header) if current_image is not None else pulse_data["volume_path"]
    
    # Convert to NIfTI
    output_path = nifti_write_3d(
        volume=input_data, #type: ignore
        out_file=str(nifti_path),
        verbose=verbose
    )
    
    # Update the processed data with the new file path
    result["processed_data"]["nifti_path"] = output_path
    
    if verbose:
        LOGGER.info(f"[RM / NIFTI EXPORT] Saved to {output_path}")
    
    # Load the exported NIfTI for further processing
    result["current_image"] = sitk.ReadImage(output_path)
    
    return result


def _process_load_mask(
    current_image: Optional[sitk.Image],
    current_mask: Optional[sitk.Image],
    pulse_data: Dict[str, Any],
    verbose: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Load the segmentation mask for the current image.
    
    Args:
        current_image: Current SimpleITK image object or None
        current_mask: Current mask image or None
        pulse_data: Dictionary containing pulse data and metadata
        verbose: Whether to print detailed processing information
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with updated processing state
    """
    result = {
        "current_image": current_image,
        "current_mask": current_mask
    }
    
    # Only proceed if we have an image to match with the mask
    if current_image is None:
        return result
    
    try:
        if verbose:
            LOGGER.info(f"[RM / MASK LOADING] Loading segmentation mask")
        
        if os.path.exists(pulse_data["mask_path"]):
            current_mask = sitk.ReadImage(pulse_data["mask_path"])
        else:
            # Generate a zero mask if the mask file does not exist
            current_mask = sitk.Image(current_image.GetSize(), sitk.sitkUInt8)
            current_mask.CopyInformation(current_image)
            LOGGER.warning(f"[RM / MASK LOADING] Mask file not found, using zero mask")
        
        if verbose:
            LOGGER.info(f"[RM / MASK LOADING] Mask loaded: size={current_mask.GetSize()}, "
                  f"dimensions={current_mask.GetDimension()}")
        
        result["current_mask"] = current_mask
            
    except Exception as e:
        LOGGER.error(f"[RM / MASK LOADING] Error loading mask: {str(e)}")
    
    return result


def _process_cast_volume(
    current_image: Optional[sitk.Image],
    current_mask: Optional[sitk.Image],
    pulse_data: Dict[str, Any],
    preprocessing_plan: Dict[str, Any],
    patient_id: str,
    pulse_name: str,
    verbose: bool = False,
    save_intermediate: bool = False,
    others_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Cast volume to float32 and mask to uint8.
    
    Args:
        current_image: Current SimpleITK image object or None
        current_mask: Current mask image or None
        pulse_data: Dictionary containing pulse data and metadata
        preprocessing_plan: Dictionary with step-specific configuration
        patient_id: Patient identifier (e.g., 'P1')
        pulse_name: Name of the pulse sequence (e.g., 'T1', 'T2', 'SUSC')
        verbose: Whether to print detailed processing information
        save_intermediate: Whether to save intermediate files
        others_dir: Directory for intermediate files
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with updated processing state and processed data
    """
    result = {
        "current_image": current_image,
        "current_mask": current_mask,
        "processed_data": {}
    }
    
    # Skip if necessary components are missing
    if current_image is None or current_mask is None:
        return result
    
    if verbose:
        LOGGER.info(f"[RM / CAST VOLUME] Casting volume to float32 and mask to uint8")
    
    try:
        # Cast volume to float32 and mask to uint8
        cast_image, cast_mask = cast_volume_and_optional_mask(
            current_image, 
            current_mask
        )
        
        result["current_image"] = cast_image
        result["current_mask"] = cast_mask
        
        # Save the cast volume and mask if requested
        if save_intermediate and others_dir:
            cast_volume_path = others_dir / f"{pulse_name}_{patient_id}_cast.nii.gz"
            cast_mask_path = others_dir / f"{pulse_name}_{patient_id}_mask_cast.nii.gz"
            
            sitk.WriteImage(cast_image, str(cast_volume_path))
            sitk.WriteImage(cast_mask, str(cast_mask_path))
            
            # Update the processed data with the new file paths
            result["processed_data"]["cast_volume_path"] = str(cast_volume_path)
            result["processed_data"]["cast_mask_path"] = str(cast_mask_path)
            
            if verbose:
                LOGGER.info(f"[RM / CAST VOLUME] Volume cast to {cast_image.GetPixelID()} and saved to {cast_volume_path}")
                LOGGER.info(f"[RM / CAST VOLUME] Mask cast to {cast_mask.GetPixelID()} and saved to {cast_mask_path}")
        
    except Exception as e:
        LOGGER.error(f"[RM / CAST VOLUME] Error: {str(e)}")
    
    return result


def _process_denoise(
    current_image: Optional[sitk.Image],
    current_mask: Optional[sitk.Image],
    preprocessing_plan: Dict[str, Any],
    patient_id: str,
    pulse_name: str,
    verbose: bool = False,
    save_intermediate: bool = False,
    others_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply SUSAN denoising to the volume.
    
    Args:
        current_image: Current SimpleITK image object or None
        current_mask: Current mask image or None
        preprocessing_plan: Dictionary with step-specific configuration
        patient_id: Patient identifier (e.g., 'P1')
        pulse_name: Name of the pulse sequence (e.g., 'T1', 'T2', 'SUSC')
        verbose: Whether to print detailed processing information
        save_intermediate: Whether to save intermediate files
        others_dir: Directory for intermediate files
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with updated processing state and processed data
    """
    result = {
        "current_image": current_image,
        "current_mask": current_mask,
        "processed_data": {}
    }
    
    # Skip if image is missing
    if current_image is None:
        return result
    
    # Check if denoising is enabled
    denoise_enabled = preprocessing_plan.get("enable", False)
    
    if not denoise_enabled:
        if verbose:
            LOGGER.info(f"[RM / DENOISE] Denoising is disabled in the preprocessing plan, skipping")
        return result
    
    if verbose:
        LOGGER.info(f"[RM / DENOISE] Applying SUSAN denoising to volume")
    
    try:
        # Get denoise parameters
        susan_params = preprocessing_plan.get("susan", {})
        brightness_threshold = susan_params.get("brightness_threshold", 0.001)
        fwhm = susan_params.get("fwhm", 0.5)
        dimension = susan_params.get("dimension", 3)
        
        if verbose:
            LOGGER.info(f"[RM / DENOISE] SUSAN parameters: brightness_threshold={brightness_threshold}, "
                  f"fwhm={fwhm}, dimension={dimension}")
        
        # Apply denoising to the volume
        denoised_image = denoise_susan(
            current_image,
            brightness_threshold=brightness_threshold,
            fwhm=fwhm,
            dimension=dimension,
            mask_sitk=current_mask,  # Optional: Use mask to constrain denoising
            verbose=verbose
        )
        
        result["current_image"] = denoised_image
        
        # Save the denoised volume if requested
        if save_intermediate and others_dir:
            denoised_volume_path = others_dir / f"{pulse_name}_{patient_id}_denoised.nii.gz"
            sitk.WriteImage(denoised_image, str(denoised_volume_path))
            
            # Update the processed data with the new file path
            result["processed_data"]["denoised_volume_path"] = str(denoised_volume_path)
            
            if verbose:
                LOGGER.info(f"[RM / DENOISE] Volume denoised and saved to {denoised_volume_path}")
        
    except Exception as e:
        LOGGER.error(f"[RM / DENOISE] Error: {str(e)}")
    
    return result


def _process_brain_mask(
    current_image: Optional[sitk.Image],
    brain_mask: Optional[sitk.Image],
    preprocessing_plan: Dict[str, Any],
    patient_id: str,
    pulse_name: str,
    verbose: bool = False,
    save_intermediate: bool = False,
    others_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Generate a brain mask for the current image.
    
    Args:
        current_image: Current SimpleITK image object or None
        brain_mask: Current brain mask or None
        preprocessing_plan: Dictionary with step-specific configuration
        patient_id: Patient identifier (e.g., 'P1')
        pulse_name: Name of the pulse sequence (e.g., 'T1', 'T2', 'SUSC')
        verbose: Whether to print detailed processing information
        save_intermediate: Whether to save intermediate files
        others_dir: Directory for intermediate files
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with updated processing state and processed data
    """
    result = {
        "current_image": current_image,
        "brain_mask": brain_mask,
        "processed_data": {}
    }
    
    # Skip if image is missing
    if current_image is None:
        return result
    
    if verbose:
        LOGGER.info(f"[RM / BRAIN MASK] Generating brain mask")
    
    try:
        # Get brain mask parameters
        threshold_method = preprocessing_plan.get("threshold_method", "li")
        structure_size_2d = preprocessing_plan.get("structure_size_2d", 7)
        iterations_2d = preprocessing_plan.get("iterations_2d", 3)
        structure_size_3d = preprocessing_plan.get("structure_size_3d", 3)
        iterations_3d = preprocessing_plan.get("iterations_3d", 1)
        
        if verbose:
            LOGGER.info(f"[RM / BRAIN MASK] Parameters: threshold_method={threshold_method}, "
                  f"structure_size_2d={structure_size_2d}, iterations_2d={iterations_2d}, "
                  f"structure_size_3d={structure_size_3d}, iterations_3d={iterations_3d}")
        
        # Generate brain mask using the current image
        _, new_brain_mask = generate_brain_mask_sitk(
            current_image,
            threshold_method=threshold_method,
            structure_size_2d=structure_size_2d,
            iterations_2d=iterations_2d,
            structure_size_3d=structure_size_3d,
            iterations_3d=iterations_3d
        )
        
        result["brain_mask"] = new_brain_mask
        
        # Save the brain mask if requested
        if save_intermediate and others_dir:
            brain_mask_path = others_dir / f"{pulse_name}_{patient_id}_brain_mask.nii.gz"
            sitk.WriteImage(new_brain_mask, str(brain_mask_path))
            
            # Update the processed data with the new file path
            result["processed_data"]["brain_mask_path"] = str(brain_mask_path)
            
            if verbose:
                LOGGER.info(f"[RM / BRAIN MASK] Brain mask generated and saved to {brain_mask_path}")
        
    except Exception as e:
        LOGGER.error(f"[RM / BRAIN MASK] Error: {str(e)}")
    
    return result


def _process_bias_correction(
    current_image: Optional[sitk.Image],
    brain_mask: Optional[sitk.Image],
    preprocessing_plan: Dict[str, Any],
    patient_id: str,
    pulse_name: str,
    verbose: bool = False,
    save_intermediate: bool = False,
    others_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply N4 bias field correction to the current image.
    
    Args:
        current_image: Current SimpleITK image object or None
        brain_mask: Current brain mask or None
        preprocessing_plan: Dictionary with step-specific configuration
        patient_id: Patient identifier (e.g., 'P1')
        pulse_name: Name of the pulse sequence (e.g., 'T1', 'T2', 'SUSC')
        verbose: Whether to print detailed processing information
        save_intermediate: Whether to save intermediate files
        others_dir: Directory for intermediate files
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with updated processing state and processed data
    """
    result = {
        "current_image": current_image,
        "brain_mask": brain_mask,
        "processed_data": {}
    }
    
    # Skip if necessary components are missing
    if current_image is None or brain_mask is None:
        return result
    
    # Check if N4 correction is configured
    if "n4" not in preprocessing_plan:
        if verbose:
            LOGGER.info(f"[RM / BIAS CORRECTION] Only N4 bias field correction is supported, skipping")
        return result
    
    if verbose:
        LOGGER.info(f"[RM / BIAS CORRECTION] Applying N4 bias field correction")
    
    try:
        # Get N4 parameters
        n4_params = preprocessing_plan["n4"]
        shrink_factor = n4_params.get("shrink_factor", 4)
        max_iterations = n4_params.get("max_iterations", 100)
        control_points = n4_params.get("control_points", 6)
        bias_field_fwhm = n4_params.get("bias_field_fwhm", 0.1)
        
        if verbose:
            LOGGER.info(f"[RM / BIAS CORRECTION] N4 parameters: shrink_factor={shrink_factor}, "
                  f"max_iterations={max_iterations}, control_points={control_points}, "
                  f"bias_field_fwhm={bias_field_fwhm}")
        
        # Apply N4 bias field correction
        corrected_image = n4_bias_field_correction(
            current_image,
            mask_sitk=brain_mask,
            shrink_factor=shrink_factor,
            max_iterations=max_iterations,
            bias_field_fwhm=bias_field_fwhm,
            control_points=control_points,
            verbose=verbose
        )
        
        result["current_image"] = corrected_image
        
        # Save the bias-corrected image if requested
        if save_intermediate and others_dir:
            bias_corrected_path = others_dir / f"{pulse_name}_{patient_id}_bias_corrected.nii.gz"
            sitk.WriteImage(corrected_image, str(bias_corrected_path))
            
            # Update the processed data with the new file path
            result["processed_data"]["bias_corrected_path"] = str(bias_corrected_path)
            
            if verbose:
                LOGGER.info(f"[RM / BIAS CORRECTION] Image bias-corrected and saved to {bias_corrected_path}")
        
    except Exception as e:
        LOGGER.error(f"[RM / BIAS CORRECTION] Error: {str(e)}")
    
    return result


def _process_registration(
    current_image: Optional[sitk.Image],
    current_mask: Optional[sitk.Image],
    preprocessing_plan: Dict[str, Any],
    patient_output_dir: Path,
    patient_id: str,
    pulse_name: str,
    *,
    verbose: bool = False,
    save_intermediate: bool = False,
    others_dir: Optional[Path] = None,
    apply_composite: bool = False,           # NEW
    **kwargs,
) -> Dict[str, Any]:
    """
    Register *current_image* (+ mask) to the SRI-24 atlas.

    Workflow
    --------
    * **T1** or ``apply_composite is False``  
      → full Rigid + Affine + SyN directly to the atlas.

    * Non-T1 **and** ``apply_composite is True``  
      1. Rigid + Affine **(secondary_modality)** to the saved *T1-native* image.  
      2. Compose that transform with the previously-saved *T1 → atlas* warp.  
      3. Resample image & mask in a *single* interpolation step.
    """
    result = {
        "current_image": current_image,
        "current_mask": current_mask,
        "processed_data": {},
    }

    # ---------- sanity ----------------------------------------------------
    if current_image is None or current_mask is None:
        return result
    if "sri24" not in preprocessing_plan:
        if verbose:
            LOGGER.info("[RM / REGISTRATION] no SRI24 config – skipping")
        return result

    sri24_cfg = preprocessing_plan["sri24"]
    if not sri24_cfg.get("enable", False):
        if verbose:
            LOGGER.info("[RM / REGISTRATION] disabled – skipping")
        return result

    yaml_path = sri24_cfg.get("config_path")
    if not yaml_path or not os.path.exists(yaml_path):
        raise FileNotFoundError(f"registration YAML not found: {yaml_path}")

    # ---------------------------------------------------------------------
    # (A) T1  OR  no composite –>  full registration
    # ---------------------------------------------------------------------
    if pulse_name == "T1" or not apply_composite:
        # save the *native* T1 (once) so later pulses can align to it
        if pulse_name == "T1":
            t1_native_path = patient_output_dir / "T1_native.nii.gz"
            if not t1_native_path.exists():
                sitk.WriteImage(current_image, str(t1_native_path))

        # run full R+A+SyN
        registered_img, registered_msk, tfm = register_image_to_sri24(
            moving=current_image,
            yaml_path=yaml_path,
            moving_mask=current_mask,
            full_registration=True,
            verbose=verbose,
        )

        # persist outputs
        img_out = patient_output_dir / f"{pulse_name}_{patient_id}_registered_sri24.nii.gz"
        msk_out = patient_output_dir / f"{pulse_name}_{patient_id}_mask_registered_sri24.nii.gz"
        sitk.WriteImage(registered_img, str(img_out))
        sitk.WriteImage(registered_msk, str(msk_out))

        tfm_json = patient_output_dir / f"transform_params_{pulse_name}.json"
        with open(tfm_json, "w") as fp:
            json.dump(tfm, fp, indent=2)

        if verbose:
            LOGGER.info(f"[RM / REGISTRATION] full warp saved → {img_out}")

        result.update(
            current_image=registered_img,
            current_mask=registered_msk,
            processed_data={
                "registered_image_path": str(img_out),
                "registered_mask_path": str(msk_out),
                "registration_transform_params": tfm,
            },
        )
        return result

    # ---------------------------------------------------------------------
    # (B) non-T1  **and**  composite =True
    # ---------------------------------------------------------------------
    try:
        # 1. load T1 → atlas transform & native T1
        t1_json = patient_output_dir / "transform_params_T1.json"
        t1_native = patient_output_dir / "T1_native.nii.gz"
        if not (t1_json.exists() and t1_native.exists()):
            raise FileNotFoundError("T1 assets missing; cannot compose transforms")

        with open(t1_json) as fp:
            t1_to_atlas = json.load(fp)

        # 2. parse YAML to get secondary-modality parameters
        with open(yaml_path) as fp:
            yaml_cfg = yaml.safe_load(fp)
        interp = yaml_cfg.get("interpolation", "Linear")
        sec_cfg = yaml_cfg.get("secondary_modality", {})

        # 3. register modality → T1 (native)   [Rigid + Affine]
        reg_tmp_dir = (
            (others_dir or patient_output_dir) / f"{pulse_name}_to_T1_reg"
        )
        reg_tmp_dir.mkdir(exist_ok=True)

        _, mod2t1 = register_to_sri24(
            moving_image=current_image,
            atlas_path=str(t1_native),
            output_dir=reg_tmp_dir,
            output_prefix=f"{pulse_name}toT1_",
            output_image_path=reg_tmp_dir / f"{pulse_name}_in_T1.nii.gz",
            full_registration=False,
            interpolation=interp,
            stage_params=sec_cfg,
            dimension=yaml_cfg.get("dimension", 3),
            histogram_matching=bool(yaml_cfg.get("histogram_matching", True)),
            winsorize_lower_quantile=float(
                yaml_cfg.get("winsorize_lower_quantile", 0.005)
            ),
            winsorize_upper_quantile=float(
                yaml_cfg.get("winsorize_upper_quantile", 0.995)
            ),
            number_threads=int(yaml_cfg.get("number_threads", 1)),
            verbose=verbose,
            cleanup=not SAVE_INTERMEDIATE,
        )

        # 4. composite warp  (modality→T1) ∘ (T1→atlas)
        img_out = patient_output_dir / f"{pulse_name}_{patient_id}_registered_sri24.nii.gz"
        registered_img = apply_composed_transforms(
            input_image=current_image,
            t1_to_atlas=t1_to_atlas,
            modality_to_t1=mod2t1,
            output_path=str(img_out),
            interpolation=interp,
            verbose=verbose,
        )

        # 5. warp mask
        msk_out = patient_output_dir / f"{pulse_name}_{patient_id}_mask_registered_sri24.nii.gz"
        registered_msk = apply_composed_transforms(
            input_image=current_mask,
            t1_to_atlas=t1_to_atlas,
            modality_to_t1=mod2t1,
            output_path=str(msk_out),
            interpolation="NearestNeighbor",
            verbose=False,
        )

        # 6. save combined transform params
        combo_json = patient_output_dir / f"transform_params_{pulse_name}.json"
        with open(combo_json, "w") as fp:
            json.dump(
                {
                    "t1_to_atlas": t1_to_atlas,
                    "modality_to_t1": mod2t1,
                },
                fp,
                indent=2,
            )

        if verbose:
            LOGGER.info(f"[RM / REGISTRATION] composite warp saved → {img_out}")

        result.update(
            current_image=registered_img,
            current_mask=registered_msk,
            processed_data={
                "registered_image_path": str(img_out),
                "registered_mask_path": str(msk_out),
                "registration_transform_params": {
                    "t1_to_atlas": t1_to_atlas,
                    "modality_to_t1": mod2t1,
                },
            },
        )
        return result

    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error(f"[RM / REGISTRATION] composite path failed – {exc}")
        # graceful fallback: run full registration
        preprocessing_plan["sri24"]["enable"] = True
        return _process_registration(
            current_image,
            current_mask,
            preprocessing_plan,
            patient_output_dir,
            patient_id,
            pulse_name,
            verbose=verbose,
            save_intermediate=save_intermediate,
            others_dir=others_dir,
            apply_composite=False,  # avoid infinite loop
        )


def _process_brain_extraction(
    current_image: Optional[sitk.Image],
    preprocessing_plan: Dict[str, Any],
    patient_id: str,
    pulse_name: str,
    verbose: bool = False,
    save_intermediate: bool = True,
    others_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply brain extraction to the current image using either FSL BET or a universal mask template.
    
    Args:
        current_image: Current SimpleITK image object or None
        preprocessing_plan: Dictionary with step-specific configuration
        patient_id: Patient identifier (e.g., 'P1')
        pulse_name: Name of the pulse sequence (e.g., 'T1', 'T2', 'SUSC')
        verbose: Whether to print detailed processing information
        save_intermediate: Whether to save intermediate files
        others_dir: Directory for intermediate files
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with updated processing state and processed data
    """
    result = {
        "current_image": current_image,
        "brain_mask": None,
        "processed_data": {}
    }
    
    # Skip if image is missing
    if current_image is None:
        return result
    
    # Check if using universal mask from SRI24 atlas
    if "universal_mask" in preprocessing_plan:
        if verbose:
            LOGGER.info(f"[RM / BRAIN EXTRACTION] Using universal mask template from SRI24 atlas")
        
        try:
            # Get path to template brain mask
            mask_path = preprocessing_plan["universal_mask"].get("path")
            if not mask_path or not os.path.exists(mask_path):
                LOGGER.error(f"[RM / BRAIN EXTRACTION] Universal mask template not found at {mask_path}")
                return result
            
            if verbose:
                LOGGER.info(f"[RM / BRAIN EXTRACTION] Loading brain template from {mask_path}")
            
            # Load the template brain image (this is already skull-stripped)
            template_brain = sitk.ReadImage(mask_path)
            
            # Create a binary mask from the brain template (1 where brain tissue exists)
            binary_threshold = sitk.BinaryThreshold(
                template_brain, 
                lowerThreshold=0.01,  # Small positive threshold to avoid numerical issues
                upperThreshold=float('inf'),
                insideValue=0,
                outsideValue=1
            )
            
            # Ensure mask is of correct type
            brain_mask = sitk.Cast(binary_threshold, sitk.sitkUInt8)
            
            # Resample mask to match the input image dimensions and orientation
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(current_image)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for binary masks
            brain_mask = resampler.Execute(brain_mask)
            
            # Apply the brain mask to the current image (retains only brain tissue)
            extracted_brain = sitk.Mask(current_image, brain_mask)
            
            # Store results
            result["current_image"] = extracted_brain  # This now contains only the brain tissue
            result["brain_mask"] = brain_mask
            
            # Always save the brain mask, even if save_intermediate is False
            if others_dir and save_intermediate:
                brain_extracted_path = others_dir / f"{pulse_name}_{patient_id}_brain_extracted.nii.gz"
                brain_extraction_mask_path = others_dir / f"{pulse_name}_{patient_id}_brain_extraction_mask.nii.gz"
                
                sitk.WriteImage(extracted_brain, str(brain_extracted_path))
                sitk.WriteImage(brain_mask, str(brain_extraction_mask_path))
                
                # Update the processed data with the new file paths
                result["processed_data"]["brain_extracted_path"] = str(brain_extracted_path)
                result["processed_data"]["brain_extraction_mask_path"] = str(brain_extraction_mask_path)
                
                if verbose:
                    LOGGER.info(f"[RM / BRAIN EXTRACTION] Brain extracted using SRI24 template and saved to {brain_extracted_path}")
                    LOGGER.info(f"[RM / BRAIN EXTRACTION] Brain mask saved to {brain_extraction_mask_path}")
                
        except Exception as e:
            LOGGER.error(f"[RM / BRAIN EXTRACTION] Error applying universal mask: {str(e)}")
            
    # Fall back to FSL BET if universal mask is not specified or failed
    elif "fsl_bet" in preprocessing_plan:
        if verbose:
            LOGGER.info(f"[RM / BRAIN EXTRACTION] Applying FSL BET brain extraction")
        
        try:
            # Get FSL BET parameters
            bet_params = preprocessing_plan["fsl_bet"]
            frac = bet_params.get("frac", 0.5)
            robust = bet_params.get("robust", True)
            vertical_gradient = bet_params.get("vertical_gradient", 0.0)
            skull = bet_params.get("skull", False)
            
            if verbose:
                LOGGER.info(f"[RM / BRAIN EXTRACTION] FSL BET parameters: frac={frac}, "
                      f"robust={robust}, vertical_gradient={vertical_gradient}, "
                      f"skull={skull}")
            
            # Apply FSL BET brain extraction
            extracted_brain, brain_mask = fsl_bet_brain_extraction(
                current_image,
                frac=frac,
                robust=robust,
                vertical_gradient=vertical_gradient,
                skull=skull,
                verbose=verbose
            )
            
            result["current_image"] = extracted_brain
            result["brain_mask"] = brain_mask
            
            # Always save the brain mask for T1, even if save_intermediate is False
            if others_dir:
                brain_extracted_path = others_dir / f"{pulse_name}_{patient_id}_brain_extracted.nii.gz"
                brain_extraction_mask_path = others_dir / f"{pulse_name}_{patient_id}_brain_extraction_mask.nii.gz"
                
                sitk.WriteImage(extracted_brain, str(brain_extracted_path))
                sitk.WriteImage(brain_mask, str(brain_extraction_mask_path))
                
                # Update the processed data with the new file paths
                result["processed_data"]["brain_extracted_path"] = str(brain_extracted_path)
                result["processed_data"]["brain_extraction_mask_path"] = str(brain_extraction_mask_path)
                
                if verbose:
                    LOGGER.info(f"[RM / BRAIN EXTRACTION] Brain extracted and saved to {brain_extracted_path}")
                    LOGGER.info(f"[RM / BRAIN EXTRACTION] Brain extraction mask saved to {brain_extraction_mask_path}")
            
        except Exception as e:
            LOGGER.error(f"[RM / BRAIN EXTRACTION] Error with FSL BET: {str(e)}")
    else:
        if verbose:
            LOGGER.info(f"[RM / BRAIN EXTRACTION] No brain extraction method specified, skipping")
    
    return result


def _process_brain_extraction_with_t1_mask(
    current_image: Optional[sitk.Image],
    t1_brain_mask_path: Optional[str],
    patient_id: str,
    pulse_name: str,
    verbose: bool = False,
    save_intermediate: bool = False,
    others_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply brain extraction using the T1 brain mask for non-T1 pulses.
    
    Args:
        current_image: Current SimpleITK image object or None
        t1_brain_mask_path: Path to T1 brain mask
        patient_id: Patient identifier (e.g., 'P1')
        pulse_name: Name of the pulse sequence (e.g., 'T1', 'T2', 'SUSC')
        verbose: Whether to print detailed processing information
        save_intermediate: Whether to save intermediate files
        others_dir: Directory for intermediate files
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary with updated processing state and processed data
    """
    result = {
        "current_image": current_image,
        "brain_mask": None,
        "processed_data": {}
    }
    
    # Skip if necessary components are missing
    if current_image is None or t1_brain_mask_path is None:
        if verbose:
            LOGGER.warning(f"[RM / BRAIN EXTRACTION] Missing image or T1 brain mask path, skipping")
        return result
    
    if verbose:
        LOGGER.info(f"[RM / BRAIN EXTRACTION] Using T1 brain mask for non-T1 pulse")
    
    try:
        # Load the T1 brain mask
        t1_brain_mask = sitk.ReadImage(t1_brain_mask_path)
        
        # Apply the T1 brain mask to the current image
        extracted_brain = sitk.Mask(current_image, t1_brain_mask)
        
        result["current_image"] = extracted_brain
        result["brain_mask"] = t1_brain_mask
        
        # Save the brain-extracted image and mask if requested
        if save_intermediate and others_dir:
            brain_extracted_path = others_dir / f"{pulse_name}_{patient_id}_brain_extracted_t1mask.nii.gz"
            
            sitk.WriteImage(extracted_brain, str(brain_extracted_path))
            
            # Update the processed data with the new file paths
            result["processed_data"]["brain_extracted_path"] = str(brain_extracted_path)
            result["processed_data"]["brain_extraction_mask_path"] = t1_brain_mask_path
            
            if verbose:
                LOGGER.info(f"[RM / BRAIN EXTRACTION] Brain extracted using T1 mask and saved to {brain_extracted_path}")
        
    except Exception as e:
        LOGGER.error(f"[RM / BRAIN EXTRACTION] Error applying T1 brain mask: {str(e)}")
    
    return result