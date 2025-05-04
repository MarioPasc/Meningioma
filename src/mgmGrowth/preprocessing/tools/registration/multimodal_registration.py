#!/usr/bin/env python3
"""
Subject-level multi-modal registration helper.

Registers a *secondary* modality (e.g. T2/FLAIR/SUSC) to the subject's
*T1* and composes that transform with the existing T1→atlas warp, so
the secondary image lands in atlas space **exactly** where T1 does.

Relies on Nipype's ``Registration`` and ``ApplyTransforms`` interfaces
(ANTS) – same dependencies as the existing ``ants_registration.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import SimpleITK as sitk
from nipype.interfaces.ants import Registration, ApplyTransforms  # type: ignore


__all__ = ["register_secondary_to_primary"]


def _rigid_affine_registration(
    fixed_path: str,
    moving_path: str,
    out_prefix: str,
    num_threads: int = 1,
) -> Dict[str, str]:
    """
    Rigid + affine registration (no SyN) returning transform filenames.
    """
    reg = Registration()
    reg.inputs.fixed_image = fixed_path
    reg.inputs.moving_image = moving_path
    reg.inputs.transforms = ["Rigid", "Affine"]
    reg.inputs.transform_parameters = [(0.1,), (0.1,)]
    reg.inputs.number_of_iterations = [[1000, 500, 250], [1000, 500, 250]]
    reg.inputs.metric = ["MI", "MI"]
    reg.inputs.metric_weight = [1.0, 1.0]
    reg.inputs.shrink_factors = [[8, 4, 2], [8, 4, 2]]
    reg.inputs.smoothing_sigmas = [[3, 2, 1], [3, 2, 1]]
    reg.inputs.output_transform_prefix = out_prefix
    reg.inputs.num_threads = num_threads
    _res = reg.run()

    return {
        "composite": _res.outputs.composite_transform,
        "affine": _res.outputs.affine_transform,
        "warped": _res.outputs.warped_image,
    }


def register_secondary_to_primary(
    secondary_image: sitk.Image,
    primary_image: sitk.Image,
    t1_to_atlas_params: Dict[str, Any],
    output_dir: str | Path,
    interpolation: str = "Linear",
    num_threads: int = 1,
) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Register *secondary_image* → *primary_image* (rigid + affine),
    then compose that transform with ``t1_to_atlas_params`` to obtain
    the secondary image in atlas space.

    Returns
    -------
    (secondary_in_atlas, info_dict)
        *secondary_in_atlas* is a SimpleITK image already copied with
        atlas geometry; *info_dict* summarises paths to all intermediate
        and final transforms.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save temporary NIfTI files
    sec_path = output_dir / "secondary.nii.gz"
    pri_path = output_dir / "primary.nii.gz"
    sitk.WriteImage(secondary_image, str(sec_path))
    sitk.WriteImage(primary_image, str(pri_path))

    # 1) Secondary → Primary
    sec2pri_prefix = str(output_dir / "sec2pri_")
    sec2pri_tf = _rigid_affine_registration(
        fixed_path=str(pri_path),
        moving_path=str(sec_path),
        out_prefix=sec2pri_prefix,
        num_threads=num_threads,
    )

    # 2) Compose with Primary (T1) → Atlas
    at = ApplyTransforms()
    at.inputs.dimension = 3
    at.inputs.input_image = sec_path
    at.inputs.reference_image = t1_to_atlas_params["fixed_image_path"]
    at.inputs.transforms = [
        t1_to_atlas_params["composite_transform"],
        sec2pri_tf["composite"],
    ]  # ANTs applies last → first
    at.inputs.output_image = str(output_dir / "secondary_in_atlas.nii.gz")
    at.inputs.interpolation = interpolation
    at.inputs.num_threads = num_threads
    _ = at.run()

    sec_in_atlas = sitk.ReadImage(at.inputs.output_image)
    atlas_img = sitk.ReadImage(t1_to_atlas_params["fixed_image_path"])
    sec_in_atlas.CopyInformation(atlas_img)

    info = {
        "sec2pri_transform": sec2pri_tf["composite"],
        "t1_to_atlas_transform": t1_to_atlas_params["composite_transform"],
        "secondary_in_atlas_path": str(at.inputs.output_image),
    }
    return sec_in_atlas, info
