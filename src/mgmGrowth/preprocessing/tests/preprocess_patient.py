#!/usr/bin/env python3
"""
Preprocess a single meningioma patient (T1, T2, FLAIR, SUSC).

* The only CLI inputs are   ▸ patient_id   ▸ planner.yaml
* The list and order of stages is taken from planner.yaml
* Intermediate results are written to   <out_dir>/<patient_id>/misc/
* Final atlas-aligned volumes end up in <out_dir>/<patient_id>/results/

-----------------------------------------------------------------------
Stage discovery & mapping
-----------------------------------------------------------------------
The keys inside   preprocessing.<MODALITY_GROUP>   of planner.yaml are
read in insertion order and mapped to **canonical internal names**:

    YAML key              →  internal stage name
    ─────────────────────────────────────────────────────
    remove_channel        →  raw
    export_nifti          →  nifti
    cast_volume           →  cast
    brain_extraction      →  mask
    bias_field_correction →  n4
    intensity_normalise   →  norm        (optional)
    registration          →  registration

If the YAML omits a step it is *not* executed.  Therefore the length
and order of `STAGES` depends entirely on the planner.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import SimpleITK as sitk
import yaml

# ─────────────────────────── project helpers ────────────────────────────
from mgmGrowth.preprocessing import LOGGER
from mgmGrowth.preprocessing.tools.remove_extra_channels import remove_first_channel
from mgmGrowth.preprocessing.tools.nrrd_to_nifti import nifti_write_3d
from mgmGrowth.preprocessing.tools.casting import cast_volume_and_optional_mask
from mgmGrowth.preprocessing.tools.intensity_normalization import zscore_normalise
from mgmGrowth.preprocessing.tools.brain_mask_utils import (
    ensure_binary_polarity,
    dilate_mask,
)
from mgmGrowth.preprocessing.tools.skull_stripping.fsl_bet import (
    fsl_bet_brain_extraction,
)
from mgmGrowth.preprocessing.tools.bias_field_corr_n4 import (
    generate_brain_mask_sitk,
    n4_bias_field_correction,
)
from mgmGrowth.preprocessing.tools.registration import (
    register_image_to_sri24,
    register_secondary_to_primary,
)
from mgmGrowth.preprocessing.tools.qa_utils import (
    IntensityStats,
    geometry_summary,
    intensity_summary,
    transform_sanity_check,
)

###############################################################################
#                    ----------  configuration  ----------                    #
###############################################################################


_YAML2INTERNAL = {
    "remove_channel": "raw",
    "export_nifti": "nifti",
    "cast_volume": "cast",
    "brain_extraction": "mask",
    "bias_field_correction": "n4",
    "intensity_normalise": "norm",  # optional, not in example planner
    "registration": "registration",
}


def _build_stage_dict(step_keys: Iterable[str]) -> OrderedDict[str, int]:
    """
    Convert the YAML list of keys into an OrderedDict mapping the
    **canonical internal stage names** to an incremental index.
    """
    stages: "OrderedDict[str, int]" = OrderedDict()
    for idx, yaml_key in enumerate(step_keys):
        internal = _YAML2INTERNAL.get(yaml_key)
        if internal is None:  # ignore unrecognised blocks silently
            continue
        stages[internal] = idx
    return stages


###############################################################################
# ------------------------------  helper I/O  -------------------------------- #
###############################################################################

def _load_checkpoint(
    misc_dir: Path,
    pulse: str,
    stage_idx: int,
    stages: "OrderedDict[str, int]",
) -> Tuple[Optional[sitk.Image], Optional[sitk.Image]]:
    """
    Load the image (and, if available, its brain-mask) stored at *stage_idx*.
    Returns (img, mask) – either may be None when not present on disk.
    """
    # image of the last completed stage
    patt = f"stage-{stage_idx:03d}_*_{pulse}.nii.gz"
    files = list(misc_dir.glob(patt))
    img = sitk.ReadImage(str(files[0])) if files else None

    # brain-mask if the mask-stage has already run
    mask: Optional[sitk.Image] = None
    if "mask" in stages and stage_idx >= stages["mask"]:
        mask_file = misc_dir / f"stage-{stages['mask']:03d}_mask_{pulse}.nii.gz"
        if mask_file.exists():
            mask = sitk.ReadImage(str(mask_file))

    return img, mask


def _save_intermediate(
    img: sitk.Image,
    misc_dir: Path,
    stage_name: str,
    pulse: str,
    counter: int,
    stages: "OrderedDict[str, int]",
) -> Path:
    """
    Write *img* to disk with a filename encoding stage index and name.
    A small text file maps indices→names for later inspection.
    """
    stage_map = misc_dir / "stage_map.txt"
    if not stage_map.exists():
        with stage_map.open("w") as fp:
            for name, num in stages.items():
                fp.write(f"{num:03d} {name}\n")

    fname = f"stage-{counter:03d}_{stage_name}_{pulse}.nii.gz"
    out_path = misc_dir / fname
    sitk.WriteImage(img, str(out_path))
    return out_path


def _last_stage_completed(misc_dir: Path, pulse: str) -> int:
    """
    Inspect filenames like  'stage-003_cast_T1.nii.gz' and return the
    highest numeric stage already present for *pulse*.
    """
    patt = re.compile(r"stage-(\d{3})_.*_" + re.escape(pulse) + r"\.nii\.gz$")
    nums = [
        int(m.group(1))
        for m in (patt.match(f.name) for f in misc_dir.glob(f"stage-*_{pulse}.nii.gz"))
        if m
    ]
    return max(nums) if nums else -1


###############################################################################
# -----------------------------  main driver  -------------------------------- #
###############################################################################


@dataclass(frozen=True)
class Planner:
    """Minimal strongly-typed view of the YAML content we need."""
    root: Path
    output: Path
    atlas_root: Path
    rm_cfg: Dict[str, Any]
    threads: int

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Planner":
        data = yaml.safe_load(yaml_path.read_text())

        root = Path(data["paths"]["root"]).expanduser()
        output = Path(data["paths"]["output"]).expanduser()
        atlas_root = Path(data["paths"]["atlas_root"]).expanduser()

        rm_cfg = data["preprocessing"]["RM"]
        threads = int(data.get("processing", {}).get("threads", 1))

        return cls(root, output, atlas_root, rm_cfg, threads)


class PatientPreprocessor:
    """
    MRI preprocessing orchestrator for a single patient.

    The concrete list and order of stages is determined at runtime from
    the supplied planner (YAML).  Each stage is implemented by a handler
    method named  _stage_<internal_name>() .
    """

    # ------------------------------------------------------------------ #
    #                             lifecycle                              #
    # ------------------------------------------------------------------ #
    def __init__(self, patient_id: str, planner: Planner) -> None:
        self.pid = patient_id
        self.plan = planner

        # dynamic stage table (e.g. {'raw':0, 'nifti':1, ...})
        self.stages = _build_stage_dict(self.plan.rm_cfg.keys())

        # folders --------------------------------------------------------
        self.patient_dir = self.plan.output / self.pid
        self.results_dir = self.patient_dir / "results"
        self.misc_dir = self.patient_dir / "misc"
        (self.results_dir).mkdir(parents=True, exist_ok=True)
        (self.misc_dir / "transforms").mkdir(parents=True, exist_ok=True)

        # QA collection
        self.qa: Dict[str, Any] = {}

        # atlas template
        self.sri24_t1 = self.plan.atlas_root / "T1.nii"  # generic name

        # place-holders used across stages
        self._t1_to_atlas_params: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    #                               API                                  #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Run the configured pipeline."""
        t1_img = self._process_modality("T1", primary=True)

        # atlas geometry QA once
        self.qa["atlas_geometry"] = geometry_summary(t1_img)

        # secondary modalities
        for pulse in ("T2", "FLAIR", "SUSC"):
            self._process_modality(pulse, reference_img=t1_img)

        # dump QA -------------------------------------------------------
        (self.misc_dir / "qa_report.json").write_text(json.dumps(self.qa, indent=2))

    # ------------------------------------------------------------------ #
    #                        modality processing                         #
    # ------------------------------------------------------------------ #
    def _process_modality(
        self,
        pulse: str,
        *,
        primary: bool = False,
        reference_img: Optional[sitk.Image] = None,
    ) -> sitk.Image:
        """
        Execute *all* stages for the given pulse.  The set and order of
        stages is contained in   self.stages  and the detailed per-stage
        parameters in   self.plan.rm_cfg .
        """
        LOGGER.info(f"[{pulse}] ✦ pipeline start")
        already = _last_stage_completed(self.misc_dir, pulse)
        LOGGER.info(f"[{pulse}] resume ➜ last completed index = {already:02d}")

        # Load previous checkpoint ------------------------------------------------------------
        img: Optional[sitk.Image] = None
        mask: Optional[sitk.Image] = None
        hdr: Dict[str, Any] = {}
        qa_pulse: Dict[str, Any] = {}

        if already >= 0:
            img, mask = _load_checkpoint(
                self.misc_dir, pulse, already, self.stages
            )
            if img is None:
                LOGGER.warning(
                    f"[{pulse}] resume requested but no checkpoint file found – "
                    "re-running the full pipeline for this modality"
                )
                already = -1

        # ---------------- main stage dispatcher -----------------
        for name, idx in self.stages.items():
            if already >= idx:  # resume → skip completed
                continue

            handler = getattr(self, f"_stage_{name}", None)
            if handler is None:
                LOGGER.warning(f"[{pulse}] no handler for stage '{name}', skipping")
                continue

            LOGGER.info(f"[{pulse}] ▸ stage {idx:02d} – {name}")
            img, hdr, mask = handler(
                pulse=pulse,
                img=img,
                hdr=hdr,
                mask=mask,
                qa=qa_pulse,
                cfg=self.plan.rm_cfg,
                stage_idx=idx,
                reference_img=reference_img,
                primary=primary,
            )

        # --------------------------------------------------------
        if img is None:
            raise RuntimeError(f"{pulse}: pipeline produced no image")

        self.qa[pulse] = qa_pulse
        return img

    # ================================================================== #
    #                           STAGE HANDLERS                           #
    # Each method updates (img, hdr, mask) and returns the new triple.   #
    # ================================================================== #
    # 1 ─ remove_channel  →  internal 'raw'
    # ------------------------------------------------------------------ #
    def _stage_raw(
        self,
        *,
        pulse: str,
        img: Optional[sitk.Image],
        hdr: Dict[str, Any],
        mask: Optional[sitk.Image],
        qa: Dict[str, Any],
        cfg: Dict[str, Any],
        stage_idx: int,
        **_,
    ) -> Tuple[sitk.Image, Dict[str, Any], Optional[sitk.Image]]:
        params = cfg["remove_channel"]
        channel = int(params.get("channel", 0))

        raw_path = (
            self.plan.root / pulse /self.pid / f"{pulse}_{self.pid}.nrrd"
        ).expanduser()

        img, hdr = remove_first_channel(raw_path, channel=channel, verbose=False)
        qa["raw_stats"] = intensity_summary(img).as_dict()

        _save_intermediate(
            img, self.misc_dir, "raw", pulse, stage_idx, stages=self.stages
        )
        return img, hdr, mask

    # 2 ─ export_nifti
    # ------------------------------------------------------------------ #
    def _stage_nifti(
        self,
        *,
        pulse: str,
        img: sitk.Image,
        hdr: Dict[str, Any],
        mask: Optional[sitk.Image],
        cfg: Dict[str, Any],
        stage_idx: int,
        **_,
    ) -> Tuple[sitk.Image, Dict[str, Any], Optional[sitk.Image]]:
        out = self.misc_dir / f"stage-{stage_idx:03d}_nifti_{pulse}.nii.gz"
        nifti_write_3d((img, hdr), out_file=str(out), verbose=False)
        return img, hdr, mask

    # 3 ─ cast_volume
    # ------------------------------------------------------------------ #
    def _stage_cast(
        self,
        *,
        pulse: str,
        img: sitk.Image,
        hdr: Dict[str, Any],
        mask: Optional[sitk.Image],
        stage_idx: int,
        **_,
    ) -> Tuple[sitk.Image, Dict[str, Any], Optional[sitk.Image]]:
        img, _ = cast_volume_and_optional_mask(img, None)
        _save_intermediate(
            img, self.misc_dir, "cast", pulse, stage_idx, stages=self.stages
        )
        return img, hdr, mask

    # 4 ─ brain_extraction
    # ------------------------------------------------------------------ #
    def _stage_mask(
        self,
        *,
        pulse: str,
        img: sitk.Image,
        hdr: Dict[str, Any],
        mask: Optional[sitk.Image],
        cfg: Dict[str, Any],
        stage_idx: int,
        qa: Dict[str, Any],
        **_,
    ) -> Tuple[sitk.Image, Dict[str, Any], Optional[sitk.Image]]:
        be_cfg = cfg["brain_extraction"]["fsl_bet"]
        bet_frac = float(be_cfg.get("frac", 0.5))

        brain, mask = fsl_bet_brain_extraction(
            img,
            frac=bet_frac,
            skull=False,
            robust=be_cfg.get("robust", True),
            verbose=False,
        )
        mask = ensure_binary_polarity(mask, brain_is_one=True)
        mask = dilate_mask(mask, radius_mm=1.5)

        _save_intermediate(
            mask, self.misc_dir, "mask", pulse, stage_idx, stages=self.stages
        )
        return brain, hdr, mask

    # 5 ─ bias_field_correction
    # ------------------------------------------------------------------ #
    def _stage_n4(
        self,
        *,
        pulse: str,
        img: sitk.Image,
        hdr: Dict[str, Any],
        mask: Optional[sitk.Image],
        cfg: Dict[str, Any],
        stage_idx: int,
        qa: Dict[str, Any],
        **_,
    ) -> Tuple[sitk.Image, Dict[str, Any], Optional[sitk.Image]]:
        n4_cfg = cfg["bias_field_correction"]["n4"]


        img = n4_bias_field_correction(
            volume_sitk=img,
            mask_sitk=None if mask is None else mask,
            shrink_factor=n4_cfg["shrink_factor"],
            max_iterations=n4_cfg["max_iterations"],
            control_points=n4_cfg["control_points"],
            bias_field_fwhm=n4_cfg.get("bias_field_fwhm", 0.15),
            verbose=False,
        )
        qa["after_n4"] = intensity_summary(img).as_dict()

        _save_intermediate(
            img, self.misc_dir, "n4", pulse, stage_idx, stages=self.stages
        )
        return img, hdr, mask

    # 6 ─ intensity_normalise  (optional – only if present in YAML)
    # ------------------------------------------------------------------ #
    def _stage_norm(
        self,
        *,
        pulse: str,
        img: sitk.Image,
        hdr: Dict[str, Any],
        mask: Optional[sitk.Image],
        stage_idx: int,
        qa: Dict[str, Any],
        **_,
    ) -> Tuple[sitk.Image, Dict[str, Any], Optional[sitk.Image]]:
        img, info = zscore_normalise(img, mask)
        qa["zscore"] = info | intensity_summary(img).as_dict()

        _save_intermediate(
            img, self.misc_dir, "norm", pulse, stage_idx, stages=self.stages
        )
        return img, hdr, mask

    # 7 ─ registration
    # ------------------------------------------------------------------ #
    def _stage_registration(
        self,
        *,
        pulse: str,
        img: sitk.Image,
        hdr: Dict[str, Any],
        mask: Optional[sitk.Image],
        cfg: Dict[str, Any],
        stage_idx: int,
        qa: Dict[str, Any],
        primary: bool,
        reference_img: Optional[sitk.Image],
        **_,
    ) -> Tuple[sitk.Image, Dict[str, Any], Optional[sitk.Image]]:
        reg_cfg = cfg["registration"]["sri24"]
        if not reg_cfg.get("enable", True):
            LOGGER.info(f"[{pulse}] registration disabled → skipping")
            return img, hdr, mask

        if primary:
            t1_reg_dir = self.misc_dir / "transforms" / "t1_to_atlas"
            t1_reg_dir.mkdir(exist_ok=True)

            registered_img, params = register_image_to_sri24(
                moving_image=img,
                moving_mask=None,
                fixed_image=sitk.ReadImage(str(self.sri24_t1)),
                config_path=Path(reg_cfg["config_path"]).expanduser(),
                
            )
            affine = params.get("affine_transform") or params["composite_transform"]
            qa["registration"] = transform_sanity_check(affine)
            self._t1_to_atlas_params = params

            sitk.WriteImage(
                registered_img,
                str(self.results_dir / f"{pulse}_{self.pid}_atlas.nii.gz"),
            )
            _save_intermediate(
                registered_img,
                self.misc_dir,
                "registration",
                pulse,
                stage_idx,
                stages=self.stages,
            )
            return registered_img, hdr, mask

        # ───────── secondary modalities ─────────
        sec_dir = self.misc_dir / "transforms" / f"{pulse.lower()}_to_atlas"
        sec_dir.mkdir(exist_ok=True)
        if reference_img is None or self._t1_to_atlas_params is None:
            raise RuntimeError("T1 must be processed before secondary pulses!")

        img_atlas, info = register_secondary_to_primary(
            secondary_image=img,
            primary_image=reference_img,
            t1_to_atlas_params=self._t1_to_atlas_params,
            output_dir=sec_dir,
            num_threads=self.plan.threads,
        )
        qa["registration"] = transform_sanity_check(info["sec2pri_transform"])
        sitk.WriteImage(
            img_atlas,
            str(self.results_dir / f"{pulse}_{self.pid}_atlas.nii.gz"),
        )
        _save_intermediate(
            img_atlas,
            self.misc_dir,
            "registration",
            pulse,
            stage_idx,
            stages=self.stages,
        )
        return img_atlas, hdr, mask


###############################################################################
# -------------------------------  CLI glue  --------------------------------- #
###############################################################################


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-patient MRI preprocessing")
    p.add_argument("--patient_id", required=True, help="Patient identifier")
    p.add_argument(
        "--planner",
        type=Path,
        required=True,
        help="Path to planner.yaml configuration file",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    planner = Planner.from_yaml(args.planner)

    prep = PatientPreprocessor(patient_id=args.patient_id, planner=planner)
    prep.run()

    print(f"✓ Finished preprocessing {args.patient_id}")
    print(f"  Results → {prep.results_dir}")
    print(f"  Misc    → {prep.misc_dir}")


if __name__ == "__main__":
    main()
