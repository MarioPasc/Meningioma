# Meningioma MRI/CT Pre-Processing Pipeline

This repository provides a *reproducible* pipeline that prepares clinical
T1-, T2-, FLAIR- and susceptibility-weighted MRI (±CT) for automatic
meningioma segmentation with a pre-trained **nnU-Net** model.  
All steps are implemented in *Pure Python + SimpleITK + Nipype* and can be
executed with a single command (see `preprocess_patient.py`).

---

## 1  Workflow Summary 

| Stage No | Stage name | Purpose (scientific rationale) |
|:--|:--|:--|
| 1 |**Raw import + channel split**|Clinical MRI are stored as multi-component NRRD. We extract the first channel and keep *all* meta-data (voxel spacing, orientation) for accurate spatial processing.|
| 2 |**NIfTI export**|Converts NRRD → NIfTI‐1 using a custom affine built from the NRRD *space directions*; the entire original header is preserved in a NIfTI extension.|
| 3 |**Datatype casting**|Volumes are cast to `float32` (mandatory for N4 and deep nets); masks—if present—are clamped to {0,1} and cast to `uint8`.|
| 4 |**Brain extraction**|We run FSL **BET** (`-R -g 0`) and automatically *fix* the polarity and slightly *dilate* the mask so extra-axial meningiomas are never cropped.|
| 5 |**N4 bias-field correction**|Corrects B<sub>1</sub> inhomogeneity (spatial intensity shading) using a mask-guided N4 filter with shrink-factor 4 and 100 iterations.|
| 6 |**Intensity normalisation**|Per-scan z-score normalisation: \\(\hat I = (I-\mu_\text{brain})/\sigma\\). Intensities are clipped to [-3, 3] SD to suppress outliers while keeping pathological hyper-intensity intact.|
| 7 |**Multi-modal registration**|***Two–step strategy (crucial for robustness):***<br>① *T1 → SRI24 atlas* (non-linear SyN)<br>② *(T2/FLAIR/SUSC) → T1* (rigid + affine)<br>The secondary → T1 transform is then **composed** with the stored T1 → atlas warp, guaranteeing *sub-voxel* multi-modal alignment **in atlas space**.|
| 8 |**Quality assurance**|For every stage we log: voxel-wise intensity stats, geometry, transform sanity (max shear/translation) → `misc/qa_report.json`.|
| 9 |**Outputs**|`results/` contains the atlas-aligned, bias-corrected, normalised, 1 mm-isotropic NIfTI volumes ready for nnU-Net. `misc/` stores **all** intermediates (`stage-XXX_*.nii.gz`) and ANTs transform files for provenance and resume-execution.|

> **Resume execution** – Each intermediate is saved as  
> `stage-<NNN>_<name>_<pulse>.nii.gz`.  
> On re-running, the pipeline inspects these files and continues from the
> first *missing* stage, allowing interruption-safe processing of large
> cohorts.

---

## 2  Folder Structure (per patient)

```

P42/
├── results/
│   ├── T1\_P42\_atlas.nii.gz
│   ├── T2\_P42\_atlas.nii.gz
│   ├── FLAIR\_P42\_atlas.nii.gz
│   └── SUSC\_P42\_atlas.nii.gz
└── misc/
├── stage-001\_raw\_T1.nii.gz
├── …                                        ← intermediates
├── transforms/
│   ├── t1\_to\_atlas/                         ← ANTs SyN (h5+mat)
│   └── t2\_to\_atlas/
└── qa\_report.json

```
All images in *results/* share:

* **Resolution** 1 x 1 x 1 mm  
* **FoV / orientation** SRI24 T1 template  
* **Intensity scale** 0-mean, unit-variance (brain voxels)  
* **Perfect intra-patient co-registration** by construction.

---

## 3  Scientific Notes on the Registration Design

1. *Why T1 as anchor?* Most MRI scanners acquire T1 with the highest
   spatial resolution and lowest distortion, making it the best reference.
2. *Why two-step instead of direct multi-modal→atlas?* Subject-level
   (rigid/affine) alignment corrects scanner-specific offsets **before**
   the high-dof SyN warp. This avoids cross-modal artefacts where texture
   differences misguide the similarity metric.
3. *Choice of SRI24* SRI24 is an adult T1 template with balanced GM/WM
   contrast and an associated atlas mask, mirroring the BraTS-MEN
   preprocessing pipeline on which our nnU-Net model was trained.
4. *Transform provenance* All forward & inverse transforms are saved,
   enabling **inverse-mapping** of nnU-Net segmentations back to each
   patient’s native space for radiologist review.

---

## 4  Running the pipeline

```bash
conda activate mgmGrowth
python preprocess_patient.py \
   --patient_id P42 \
   --root      /path/to/Raw/RM \
   --out_dir   /path/to/PreprocOut \
   --sri24_t1  /atlas/sri24_spm8/templates/T1.nii \
   --bet_frac  0.5
````
