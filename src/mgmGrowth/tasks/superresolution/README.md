# Super-resolution (MRI) – SMORE, ECLARE, UNIRES and baselines

This subpackage provides a complete toolkit to create anisotropic MR datasets, run multiple super-resolution (SR) models, and evaluate them quantitatively and qualitatively. It contains thin runners around external CLIs (SMORE, ECLARE, UNIRES), interpolation baselines, pipelines, and metrics.

Highlights
- Data prep: subset BraTS-Men, downsample to 3/5/7 mm (z-only)
- Models: SMORE, ECLARE, UNIRES, plus interpolation baselines (NN/Linear/BSpline/Lanczos/Gaussian)
- Pipelines: batch experiments over cohorts and resolutions
- Metrics: PSNR, SSIM, Bhattacharyya, LPIPS (optional), ROI-aware
- Outputs: consistent directory layout per model and per resolution


## Package layout

```
tasks/superresolution/
  ├─ cli/                     # User-facing entry points
  │   ├─ downsample.py        # Create LR cohorts (z-downsample)
  │   ├─ prepare_brats_data.py# Subset + downsample in one go
  │   ├─ smore.py             # Train/Infer SMORE via YAML config
  │   ├─ eclare.py            # Train/Infer ECLARE via YAML config
  │   └─ unires.py            # Batch UNIRES driver (external CLI)
  ├─ engine/                  # Thin wrappers around model CLIs
  │   ├─ smore_runner.py      # smore-train / smore-test / run-smore
  │   ├─ eclare_runner.py     # eclare-train / eclare-test / run-eclare
  │   └─ interpolation_runner.py  # SimpleITK resampling helpers
  ├─ pipelines/
  │   ├─ superresolution_brats_experiment.py   # SMORE batch (per volume)
  │   └─ interpolation_brats_experiment.py     # Baselines over cohorts
  ├─ statistics/
  │   └─ metrics.py            # PSNR/SSIM/Bhattacharyya/LPIPS (+ROI)
  ├─ tools/                    # I/O, paths, parallelism, metrics utils
  ├─ utils/                    # NIfTI/LPS IO helpers, BRATS sorting
  ├─ cfg/
  │   ├─ smore_cfg.yaml        # Example SMORE config
  │   └─ eclare_cfg.yaml       # Example ECLARE config
  └─ README.md                 # This document
```


## Data layout assumptions

We use BraTS-Men naming and folder structure. Each patient folder contains pulses and a segmentation:

- `<PID>/<PID>-t1c.nii.gz`
- `<PID>/<PID>-t2w.nii.gz`
- `<PID>/<PID>-t2f.nii.gz` (FLAIR)
- `<PID>/<PID>-seg.nii.gz`

LR cohorts are organized by slice thickness: `low_res/{3mm,5mm,7mm}/<PID>/*.nii.gz`.


## Models included

1) SMORE – Self-super-resolution from through-plane slices
- Entry point: `python -m src.mgmGrowth.tasks.superresolution.cli.smore --config cfg/smore_cfg.yaml`
- Backed by external CLIs: `smore-train`, `smore-test`, `run-smore`
- Mode "train": train per-volume and immediately infer, storing
  - `out_root/SMORE/{3mm|5mm|7mm}/weights/<PID>-<pulse>.pt`
  - `out_root/SMORE/{3mm|5mm|7mm}/output_volumes/<PID>-<pulse>.nii.gz`

2) ECLARE – Self-super-resolution with explicit slice profile
- Entry point: `python -m src.mgmGrowth.tasks.superresolution.cli.eclare --config cfg/eclare_cfg.yaml`
- Backed by external CLIs: `eclare-train`, `eclare-test`, `run-eclare`
- Mirrors the SMORE interface and output layout under `out_root/ECLARE/...`

3) UNIRES – Multi-contrast reconstruction (external package)
- Batch driver: `python -m src.mgmGrowth.tasks.superresolution.cli.unires --input-dir <LR/res> --output-dir <dest> --device cuda`
- Expects one triplet per subject: `-t1c`, `-t2w`, `-t2f`
- Calls the external `unires` binary; results are written per-subject under `--output-dir/<PID>`

4) Interpolation baselines – SimpleITK resampling to 1x1x1 mm³
- Pipeline: `python -m src.mgmGrowth.tasks.superresolution.pipelines.interpolation_brats_experiment --interp {nn|linear|bspline|lanczos|gaussian}`
- Uses `engine/interpolation_runner.py` helpers and writes results under a model-like folder (e.g., `results/models/BSPLINE/{3mm,5mm,7mm}/output_volumes`).

Note: SMORE/ECLARE/UNIRES binaries must be available in your PATH. This repo intentionally wraps them without vendoring their code.


## Quick start

1) Create a curated subset and LR cohorts

```bash
python -m src.mgmGrowth.tasks.superresolution.cli.prepare_brats_data \
  --src-root  /path/to/BraTS_Men_Train \
  --out-root  /path/to/BraTS/super_resolution \
  --num-patients 50 \
  --pulses t1c t2w t2f \
  --resolutions 3 5 7 \
  --jobs 8
```

Alternatively, just downsample an existing cohort directory:

```bash
python -m src.mgmGrowth.tasks.superresolution.cli.downsample \
  --src-root  /path/to/subset \
  --out-root  /path/to/low_res/5mm \
  --target-dz 5 \
  --jobs 8
```

2) Run SMORE across pulses and resolutions via YAML

Edit `cfg/smore_cfg.yaml` (paths, pulses, low_res_slices) and run:

```bash
python -m src.mgmGrowth.tasks.superresolution.cli.smore \
  --config src/mgmGrowth/tasks/superresolution/cfg/smore_cfg.yaml
```

3) Run ECLARE (same idea, different out-root)

```bash
python -m src.mgmGrowth.tasks.superresolution.cli.eclare \
  --config src/mgmGrowth/tasks/superresolution/cfg/eclare_cfg.yaml
```

4) Run the SMORE BraTS experiment pipeline (per-volume wrapper)

```bash
python -m src.mgmGrowth.tasks.superresolution.pipelines.superresolution_brats_experiment \
  --orig-root /path/to/BraTS/source_HR \
  --lr-root   /path/to/low_res/5mm \
  --out-root  /path/to/results/SMORE_5mm \
  --pulses    t1c t2w \
  --slice-dz  5 \
  --gpu-id    0
```

5) Interpolation baselines over the whole cohort

```bash
python -m src.mgmGrowth.tasks.superresolution.pipelines.interpolation_brats_experiment \
  --interp bspline
```

6) UNIRES batch driver (external binary)

```bash
python -m src.mgmGrowth.tasks.superresolution.cli.unires \
  --input-dir  /path/to/low_res/5mm \
  --output-dir /path/to/results/UNIRES/5mm \
  --device cuda \
  --threads 8
```


## Output structure

SMORE/ECLARE (config runners)
```
<out_root>/
  SMORE/ or ECLARE/
    3mm/
      weights/         # <PID>-<pulse>.pt
      output_volumes/  # <PID>-<pulse>.nii.gz
    5mm/
      ...
    7mm/
      ...
```

SMORE pipeline (`superresolution_brats_experiment.py`)
```
<out_root>/
  weights/          # flat: <PID>-<pulse>.pt
  output_volumes/   # flat: <PID>-<pulse>.nii.gz
  metrics_<tag>.npz # patient_ids only (metrics computed separately)
```

Interpolation pipeline (`interpolation_brats_experiment.py`)
```
.../results/models/<ALGO>/{3mm,5mm,7mm}/output_volumes/<PID>-<pulse>.nii.gz
```

UNIRES driver (`cli/unires.py`)
```
--output-dir/<PID>/  # per-subject outputs written by the external tool
```


## Evaluation and metrics

We compute robust per-slice, per-ROI metrics against the original HR volumes:
- PSNR (dB)
- SSIM (Gaussian, masked by ROI)
- Bhattacharyya distance (histogram overlap)
- LPIPS (AlexNet) – optional; requires `torch` and `lpips`

Run:

```bash
python -m src.mgmGrowth.tasks.superresolution.statistics.metrics \
  --hr_root      /path/to/BraTS/super_resolution/subset \
  --results_root /path/to/BraTS/super_resolution/results/models \
  --pulse        all \
  --slice-window 10 140 \
  --workers      8 \
  --out          /path/to/BraTS/super_resolution/results/metrics/metrics.npz
```

The NPZ includes `metrics` with shape `(P, 3, 3, M, 4, 4, 2)` and metadata arrays (`patient_ids`, `pulses`, `resolutions_mm`, `models`, `metric_names`, `roi_labels`, `stat_names`). See `statistics/metrics.py` for details.


## Configuration (YAML)

Both SMORE and ECLARE runners accept the same structure:

```yaml
mode: train # or inference
data:
  train_root: /path/to/low_res
  test_root:  /path/to/low_res/test
  out_root:   /path/to/results
processing:
  low_res_slices: ["3mm", "5mm", "7mm"]
  pulses: ["t1c", "t2w", "t2f"]
  gpu_id: 0
network:
  patch_size: 48
  n_blocks: 16
  n_channels: 32
  batch_size: 32
  n_patches: 832000
  n_rots: 2
```

In "inference" mode, the runners will look for weights produced during training under the same `out_root/{MODEL}/{res}/weights/` directory.


## Tips and troubleshooting

- External CLIs: ensure `smore-*`, `run-smore`, `eclare-*`, `run-eclare`, and `unires` are in PATH on the target machine.
- GPU selection: both SMORE/ECLARE runners accept `processing.gpu_id`.
- Geometry mismatches: metrics logic tolerates ≤2 voxels padding per axis when aligning HR/SR/SEG (see `statistics/metrics.py`). Larger mismatches are skipped and reported.
- LPIPS dependency: if `lpips` isn’t installed, LPIPS is returned as NaN (others remain valid).
- Resampling window: adjust `--slice-window` to skip empty/extreme slices.


## Citations

- SMORE: Zeng et al., “Self Super-Resolution for Magnetic Resonance Images,” MICCAI 2018.
- ECLARE: Remedios et al., “Self-supervised super-resolution for anisotropic MR images with and without slice gap,” SASHIMI 2023.
- UNIRES: Brudfors et al., “UniRes: ...” (see the official repository/documentation for citation details).


## License and acknowledgements

This subpackage wraps external research code via their CLIs. Please consult their licenses. The surrounding pipeline code here is distributed under this repository’s license.
