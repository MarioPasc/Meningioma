# CLI

## Commands

### 1. Downsample

python -m src.mgmGrowth.tasks.superresolution.cli.downsample \
  --src-root  ~/Python/Datasets/Meningiomas/BraTS/BraTS_Men_Train \
  --out-root  ~/Python/Datasets/Meningiomas/BraTS/SR/downsampled_brats_5mm \
  --target-dz 5

### 2. Split

python -m src.mgmGrowth.tasks.superresolution.cli.split \
  --ds-root  ~/Python/Datasets/Meningiomas/BraTS/SR/downsampled_brats_5mm \
  --out-root ~/Python/Datasets/Meningiomas/BraTS/SR/downsampled_brats_5mm_split

### 3. Train SMORE

python -m src.mgmGrowth.tasks.superresolution.cli.train_smore \
  --train-root ~/Python/Datasets/Meningiomas/BraTS/SR/downsampled_brats_5mm_split/train \
  --smore-root ~/Python/Projects/Meningioma/src/mgmGrowth/external/SMORE \
  --slice-dz   5

### 4. Infer

python -m src.mgmGrowth.tasks.superresolution.cli.infer_smore \
  --test-root    ~/Datasets/downsampled_brats_5mm_split/test \
  --weights-root ~/Datasets/downsampled_brats_5mm_split/train/_smore_weights \
  --smore-root   ~/Repos/smore-main \
  --out-root     ~/Datasets/SR_results_5mm
