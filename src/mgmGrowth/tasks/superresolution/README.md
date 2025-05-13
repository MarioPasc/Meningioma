# Superresolution

The main idea is to train a SMORE model with the BraTS-Men 2023 dataset for the T1 and T2 pulses. Then, we will fine-tune the final SMORE model on our own T1 and T2 pulses from the MenGrowth dataset, to then, predict on the own trained volumes. For the SWI/SUSC pulses we must proceed in a different way, since there is no available dataset for this pulse in the BraTS-Men 2023 dataset, therefore, we can choose to fine-tune the T2 SMOTE model on the SWI or train from scratch and predict on the little SWI images.

## Commands

### 1. Downsample

python -m src.mgmGrowth.tasks.superresolution.cli.downsample \
  --src-root  ~/Python/Datasets/Meningiomas/BraTS/BraTS_Men_Train \
  --out-root  ~/Python/Datasets/Meningiomas/BraTS/SR/downsampled_brats_5mm \
  --target-dz 5

### 2. Perform val

python -m src.mgmGrowth.tasks.superresolution.pipelines.superresolution_brats_experiment \
  --data-root   ~/Python/Datasets/Meningiomas/BraTS/SR/downsampled_brats_5mm \
  --orig-root   ~/Python/Datasets/Meningiomas/BraTS/BraTS_Men_Train \
  --pulses      t1c \
  --slice-dz    5 \
  --val-frequency 2 \
  --holdout-ratio 0.2 \
  --gpu 0
