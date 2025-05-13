# Superresolution

The main idea is to train a SMORE model with the BraTS-Men 2023 dataset for the T1 and T2 pulses. Then, we will fine-tune the final SMORE model on our own T1 and T2 pulses from the MenGrowth dataset, to then, predict on the own trained volumes. For the SWI/SUSC pulses we must proceed in a different way, since there is no available dataset for this pulse in the BraTS-Men 2023 dataset, therefore, we can choose to fine-tune the T2 SMOTE model on the SWI or train from scratch and predict on the little SWI images.

## Experiment design

1. Perform a MenGrowth raw data analysis to identify at least 3 slice thickness physical voxel resolution (in mm) that we are interested in testing. Maybe 3mm, 5mm and 7mm
2. Downsample the BraTS 2023 Men Dataset (a subset of 100 patients for now) to the desired physical resolution.
3. Train a SMORE model with the BraTS-Men 2023 dataset. One model per pulse: T1c and T2w. The validation is as follows:
   1. We train the model 

## Commands

### 1. Downsample

python -m src.mgmGrowth.tasks.superresolution.cli.downsample \
  --src-root  ~/Python/Datasets/Meningiomas/BraTS/BraTS_Men_Train \
  --out-root  ~/Python/Datasets/Meningiomas/BraTS/SR/downsampled_brats_5mm \
  --target-dz 5

### 2. Perform val

python -m src.mgmGrowth.tasks.superresolution.pipelines.superresolution_brats_experiment \
  --lr-root   ~/Python/Datasets/Meningiomas/BraTS/SR/downsampled_brats_5mm \
  --orig-root ~/Python/Datasets/Meningiomas/BraTS/BraTS_Men_Train \
  --out-root  ~/Python/Datasets/Meningiomas/BraTS/SR/SMORE_Results \
  --pulses    t1c \
  --slice-dz  5 \
  --gpu-id    0

run-smore --in-fpath /home/mariopasc/Python/Datasets/Meningiomas/raw/Meningioma_Adquisition/RM/SUSC/P1/SUSC_P1.nii.gz --out-dir /home/mariopasc/Python/Datasets/Meningiomas/raw/Meningioma_Adquisition/RM/SUSC/P1

python -m src.mgmGrowth.tasks.superresolution.statistics.analysis \
  --hr  ~/Python/Datasets/Meningiomas/BraTS/BraTS_Men_Train/BraTS-MEN-00012-000/BraTS-MEN-00012-000-t1c.nii.gz \
  --sr  ~/Python/Datasets/Meningiomas/BraTS/SR/SMORE_Results/output_volumes/BraTS-MEN-00012-000-t1c.nii.gz \
  --seg ~/Python/Datasets/Meningiomas/BraTS/BraTS_Men_Train/BraTS-MEN-00012-000/BraTS-MEN-00012-000-seg.nii.gz \
  --out-dir ~/Python/Datasets/Meningiomas/BraTS/SR/SMORE_Results
