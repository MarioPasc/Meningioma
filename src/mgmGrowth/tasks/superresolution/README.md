# Superresolution

The main idea is to train a SMORE model with the BraTS-Men 2023 dataset for the T1 and T2 pulses. Then, we will fine-tune the final SMORE model on our own T1 and T2 pulses from the MenGrowth dataset, to then, predict on the own trained volumes. For the SWI/SUSC pulses we must proceed in a different way, since there is no available dataset for this pulse in the BraTS-Men 2023 dataset, therefore, we can choose to fine-tune the T2 SMOTE model on the SWI or train from scratch and predict on the little SWI images.

## Experimental design

1. First we extract the t1c and t2w, and segmentation files from the BraTS dataset from 50 patients, where all the tumour labels are present. This is going to be the cohort of the study to measure the performance of SMORE + Interolation algorithms.
2. We are going to downsample this cohort to 3 low-resolution datasets: 1x1x3mm, 1x1x5mm and 1x1x7mm; The algorithm will try to recreate the HR-image of 1x1x1mm.
3. We are going to perform two different evaluation strategies:
    - **Cuantitative Evaluation**. Usage of PSNR, SSIM and MI per region + whole volume. We are going to display a 1x3 violin plot per pulse, where each region will contain 3 violins, one for each downsampled resolution.
    - **Cualitative Evaluation**. Two different visualizations:
        - *BraTS validation*. Show a 3D-Cubic visualization of coronal, sagital and axial slices of the reconstructed volume, per-model (x-axis) and per-downsampling (y-axis; 3mm, 5mm, 7mm). We must select a slice where we can see the 3 labels: Enhancing Tumor Core, Edema, and Surrounding Tumor. We can have 2 lines per downsampled category, where in the first one we show the full slice, and in the second one we zoom into the tumor to highlight differences. The second visualization could be analogous to this one, showing the same slices and same layout, but, instead of showing the images, we showcase the intensity difference maps. We can add a tag to each image showcasing the dB of the PSNR for the zoomed tumor region. (p-value?)
        - *MenGrowth validation*. Since we can't quantify the similarity between the LR volumes and the generated HR volumes, we can show a comparison between: (x-axis) LR-image + Model-generated images; (y-axis) T1, T2, SWI pulses.     


## Commands

### 1. Downsample

Local
```bash
python -m src.mgmGrowth.tasks.superresolution.cli.downsample \
  --src-root  ~/Python/Datasets/Meningiomas/BraTS/SR/subset \
  --out-root  ~/Python/Datasets/Meningiomas/BraTS/SR/low_res/5mm \
  --target-dz 5 \
  --jobs 8
```
Server
```bash
python -m src.mgmGrowth.tasks.superresolution.cli.downsample \
  --src-root  /media/hddb/mario/data/Meningiomas/Brats \
  --out-root  /media/hddb/mario/data/Meningiomas/downsampled_brats_5mm \
  --target-dz 5
```

python -m src.mgmGrowth.tasks.superresolution.cli.downsample \
  --src-root  /media/hddb/mario/data/Meningiomas/Brats \
  --out-root  /media/hddb/mario/data/Meningiomas/downsampled_brats_5mm \
  --target-dz 5

### 2. Perform val

Local
```bash
python -m src.mgmGrowth.tasks.superresolution.pipelines.superresolution_brats_experiment \
  --orig-root /media/hddb/mario/data/Meningiomas/Brats \
  --lr-root   /media/hddb/mario/data/Meningiomas/downsampled_brats_5mm \
  --out-root  /home/mariopascual/Projects/MENINGIOMA/SR/SMORE_Results \
  --pulses    t1c \
  --slice-dz  5 \
  --gpu-id    0

python -m src.mgmGrowth.tasks.superresolution.pipelines.superresolution_brats_experiment \
--orig-root /media/hddb/mario/data/Meningiomas/Brats \
--lr-root   /media/hddb/mario/data/Meningiomas/downsampled_brats_5mm \
--out-root  /home/mariopascual/Projects/MENINGIOMA/SR/SMORE_Results \
--pulses    t2w \
--slice-dz  5 \
--gpu-id    1

run-smore --in-fpath /home/mariopasc/Python/Datasets/Meningiomas/raw/Meningioma_Adquisition/RM/SUSC/P1/SUSC_P1.nii.gz --out-dir /home/mariopasc/Python/Datasets/Meningiomas/raw/Meningioma_Adquisition/RM/SUSC/P1

python -m src.mgmGrowth.tasks.superresolution.statistics.analysis \
  --hr  ~/Python/Datasets/Meningiomas/BraTS/BraTS_Men_Train/BraTS-MEN-00012-000/BraTS-MEN-00012-000-t1c.nii.gz \
  --sr  ~/Python/Datasets/Meningiomas/BraTS/SR/SMORE_Results/output_volumes/BraTS-MEN-00012-000-t1c.nii.gz \
  --seg ~/Python/Datasets/Meningiomas/BraTS/BraTS_Men_Train/BraTS-MEN-00012-000/BraTS-MEN-00012-000-seg.nii.gz \
  --out-dir ~/Python/Datasets/Meningiomas/BraTS/SR/SMORE_Results
```

Server
```bash
python -m src.mgmGrowth.tasks.superresolution.pipelines.superresolution_brats_experiment \
  --lr-root   /media/hddb/mario/data/Meningiomas/downsampled_brats_7mm \
  --orig-root /media/hddb/mario/data/Meningiomas/Brats \
  --out-root  /home/mariopascual/Projects/MENINGIOMA/SR \
  --pulses    t2w \
  --slice-dz  7 \
  --gpu-id    1
```
