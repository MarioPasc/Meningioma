Below is a drop-in README section for the new folder. It documents the exact model we coded, with equations, design rationale, training scheme, and CLI usage.

---

# Spatially-Adaptive Gated Nonlinear Ensemble Fusion (SAGNEF)

SAGNEF learns **per-voxel convex weights** to fuse four SR experts (BSPLINE, ECLARE, SMORE, UNIRES) and optionally the LR input, **per (spacing, pulse)**. A light 3D U-Net acts as the **gating** network that outputs weight maps; the fused HR prediction is their weighted sum. Mixture-of-Experts with input-dependent softmax gating is a standard, well-studied formulation; our gating is a spatial, dense version for volumes. ([www2.bcs.rochester.edu][1])

---

## Data interface

* **Inputs (per subject, fixed spacing & pulse):**
  Channels = `[LR?] + SR_BSPLINE + SR_ECLARE + SR_SMORE + SR_UNIRES` → shape `(C, D, H, W)`.
  Volumes are already aligned (one-time sanity align can resample tiny mismatches up to 2–3 voxels).

* **Target:** HR volume `(1, D, H, W)`.

* **Normalization:** z-score or percentile clipping per volume before batching.

* **Cross-validation:** use `kfolds_manifest.json` to iterate folds split by patient. See `cfgs/model_configuration.yaml`.

Folder outputs follow:

```
{OUT_ROOT}/{spacing}/
  output_volumes/                      # blended NIfTI for this spacing, all folds' test subjects
  model_data/{pulse}/{fold}/
    model_epochXXX_val{loss}.pt        # checkpoints
    training_log.csv                   # per-epoch losses and components
    test_report.json                   # per-subject PSNR/RMSE/SSIM and file paths
```

---

## Model

### Per-voxel mixture

Let (M=4) experts, indexed by (m). For voxel location (r),
[
\hat x_E(r) ;=; \sum_{m=1}^{M} w_m(r), \hat x_m(r),
\qquad w_m(r)\ge 0,;; \sum_{m=1}^{M} w_m(r)=1,
]
where ({\hat x_m}) are the expert SR volumes and ({w_m}) are gating weights produced by a CNN and normalized with a voxelwise softmax. This is classic input-dependent MoE gating, adapted densely in space. ([www2.bcs.rochester.edu][1])

### Gating U-Net (3D)

We use a **small 3D U-Net** for context aggregation and precise localization:

* **Encoder:** two levels
  `Conv3d→GN→LeakyReLU→Conv3d→GN→LeakyReLU` (widths `wf, 2wf`), strided conv downsampling (×2) between levels.
* **Bottleneck:** width `4wf`.
* **Decoder:** transpose conv upsample (×2) and skip concatenation with encoder features at each scale; same conv block widths `2wf → wf`.
* **Head:** `1×1×1` conv to `M` channels, followed by `softmax(dim=1)` to ensure a convex simplex at every voxel.
* **Defaults:** `wf=24`, GroupNorm(8), LeakyReLU. Fully convolutional, works on arbitrary patch sizes.

U-Net captures multi-scale context and precise localization with few parameters; its 3D variant is standard for volumetric medical data. ([arXiv][2])

**Forward pass** (PyTorch sketch):

```python
# x: (B, C, D, H, W) where C = (#experts) + (1 if LR included)
W = gate(x)                 # (B, M, D, H, W), voxel-wise softmax
experts = x[:, -M:, ...]    # last M channels are the SR experts
y_hat = (W * experts).sum(1, keepdim=True)  # (B, 1, D, H, W)
```

---

## Training objective

Total loss per mini-batch:
[
\mathcal{L} ;=;
\underbrace{\alpha,|,\hat x_E - x_{\text{HR}},|*2^2}*{\text{image MSE}}
;+;
\underbrace{\beta,(1-\operatorname{SSIM}(\hat x_E, x_{\text{HR}}))}*{\text{structural term}}
;+;
\underbrace{\gamma,\operatorname{TV}(W)}*{\text{spatial smoothness on weights}}
;+;
\underbrace{\delta,\mathbb{E}[H(W)]}*{\text{entropy of weights}},
]
where (H(W) = -\sum*{m} w_m \log w_m) per voxel and averaged.
SSIM complements MSE by enforcing structural fidelity; TV discourages noisy, per-voxel switching in the weight maps; entropy regularization discourages degenerate single-expert gating and improves usage diversity. ([cns.nyu.edu][3])

**Terms:**

* **MSE:** ( | \hat x_E - x_{\text{HR}} |_2^2 ).
* **SSIM (3D):** Gaussian-window SSIM computed volumetrically; loss is (1-\text{SSIM}). ([cns.nyu.edu][4])
* **TV on (W):** isotropic TV on the (M) channels:
  (\operatorname{TV}(W)=\sum_{\text{axes}} | \nabla W |_2) approximated by squared-differences and square-rooted mean. Total variation regularization is the standard edge-preserving smoothness prior. ([ScienceDirect][5])
* **Entropy:** ( H(W) = -\sum_m w_m \log w_m ) per voxel, averaged over space; encourages non-peaky, better-calibrated gating.

**Default weights (YAML):** `w_mse=1.0, w_ssim=0.2, w_tv=0.01, w_entropy=0.001`. Tunable per spacing/pulse.

---

## Optimization and regularization

* **Optimizer:** Adam with default betas, LR (1\times10^{-3}). Adam is efficient for noisy, non-stationary gradients and needs little tuning. ([arXiv][6])
* **Gradient clipping:** (\ell_2) clip at 1.0 to stabilize 3D training.
* **AMP:** optional mixed precision for speed on RTX 4090/A100.
* **Early stopping:** monitor validation loss; stop after patience epochs without improvement. Early stopping is a standard regularizer to avoid overfitting. ([page.mi.fu-berlin.de][7])
* **Data augmentation:** random 3D flips/rotations recommended (add in `Dataset` if desired).

> Scheduler: not required by default; you can add cosine decay or step LR in the config if validation flattens.

---

## Patch-based learning and inference

* **Training patches:** random (64^3) 3D crops; per volume we draw (N) patches each epoch to increase sample count and reduce memory.
* **Validation patches:** similarly cropped without augmentation.
* **Inference (full volume):** sliding-window with overlap, **Hann** blend to avoid seams; overlap-add reconstruction re-normalizes by the accumulated window weights.

This keeps memory bounded and enables batch size 1–2 on a single GPU while covering full volumes.

---

## Hyperparameters

All set in `cfgs/model_configuration.yaml`.

* **Data:** spacings, pulses, experts; include LR or not; normalization mode; sanity alignment with max voxel tolerance.
* **Patch:** `patch_size=[64,64,64]`, `stride=[32,32,32]`, Hann blending on/off.
* **Loss:** `w_mse, w_ssim, w_tv, w_entropy`, SSIM window size ((7)), Gaussian (\sigma).
* **Train:** epochs, batch size, LR, weight decay, grad clip, patience, save cadence, workers.
* **Out paths:** `out_root` determines `{spacing}/output_volumes` and `{spacing}/model_data/{pulse}/{fold}` trees.

---

## Metrics and reporting

* **On-the-fly (patch level):** total loss and each component (MSE, SSIM-loss, TV, entropy) logged per epoch to `training_log.csv`.
* **Held-out (full volume):** PSNR, RMSE, SSIM are computed per subject and written to `test_report.json`. SSIM is reported with the same window as the training loss. SSIM is a perceptual full-reference metric that correlates with perceived structural fidelity. ([imatest.com][8])

---

## CLI

### Train per spacing & pulse across folds

```bash
python -m mri_sr_sagnef.cli.train \
  --cfg mri_sr_sagnef/cfgs/model_configuration.yaml \
  --spacings 3mm,5mm,7mm \
  --pulses t1c,t1n,t2f,t2w
```

### Predict a single subject from expert files

```bash
python -m mri_sr_sagnef.cli.predict \
  --cfg mri_sr_sagnef/cfgs/model_configuration.yaml \
  --ckpt /.../model_epoch050_val0.123456.pt \
  --spacing 3mm --pulse t1c \
  --experts /path/BSPLINE.nii.gz,/path/ECLARE.nii.gz,/path/SMORE.nii.gz,/path/UNIRES.nii.gz \
  --lr /path/LR.nii.gz \
  --ref /path/HR_like_ref.nii.gz \
  --out /out/BraTS-MEN-XXXX-000-t1c.nii.gz
```

---

## Design justifications (concise)

* **Spatial MoE necessity:** global linear pools saturate when experts are highly correlated. Spatial gating lets different experts dominate in different regions. Softmax gating is the canonical MoE mechanism. ([www2.bcs.rochester.edu][1])
* **3D U-Net gate:** encoder–decoder with skips provides sufficient receptive field and localization with few parameters; proven effective on volumetric medical data. ([arXiv][9])
* **SSIM term:** complements MSE by emphasizing structural similarity; widely used for perceptual fidelity. ([cns.nyu.edu][4])
* **TV on weights:** suppresses noisy pixel-wise switching, producing smooth, interpretable weight maps. ([ScienceDirect][5])
* **Adam + early stopping:** practical, stable optimization for small data, with standard validation-based stopping. ([arXiv][6])

---

## Minimal code example (Python API)

```python
import torch
from mri_sr_sagnef.engine.model import SAGNEF
from mri_sr_sagnef.engine.losses import SAGNEFLoss, LossConfig

# Example: no LR channel, 4 experts → in_ch=4, n_experts=4
net = SAGNEF(in_ch=4, n_experts=4, wf=24).cuda().eval()

# fake batch: (B=1, C=4, D=64, H=64, W=64)
x = torch.randn(1, 4, 64, 64, 64, device="cuda")
y_hat, W = net(x)                        # y_hat: (1,1,64,64,64), W: (1,4,64,64,64)

cfg = LossConfig(w_mse=1.0, w_ssim=0.2, w_tv=0.01, w_entropy=0.001,
                 ssim_window=7, ssim_sigma=1.5, ssim_K1=0.01, ssim_K2=0.03)
criterion = SAGNEFLoss(cfg)

# dummy target
y = torch.randn_like(y_hat)
losses = criterion(y_hat, y, W)
print(losses["total"], losses["mse"], losses["ssim"], losses["tv"], losses["entropy"])
```

---

## References

* **Mixture-of-Experts / Gating:** Jacobs et al. 1991; Jordan & Jacobs 1994. ([www2.bcs.rochester.edu][1])
* **U-Net / 3D U-Net:** Ronneberger et al. 2015; Çiçek et al. 2016. ([arXiv][2])
* **SSIM:** Wang et al. 2004. ([cns.nyu.edu][4])
* **Total Variation:** Rudin–Osher–Fatemi 1992; modern overviews. ([web.eecs.utk.edu][10])
* **Adam:** Kingma & Ba 2014/ICLR 2015. ([arXiv][6])
* **Early stopping:** Prechelt 1997/1998. ([page.mi.fu-berlin.de][7])

---

[1]: https://www2.bcs.rochester.edu/sites/jacobslab/cheat_sheet/mixture_experts.pdf "Mixtures-of-Experts Robert Jacobs Department of Brain ..."
[2]: https://arxiv.org/abs/1505.04597 "U-Net: Convolutional Networks for Biomedical Image Segmentation"
[3]: https://www.cns.nyu.edu/pub/lcv/wang03-preprint.pdf "Image Quality Assessment: From Error Visibility to Structural ..."
[4]: https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf "From Error Visibility to Structural Similarity"
[5]: https://www.sciencedirect.com/science/article/pii/016727899290242F "Nonlinear total variation based noise removal algorithms"
[6]: https://arxiv.org/abs/1412.6980 "Adam: A Method for Stochastic Optimization"
[7]: https://page.mi.fu-berlin.de/prechelt/Biblio/stop_tricks1997.pdf "Early Stopping | but when?"
[8]: https://www.imatest.com/docs/ssim/ "SSIM: Structural Similarity Index"
[9]: https://arxiv.org/abs/1606.06650 "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
[10]: https://web.eecs.utk.edu/~hqi/ece692/references/noise-TV-PhysicaD92.pdf "Nonlinear total variation based noise removal algorithms*"
