README.md for the MoE folder below.

# MoE: Cross-validated manifest + Linear Super Learner

## Overview

This folder hosts two components:

1. A **patient-level K-fold manifest** that fixes the train/test splits and records absolute paths to all inputs and outputs.
2. A **global linear Super Learner** that learns one convex weight vector per *(spacing, pulse)* to blend BSPLINE, ECLARE, SMORE, and UNIRES reconstructions. The method is stacked generalization a.k.a. Super Learner. ([ScienceDirect][1])

---

## 1) Cross-validated data manifest (`folds_builder.py`)

**Purpose.** Build a JSON manifest that:

* Splits subjects into K folds at the **patient** level (no leakage).
* For each fold → `{train, test}` → each patient → records absolute file paths for:

  * Per-model SR outputs at `3mm, 5mm, 7mm` for four pulses `{t1c,t1n,t2w,t2f}`.
  * LR inputs at each spacing.
  * HR targets (duplicated under each spacing for convenience).

**Why JSON first.** Stacking needs out-of-fold evaluation; a frozen manifest guarantees reproducible folds and file paths for every downstream MoE. This mirrors the Super Learner protocol that learns on *K−1* folds and evaluates on the held-out fold. ([ResearchGate][2])

**Patient key.** Patient IDs are the BraTS-MEN stems without the pulse, e.g., `BraTS-MEN-00353-000`.

**Schema.** High-level structure:

```
{
  "fold_1": {
    "train": {
      "BraTS-MEN-XXXX-000": {
        "ECLARE": { "3mm": {"t1c": "...", ...}, "5mm": {...}, "7mm": {...} },
        "SMORE":  { ... },
        "UNIRES": { ... },
        "BSPLINE":{ ... },
        "LR":     { "3mm": {"t1c": "...", ...}, "5mm": {...}, "7mm": {...} },
        "HR":     { "3mm": {"t1c": "...", ...}, "5mm": {...}, "7mm": {...} }
      },
      ...
    },
    "test": { ... }
  },
  "fold_2": { ... },
  ...
}
```

**Coverage check.** The builder scans the filesystem and **keeps only** patients that have all required files across models, spacings, and pulses, plus LR and HR. Missing any file → patient excluded from all folds to avoid partial tensors.

**Reproducibility.** Deterministic shuffling with a seed; stratification is unnecessary because learning occurs per *(spacing, pulse)* with patient-level independence.

**Outputs.**

* `moe_cv_manifest.json` at the user-chosen path.

**Notes on formats.** All I/O uses NIfTI via NiBabel; the training code relies on `nib.load(...)` and `nib.save(...)`. ([nipy.org][3])

---

## 2) Linear Super Learner (global convex stacking) (`train_linear_super_learner.py`)

**Goal.** For each *(spacing, pulse)* learn constant, non-negative weights that sum to one across the four experts and apply them to all subjects in each fold’s test split. This is classic stacked generalization with a linear meta-learner a.k.a. Super Learner. ([ScienceDirect][1])

**Model.** Let $\hat{x}_m(r)$ be the voxel value at location (r) from expert (m\in{\text{BSPLINE},\text{ECLARE},\text{SMORE},\text{UNIRES}}). The ensemble is
$$
\hat{x}*E(r)=\sum*{m=1}^{M} w_m,\hat{x}_m(r),\quad w_m\ge 0,\quad \sum_m w_m=1.
$$
Weights are **global** for a fixed *(spacing, pulse)*. The simplex constraint yields a stable, interpretable linear pool. ([LSE Personal Pages][4])

**Objective.** Over training subjects and all voxels:
$$
\min_{w\in\Delta}; |Xw - y|_2^2
\quad\text{with}\quad
\Delta={w\in\mathbb{R}^M: w\ge 0,\ \mathbf{1}^\top w=1}.
$$
Here, columns of (X) are vectorized expert outputs; (y) is the HR volume vector.

**Optimization.** Solved with SLSQP (`scipy.optimize.minimize`) using an equality constraint $mathbf{1}^\top w=1$ and bound constraints $w\ge 0$. SLSQP handles general smooth objectives with linear constraints. ([SciPy Documentation][5])

**Scalable implementation.** The trainer streams subjects and accumulates sufficient statistics
$Q=X^\top X$, $b=X^\top y$, $c=y^\top y$ to avoid storing large voxel matrices. The convex QP in $M=4$ variables is then solved once per *(spacing, pulse, fold)*.

**Cross-validation protocol.**

* For each fold:

  * Fit weights on **train** entries only.
  * Report MSE on train and **held-out test**.
  * Save blended NIfTI for all test subjects (and optionally train).

**Console + file outputs.**

* **Real-time logs:** per fold show learned weights and mean train/test MSE.
* **Predictions:**
  `LINEAR_SUPER_LEARNER/{spacing}/output_volumes/BraTS-MEN-....-<pulse>.nii.gz`
* **Per-pulse model data:**
  `LINEAR_SUPER_LEARNER/{spacing}/model_data/{pulse}/fold_*/weights.json`
  `LINEAR_SUPER_LEARNER/{spacing}/model_data/{pulse}/metrics.csv`
  The CSV has columns: `spacing,pulse,fold,n_train,n_test,train_mse,test_mse,w_BSPLINE,w_ECLARE,w_SMORE,w_UNIRES`.

**CLI.**

* Build manifest:

  ```
  python folds_builder.py \
    --models-root /media/.../results/models \
    --hr-root /media/.../high_resolution \
    --lr-root ~/research/datasets/.../low_res \
    --kfolds 5 --seed 17 \
    --out /path/to/moe_cv_manifest.json
  ```
* Train linear combiner:

  ```
  python train_linear_super_learner.py \
    --cv-json /path/to/moe_cv_manifest.json \
    --out-root /media/.../tasks/superresolution/results/models \
    --spacings 3mm,5mm,7mm \
    --pulses t1c,t1n,t2w,t2f \
    --models BSPLINE,ECLARE,SMORE,UNIRES \
    --save-train-outputs
  ```

**Interpretation.**

* Weights quantify each expert’s global contribution at a fixed *(spacing, pulse)*.
* Because the objective is quadratic and the constraint is a simplex, the solution is the **best linear opinion pool under MSE**. Expect modest but consistent gains when experts have complementary biases. ([LSE Personal Pages][4])

---

## Folder layout produced by training

```
LINEAR_SUPER_LEARNER/
  3mm/
    output_volumes/   # blended NIfTI for test (and optionally train)
    model_data/
      t1c/ fold_*/weights.json, metrics.csv
      t1n/ ...
      t2w/ ...
      t2f/ ...
  5mm/
    ...
  7mm/
    ...
```

---

## References

* Wolpert, D. H. “Stacked Generalization.” *Neural Networks*, 5(2):241–259, 1992. ([ScienceDirect][1])
* van der Laan, M. J., Polley, E. C., Hubbard, A. E. “Super Learner.” *Statistical Applications in Genetics and Molecular Biology*, 6(1), 2007. ([ResearchGate][2])
* Dietrich, F. “Probabilistic Opinion Pooling.” Review incl. linear pools and Genest & Zidek (1986). ([LSE Personal Pages][4])
* Genest, C., Zidek, J. “Combining Probability Distributions: A Critique and an Annotated Bibliography.” *Statistical Science*, 1986. ([JSTOR][6])
* SciPy `minimize` SLSQP documentation. ([SciPy Documentation][5])
* NiBabel docs for NIfTI I/O. ([nipy.org][3])

[1]: https://www.sciencedirect.com/science/article/pii/S0893608005800231/pdf?md5=96868168e69892d774f00354ed8f287f&pid=1-s2.0-S0893608005800231-main.pdf&utm_source=chatgpt.com "Stacked Generalization"
[2]: https://www.researchgate.net/profile/Mark-Laan/publication/5933560_Super_Learner/links/02e7e51f437399127d000000/Super-Learner.pdf "Super Learner"
[3]: https://nipy.org/nibabel/nibabel_images.html "Nibabel images"
[4]: https://personal.lse.ac.uk/list/PDF-files/OpinionPoolingReview.pdf "Probabilistic Opinion Pooling"
[5]: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html "minimize(method='SLSQP') — SciPy v1.16.2 Manual"
[6]: https://www.jstor.org/stable/2245510 "Combining Probability Distributions: A Critique and an ..."
