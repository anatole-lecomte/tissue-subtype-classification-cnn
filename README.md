# Molecular Subtype Classification from Whole-Slide Tissue Images

CNN-based classification of diseased human tissue into four molecular subtypes from low-magnification whole-slide images, comparing three training strategies and the impact of mask-guided input processing.

*University project — Artificial Neural Networks and Deep Learning (AN2DL), Politecnico di Milano (2025–2026). Team "The Underfitters", 3 students.*

## Overview

The task is to predict the molecular subtype of diseased tissue — **Luminal A**, **Luminal B**, **HER2(+)**, or **Triple Negative** — from low-magnification whole-slide images, each paired with an optional binary mask highlighting regions likely to contain diseased tissue. Unlike the team's first AN2DL challenge (models trained from scratch only), this challenge allowed pretrained architectures, and was evaluated with F1-score to reward balanced performance across the four subtypes.

**Dataset**: 691 image/mask pairs, variable image sizes, with a noisy real-world twist: some samples were fully replaced by unrelated content, and others carried localized greenish stain-like artifacts requiring targeted cleaning.

## Key results

| Strategy | MobileNetV3Small | EfficientNetB0 | ResNet18 | EfficientNetB0 + Masks |
|---|---|---|---|---|
| From scratch | 0.181 ± 0.000 | 0.234 ± 0.043 | 0.385 ± 0.022 | 0.295 ± 0.023 |
| Transfer learning | 0.326 ± 0.019 | 0.347 ± 0.010 | 0.362 ± 0.012 | 0.333 ± 0.043 |
| Fine-tuning | 0.365 ± 0.036 | **0.405 ± 0.025** | 0.370 ± 0.009 | 0.389 ± 0.027 |

*(F1-score, mean ± std across k-fold cross-validation splits)*

- **Best model on the held-out test set: EfficientNetB0 with transfer learning**, reaching an **F1-score of 32.99%** (learning rate 1e-3, batch size 64, dropout 0.4, early-stopping patience 30).
- **Fine-tuning consistently outperformed both from-scratch and frozen-backbone transfer learning** across all three architectures — expected given the limited dataset size (691 samples).
- **Masking the input to keep only the diseased region actually hurt performance slightly** compared to using the full image (0.405 vs. 0.389 F1 for EfficientNetB0 fine-tuned), suggesting the surrounding tissue context carries useful signal that masking discards.
- Some architecture/strategy combinations failed to converge meaningfully, with flat or unstable training curves.

## Method highlights

- **Artifact handling**: samples fully replaced by unrelated content were detected via histogram analysis and removed; images with localized greenish stain artifacts were kept, but the artifact regions were removed from the corresponding masks.
- **Preprocessing**: all images resized to 224×224, ImageNet normalization when feeding pretrained backbones, stratified 70/30 train/validation split.
- **Model architectures**: MobileNetV3Small, EfficientNetB0, and ResNet18, each evaluated under three training strategies (from scratch, transfer learning with frozen backbone, full fine-tuning).
- **Data augmentation**: random flipping, affine transformations, and random erasing applied to the training set.
- **Mask-guided input**: an additional experiment applying the binary mask directly to the image (keeping only the masked region) to test whether focusing the model on the diseased area would help.
- **Hyperparameter search**: manual exploration followed by a systematic grid search over learning rate, batch size, dropout, and early-stopping patience.
- **K-fold cross-validation**: used throughout to reduce the bias of any single train/validation split.

## Honest limitations (from the report's discussion)

- The full combinatorial space (3 architectures × 3 strategies × mask usage × 4 hyperparameters) was too large to search exhaustively given compute constraints — a better configuration likely exists.
- Masking discarded all background context; feeding the mask as an additional input channel alongside the full RGB image (rather than multiplying it in) is flagged as a promising alternative for future work.
- Greenish stain artifacts were cleaned out of the masks but not out of the images themselves, leaving a source of noise in the visual input.
- A broader augmentation study (beyond flips, affine transforms, and random erasing) was not explored due to time constraints.

## Tech stack

`PyTorch` / `torchvision` (MobileNetV3Small, EfficientNetB0, ResNet18) · `scikit-learn` (stratified splitting, k-fold) · `TensorBoard`

## Repository structure

```
.
├── README.md
├── AN2DL_Challenge2_The_Underfitters.ipynb   # Full notebook: data cleaning, all architectures/strategies, mask experiments
└── AN2DL_2e_Challenge.pdf                     # Full write-up: problem analysis, method, results table, discussion
```

The notebook was built for Google Colab (Kaggle API download, GPU runtime) and expects a Kaggle API token to fetch the competition dataset, which is not included in this repository.

## Authors

Ilyas Benkirane, Anatole Lecomte, Nathan Luneau
Politecnico di Milano — Artificial Neural Networks and Deep Learning
