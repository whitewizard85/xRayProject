# Pre-registration of the extension experiments (v2, November 2026)

**Study:** Robustness of attribution and calibration evaluations for deep learning chest radiograph classifiers.
**Status at registration:** the 100 models of the original study (ConvNeXt-Base factorial, loss block,
DenseNet-121 block) have been trained and analysed. **None of the experiments below has been run.**
This document fixes their design, end points and analysis before any of them is started.
Suggested registration: upload this file unchanged to OSF (or another time-stamped registry) and record the
registration date and DOI here: `______`.

## Rationale

The original analysis left four objections open: (a) only convolutional architectures were studied; (b) all
models were selected at epoch 2–3, so the conclusions might be specific to an early-stopping regime; (c) the
randomization test re-initialized only the last stage and the head; (d) Integrated Gradients (IG) used a single
(black) baseline. The extension addresses each one with a fixed design.

## Design

All new models are trained on the same NIH ChestX-ray14 patient-level split and tested on the same internal test
set and the same 15 000 VinDr-CXR external images, with the same five seeds (42, 123, 2026, 7, 314), the same
preprocessing, the same evaluation code and the same fixed subsets of image–finding pairs as the original study.

| Block | Model | Recipe | Losses | Runs |
|---|---|---|---|---:|
| `swin` | Swin-B (`swin_base_patch4_window12_384.ms_in22k`, ImageNet-22k, 384 px) | paper recipe at the reference cell: no crop, flip augmentation, AdamW lr 3e-5, wd 0.01, cosine 10 epochs, batch 16 | BCE, ASL | 10 |
| `regime` | ConvNeXt-Base (ImageNet-22k) | alternative regime: AdamW lr 1e-5, wd 0.05, 1-epoch linear warm-up + cosine to 15 epochs, strong augmentation (flip, affine ±7°, ±4 % translation, ±7 % scale, brightness/contrast ±0.15), no crop | BCE, ASL | 10 |

Model selection is unchanged (checkpoint with the highest validation macro AUC).
Grad-CAM/Grad-CAM++ for Swin-B use the output of the last hierarchical stage.

**Extended attribution** (all 50 reference-cell models: ConvNeXt-Base BCE/ASL, wBCE, focal; DenseNet-121
BCE/ASL; Swin-B BCE/ASL; regime BCE/ASL; five seeds each):
- *Full cascading randomization* (Adebayo et al., 2018): layers re-initialized cumulatively from the top
  (head, last stage, …, stem), with the same re-initialization for every architecture (truncated normal
  σ = 0.02 for weights, ones/zeros for normalization parameters). After each level, the Spearman correlation
  between original and randomized maps is computed on the 92 external pairs of the original randomization subset.
- *Second IG baseline*: a Gaussian-blurred copy of the image (σ = 16 px), computed on the original IG subset
  (600 pairs; localization), deletion subset (300 pairs; faithfulness) and randomization subset.

**Demographics:** age and sex for every NIH partition (per image and per patient) and for VinDr-CXR
(per image, from DICOM headers; missing values reported, not imputed).

## Primary hypotheses (family of seven tests, Holm correction, α = .05)

| ID | Hypothesis | Test |
|---|---|---|
| H1-swin | External/internal seed-to-seed SD ratio of matched-class AUC > 1 for Swin-B | one-sided; log-ratio with leave-one-seed-out jackknife, *t* with 4 df |
| H1-regime | Same, for the alternative regime | as above |
| H2-swin | With ASL, internal ECE after temperature scaling exceeds internal ECE after Platt scaling (Swin-B) | one-sided paired *t* across 5 seeds |
| H2-regime | Same, for the alternative regime | as above |
| H3-swin | The ASL − BCE effect on external Grad-CAM energy-in-box differs from the ASL − BCE effect on Grad-CAM++ energy-in-box (Swin-B) | two-sided paired *t* across 5 seeds on the difference of the two effects |
| H4 | After full cascading randomization, the original-vs-randomized correlation is higher for IG (black baseline) than for Grad-CAM | one-sided paired *t* across the 50 reference models |
| H5-regime | The selected epoch is later under the alternative regime than under the paper recipe (manipulation check) | one-sided Mann–Whitney, regime vs ConvNeXt reference cell |

All other quantities (per-group AUC, calibration intercepts and slopes, localization means, seed agreement,
cascade curves, IG black vs blurred) are **descriptive**, reported with 95 % CIs and not interpreted as tests.

## What will be reported regardless of outcome

Every hypothesis above, with estimate, CI, raw and Holm-adjusted *P*, including those that are not supported.
If H5 fails (the alternative regime also peaks at epoch 2–3), H1-regime and H2-regime are still reported, with
the explicit statement that the regime did not change the selection behaviour.

## Analysis code

`pipeline/08_extension_analysis.py` (written before any extension experiment was run); demographics:
`pipeline/09_demographics.py`; extended attribution: `pipeline/05b_attribution_ext.py`.
