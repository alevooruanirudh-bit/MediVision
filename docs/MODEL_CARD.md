# Model card — MediVision AI

**Research & educational prototype — not a medical device. Not for clinical use.**

## Overview

| Model | Architecture (ours) | Init | Data | Output |
|---|---|---|---|---|
| Chest ensemble | DenseNet-121, EfficientNet-B2, ConvNeXt-Tiny backbones + our 2-layer MC-dropout heads | generic ImageNet | ChestMNIST+ (starter) / NIH ChestX-ray14 (full) | 14 calibrated probabilities, uncertainty, thresholds |
| Lung U-Net | U-Net (base 16, depth 4); full: ResNet-34 encoder | random / ImageNet encoder | COVID-19 Radiography lung masks (+ Montgomery/Shenzhen, COVID-QU-Ex full) | lung masks → CTR, areas, zones |
| Fracture classifier | EfficientNet-B0 + head | ImageNet | FracAtlas | fracture probability |
| Fracture detector | Faster R-CNN MobileNetV3-Large-FPN | generic COCO | FracAtlas boxes | boxes + scores |
| Brain classifier | EfficientNet-B0 + head | ImageNet | Brain Tumor MRI (4 classes) | glioma / meningioma / pituitary / no tumour |
| Brain segmenter | Attention U-Net (base 16/32) | random | LGG FLAIR slices | tumour mask, area |
| Modality router | SmallCNN (4 blocks, ~0.4 M params) | random | ChestMNIST, lungs, FracAtlas, brain MRI, LGG, Organ/Breast/PneumoniaMNIST, CIFAR-10 | 5 classes |
| Denoiser | DnCNN (10 layers / 17 full) residual | random | ChestMNIST + synthetic Poisson + Gaussian noise | denoised image |
| Report generator (experimental) | DenseNet-121 encoder (from our chest model) + 4-layer Transformer decoder (d=384, ~10 M params decoder), BPE trained by us | ours | IU X-ray reports | free-text report |
| Copilot intent | TF-IDF (word + char) + logistic regression + keyword-group features | — | 22 intents × ≥40 hand-written paraphrases | intent |
| Pneumonia localiser (full only) | Faster R-CNN MobileNetV3-FPN | generic COCO | RSNA | boxes |

No third-party medical weights are used. `PRETRAINED_BACKBONE=false` retrains everything from scratch.

## Intended use

Research, teaching and portfolio demonstration of AI-assisted imaging workflows (explainability, triage,
reporting, human review, federated learning). **Out of scope:** diagnosis, treatment, screening, any
clinical or commercial use, paediatric imaging, modalities other than frontal chest X-ray, musculoskeletal
X-ray and single axial brain MRI slices.

## Metrics

Measured on held-out test splits — see [METRICS.md](METRICS.md) (generated from `artifacts/`) and the Lab
page. A finding is flagged when its calibrated probability ≥ max(Youden threshold, 0.20) (D-18). Calibration: per-label, per-member temperature scaling on validation; reliability diagrams and ECE
before/after. Operating thresholds: Youden's J on validation.

## Fairness

Macro AUROC by sex, age band (<40, 40–59, 60+) and view (PA/AP). Starter: NIH sample restricted to
official-test-list patients (never in training). Full: NIH official test split. Small subgroup sizes in the
starter set make the gaps noisy; no claim of fairness is made.

## Triage weights

Defined in `backend/app/inference/triage_weights.yaml` and shown on the Model Card page:
Pneumothorax 1.00 · Glioma 0.95 · Mass 0.85 · Pneumonia 0.85 · Consolidation 0.80 · Edema 0.80 ·
Fracture 0.80 · Meningioma 0.80 · Effusion 0.75 · Pituitary tumour 0.75 · Nodule 0.60 · Cardiomegaly 0.55 ·
Atelectasis 0.50 · Infiltration 0.50 · Emphysema 0.40 · Fibrosis 0.40 · Pleural thickening 0.30 · Hernia 0.30.

`cᵢ = wᵢ · evidenceᵢ · confidenceᵢ`, sorted descending, `score = 100 · min(1, c₁ + 0.25·c₂ + 0.10·c₃)`; evidence = 0.5 + 0.5 · clip((p − t)/(1 − t)),
confidence = 1 − min(2 · uncertainty, 0.5). CRITICAL ≥ 75, URGENT ≥ 45, ROUTINE = any positive or doubtful
input, NORMAL otherwise. The weights are the author's judgement of time-criticality, not clinically
validated.

## Limitations & risks

- Label noise (NIH labels are NLP-mined); dataset shift across scanners, institutions, populations.
- Starter models are low-resolution and briefly trained.
- Random splits for datasets without patient IDs (brain MRI, lung masks) may inflate scores.
- The starter chest ensemble's JointEnergy OOD score does not separate photos / MRI / bone images from chest
  X-rays (AUROC ≈ 0.45); out-of-distribution protection relies on the modality router (99.7% test accuracy)
  plus a grayscale check.
- Grad-CAM can highlight shortcuts (text markers, devices); CTR is an estimate.
- The experimental generator can hallucinate findings — a consistency check against the classifier is shown.
- Over-reliance / automation bias: outputs are drafts that require qualified review.

## Data provenance & privacy

All training data are public research datasets ([DATASETS.md](DATASETS.md)). Uploaded DICOMs are
de-identified (PHI tags + private tags removed, `PatientIdentityRemoved=YES`) and pseudonymised with a salted
hash; inference is local; every action is audited.
