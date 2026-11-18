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
| Head-CT hemorrhage | EfficientNet-B0 + MC-dropout head | ImageNet | Head CT – hemorrhage (Kitamura; full: RSNA ICH) | hemorrhage probability + Grad-CAM |
| CT nodule malignancy | SmallCNN3D (4 stages, 16–128 ch) + 3-D Grad-CAM; classical 3-D segmentation | random | NoduleMNIST3D 64³ (LIDC-IDRI) | malignancy probability, volume, sphericity |
| Breast-US malignancy | EfficientNet-B0 + MC-dropout head | ImageNet | BreastMNIST+ (BUSI) | malignancy probability + suspicion category |
| Fundus DR grader | EfficientNet-B0 (colour input) | ImageNet | RetinaMNIST+ (DeepDRiD) | ICDR grade 0–4, referable-DR probability |
| Super-resolution ×2 (experimental) | residual generator + PatchGAN (SRGAN-style, L1 + 0.003·LSGAN) | random | ChestMNIST+ HR/LR pairs | ×2 image |
| Radiogenomics (analysis) | expert-mask radiomics → logistic regression | — | LGG + TCGA `data.csv` | CV AUROC vs permutation baseline |

Modality router v2 has 8 classes: chest X-ray · bone X-ray · brain MRI · head CT · breast US · fundus · other
medical · not medical. Colour is allowed only for fundus (D-22).

No third-party medical weights are used. `PRETRAINED_BACKBONE=false` retrains everything from scratch.

## Intended use

Research, teaching and portfolio demonstration of AI-assisted imaging workflows (explainability, triage,
reporting, human review, federated learning). **Out of scope:** diagnosis, treatment, screening, any
clinical or commercial use, paediatric imaging, and anything outside the trained inputs: frontal chest X-ray,
musculoskeletal X-ray, single axial brain-MRI slices, single axial head-CT slices (brain window), breast
ultrasound images, colour fundus photographs and 64³-resampled chest-CT nodule patches. Not covered: whole-scan
nodule detection, stroke LVO / pulmonary embolism, mammography, wearables.

**Intended users:** developers, students and researchers. **Not** clinicians making decisions.

## Metrics

Measured on held-out test splits — see [METRICS.md](METRICS.md) (generated from `artifacts/`) and the Lab
page. A finding is flagged when its calibrated probability ≥ max(Youden threshold, 0.20) (D-18). Calibration: per-label, per-member temperature scaling on validation; reliability diagrams and ECE
before/after. Operating thresholds: Youden's J on validation.

Second-reader operating points (sensitivity, specificity, FPR = 1 − spec, FNR = 1 − sens, PPV, NPV, confusion
counts) are reported for every classifier in the Lab ("Second-reader summary") and in METRICS.md.

## Transparency & provenance

- Every finding has a heatmap (Grad-CAM / Grad-CAM++ / occlusion; 3-D Grad-CAM for CT volumes) **and**
  plain-language reasons: its score vs. the threshold, model agreement, location, and whether a measurement
  supports it (D-24).
- Every result records each model's weights hash, profile and training date. Findings record AI source vs.
  human reviewer (name, role, time). Reports record the AI drafter, the last human editor and the signer
  (name, role) — in the UI, PDF, FHIR and DICOM (D-26).
- The audit log is hash-chained (tamper-evident) and stores input hash, model versions, outputs, actor and
  role (D-28).

## Fairness

Chest: macro AUROC by sex, age band (<40, 40–59, 60+) and view (PA/AP). Brain segmentation (LGG): per-patient
Dice by sex, age band and race. Race is shown as the dataset's numeric codes; it is the only dataset here that
records race. Federated: AUROC per hospital scanner style (worst-site AUROC, site gap). Starter: NIH sample restricted to
official-test-list patients (never in training). Full: NIH official test split. Small subgroup sizes in the
starter set make the gaps noisy; no claim of fairness is made.

## Regulatory status, security & consent

- **Not a medical device**: no FDA / CE / UKCA clearance. The disclaimer is on every screen, in the API header
  and in every export.
- Model updates change behaviour, while approval frameworks assume static software. Versions are therefore
  pinned per result (weights hash) and shown on the Model Card page.
- Access control: roles admin / radiologist / resident / viewer, with only radiologists and admins able to
  sign. Bearer tokens are used when `AUTH_ENABLED=true`; in the default local demo the acting role is chosen
  in Settings.
- Uploads require a consent / lawful-basis statement (`REQUIRE_CONSENT=true`). DICOM is de-identified and
  patients are pseudonymised.
- Federated learning shares weights only. HIPAA / GDPR duties still apply, and a real deployment would add
  secure aggregation and differential privacy.

## Autonomous mode (`AUTONOMOUS_TRIAGE=true`)

Two well-defined tasks only:
- clear NORMAL studies (every output ≤ 40 % of its threshold, measured uncertainty ≤ 0.08, in-distribution)
  go to a "confirm normal" queue;
- CRITICAL studies raise an alert that stays visible until acknowledged.

A human always confirms, and nothing is signed automatically.

## Triage weights

Defined in `backend/app/inference/triage_weights.yaml` and shown on the Model Card page:
Pneumothorax 1.00 · Glioma 0.95 · Mass 0.85 · Pneumonia 0.85 · Consolidation 0.80 · Edema 0.80 ·
Fracture 0.80 · Meningioma 0.80 · Effusion 0.75 · Pituitary tumour 0.75 · Nodule 0.60 · Cardiomegaly 0.55 ·
Atelectasis 0.50 · Infiltration 0.50 · Emphysema 0.40 · Fibrosis 0.40 · Pleural thickening 0.30 · Hernia 0.30 ·
Intracranial hemorrhage 1.00 · Malignancy-suspicious lung nodule 0.60 · Suspicious breast lesion 0.55 ·
Referable diabetic retinopathy 0.35.

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
- New-modality starter sets are small: head CT has 40 test images, so AUROC 0.995 has a wide CI. The fundus
  grader's 5-class accuracy is modest (referable-DR AUROC is the meaningful number).
- CT nodule volume assumes 1 mm voxels when the file has no spacing. The 3-D view is MIP/MPR, not a surgical
  reconstruction.
- Radiomics and radiogenomics are exploratory (n ≈ 100) and are not biomarkers.
- SR-GAN output can contain invented texture. It is for viewing only, never for diagnosis.

## Data provenance & privacy

All training data are public research datasets ([DATASETS.md](DATASETS.md)). Uploaded DICOMs are
de-identified (PHI tags + private tags removed, `PatientIdentityRemoved=YES`) and pseudonymised with a salted
hash; inference is local; every action is audited.
