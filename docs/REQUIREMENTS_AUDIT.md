# Requirements audit — "AI-Powered Radiology: Innovations and Challenges in Medical Imaging" (JECIR 2023)

**Before** = commit `4780ae7` (first complete build). **After** = round 1 (`66bd164`) plus round 2 (this
commit: mammography, PET-CT fusion, fMRI, whole-scan nodule detection). Status: ✅ implemented · 🟡 partial ·
❌ missing. Metrics are starter-profile, held-out test sets (see METRICS.md).

**Scope change (D-37).** C3 (stroke LVO / PE on CT angiography) and C8 (wearables) are removed. There is no
small, free, suitable dataset for C3 (RSNA PE is ~900 GB), and C8 is outside imaging. The COULD list is now 8
items.

## Coverage

| Priority | Before ✅ / 🟡 / ❌ | After ✅ / 🟡 / ❌ |
|---|---|---|
| MUST (34) | 19 / 12 / 3 | **34 / 0 / 0** |
| SHOULD (7) | 4 / 3 / 0 | **7 / 0 / 0** |
| COULD (8, after removing C3 and C8) | 0 / 2 / 6 | **8 / 0 / 0** |

## MUST

| ID | Requirement | Before | After | Evidence | Remaining gap |
|---|---|---|---|---|---|
| M1 | Deep-learning (CNN) abnormality detection | ✅ | ✅ | `ml/models/`, 13 trained CNNs | — |
| M2 | Tumour detection | ✅ | ✅ | brain_cls (acc 0.953), brain_seg (Dice 0.75), CXR Mass/Nodule | — |
| M3 | Fracture detection | ✅ | ✅ | fracture_cls AUROC 0.901; Faster R-CNN detector | detector weak (mAP 0.048) |
| M4 | Hemorrhage detection (head CT) | ❌ | ✅ | `head_ct_cls` AUROC 0.995, sens/spec 0.95/0.95 (n=40) | small test set; RSNA ICH = full profile |
| M5 | Lung nodule (CT) | 🟡 | ✅ | whole-scan detector on LUNA16 (lung mask → LoG candidates → 3-D CNN): candidate recall 0.94, FROC CPM 0.223, sens 0.12 @ 1 FP / 0.50 @ 8 FP per scan; then `nodule3d` malignancy (AUROC 0.900) + size | detector is weak (trained on 25 scans) |
| M6 | Chest disease on X-ray | ✅ | ✅ | 14-label ensemble, macro AUROC 0.791 | — |
| M7 | Second reader, FP/FN rates | 🟡 | ✅ | sens/spec/FPR/FNR/PPV/NPV per model: Lab "Second-reader summary", METRICS.md | — |
| M8 | Multiple modalities (X-ray, US, MRI, CT) | 🟡 | ✅ | router v3 (9 classes, acc 0.993): X-ray, MRI, CT, US, mammography, fundus; CT and PET-CT volumes | — |
| M9 | Automate repetitive tasks | ✅ | ✅ | upload → full pipeline → report draft | — |
| M10 | Segmentation | ✅ | ✅ | lung U-Net (Dice 0.974), Attention U-Net, 3-D nodule segmentation | — |
| M11 | Measurements / volumetrics | 🟡 | ✅ | CTR, lung areas, tumour area, nodule volume / diameter / sphericity | 1 mm spacing assumed if absent |
| M12 | Triage / worklist by urgency | ✅ | ✅ | `triage.py`, worklist | — |
| M13 | Critical-finding alerting | 🟡 | ✅ | `/api/alerts`, red bar on every page, acknowledge (audited) | — |
| M14 | Findings → coherent report | ✅ | ✅ | `report_writer.py` (all 7 pipelines) | — |
| M15 | Editable, exportable report | ✅ | ✅ | editor, PDF, FHIR, DICOM | — |
| M16 | Quality enhancement (denoise / low-res) | 🟡 | ✅ | DnCNN; SR-GAN ×2 (PSNR 36.9 → 37.9 dB vs bicubic) | — |
| M17 | Low-dose reconstruction | ✅ | ✅ | simulated low dose + DnCNN (22.9 → 32.8 dB) | — |
| M18 | Radiomics features | ❌ | ✅ | `radiomics.py` shape / first-order / GLCM; Measure tab | descriptive only |
| M19 | Progression / risk over time | 🟡 | ✅ | `/api/patients/{id}/timeline`, compare page; nodule / breast malignancy risk | descriptive, not prognostic |
| M20 | Data quality, documentation, imbalance | ✅ | ✅ | DATASETS.md (sizes, licences, imbalance handling) | — |
| M21 | DICOM in, standard export | 🟡 | ✅ | DICOM in; FHIR, DICOM Secondary Capture, DICOM Encapsulated PDF out | no DICOM-SR |
| M22 | No black box | ✅ | ✅ | Grad-CAM / ++ / occlusion / 3-D Grad-CAM + reasons | — |
| M23 | OOD warning | ✅ | ✅ | router + colour guard; energy score | energy score weak (AUROC 0.448) |
| M24 | Auditability (inputs, version, outputs, actions) | 🟡 | ✅ | audit stores input SHA-256, model versions, outputs, actor + role | — |
| M25 | Fits radiologist workflow | ✅ | ✅ | worklist → viewer → report → sign | — |
| M26 | Human oversight, sign-off | ✅ | ✅ | accept / reject / edit; sign with attestation | — |
| M27 | Liability: who produced / approved | 🟡 | ✅ | per-finding provenance; report drafter / editor / signer + role | — |
| M28 | Confidence per finding | ✅ | ✅ | probability, uncertainty, certainty | — |
| M29 | Bias across demographics | 🟡 | ✅ | chest sex / age / view; brain-seg sex / age / race codes | small subgroups |
| M30 | De-identification | ✅ | ✅ | `deid.py` | — |
| M31 | Access control + consent | ❌ | ✅ | roles + token auth (`AUTH_ENABLED`), consent required at upload | auth off by default (demo) |
| M32 | Intended use, disclaimer, model card, versioning | 🟡 | ✅ | Model Card page; weights hash on every result | — |
| M33 | Zero cost | ✅ | ✅ | free data, local training, Kaggle free tier | — |
| M34 | CPU / offline | ✅ | ✅ | ONNX CPU; 17 models loaded | — |

## SHOULD

| ID | Requirement | Before | After | Evidence |
|---|---|---|---|---|
| S1 | Saliency / attention maps | ✅ | ✅ | Grad-CAM, Grad-CAM++, occlusion, 3-D Grad-CAM |
| S2 | Plain-language reasons | 🟡 | ✅ | "Why the AI flagged this" per finding; Copilot "why" uses it |
| S3 | Federated learning + HIPAA / GDPR | 🟡 | ✅ | FedAvg demo + privacy notes (UI, KB, MODEL_CARD) |
| S4 | Federation improves generalisation / bias | 🟡 | ✅ | per-site AUROC: rural clinic 0.778 local → 0.879 FedAvg; FedAvg worst-site AUROC highest among multi-site models |
| S5 | Augments, does not replace | ✅ | ✅ | human sign-off; only radiologists sign |
| S6 | Equity visible in UI | ✅ | ✅ | Lab fairness charts + brain-seg fairness table |
| S7 | Honest evaluation | ✅ | ✅ | baselines, permutation tests, misses documented |

## COULD

| ID | Requirement | Before | After | Evidence / reason |
|---|---|---|---|---|
| C1 | Breast cancer screening | ❌ | ✅ | mammography (Mini-DDSM): exam AUROC 0.892 (sens 0.64 / spec 0.91), cancer vs normal 0.961, cancer vs benign 0.836; plus breast US AUROC 0.907 |
| C2 | Diabetic retinopathy | ❌ | ✅ | `retina_dr` referable-DR AUROC 0.882, κ 0.688 |
| C4 | 3-D view | ❌ | ✅ | rotating MIP + MPR (not AR/VR) |
| C5 | Tamper-evident traceability | 🟡 | ✅ | hash-chained audit + `/api/audit/verify` |
| C6 | Radiogenomics | ❌ | ✅ | LGG / TCGA: 5 / 6 molecular targets above chance |
| C7 | Multimodal fusion / fMRI | ❌ | ✅ | PET-CT (TCIA sarcoma, 7 patients, leave-one-out Dice): CT 0.00 / PET 0.30 / PET+CT 0.34, with SUVmax, MTV and TLG; fMRI GLM activation maps (ds000114) |
| C9 | Generative enhancement | ❌ | ✅ | SR-GAN ×2 (experimental) |
| C10 | Autonomous mode with review | 🟡 | ✅ | pre-clear only clear normals; critical auto-alert |

## Round 2 — what the data showed (honest)

- **PET-CT fusion.** Fusion beats PET alone on 4 of 7 patients (+0.04 mean Dice); CT alone finds nothing.
  The demo patient is held out of the deployed model: Dice 0.69, MTV 686 ml vs 506 ml expert. With 7
  patients this is indicative only.
- **fMRI.** sub-01 reproduces the motor-homunculus order, but only the lips peak survives Bonferroni; sub-02
  does not reproduce it. The task signal in this release is weak (task > rest t ≈ 4–8).
- **Mammography.** Labels are per exam (no lesion side), so view-level scores are noisy. Cancer vs benign
  is clearly harder than cancer vs normal.
- **Whole-scan detection.** The candidate generator finds 94 % of test nodules, but the 3-D CNN ranks them
  poorly. The demo scan's 6.4 mm nodule is a candidate (score 0.88, rank 10) but falls below the 0.91
  operating threshold. The top detections there are chest-wall false positives, and their sizes are
  reported as "indeterminate" rather than guessed.

## Found during verification

- The first autonomous rule pre-cleared a head CT just below threshold and models without uncertainty. It
  was tightened, and a test was added (D-29).
- Re-analysing a reviewed study crashed on a foreign key. Reviewed findings are now kept as `superseded`.
- Model misses on demo samples (honest): malignant breast US (p 0.34 < 0.75), malignant CT nodule
  (p 0.19 < 0.60), wrist fracture (p 0.30 < 0.41). These are expected at starter-profile sensitivities.
