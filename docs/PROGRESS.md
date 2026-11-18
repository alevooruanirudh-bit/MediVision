# Build progress

## Phase 1 — Scaffold
- [x] Repo, CLAUDE.md, PROGRESS.md, DECISIONS.md, Makefile, .env.example, .gitignore, pyproject

## Phase 2 — Data layer
- [x] download.py (Kaggle API + MedMNIST+ + mirrors; resumable; SHA-256 checksums; SOURCES.md)
- [x] prepare.py: zip/dir readers → compact caches; dataset classes for every dataset
- [x] Patient-level splits saved as CSV (no leakage) + leakage tests
- [x] Augmentation pipelines (no horizontal flips for chest; CLAHE; joint image/mask)

## Phase 3 — Training pipelines
- [x] Config-driven trainer (AMP, early stopping, cosine LR + warm-up, ASL / weighted BCE / CE / BCE+Dice, best/last ckpt, resume, seeding, history)
- [x] Chest ensemble (DenseNet-121, EfficientNet-B2, ConvNeXt-Tiny) with MC-dropout heads
- [x] Pneumonia localiser (Faster R-CNN MobileNetV3-FPN) — config + RSNA preparer + Kaggle notebook (full only, D-05)
- [x] Lung U-Net
- [x] Fracture detector + fracture classifier
- [x] Brain MRI classifier + Attention U-Net segmenter
- [x] Modality router + JointEnergy OOD score
- [x] Denoising autoencoder (DnCNN-style)
- [x] Copilot intent model + intents.jsonl (22 intents × ≥40 paraphrases) + held-out test set
- [x] Experimental report generator (BPE + Transformer decoder)

## Phase 4 — Starter training (CPU/MPS)
- [x] All starter models trained, metrics JSON + plots in artifacts/

## Phase 5 — Kaggle notebooks
- [x] Notebook + kernel-metadata.json per full model; `make kaggle-train` push/poll/pull/export

## Phase 6 — Evaluation
- [x] AUROC / F1 / sens-spec, temperature scaling, reliability diagrams, fairness splits, ONNX export + parity check

## Phase 7 — Inference + XAI + triage + Copilot
- [x] Pipeline (de-id → router/OOD → modality → XAI → measurements → triage → report), timed per stage
- [x] Grad-CAM / Grad-CAM++ / occlusion overlays + hottest box + lung zone
- [x] Triage scoring (weights YAML)
- [x] Copilot KB, phrase bank (mined from IU X-ray), report writer, 22-intent chat (SSE), rewrite helpers, optional Ollama adapter (off)
- [x] Compare, Enhancement Lab, Federated FedAvg

## Phase 8 — FastAPI backend
- [x] All endpoints, SQLite/SQLModel, audit log, PDF + FHIR export, consistent errors, OpenAPI

## Phase 9 — Frontend
- [x] 12 pages, design system, dark/light, responsive, keyboard shortcuts

## Phase 10 — Quality & ship
- [x] pytest suite green, ruff/black, eslint/tsc, vitest, Playwright E2E
- [x] Screenshots of every page in docs/screenshots/
- [x] Docs: README, ARCHITECTURE, DATASETS, TRAINING, MODEL_CARD, METRICS, DEPLOYMENT, DEMO_SCRIPT, API
- [x] Dockerfiles (HF Spaces compatible) + docker-compose; Vercel-ready frontend (Docker build not run locally — D-16)
- [x] Final commit

## Phase 11 — Paper requirements audit (JECIR 2023) — see REQUIREMENTS_AUDIT.md
- [x] Audit of 34 MUST / 7 SHOULD / 10 COULD requirements (before: 19 / 4 / 0 fully implemented)
- [x] New modalities: head-CT hemorrhage, CT nodule 3-D CNN + volumetrics, breast ultrasound, fundus DR grading (D-21)
- [x] Router v2 (8 classes, fundus colour exemption) (D-22); CT volume upload + 3-D MIP/MPR view (D-23)
- [x] Second-reader metrics (sens/spec/FPR/FNR/PPV/NPV) for every classifier (Lab + METRICS.md)
- [x] Plain-language rationale per finding (D-24); radiomics panel (D-25)
- [x] Model versioning (weights hash), per-finding AI/human provenance, report drafter/editor/signer (D-26)
- [x] Role-based access + optional token auth, consent capture at upload (D-27)
- [x] Hash-chained tamper-evident audit log + verify endpoint/UI (D-28)
- [x] Critical-finding alerts with acknowledgement; autonomous pre-clear mode (D-29)
- [x] DICOM Secondary Capture + Encapsulated PDF export (D-30)
- [x] Progression timeline; SR-GAN ×2 super-resolution; image fusion (D-31)
- [x] Radiogenomics (LGG/TCGA) + segmentation fairness by sex/age/race; federated per-site generalisation + HIPAA/GDPR notes (D-32)
- [x] Tests: backend/tests/test_requirements.py; docs updated (README, MODEL_CARD, DATASETS, DECISIONS)
- [x] Removed from scope (D-37): stroke LVO / PE (C3), wearables (C8) — no suitable free data / not imaging

## Phase 12 — Round 2: partial items completed (D-33 … D-36)
- [x] Mammography (C1): Mini-DDSM classifier, exam-level metrics, letterbox-aware Grad-CAM, router v3 (9 classes)
- [x] PET-CT fusion (C7): CT / PET / PET+CT leave-one-patient-out study, deployed segmenter, SUV / MTV / TLG, fused viewer
- [x] fMRI (C7): GLM activation maps for 2 subjects, somatotopy check, Enhancement Lab card + `/api/fmri`
- [x] Whole-scan lung-nodule detection (M5): LUNA16 subset, lung mask + LoG candidates + 3-D CNN, FROC; detections feed nodule3d
- [x] Held-out demo samples (mammograms, PET-CT case, chest-CT slab); tests in backend/tests/test_round2.py; docs updated
