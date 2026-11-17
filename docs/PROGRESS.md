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
