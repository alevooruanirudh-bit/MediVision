# Build progress

## Phase 1 — Scaffold
- [ ] Repo, CLAUDE.md, PROGRESS.md, DECISIONS.md, Makefile, .env.example, .gitignore, pyproject

## Phase 2 — Data layer
- [ ] download.py (Kaggle API + MedMNIST+ + Figshare; resumable; checksums; SOURCES.md)
- [ ] Dataset classes for every dataset
- [ ] Patient-level splits saved as CSV (no leakage) + test
- [ ] Augmentation pipelines

## Phase 3 — Training pipelines
- [ ] Config-driven trainer (AMP, early stopping, cosine LR, imbalance-aware loss, ckpt, resume, seed)
- [ ] Chest ensemble (DenseNet-121, EfficientNet-B2, ConvNeXt-Tiny)
- [ ] Pneumonia localizer (Faster R-CNN MobileNetV3-FPN)
- [ ] Lung U-Net
- [ ] Fracture detector + fracture classifier
- [ ] Brain MRI classifier + Attention U-Net segmenter
- [ ] Modality router + OOD energy score
- [ ] Denoising autoencoder (DnCNN-style)
- [ ] Copilot intent model + intents.jsonl
- [ ] Experimental report generator (BPE + Transformer decoder)

## Phase 4 — Starter training (CPU/MPS)
- [ ] All starter models trained, metrics JSON + plots in artifacts/

## Phase 5 — Kaggle notebooks
- [ ] Notebook + kernel-metadata.json per full model; `make kaggle-train` push/poll/pull

## Phase 6 — Evaluation
- [ ] AUROC/F1/sens/spec, temperature scaling, reliability diagrams, fairness splits, ONNX export

## Phase 7 — Inference + XAI + triage + Copilot
- [ ] Pipeline (de-id → router/OOD → modality → XAI → measurements → triage → report)
- [ ] Grad-CAM / Grad-CAM++ / occlusion
- [ ] Triage scoring
- [ ] Copilot KB, report writer, chat (≥20 intents, SSE), rewrite helpers, optional Ollama adapter
- [ ] Compare, Enhancement Lab, Federated FedAvg

## Phase 8 — FastAPI backend
- [ ] All endpoints, SQLite/SQLModel, audit log, PDF + FHIR export

## Phase 9 — Frontend
- [ ] 12 pages, design system, dark/light, responsive

## Phase 10 — Quality & ship
- [ ] pytest suite green, ruff/black, eslint/tsc, vitest, Playwright E2E
- [ ] Screenshots of every page in docs/screenshots/
- [ ] Docs: README, ARCHITECTURE, DATASETS, TRAINING, MODEL_CARD, DEPLOYMENT, DEMO_SCRIPT, API
- [ ] Docker files, HF Spaces Dockerfile, Vercel-ready frontend
- [ ] Final commit
