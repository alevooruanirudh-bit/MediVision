# MediVision AI — Intelligent Medical Imaging Copilot

> **Research & educational prototype — not a medical device. Not for clinical use.**

MediVision AI is a full-stack imaging copilot in which **every medical model is designed and trained by
this project on free public datasets**, inference runs **fully offline on CPU**, and the "Copilot" is a
**self-built clinical language engine — no LLM, no external AI API**. Total cost: zero.

![Study viewer](docs/screenshots/04-study-viewer.png)

## What it does

| Paper section | Implemented as |
|---|---|
| III-A Diagnostic accuracy | Our DenseNet-121 + EfficientNet-B2 + ConvNeXt-Tiny **ensemble** for 14 chest findings (NIH ChestX-ray14); Faster R-CNN **fracture detector** + classifier (FracAtlas); 4-class **brain-tumour MRI classifier** |
| III-C Segmentation | Our **U-Net** (lungs) and **Attention U-Net** (brain tumour); automatic **cardiothoracic ratio**, lung areas, symmetry, tumour area; heatmap peak → **lung zone** |
| III-D Triage | Urgency 0–100 from severity weights × calibrated exceedance × confidence (strongest finding + diminishing credit for others) → CRITICAL / URGENT / ROUTINE / NORMAL worklist |
| III-E Reporting & NLP | **MediVision Copilot**: structured report writer (content selection → sentence planning → realisation with a phrase bank mined from IU X-ray), 22-intent chat with our intent model + BM25 knowledge base, streamed over SSE with UI action chips; rewrite as concise / detailed / patient-friendly; **experimental** CNN + Transformer report generator trained from scratch |
| III-F Advanced imaging | Our **DnCNN denoiser** trained on simulated low-dose noise + CLAHE / unsharp / gamma, PSNR / SSIM |
| III-G Personalised | Longitudinal **compare**: per-finding deltas, CTR change, heatmap difference, interval-change paragraph |
| IV Black box | **Grad-CAM, Grad-CAM++, occlusion**; ensemble + **MC-dropout** uncertainty; per-label **temperature scaling** |
| IV Bias & data quality | **Fairness** by sex / age / view; **modality router** + **energy-score OOD guard** |
| IV Privacy | **DICOM de-identification** + pseudonyms, local inference, **audit log** |
| IV Interoperability | DICOM in; **FHIR R4-style** DiagnosticReport + **PDF** out |
| V Federated learning | Real **FedAvg** across 3 simulated non-IID hospitals, live SSE charts |
| V Human–AI collaboration | Accept / reject / edit each finding; disagreements export as a **re-training CSV** |

## Measured metrics (starter models, laptop-trained)

<!-- METRICS:START -->
| Model | Metric (held-out test) | Value | CPU ms/img (ONNX) |
|---|---|---|---|
| Chest X-ray ensemble (14 labels) | macro AUROC | 0.791 | — |
| · DenseNet-121 | macro AUROC | 0.769 | 7.3 |
| · EfficientNet-B2 | macro AUROC | 0.767 | 4.4 |
| · ConvNeXt-Tiny | macro AUROC | 0.786 | 9.6 |
| Lung U-Net | Dice | 0.974 | 19.2 |
| Fracture classifier | AUROC | 0.901 | 4.5 |
| Fracture detector | mAP@0.5 | 0.048 | — |
| Brain MRI classifier | accuracy / macro-F1 | 0.953 / 0.952 | 2.7 |
| Brain tumour Attention U-Net | Dice (tumour slices) | 0.752 | 5.4 |
| Modality router | accuracy / macro-F1 | 0.997 / 0.997 | 1.8 |
| DnCNN denoiser | PSNR noisy → ours (dB) | 22.9 → 32.8 | 7.9 |
| Report generator (experimental) | BLEU-4 / ROUGE-L | 0.095 / 0.248 | — |
| Copilot intent model | held-out accuracy | 0.964 | — |
<!-- METRICS:END -->

Starter models are trained on small images for a few epochs on a laptop, so they are clearly weaker
than full models. Full-profile numbers appear in [docs/METRICS.md](docs/METRICS.md) and on the Lab page
after `make kaggle-train-all`. Full details, fairness tables and the federated run:
[docs/METRICS.md](docs/METRICS.md).

## Quick start

```bash
make setup            # Python 3.11 venv + requirements, npm install
make data-starter     # free datasets via MedMNIST+ and the Kaggle API (~/.kaggle token)
make train-starter    # trains, evaluates and exports EVERY starter model
make seed             # 12 demo studies, already analysed
make dev              # API http://localhost:8000/docs · UI http://localhost:3000
```
No API keys for any AI service are needed. After the downloads, everything works offline.
Upgrade to full accuracy on Kaggle's free GPU: [docs/TRAINING.md](docs/TRAINING.md).

Other commands: `make test`, `make lint`, `make e2e`, `make screenshots`, `make kaggle-train MODEL=brain_cls`,
`make docker` (see `make help`).

## Architecture

```mermaid
flowchart LR
  U[PNG / JPG / DICOM] --> D[De-identify] --> R[Modality router + OOD]
  R -->|chest| C[3-net ensemble · MC-dropout · calibration]
  R -->|bone| F[Fracture cls + Faster R-CNN]
  R -->|brain| B[Tumour classifier + Attention U-Net]
  C & F & B --> X[Grad-CAM / ++ / occlusion] --> M[CTR · areas · zones] --> T[Triage] --> P[Copilot report]
  P --> DB[(SQLite + audit)] --> UI[Next.js UI]
```
More in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Stack: PyTorch · timm · ONNX Runtime · FastAPI ·
SQLModel · scikit-learn · Next.js 15 · TypeScript · Tailwind · Radix · Framer Motion · Recharts ·
TanStack Query · zustand.

## Screenshots

| | |
|---|---|
| ![Landing](docs/screenshots/01-landing.png) | ![Dashboard](docs/screenshots/02-dashboard.png) |
| ![Worklist](docs/screenshots/03-worklist.png) | ![Copilot](docs/screenshots/04b-study-copilot.png) |
| ![Report](docs/screenshots/05-report-editor.png) | ![Compare](docs/screenshots/06-compare.png) |
| ![Enhancement](docs/screenshots/07-enhancement-lab.png) | ![Federated](docs/screenshots/08-federated.png) |
| ![Lab](docs/screenshots/09-lab.png) | ![Model card](docs/screenshots/10-model-card.png) |

## Limitations

- Starter models: low resolution (96–192 px), subsets, few epochs → modest accuracy.
- NIH labels are NLP-mined and noisy; public datasets under-represent many populations and scanners.
- Single 2-D images; no clinical history; the brain model knows three tumour types only.
- Brain-MRI and lung-mask splits are random (no patient IDs published) — test scores may be optimistic.
- Heatmaps are approximate; the CTR is an estimate from segmentation masks.
- The chest ensemble's energy score is a weak OOD signal on its own (starter AUROC < 0.5 vs photos/MRI/bone); the
  modality router + grayscale check carry the OOD guard. Both are reported honestly in the Lab.
- The experimental report generator is small and can hallucinate; the rule-based report is the default.

## Ethics

This is a research and teaching prototype. It must not be used to diagnose or treat anyone. All data are
public, de-identified research datasets used under their licenses; uploaded DICOMs are de-identified and
processed locally only. Every AI output is a draft for qualified human review, every action is audited,
and the Copilot never states a definitive diagnosis. See [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Dataset credits

NIH ChestX-ray14 (Wang et al., CVPR 2017) · MedMNIST+ (Yang et al., Sci Data 2023) · FracAtlas (Abedeen et
al., Sci Data 2023) · Brain Tumor MRI (Nickparvar 2021) · LGG MRI Segmentation (Buda et al., CBM 2019) ·
Indiana University Chest X-rays / OpenI (Demner-Fushman et al., JAMIA 2016) · COVID-19 Radiography
Database (Chowdhury et al. 2020; Rahman et al. 2021) · RSNA Pneumonia (Shih et al. 2019) · CIFAR-10
(Krizhevsky 2009). Licenses: [docs/DATASETS.md](docs/DATASETS.md).

## Docs

[ARCHITECTURE](docs/ARCHITECTURE.md) · [DATASETS](docs/DATASETS.md) · [TRAINING](docs/TRAINING.md) ·
[MODEL_CARD](docs/MODEL_CARD.md) · [METRICS](docs/METRICS.md) · [API](docs/API.md) ·
[DEPLOYMENT](docs/DEPLOYMENT.md) · [DEMO_SCRIPT](docs/DEMO_SCRIPT.md) · [DECISIONS](docs/DECISIONS.md) ·
[PROGRESS](docs/PROGRESS.md)
