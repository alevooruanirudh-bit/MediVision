# Architecture

MediVision AI is three cooperating parts that share one Python package (`ml/`):

```mermaid
flowchart LR
  subgraph Data["ml/data — free public datasets"]
    K[Kaggle API] --> R[data/raw]
    M[MedMNIST+] --> R
    R --> P[prepare.py → compact caches + patient-level split CSVs]
  end
  subgraph Train["ml/ — our models"]
    P --> T[train.py (config-driven)]
    T --> E[evaluate.py: AUROC, calibration, fairness, plots]
    E --> X[export_onnx.py]
    KG[kaggle/generate.py + run.py] -. full profile on free GPU .-> T
  end
  X --> A[(artifacts/&lt;model&gt;/&lt;profile&gt;)]
  subgraph API["backend/app — FastAPI (CPU, offline)"]
    A --> REG[registry: full > starter, ONNX > PyTorch]
    U[Upload PNG/JPG/DICOM] --> D[de-identify + pseudonymise]
    D --> RT[modality router + OOD]
    RT --> PIPE{pipeline}
    PIPE -->|chest| CH[3-model ensemble + MC-dropout + calibration + energy OOD]
    PIPE -->|bone| FR[fracture classifier + Faster R-CNN]
    PIPE -->|brain| BR[4-class classifier + Attention U-Net]
    CH & FR & BR --> XAI[Grad-CAM / Grad-CAM++ / occlusion]
    XAI --> MEAS[measurements: CTR, lung areas, zones, tumour area]
    MEAS --> TRI[triage 0–100]
    TRI --> COP[Copilot: report writer · intent model · KB/BM25 · rewrite]
    COP --> DB[(SQLite: studies, findings, reviews, reports, chat, audit)]
  end
  subgraph UI["frontend — Next.js 15"]
    DB --> W[Worklist / Viewer / Report / Compare / Lab / …]
  end
```

## Request flow for one upload

1. `POST /api/studies` — `services/deid.py` reads PNG/JPG/DICOM, strips PHI tags (PS3.15 basic-profile
   subset + private tags), hashes the patient identifier into a pseudonym (`PT-XXXXXX`), stores a
   display PNG (≤1024 px) and the de-identified DICOM.
2. `inference/pipeline.py` runs, timing every stage:
   - **router** (`inference/router.py`) — our SmallCNN picks chest / bone / brain / other-medical /
     not-medical; colour photos and non-medical images are stopped here (OOD guard #1).
   - **model** — chest: three ONNX classifiers return logits *and* pooled features; MC-dropout is
     sampled on the head in numpy; per-member per-label temperature scaling; ensemble mean ± combined
     uncertainty; per-label Youden thresholds; JointEnergy OOD score (guard #2).
   - **measurements** — lung U-Net → two largest components → thoracic width, heart width (medial
     gap, lower half), CTR, areas, symmetry; brain → Attention U-Net lesion area.
   - **XAI** — Grad-CAM on our PyTorch model for the top-3 findings; peak → lung zone.
   - **triage** — weights in `inference/triage_weights.yaml`.
   - **report** — `copilot/report_writer.py` drafts the structured report.
3. Results JSON + `Finding` rows are stored; the UI renders them.

## Copilot engine (no LLM)

| Piece | File | How |
|---|---|---|
| Intent model | `ml/copilot/train_intent.py` | TF-IDF (word + char) + logistic regression over 22 hand-written intents, keyword-group features from `ml/copilot/text.py` |
| Entities | `copilot/entities.py` | gazetteer from the knowledge base + rapidfuzz typo matching; sides, zones, sections, styles, XAI methods |
| Knowledge | `copilot/knowledge_base.json`, `copilot/kb.py` | curated entries per finding; BM25 retrieval |
| Answers | `copilot/chat.py` | 22 composers grounded in the study's numbers; action chips drive the UI |
| Report | `copilot/report_writer.py` | content selection → sentence planning → realisation (phrase bank mined from IU X-ray) |
| Rewrite | `copilot/rewrite.py` | concise / detailed / patient-friendly, lay glossary |
| Optional | `copilot/local_llm.py` | Ollama on localhost only, wording polish, OFF by default |

## Key design choices
See [DECISIONS.md](DECISIONS.md).
