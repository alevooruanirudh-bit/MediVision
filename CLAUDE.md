# MediVision AI — Claude Code guide

Research & educational prototype — **not a medical device. Not for clinical use.**

## Hard rules
- **No external AI APIs** (OpenAI, Gemini, Anthropic, Groq, HF Inference, Copilot…). Enforced by
  `backend/tests/test_no_external_ai.py`. The only optional LLM hook is a localhost Ollama adapter,
  off by default (`COPILOT_LOCAL_LLM=false`).
- **All medical models are trained by this repo** on free public datasets. Only generic ImageNet
  backbone init (torchvision/timm) is allowed (`PRETRAINED_BACKBONE=true`); `false` → from scratch.
- Zero cost: free datasets, Kaggle free GPU, free hosting tiers.
- Every UI screen shows the disclaimer banner.

## Layout
```
ml/            data download/loaders, model architectures, trainer, evaluation, ONNX export,
               Kaggle notebook generation, federated learning, copilot intent model
artifacts/     <model>/<profile>/{model.pt, model.onnx, metrics.json, plots/, card.md}
backend/app/   FastAPI: api/, inference/, copilot/, services/, models/ (SQLModel tables)
backend/tests/ pytest suite
frontend/      Next.js 15 App Router + TS strict + Tailwind + shadcn-style UI
docs/          ARCHITECTURE, DATASETS, TRAINING, MODEL_CARD, DECISIONS, PROGRESS, DEPLOYMENT, API…
```

## Commands
```
make setup            # venv (.venv, py3.11) + pip deps + npm install
make data-starter     # downloads small starter datasets into data/ (git-ignored)
make train-starter    # trains ALL starter models -> artifacts/*/starter
make evaluate export  # metrics JSON/plots + ONNX
make seed             # demo patients/studies into backend SQLite
make dev              # backend :8000 + frontend :3000
make test lint        # pytest + ruff/black + eslint/tsc + vitest
make kaggle-train MODEL=<name>   # full-profile training on Kaggle free GPU
```
`PYTHONPATH=.:backend` is set by the Makefile so `ml` and `app` are importable.

## Conventions
- Python 3.11, type hints, Pydantic v2, ruff + black (line length 100).
- One YAML per model in `ml/configs/`, each with `starter` and `full` profiles.
- Model registry (`backend/app/inference/registry.py`) prefers `full` weights, falls back to `starter`;
  ONNX runtime first, PyTorch fallback (PyTorch is always used for Grad-CAM).
- Device: cuda > mps > cpu for training; inference is CPU.
- Log every judgement call in `docs/DECISIONS.md`; tick `docs/PROGRESS.md`; commit per phase.
