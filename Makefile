SHELL := /bin/bash
PY ?= .venv/bin/python
export PYTHONPATH := $(CURDIR):$(CURDIR)/backend
PROFILE ?= starter
MODEL ?=

STARTER_MODELS := chest_densenet121 chest_efficientnet_b2 chest_convnext_tiny lung_unet \
	fracture_cls fracture_det brain_cls brain_seg router denoiser report_gen
FULL_MODELS := $(STARTER_MODELS) pneumonia_det

.PHONY: help setup data-starter prepare train-starter train evaluate export seed dev backend frontend \
	test test-backend test-frontend e2e lint format kaggle-notebooks kaggle-train kaggle-train-all \
	federated docker screenshots clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n",$$1,$$2}'

setup: ## Create .venv (Python 3.11) and install backend + frontend deps
	@command -v uv >/dev/null 2>&1 && uv venv -p 3.11 .venv || python3.11 -m venv .venv
	@command -v uv >/dev/null 2>&1 && VIRTUAL_ENV=.venv uv pip install -r requirements.txt \
		|| .venv/bin/pip install -r requirements.txt
	cd frontend && npm install

data-starter: ## Download the small starter datasets (MedMNIST+, Kaggle subsets) and build caches
	$(PY) -m ml.data.download --profile starter
	$(PY) -m ml.data.prepare --all

prepare: ## Rebuild processed caches + patient-level splits from data/raw
	$(PY) -m ml.data.prepare --all

train-starter: ## Train ALL starter models (CPU/MPS, minutes each) + evaluate + export ONNX
	$(PY) -m ml.train_all --profile starter

train: ## Train one model: make train MODEL=brain_cls PROFILE=starter
	$(PY) -m ml.train --model $(MODEL) --profile $(PROFILE)

evaluate: ## Evaluate all models of PROFILE -> artifacts/*/PROFILE/metrics.json
	$(PY) -m ml.evaluate --all --profile $(PROFILE)

export: ## Export all trained models of PROFILE to ONNX
	$(PY) -m ml.export_onnx --all --profile $(PROFILE)

federated: ## Run the FedAvg demo from the CLI
	$(PY) -m ml.federated.fedavg

kaggle-notebooks: ## (Re)generate Kaggle notebooks for every full-profile model
	$(PY) -m ml.kaggle.generate

kaggle-train: ## Train MODEL on Kaggle free GPU (needs ~/.kaggle token), pull weights, export ONNX
	$(PY) -m ml.kaggle.run --model $(MODEL)

kaggle-train-all: ## Train every full-profile model on Kaggle, one after another
	@for m in $(FULL_MODELS); do $(PY) -m ml.kaggle.run --model $$m || exit 1; done

seed: ## Create demo patients/studies (already analysed) in the backend DB
	$(PY) -m app.seed

backend: ## Run the FastAPI backend on :8000
	$(PY) -m uvicorn app.main:app --reload --port 8000 --app-dir backend

frontend: ## Run the Next.js frontend on :3000
	cd frontend && npm run dev

dev: ## Backend :8000 + frontend :3000
	@trap 'kill 0' EXIT; \
	$(PY) -m uvicorn app.main:app --port 8000 --app-dir backend & \
	(cd frontend && npm run dev) & wait

test: test-backend test-frontend ## All tests

test-backend:
	$(PY) -m pytest

test-frontend:
	cd frontend && npm run test --silent

e2e: ## Playwright E2E (starts both servers)
	cd frontend && npx playwright test

screenshots: ## Playwright screenshots of every page into docs/screenshots/
	cd frontend && npx playwright test e2e/screenshots.spec.ts

lint: ## ruff + black --check + eslint + tsc
	$(PY) -m ruff check ml backend
	$(PY) -m black --check ml backend
	cd frontend && npm run lint && npx tsc --noEmit

format:
	$(PY) -m ruff check --fix ml backend
	$(PY) -m black ml backend

docker: ## Build and run with docker compose
	docker compose up --build

clean:
	rm -rf backend/storage backend/medivision.db frontend/.next
