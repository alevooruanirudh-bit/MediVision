# Decisions log

Format: **D-NN — title.** Context → decision → consequence.

**D-01 — Repo root = workspace folder.** The spec shows a `medivision-ai/` tree; the workspace folder
itself is the repo root (it was empty), so the tree lives directly here.

**D-02 — Training device.** Trainer picks cuda > mps > cpu. On the build machine (Apple Silicon) starter
models train on MPS where ops are supported; detectors train on CPU (torchvision detection ops are
unreliable on MPS). Inference in the app is always CPU (ONNX Runtime).

**D-03 — Disk budget.** The build machine had ~15 GB free. Raw archives are streamed/converted to compact
`.npz` caches in `data/processed/` and the raw zips are deleted after conversion. Full-resolution datasets
(NIH 45 GB, RSNA 4 GB) are only used on Kaggle via `dataset_sources`, never downloaded locally.
