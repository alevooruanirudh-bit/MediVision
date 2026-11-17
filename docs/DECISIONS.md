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

**D-04 — Chest starter data = ChestMNIST+ 128 px; NIH sample for fairness.** Downloading a large NIH subset
file-by-file ran at ~1.6 images/s. ChestMNIST+ contains the same NIH images with the official
patient-wise split, so the starter ensemble trains on a 24k subset of it. Fairness (age/sex/view) is
evaluated on the NIH random sample restricted to official-test-list patients, which never appear in the
ChestMNIST training split. The full profile trains on NIH directly with our own patient-grouped split.

**D-05 — RSNA localiser is full-profile only.** The competition rules were not accepted on the build
account, so the download returned 403. Per the spec the build continues: `pneumonia_det` has a config and a
Kaggle notebook; in the starter app pneumonia is localised with Grad-CAM + lung-zone mapping.

**D-06 — Lung masks from the COVID-19 Radiography Database.** Montgomery/Shenzhen with masks is ~4 GB of
large PNGs; the COVID-19 Radiography Database (same Qatar University/Dhaka group as COVID-QU-Ex) ships
256 px lung masks for 21k images in 0.8 GB. Starter uses 3,100 pairs; the full notebook uses all of them
plus Montgomery/Shenzhen and COVID-QU-Ex as dataset sources.

**D-07 — Random (stratified) splits where no patient IDs exist.** Brain Tumor MRI and the lung-mask
database publish no patient identifiers, so we split per class at random. This can leak near-duplicate
slices between splits and may inflate brain-classifier test accuracy — documented in the model card.

**D-08 — IU X-ray from a 256 px cleaned Kaggle copy.** The original Kaggle copy is 14 GB; the cleaned copy
(2.2 GB) contains resized images + concatenated findings/impression captions. The full notebook uses the
original (`raddar/chest-xrays-indiana-university`), and the preparer supports both schemas.

**D-09 — Fracture detector = torchvision Faster R-CNN (MobileNetV3-FPN), not YOLOv8.** Avoids the AGPL
licence of ultralytics in the app; uses generic COCO initialisation (allowed). Trains on CPU because
torchvision detection ops are unreliable on MPS.

**D-10 — MC-dropout on the classifier head in numpy.** ONNX exports `(logits, features)`; the head weights
are saved to `head.npz` and sampled with dropout masks in numpy, so uncertainty costs microseconds and the
backbone runs once.

**D-11 — Intent-model evaluation.** Accuracy is reported on 110 hand-written natural questions that were
never used for training (`intents_test.jsonl`), plus a stricter 5-fold held-out-*template* CV (≈0.89),
both shown in the Lab. Keyword-group features (`ml/copilot/text.py`) lifted template-CV from 0.66 to 0.89.

**D-12 — CIFAR-10 from a Kaggle mirror.** The Toronto host served ~5 KB/s; `download.py` now tries the
Kaggle mirror first and falls back to torchvision.

**D-13 — Router "other medical" class** uses OrganAMNIST (CT), BreastMNIST (ultrasound) and PneumoniaMNIST
64 px. A colourfulness check backs up the router for colour photographs.

**D-14 — Brain segmentation input = FLAIR channel.** LGG slices are 3-channel (pre-contrast, FLAIR,
post-contrast); uploaded MRIs are single grayscale images, so the Attention U-Net is trained on FLAIR only.
Domain shift between LGG FLAIR and the Nickparvar T1/T2 images remains a limitation.

**D-15 — Next.js 15 + Tailwind v4 with hand-written shadcn-style primitives** (Radix UI under the hood)
instead of running the shadcn CLI, which is interactive. Fonts are self-hosted via @fontsource so the UI
works offline.

**D-16 — Docker not available on the build machine.** The Dockerfiles and compose file were written
following the same commands the Makefile runs and tested indirectly (`next build` standalone output,
uvicorn start); a real `docker compose build` must be run on a machine with Docker.

**D-17 — Kaggle automation verified end to end (smoke run).** `python -m ml.kaggle.run --model brain_cls --smoke`
pushed a private kernel, polled it to `complete`, and pulled the weights + metrics into
`artifacts/brain_cls/kaggle_smoke/`. The build account's kernels had **no internet and no GPU** (phone not
verified), so the notebook now detects this, prints a phone-verification hint, and falls back to
`PRETRAINED_BACKBONE=false` on CPU (smoke result: 73% accuracy from scratch). Two earlier attempts surfaced
real bugs that were fixed: nested `/kaggle/input` mount paths and an unconditional `medmnist` import.

**D-18 — Flagging floor + non-saturating triage.** On the starter ensemble, Youden thresholds for rare labels
were ~0–0.08 (e.g. Hernia 0.00), so 8–11 labels were flagged per image and the noisy-OR triage saturated
(six demo studies CRITICAL). Fix: a finding is *flagged* only if p ≥ max(Youden threshold, 0.20) — thresholds
themselves are unchanged and still used for reported sensitivity/specificity; "cannot be excluded" requires
p ≥ 0.12; triage now uses the strongest finding plus diminishing credit (c₁ + 0.25·c₂ + 0.10·c₃) with
CRITICAL ≥ 75.

**D-19 — Report generator decoding.** Beam search with a weak length penalty stopped after ~10 tokens
(BLEU-4 0.009 due to the brevity penalty). Decoding now forbids EOS before 24 tokens and uses length
normalisation α = 1.0 (BLEU-4 0.095, ROUGE-L 0.248 — on par with the "most common report" baseline, which
is reported next to it; the model is experimental).

**D-20 — Federated demo speed.** The CNN averages 64 px inputs down to 32 px first; 10 rounds + both baselines
now run in ~26 s on CPU (was 228 s).
