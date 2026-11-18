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

## Requirements audit against the source paper (JECIR 2023) — D-21 … D-32

**D-21 — New modalities from small free datasets.** The paper names CT, ultrasound, hemorrhage and lung-nodule
detection, which the first build did not cover. Added, all trained by us with the same trainer / evaluator / ONNX
export:
- head-CT hemorrhage on the Kitamura 200-slice set. Each image is a different patient, so a stratified random
  split is patient-disjoint. RSNA ICH is the full profile; it needs the competition rules to be accepted.
- a 3-D CNN for chest-CT nodule malignancy on NoduleMNIST3D (LIDC-IDRI patches, official split).
- breast-ultrasound malignancy on BreastMNIST+.
- diabetic-retinopathy grading on RetinaMNIST+ colour fundus photos.

Nodule *detection* on whole CT scans (LUNA16, ~120 GB) is out of reach at zero cost. We characterise a supplied
nodule patch instead, and the docs say so.

**D-22 — Router v2 (8 classes) and the colour guard.** The router now has three extra classes: head CT,
breast US and fundus. Fundus photographs are colourful by nature, so the colour guard (D-18 era) exempts the
fundus class only. Every other colourful image is still treated as not medical. Older 5-class router artefacts
keep working, because the class list is read from their config.

**D-23 — CT volumes.** `.npy` / `.npz` volumes (loaded with `allow_pickle=False`) and multi-frame CT DICOM are
accepted. Any volume is windowed to the lung window and resampled to 64³. The nodule is segmented classically
(Otsu thresholding, keeping the connected component nearest the centre), and volume / equivalent diameter /
sphericity are computed from that mask. A rotating MIP and three MPR slices provide the "3-D view". This is
not AR/VR, and voxel spacing is assumed to be 1 mm when the file does not carry it (flagged in the UI).

**D-24 — Plain-language rationale.** Each finding carries 2–5 extracted reasons:
- its probability vs. the operating threshold;
- ensemble / MC-dropout agreement;
- the heatmap zone;
- whether an independent measurement supports or contradicts it (CTR, lung asymmetry, tumour mask, detector
  box, nodule volume).

This summarises evidence; it is not a causal explanation of the network.

**D-25 — Radiomics.** An IBSI-style subset (shape, first-order, GLCM with 32 levels and 4 directions) is computed
with scikit-image inside the lung / tumour / nodule mask, or over the whole image when no mask exists. It is
labelled descriptive. We chose this over pyradiomics to avoid a heavy C-extension dependency.

**D-26 — Provenance, versioning, liability.**
- Every result stores each model's weights hash (SHA-256, 12 hex characters), its profile and its training
  date.
- Findings record `source`, `model_version`, `reviewed_by`, `reviewed_role` and `reviewed_at`.
- Reports record `drafted_by` (AI engine), `edited_by` and `signed_by` / `signed_role`.
- PDF, FHIR (`resultsInterpreter`, Observation `device` / `performer`) and DICOM outputs carry the same
  information.

**D-27 — Access control and consent.**
- Local users with four roles (admin / radiologist / resident / viewer) and PBKDF2 password hashes. Bearer
  tokens are stored only as SHA-256 hashes.
- `AUTH_ENABLED=false` by default, so the single-user demo keeps working. In that mode the UI sends
  `X-Demo-Role` so the permission checks can be tried out. Enable auth for any shared deployment.
- Uploads must state a consent / lawful basis (`REQUIRE_CONSENT=true`, recorded per study and in the audit
  log).

**D-28 — Hash-chained audit log.** This is the "blockchain-style traceability" the paper asks for, without a
blockchain. Each entry stores `prev_hash` and `sha256(prev_hash | canonical entry)`, and `GET /api/audit/verify`
detects edits or deletions. Rows written before the chain existed are reported as legacy and are not chained.

**D-29 — Critical alerts and autonomous mode.**
- CRITICAL studies raise an alert. It shows as a red bar on every page and pins the study to the top of the
  worklist until someone with the `alert.ack` permission acknowledges it (audited).
- `AUTONOMOUS_TRIAGE=true` pre-clears a NORMAL study into a "confirm normal" queue only when every output is
  ≤ 40 % of its threshold AND every model produced a real uncertainty estimate ≤ 0.08. Verification showed
  the first rule (uncertainty only) pre-cleared a head CT at p 0.905 vs threshold 0.915, and single models
  that have no uncertainty estimate (reported as 0.0); both are now excluded.
  Every such study still needs human confirmation, and no report is ever finalised automatically.

**D-30 — DICOM outputs.** We export a Secondary Capture (the image with the AI overlay and findings burned in)
and the report as an Encapsulated PDF. A full DICOM-SR TID 1500 encoding was judged too large for the benefit;
FHIR DiagnosticReport remains the structured output.

**D-31 — Progression and generative enhancement.**
- `/api/patients/{id}/timeline` fits a linear trend per finding (probability per 30 days) over all of a
  patient's studies and makes a rule-based progression call. It is labelled descriptive and is not a
  prognostic model.
- A small SRGAN-style ×2 super-resolution generator: a residual network over bicubic upsampling, trained with
  L1 plus an LSGAN PatchGAN term weighted 0.003. It is labelled experimental, with the warning that GAN
  texture can be invented.
- A diffusion model was rejected for CPU cost.

**D-32 — Radiogenomics and the federated site analysis.**
- Radiogenomics uses expert-mask radiomics of the 110 LGG patients against TCGA clusters. Repeated stratified
  5-fold logistic regression is compared with a 10-shuffle label-permutation baseline, and associations are
  reported only as "above chance".
- Brain-segmentation Dice is broken down by sex / age / race codes. This is the only dataset in the project
  with race.
- FedAvg is now also evaluated per hospital scanner style. Worst-site AUROC and the site gap are reported to
  test the paper's claim that diverse training improves generalisation and reduces bias.

**D-33 — Mammography (C1).** Mini-DDSM (2,414 screening exams × 4 views) is the only free mammography set
that fits a laptop (CBIS-DDSM is 5–160 GB, VinDr-Mammo needs credentialed access). Its labels are per exam:
all four views of a cancer exam are filed as Cancer, including the healthy breast, and this release has no
lesion side or ROI. So the model is trained per view on exam labels (noisy by construction), and it is
evaluated per view and per exam (mean of the 4 views). Cancer vs normal and cancer vs benign are reported
separately, because the second is the harder clinical question. Splits are by exam, so no woman spans two
splits. Mammograms are letterboxed (not stretched), and Grad-CAM maps are un-letterboxed back onto the
original frame. The router gains a ninth class, mammography; the 8-class label set stays registered so older
weights still load.

**D-34 — PET-CT fusion (C7).** The 4quant release of TCIA Soft-tissue-Sarcoma provides co-registered PET (SUV)
and CT at 5 mm with expert contours, for 7 patients only. The same 2.5-D U-Net is trained on CT only, PET
only, and PET+CT (early fusion, 6 channels), each evaluated leave-one-patient-out. The deployed model is
trained on 6 patients: STS_031, the last id and fixed in advance, is held out as the demo case. Read-outs are
SUVmax, SUVmean, MTV and TLG from the largest predicted component. The registration-only fusion tool from
round 1 stays as a generic 2-D utility.

**D-35 — Functional MRI (C7).** OpenNeuro ds000114 (finger / foot / lips), already preprocessed in MNI space.
We fit our own GLM in numpy: double-gamma HRF, DCT high-pass, 6 mm smoothing inside an eroded brain mask,
and motion regressors where the release provides them (sub-02 only). Thresholds are Bonferroni. This is
statistics, not a learned model.
- The first unsmoothed, single-subject run put the finger peak in the cerebellum and the foot peak at the
  mask edge. Smoothing, mask erosion and a cerebrum-only peak search (z > 0) were then fixed before the
  analysis was extended to both subjects.
- Both subjects are reported. Choosing one after seeing results would be cherry-picking.
- A pre-declared sanity check tests the motor homunculus ordering (lips below finger below foot; foot most
  medial).

**D-36 — Whole-scan lung-nodule detection (M5).** LUNA16 is ~120 GB, so only 44 subset0 scans are downloaded
(~3 GB; ~80 % with nodules, smallest first), split by scan into train / val / test.
- Pipeline: 1 mm resampling → threshold lung mask (only in-plane borders count as "outside" — the trachea
  leaves through the top slice) → multi-scale LoG candidates (≤ 600 per scan) → our 3-D CNN on 32 mm cubes.
- Positives are annotated nodules with jittered 48 mm crops plus candidates inside a nodule. All other
  candidates are hard negatives.
- Evaluation is LUNA16 FROC on held-out scans. "Excluded" findings count as false positives, which is
  stricter than the official script.
- The operating threshold is set at 1 FP/scan on the validation scans.
- Each detection is passed to the existing nodule3d malignancy model and the volumetrics.

**D-37 — Scope trimmed.** C3 (stroke LVO / PE on CT angiography) and C8 (wearables) are removed from scope:
there is no small, free, suitable dataset (RSNA PE is ~900 GB), and wearables are outside imaging. They are
no longer listed as requirements.
