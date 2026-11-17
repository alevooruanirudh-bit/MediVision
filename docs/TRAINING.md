# Training

Every model is defined by `ml/configs/<model>.yaml` with two profiles:

| Profile | Where | Images | Purpose |
|---|---|---|---|
| `starter` | your laptop (CPU / Apple MPS) | 96–384 px, subsets, few epochs | the whole app works today |
| `full` | Kaggle free GPU (T4/P100) | 256–640 px, full datasets | best accuracy |

`PRETRAINED_BACKBONE=true` (default) starts backbones from *generic* ImageNet weights (detectors:
generic COCO weights). `PRETRAINED_BACKBONE=false` trains everything from scratch. No third-party
medical weights are ever downloaded.

## Starter (local)

```bash
make setup            # .venv + deps + npm install
make data-starter     # ~10 GB transfer, ~3 GB on disk after caching (raw zips are deleted)
make train-starter    # trains + evaluates + exports every starter model (~1.5 h on an M-series laptop)
```

Individual model: `make train MODEL=brain_cls PROFILE=starter`, then `make evaluate export`.
Resume an interrupted run: `python -m ml.train --model chest_densenet121 --resume`.

## Full models on Kaggle's free GPU

1. Create a free account at <https://www.kaggle.com> and **verify your phone number**
   (Settings → Phone verification) — required for GPU and internet in notebooks.
2. Settings → API → **Create New Token**. Save `kaggle.json` to `~/.kaggle/` and `chmod 600 ~/.kaggle/kaggle.json`
   (the newer `~/.kaggle/access_token` format also works).
3. For the pneumonia localiser, open
   <https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules> and click
   **I Understand and Accept**.
4. Run one model or all of them:

```bash
make kaggle-train MODEL=brain_cls      # push → poll → pull → export ONNX
make kaggle-train-all                  # one after another
```

What happens: `ml/kaggle/generate.py` writes `ml/kaggle/<model>/notebook.ipynb` +
`kernel-metadata.json` (`enable_gpu: true`, `dataset_sources` = the Kaggle datasets, so nothing big is
downloaded to your PC). The notebook embeds this repo's `ml/` package, symlinks the mounted datasets,
prepares caches, trains, evaluates and exports. `ml/kaggle/run.py` pushes it, polls
`kaggle kernels status` every minute, downloads `kaggle kernels output`, copies the weights into
`artifacts/<model>/full/` and exports ONNX. Restart the backend (or Settings → Reload) and the app
uses the full weights automatically; the UI badge switches from *Starter* to *Full*.

A quick, cheap check that your account works: `python -m ml.kaggle.run --model brain_cls --smoke`
(starter profile on Kaggle, a few minutes of GPU).

Without a token the command prints manual instructions: upload the notebook (File → Import), add the
datasets from `kernel-metadata.json`, set the accelerator to GPU, *Save & Run All*, download
`/kaggle/working/output/` into `artifacts/`.

### GPU budget (≈30 free GPU hours per week)

| Notebook | Models | Input | Est. GPU hours |
|---|---|---|---|
| chest_ensemble | DenseNet-121 + EfficientNet-B2 + ConvNeXt-Tiny on NIH (112k) | 384 px (512 optional) | 7.0 |
| pneumonia_det | Faster R-CNN on RSNA (26k) | 512 px | 3.0 |
| lung_unet | U-Net (ResNet-34 encoder) on 21k masks | 384 px | 1.5 |
| fracture_det | Faster R-CNN on FracAtlas | 640 px | 1.5 |
| denoiser | DnCNN-17 on 40k chest crops | 256 px | 1.5 |
| fracture_cls | EfficientNet-B0 | 512 px | 1.0 |
| brain_seg | Attention U-Net on LGG | 256 px | 1.0 |
| report_gen | CNN + Transformer decoder on IU X-ray | 224 px | 1.0 |
| brain_cls | EfficientNet-B0 | 320 px | 0.7 |
| router | SmallCNN | 128 px | 0.5 |
| **Total** | | | **≈ 18.7 h** (fits one week's quota) |

Each Kaggle session is limited to 12 h; the largest notebook (chest ensemble) needs ~7 h.

## Evaluation outputs

`ml/evaluate.py` writes `artifacts/<model>/<profile>/metrics.json` + `plots/*.png` + `card.md`.
The chest ensemble additionally writes `calibration.json` (per-member per-label temperatures,
Youden thresholds, energy OOD cut-off) which the backend loads. `python -m ml.metrics_table`
regenerates `docs/METRICS.md` and the README table.

## Optional datasets (off by default)

CheXpert, MURA and VinDr-CXR need free registration and are listed in `ml/data/download.py` as
`optional` sources; download them manually and add a config profile to use them.
