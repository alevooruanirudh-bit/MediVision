# Datasets

All datasets are free. Licenses and citations are also generated into `ml/data/SOURCES.md` and shown
on the Model Card page. Images are never committed to git; only split CSVs and a handful of
test-split samples (`sample_data/`, see its `SOURCES.md`).

| Use | Dataset | License | Citation | Local (starter) | Kaggle (full) |
|---|---|---|---|---|---|
| Chest 14 labels (training) | ChestMNIST+ 128 px (MedMNIST+) — NIH images, official patient-wise split | CC BY 4.0 | Yang et al., *Sci Data* 2023; Wang et al., CVPR 2017 | ✓ 24k-image subset | — |
| Chest 14 labels (full) | NIH ChestX-ray14 (112,120 images) | CC0 | Wang et al., CVPR 2017 | metadata only | ✓ |
| Chest fairness + demo | NIH ChestX-ray14 random sample (5,606 images, age/sex/view) | CC0 | Wang et al., CVPR 2017 | ✓ | — |
| Lung masks | COVID-19 Radiography Database (QU/Dhaka team, same group as COVID-QU-Ex) | research use, cite | Chowdhury et al., IEEE Access 2020; Rahman et al., CBM 2021 | ✓ 3,100 pairs | ✓ all |
| Lung masks (full extra) | Montgomery + Shenzhen (NLM); COVID-QU-Ex | public / CC BY-SA | Jaeger et al., QIMS 2014; Tahir et al., CBM 2021 | — | ✓ |
| Fracture | FracAtlas (4,083 images, 717 fractured, YOLO boxes) | CC BY 4.0 | Abedeen et al., *Sci Data* 2023 | ✓ | ✓ |
| Brain tumour class | Brain Tumor MRI Dataset (Nickparvar, 4 classes) | CC BY 4.0 | Nickparvar, Kaggle 2021 | ✓ | ✓ |
| Brain tumour masks | LGG MRI Segmentation (110 patients) | CC BY-NC-SA 4.0 | Buda et al., CBM 2019 | ✓ | ✓ |
| Reports | Indiana University CXR (OpenI) — 256 px cleaned copy | CC BY-NC-SA 4.0 (derived) | Demner-Fushman et al., JAMIA 2016 | ✓ 3,298 frontal | ✓ originals |
| Pneumonia boxes | RSNA Pneumonia Detection Challenge | competition rules | Shih et al., Radiol AI 2019 | — (rules not accepted) | ✓ |
| Federated demo | PneumoniaMNIST 64 px | CC BY 4.0 | Kermany et al., Cell 2018 | ✓ | — |
| Head-CT hemorrhage | Head CT – hemorrhage (Kitamura): 200 axial slices, 100 / 100, one per patient | see Kaggle page (research use) | Kitamura, Kaggle 2018 | ✓ 120/40/40 split | ✓ (+ RSNA ICH, rules required) |
| Head-CT hemorrhage (full) | RSNA Intracranial Hemorrhage Detection (~750k slices, 5 subtypes) | competition rules | Flanders et al., Radiol AI 2020 | — | ✓ (rules required) |
| Lung nodule (CT) | NoduleMNIST3D 64³ — LIDC-IDRI nodule patches, benign / malignant (1,633) | CC BY 4.0 | Armato et al., Med Phys 2011; Yang et al. 2023 | ✓ official split | ✓ |
| Breast ultrasound | BreastMNIST+ 128 px — BUSI (780 images, malignant vs benign/normal) | CC BY 4.0 | Al-Dhabyani et al., Data in Brief 2020 | ✓ official split | ✓ |
| Fundus (DR) | RetinaMNIST+ 128 px — DeepDRiD colour fundus (1,600, ICDR 0–4) | CC BY 4.0 | Liu et al., Patterns 2022; Yang et al. 2023 | ✓ official split | ✓ |
| Radiogenomics | LGG `data.csv` — TCGA molecular clusters, grade, survival, sex/age/race codes (110 patients) | CC BY-NC-SA 4.0 | Buda et al., CBM 2019; TCGA, NEJM 2015 | ✓ | ✓ |
| Router "other medical" | OrganAMNIST (abdominal CT), PneumoniaMNIST (paediatric CXR), NoduleMNIST3D mid-slices | CC BY 4.0 | Yang et al. 2023 | ✓ | ✓ |
| Router "not medical" | CIFAR-10 | Krizhevsky 2009 terms | Krizhevsky 2009 | ✓ | ✓ |

**Class imbalance / rare conditions.** Chest labels use Asymmetric Loss; binary models (fracture, hemorrhage,
nodule, breast) use positive-weighted BCE (weight = negatives / positives, capped at 30); multiclass models use
class-weighted cross-entropy; operating thresholds are chosen per label on validation (Youden's J), and the
Lab reports prevalence next to every sensitivity / specificity.

**Demographic codes (LGG).** The Kaggle copy of `data.csv` stores sex and race as numeric codes without a
documented mapping, so fairness tables show the codes ("sex: code 1") rather than guessed categories.

## Splits (no patient leakage)

`ml/data/splits/*.csv` are written by `ml/data/prepare.py` and tested in
`backend/tests/test_ml_core.py::test_no_patient_leakage`.

- **NIH full**: official test list (patient-disjoint by construction) + our 90/10 train/val split
  *grouped by Patient ID*.
- **NIH sample**: images of official-test-list patients are the external fairness set — none of them
  are in the ChestMNIST training split (which is the official train_val list).
- **LGG**: grouped by patient folder. **IU X-ray**: grouped by report UID. **RSNA**: one image per patient.
- **FracAtlas**: the authors' official fracture split + stratified random split of negatives.
- **Brain MRI / lung masks**: no patient IDs are published → stratified random split (documented
  limitation, D-07).

## Disk usage on the build machine

Raw archives are converted to compact `.npy`/`.npz` caches and deleted (`--delete-raw`); the starter
setup needs about 3 GB on disk after preparation.
