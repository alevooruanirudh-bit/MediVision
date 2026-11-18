# Sample data sources

Test-split images only (never used for training). Used by `app/seed.py` and the E2E tests.

| File | Source | Original | License | Ground truth |
|---|---|---|---|---|
| chest_followup_prior.png | NIH ChestX-ray14 (official test-list patient) | 00000246_016.png | CC0 1.0 | Consolidation|Mass |
| chest_followup_current.png | NIH ChestX-ray14 (official test-list patient) | 00000246_018.png | CC0 1.0 | No Finding |
| chest_pneumothorax.png | NIH ChestX-ray14 (official test-list patient) | 00000468_005.png | CC0 1.0 | Emphysema|Pneumothorax |
| chest_cardiomegaly_effusion.png | NIH ChestX-ray14 (official test-list patient) | 00001075_046.png | CC0 1.0 | Cardiomegaly|Effusion |
| chest_effusion.png | NIH ChestX-ray14 (official test-list patient) | 00001248_017.png | CC0 1.0 | Effusion |
| chest_cardiomegaly_edema.png | NIH ChestX-ray14 (official test-list patient) | 00004344_037.png | CC0 1.0 | Cardiomegaly|Edema |
| bone_fracture.jpg | FracAtlas (test split) | IMG0003297.jpg | CC BY 4.0 | fracture |
| bone_no_fracture.jpg | FracAtlas (test split) | IMG0000045.jpg | CC BY 4.0 | no fracture |
| brain_glioma.png | Brain Tumor MRI Dataset (Nickparvar), Testing | glioma_Te-gl_101 | CC BY 4.0 | glioma |
| brain_meningioma.png | Brain Tumor MRI Dataset (Nickparvar), Testing | meningioma_Te-aug-me_101 | CC BY 4.0 | meningioma |
| brain_notumor.png | Brain Tumor MRI Dataset (Nickparvar), Testing | notumor_Te-no_101 | CC BY 4.0 | notumor |
| not_medical_photo.png | CIFAR-10 test set (upscaled) | test[13] (horse) | CIFAR-10 terms (Krizhevsky 2009) | not a medical image |
| headct_hemorrhage.png | Head CT - hemorrhage (Kitamura, Kaggle), test split | headct_001 | see Kaggle page (research use) | hemorrhage |
| headct_normal.png | Head CT - hemorrhage (Kitamura, Kaggle), test split | headct_106 | see Kaggle page (research use) | no hemorrhage |
| breast_us_malignant.png | BreastMNIST+ (BUSI, Al-Dhabyani 2020), test split | test[0] | CC BY 4.0 | malignant |
| fundus_dr.png | RetinaMNIST+ (DeepDRiD), test split | test[0] | CC BY 4.0 | ICDR grade 2 |
| ct_nodule_volume.npy | NoduleMNIST3D (LIDC-IDRI), test split — 64³ chest-CT patch | test[7] | CC BY 4.0 | malignant nodule |
| mammo_cancer.png | Mini-DDSM (DDSM screening mammography), test split | 624_A_1000_1.LEFT_CC.LJPEG.1_highpass.png | DDSM - free for research, cite (see Kaggle page) | cancer exam (exam-level label) |
| mammo_normal.png | Mini-DDSM (DDSM screening mammography), test split | 7_A_0003_1.LEFT_CC.LJPEG.1_highpass.png | DDSM - free for research, cite (see Kaggle page) | normal exam |
| petct_sarcoma.npz | TCIA Soft-tissue-Sarcoma FDG-PET/CT (4quant 5 mm release); held out of the deployed model | STS_031, slices 0-55 | CC BY 3.0 | soft-tissue sarcoma (expert contour) |
| chest_ct_scan.npz | LUNA16 (LIDC-IDRI) test-split scan, 90 mm slab around the first annotated nodule | 767743874565 (52 slices) | CC BY 4.0 | nodule, diameter 6.4 mm |

Citations: Wang X. et al. CVPR 2017 (NIH ChestX-ray14); Abedeen I. et al. Sci Data 2023 (FracAtlas); Nickparvar M. Kaggle 2021 (Brain Tumor MRI); Krizhevsky A. 2009 (CIFAR-10).
