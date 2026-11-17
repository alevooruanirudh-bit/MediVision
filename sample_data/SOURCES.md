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

Citations: Wang X. et al. CVPR 2017 (NIH ChestX-ray14); Abedeen I. et al. Sci Data 2023 (FracAtlas); Nickparvar M. Kaggle 2021 (Brain Tumor MRI); Krizhevsky A. 2009 (CIFAR-10).
