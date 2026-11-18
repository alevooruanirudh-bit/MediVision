# petct_seg - PET-CT fusion tumour segmenter (ours)

2.5-D U-Net, 6 input channels (3 CT + 3 PET slices), trained from scratch on 7 TCIA
soft-tissue-sarcoma patients (co-registered FDG-PET/CT, 5 mm); the deployed model uses 6
(patient STS_031 is held out as the demo case). Leave-one-patient-out Dice:
- ct: 0.0 (sd 0.0)
- pet: 0.296 (sd 0.265)
- petct: 0.337 (sd 0.256)

Research prototype; tiny cohort; not for clinical use.
