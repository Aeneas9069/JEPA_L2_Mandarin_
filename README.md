# JEPA-L2_Mandarin_SLA_Phonology

JEPA-L2_Mandarin_SLA_Phonology/
│
├── data/
│   ├── raw_audio/          # Mandarin L2 30-hour corpus (.wav)
│   ├── mfa_aligned/        # TextGrids from Montreal Forced Aligner
│   └── praat_continua/     # Synthesized Tone 2->Tone 3 continua
│
├── src/
│   ├── 01_praat_mfa_extract.py   # Forced alignment and F0/duration extraction
│   ├── 02_jepa_inference.py      # Audio-JEPA latent vector extraction
│   ├── 03_fep_hessian_math.py    # Covariance and Precision matrix calculations
│   └── 04_statistical_lmem.py    # Mixed-effects modeling and ANOVA
│
├── requirements.txt
└── README.md
