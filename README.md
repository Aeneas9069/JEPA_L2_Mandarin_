# JEPA-Phonology_L2_Mandarin: Precision Phonology in the Latent Space

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the computational pipeline for the **UMONS-JEPA Protocol**, an empirical framework bridging applied linguistics, foundation AI, and cognitive neuroscience. 

By analyzing a 30-hour L2 Mandarin acquisition corpus (featuring a $2 \times 2$ High-Variability Phonetic Training design) through a continuous self-supervised foundation model (`Data2Vec-Audio`), this pipeline calculates the **Free Energy Principle (FEP) Hessian matrix**. This allows researchers to mathematically quantify the cognitive precision of L2 phonological categories independently of purely physical acoustic metrics (like DTW or F0 extraction).

---

## 🧠 Theoretical Framework

Traditional assessments of L2 phonetic acquisition rely on reductionist acoustic measurements. This project argues that human lexical tone categories are stored as high-dimensional, multimodal clusters within a cognitive latent space. 

1. **The Model:** We utilize Meta's `data2vec-audio-large` as a theoretical proxy for Yann LeCun's Joint-Embedding Predictive Architecture (JEPA). Because it predicts continuous latent representations rather than reconstructing raw audio, its latent geometry mirrors entangled human semantic perception.
2. **The Mathematics:** Under the Free Energy Principle, a recognized phonological category is an "attractor state" (an energy minimum). This pipeline calculates the **Hessian matrix** ($H = \Sigma^{-1}$) of the learner's latent vectors. A large Hessian determinant represents high cognitive precision (Native L1 schema), while a small determinant represents ambiguity (L2 learner schema).

---

## 📂 Repository Structure

```text
UMONS-JEPA-Phonology/
│
├── data/
│   ├── raw_audio/          # UMONS 30-hour corpus (.wav) (Not included in repo)
│   ├── mfa_aligned/        # TextGrids from Montreal Forced Aligner
│   └── praat_continua/     # Synthesized Tone 2->Tone 3 continua
│
├── src/
│   ├── 01_praat_mfa_extract.py   # Forced alignment and F0/duration extraction
│   ├── 02_jepa_inference.py      # Data2Vec latent vector extraction
│   ├── 03_fep_hessian_math.py    # Covariance, Shrinkage, and Precision matrix calculations
│   └── 04_statistical_lmem.py    # Mixed-effects modeling and ANOVA
│
├── requirements.txt
└── README.md
