# Comparative Analysis of Vector-Level Multimodal Fusion Strategies for Medical Diagnostic Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: Indiana Chest X-ray](https://img.shields.io/badge/Dataset-Indiana_Chest_X--ray-blue)](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.19067913-1682d4?logo=zenodo)](https://doi.org/10.5281/zenodo.19067913)


This repository contains the official implementation for the paper: **"Comparative Analysis of Vector-Level Multimodal Fusion Strategies for Medical Diagnostic Classification"**.

This study provides a rigorous, parameter-balanced comparative analysis of five distinct multimodal fusion architectures for automated binary classification (Normal vs. Abnormal) of chest X-rays using both radiographic images and clinical reports from the Indiana University Chest X-ray Collection.

---

## Table of Contents
- [Overview & Methodology](#Overview-&-Methodology)
- [Dataset](#Dataset)
- [Repository Structure](#Repository-Structure)
- [Key Results](#Key-Results)
- [How to Run](#How-to-Run)
- [Citation](#Citation)

---

## Overview & Methodology

Automated diagnosis from multimodal medical data (imaging + clinical text) is critical for robust clinical decision support. This repository implements a **unified, parameter-balanced framework** comparing five vector-level multimodal fusion strategies:

| Method | Description | Reference |
|--------|-------------|-----------|
| **EarlyConcat** | Simple concatenation of unimodal features (baseline) | Baseline |
| **GMU** | Gated Multimodal Unit with adaptive modality weighting | Arevalo et al., 2017 |
| **BilinearFusion** | Element-wise product of projected features (multiplicative interaction) | - |
| **JointTransformer** | Self-attention over concatenated modality tokens | Vaswani et al., 2017 |
| **LowRankFusion** | Low-rank tensor decomposition for efficient bilinear interaction | Liu et al., 2018 |

### Unified Experimental Framework
To ensure strictly fair comparison, all fusion methods share:

- **Identical frozen encoders:** DenseNet-121 (ImageNet) for visual features, BioClinicalBERT for clinical text embeddings
- **Strict parameter budget:** ~525K trainable parameters per fusion module (max/min ratio 1.05:1)
- **Unified classification head:** Shared 66K-parameter classifier with dropout (0.3)
- **Identical training conditions:** Adam optimizer (lr=1e-4), class-weighted BCE loss, early stopping (patience=5)
- **Reproducible splits:** Stratified 70/15/15 train/val/test with fixed seed (42)

---

## Dataset

Experiments were conducted on the **Indiana University Chest X-ray Collection**, containing paired frontal chest radiographs and free-text radiology reports.

- **Link:** [Indiana Chest X-ray on Kaggle](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
- **Samples:** 3,818 frontal (PA) image-report pairs after preprocessing
- **Class Distribution:** 3,028 abnormal (79.3%), 790 normal (20.7%) — clinically realistic imbalance
- **Splits:** 70% Train (2,673), 15% Validation (572), 15% Test (573) — stratified by label
- **Preprocessing:** 
  - Images: Resized to 224×224, normalized to ImageNet statistics
  - Reports: Tokenized with BioClinicalBERT (max_len=256)

---

## Repository Structure

├── models/ # Saved model checkpoints  
│ ├── EarlyConcat_best.pt  
│ ├── GMU_best.pt  
│ ├── BilinearFusion_best.pt  
│ ├── JointTransformer_best.pt  
│ └── LowRankFusion_best.pt  
│  
├── plots/ # Generated visualizations   
│ ├── EarlyConcat_training_curves.png  
│ ├── GMU_training_curves.png  
│ ├── model_comparison.png   
│ ├── model_comparison_radar_log.png   
│ └── confusion_matrices.png   
│ └── ...  
│  
├── logs.txt    
└── results.json  



---

## Key Results

Our findings establish that **multiplicative interaction methods** (Low-Rank Fusion, Bilinear Fusion) significantly outperform additive and attention-based approaches for vector-level multimodal medical classification under strict parameter constraints.

### Test Set Performance

| Method | AUC [95% CI] | F1-Score [95% CI] | Parameters | Best Epoch |
|--------|--------------|-------------------|------------|------------|
| **LowRankFusion** | **0.970 [0.950–0.987]** | 0.967 [0.956–0.979] | 550,145 | 10 (early stop 16) |
| **BilinearFusion** | 0.967 [0.945–0.986] | **0.971 [0.960–0.982]** | 525,313 | 20 |
| GMU | 0.961 [0.934–0.984] | 0.966 [0.954–0.978] | 522,581 | 10 |
| JointTransformer | 0.957 [0.928–0.983] | 0.970 [0.958–0.981] | 526,977 | 10 (early stop 20) |
| EarlyConcat | 0.955 [0.932–0.976] | 0.949 [0.934–0.964] | 525,057 | 20 |

### Critical Findings

1. **Low-Rank Fusion achieves highest AUC (0.970):** Tensor decomposition effectively captures complex diagnostic correlations with excellent generalization (early stopping at epoch 16 prevents overfitting).

2. **Bilinear Fusion achieves highest F1 (0.971):** Direct multiplicative consensus provides optimal precision-recall balance with remarkable training stability (monotonic improvement across all 20 epochs).

3. **Joint Transformer exhibits severe overfitting:** Despite strong early convergence (val AUC 0.969 at epoch 10), validation performance degrades to 0.964 by epoch 18, resulting in lowest test AUC (0.957) among interaction-based methods. This highlights the risk of attention mechanisms in parameter-constrained regimes.

4. **Parameter balance verified:** All methods operate within 1.05:1 parameter ratio, ensuring mathematical fairness of comparison.

### Training Dynamics Summary

| Method | Peak Val AUC | Final Val AUC | Behavior |
|--------|--------------|---------------|----------|
| LowRankFusion | 0.975 (epoch 10) | 0.973 (epoch 16) | Stable, early stop |
| BilinearFusion | 0.973 (epoch 20) | 0.973 (epoch 20) | Monotonic increase |
| GMU | 0.972 (epoch 20) | 0.972 (epoch 20) | Rapid early convergence |
| JointTransformer | 0.969 (epoch 10) | 0.964 (epoch 18) | **Overfitting after epoch 10** |
| EarlyConcat | 0.959 (epoch 20) | 0.959 (epoch 20) | Slow, steady improvement |

---

## How to Run

### Prerequisites
Ensure you have Python 3.8+ installed along with the required libraries:  
!pip install tensorflow pandas numpy scikit-learn matplotlib seaborn  
### Dataset Setup
Download the dataset from Kaggle and update the DATA_PATH variable inside the .py script of the model you wish to run to point to your local dataset directory.  
### Execution
Navigate to the file.py within the root folder and execute the script.

  
@inproceedings{gherbi2026multimodal,
  author={Gherbi, Djaafer and Benomar, Mohammed Lamine},
  booktitle={2026 International Conference on Speech, Multimodal and Advanced Communication Systems (ICSMACS)}, 
  title={Comparative Analysis of Vector-Level Multimodal Fusion Strategies for Medical Diagnostic Classification}, 
  year={2026},
  pages={6},
  doi={--},
}

For questions or collaborations, please reach out via the provided university email addresses in the paper.
