# MedGuard: Uncertainty-Aware Hybrid Classification for Robust Medical Vision

## Problem Statement

Medical image classification requires not only high accuracy but also reliable uncertainty estimation. Misclassifying a pathology image can have severe consequences, making it critical to identify when a model is uncertain or encounters out-of-distribution (OOD) samples. This project implements three complementary approaches and combines deep learning embeddings with Gaussian Mixture Models for OOD detection.

## Architecture Overview

```
PathMNIST (28x28, 9 classes)
        |
        +---> [HOG Features] ---> SVM (RBF) ---> Baseline Predictions
        |
        +---> [ResNet18 Fine-tuned] ---> DL Predictions
        |           |
        |           +---> [512-dim Embeddings]
        |                        |
        |                        +---> GMM (9 components)
        |                                    |
        |                                    +---> Log-Likelihood Score
        |                                    +---> OOD Detection (5th percentile threshold)
        |
        +---> Unified Evaluation & Ablation Study
```

### Models

1. **Baseline ML (HOG + SVM):** Histogram of Oriented Gradients features with RBF-kernel Support Vector Machine
2. **Deep Learning (ResNet18):** Fine-tuned pretrained ResNet18 with class-weighted loss and early stopping
3. **Hybrid GMM:** Gaussian Mixture Model fitted on ResNet18 embeddings for uncertainty-aware classification and OOD detection

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Full Phase 1 demo (presentation mode, no retraining)
python run.py --phase 1 --demo

# Full Phase 1 with training
python run.py --phase 1

# Full Phase 2 with training
python run.py --phase 2

# Force rerun everything
python run.py --phase 1 --force
```

### Individual scripts (also runnable standalone)

```bash
python src/data_exploration.py
python src/baseline_ml.py
python src/dl_model.py
python src/hybrid_gmm.py
python src/evaluate.py
python src/ablation_study.py
```

## Results

Phase-specific results are saved to `results/phase1/`, `results/phase2/`, etc. Model weights are saved to `models/`.

## Dataset

PathMNIST from the MedMNIST benchmark: 9-class colon pathology classification with 107,180 images at 28x28 resolution.
