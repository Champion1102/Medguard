"""
Centralized configuration for MedGuard project.
All paths, hyperparameters, and dataset settings in one place.
"""

import os

# === Paths ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Results directories per phase
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PHASE1_RESULTS = os.path.join(RESULTS_DIR, "phase1")
PHASE1_EDA = os.path.join(PHASE1_RESULTS, "eda")
PHASE1_BASELINE = os.path.join(PHASE1_RESULTS, "baseline")
PHASE1_DL = os.path.join(PHASE1_RESULTS, "dl")

PHASE2_RESULTS = os.path.join(RESULTS_DIR, "phase2")
PHASE2_TSNE = os.path.join(PHASE2_RESULTS, "tsne")
PHASE2_ABLATION = os.path.join(PHASE2_RESULTS, "ablation")

# Saved model paths
SVM_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_svm.pkl")
RESNET_MODEL_PATH = os.path.join(MODELS_DIR, "resnet18_pathmnist.pth")
GMM_MODEL_PATH = os.path.join(MODELS_DIR, "hybrid_gmm.pkl")

# === Dataset ===
NUM_CLASSES = 9
IMAGE_SIZE = 64
DATASET_NAME = "pathmnist"

# === Baseline ML Hyperparameters ===
SVM_C = 10
SVM_KERNEL = "rbf"
SVM_MAX_PER_CLASS = 2000  # subsample for SVM scalability
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# === Deep Learning Hyperparameters ===
DL_BATCH_SIZE = 64
DL_LEARNING_RATE = 1e-4
DL_WEIGHT_DECAY = 1e-5
DL_EPOCHS = 30
DL_PATIENCE = 5
DL_DROPOUT = 0.3
DL_LR_FACTOR = 0.5
DL_LR_PATIENCE = 2

# === Hybrid GMM Hyperparameters ===
GMM_COMPONENTS = 9
GMM_COVARIANCE_TYPE = "full"
OOD_PERCENTILE = 5
TSNE_PERPLEXITY = 30
TSNE_MAX_ITER = 1000
TSNE_MAX_SAMPLES = 5000
