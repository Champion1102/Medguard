"""
Baseline ML model: HOG + SVM (RBF kernel) for PathMNIST classification.

HOG (Histogram of Oriented Gradients):
    Computes gradient magnitudes and orientations in local image cells,
    then bins them into histograms. This captures edge and texture information
    invariant to illumination changes. For each cell of size (pixels_per_cell),
    gradients are computed as:
        Gx = I(x+1, y) - I(x-1, y)
        Gy = I(x, y+1) - I(x, y-1)
        magnitude = sqrt(Gx^2 + Gy^2)
        orientation = arctan(Gy / Gx)
    Orientations are binned into `orientations` bins per cell, then
    block-normalized for contrast invariance.

SVM with RBF kernel:
    The decision function for RBF-kernel SVM is:
        f(x) = sum_i(alpha_i * y_i * K(x_i, x)) + b
    where the RBF kernel is:
        K(x, x') = exp(-gamma * ||x - x'||^2)
    gamma controls the influence radius of each support vector.
    Larger gamma = tighter decision boundary (risk of overfitting).
"""

import os
import sys
import json
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import get_raw_data, CLASS_NAMES

RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def extract_hog_features(images):
    """Extract HOG features from a batch of images."""
    features = []
    for img in images:
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = np.mean(img, axis=2).astype(np.uint8)
        else:
            gray = img

        gray = resize(gray, (64, 64), anti_aliasing=True)

        feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), feature_vector=True)
        features.append(feat)
    return np.array(features)


def train_svm(X_train, y_train):
    """Train SVM with RBF kernel."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print("Training SVM (RBF kernel)... This may take a few minutes.")
    svm = SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced",
              random_state=42, verbose=False)
    svm.fit(X_train_scaled, y_train)

    return svm, scaler


def evaluate_and_save(svm, scaler, X_test, y_test):
    """Evaluate SVM and save results."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = svm.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES,
                                   output_dict=True)

    print(f"\nBaseline SVM Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Baseline SVM Confusion Matrix (Acc={acc:.3f})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "baseline_confusion_matrix.png"), dpi=150)
    plt.close()

    # Save results JSON
    results = {
        "model": "HOG + SVM (RBF)",
        "accuracy": float(acc),
        "macro_f1": float(f1_macro),
        "per_class": {name: {
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1-score": report[name]["f1-score"],
        } for name in CLASS_NAMES}
    }
    with open(os.path.join(RESULTS_DIR, "baseline_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved confusion matrix to {RESULTS_DIR}/baseline_confusion_matrix.png")
    print(f"Saved results to {RESULTS_DIR}/baseline_results.json")
    return results


def subsample_balanced(X, y, max_per_class=2000, seed=42):
    """Subsample to at most max_per_class samples per class for SVM scalability."""
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        if len(c_idx) > max_per_class:
            c_idx = rng.choice(c_idx, max_per_class, replace=False)
        idx.extend(c_idx)
    idx = np.sort(idx)
    return X[idx], y[idx]


if __name__ == "__main__":
    print("Loading PathMNIST data...")
    train_images, train_labels = get_raw_data("train")
    test_images, test_labels = get_raw_data("test")

    print(f"Extracting HOG features from {len(train_images)} training images...")
    X_train = extract_hog_features(train_images)
    print(f"Extracting HOG features from {len(test_images)} test images...")
    X_test = extract_hog_features(test_images)

    print(f"HOG feature dimension: {X_train.shape[1]}")

    # Subsample training data for SVM scalability (RBF SVM is O(n^2) to O(n^3))
    X_train_sub, y_train_sub = subsample_balanced(X_train, train_labels, max_per_class=2000)
    print(f"Subsampled training set: {len(X_train_sub)} samples (from {len(X_train)})")

    svm, scaler = train_svm(X_train_sub, y_train_sub)

    # Save model
    joblib.dump({"svm": svm, "scaler": scaler},
                os.path.join(MODELS_DIR, "baseline_svm.pkl"))

    evaluate_and_save(svm, scaler, X_test, test_labels)
    print("\nDone!")
