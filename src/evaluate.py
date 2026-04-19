"""
Unified evaluation across all 3 models on the same test set.

Computes Accuracy, Macro F1, AUROC, and OOD detection rate (hybrid only).
Outputs a comparison table and saves results as JSON/CSV.
"""

import os
import sys
import json
import csv
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_pathmnist, get_raw_data, CLASS_NAMES
from src.baseline_ml import extract_hog_features

from config import RESULTS_DIR, MODELS_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")


def evaluate_baseline():
    """Evaluate baseline SVM on test set."""
    model_path = os.path.join(MODELS_DIR, "baseline_svm.pkl")
    if not os.path.exists(model_path):
        print("Baseline SVM model not found. Run baseline_ml.py first.")
        return None

    data = joblib.load(model_path)
    svm, scaler = data["svm"], data["scaler"]

    test_images, test_labels = get_raw_data("test")
    X_test = extract_hog_features(test_images)
    X_test_scaled = scaler.transform(X_test)

    y_pred = svm.predict(X_test_scaled)
    acc = accuracy_score(test_labels, y_pred)
    f1 = f1_score(test_labels, y_pred, average="macro")

    # AUROC (one-vs-rest using decision function)
    try:
        decision = svm.decision_function(X_test_scaled)
        y_bin = label_binarize(test_labels, classes=range(9))
        auroc = roc_auc_score(y_bin, decision, multi_class="ovr", average="macro")
    except Exception:
        auroc = None

    return {"model": "HOG + SVM", "accuracy": acc, "macro_f1": f1,
            "auroc": auroc, "ood_rate": "N/A"}


def evaluate_dl():
    """Evaluate ResNet18 on test set."""
    model_path = os.path.join(MODELS_DIR, "resnet18_pathmnist.pth")
    if not os.path.exists(model_path):
        print("ResNet18 model not found. Run dl_model.py first.")
        return None

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 9))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                     weights_only=True))
    model = model.to(DEVICE)
    model.eval()

    _, _, test_loader, _ = load_pathmnist(mode="dl", batch_size=128)

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.squeeze().numpy())
            all_probs.append(probs)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    y_bin = label_binarize(all_labels, classes=range(9))
    auroc = roc_auc_score(y_bin, all_probs, multi_class="ovr", average="macro")

    return {"model": "ResNet18", "accuracy": acc, "macro_f1": f1,
            "auroc": auroc, "ood_rate": "N/A"}


def evaluate_hybrid():
    """Load hybrid GMM results from saved JSON."""
    candidates = [
        os.path.join(RESULTS_DIR, "phase2", "hybrid_results.json"),
        os.path.join(RESULTS_DIR, "hybrid_results.json"),
    ]
    results_path = None
    for path in candidates:
        if os.path.exists(path):
            results_path = path
            break

    if results_path is None:
        print("Hybrid GMM results not found. Run hybrid_gmm.py first.")
        return None

    with open(results_path) as f:
        data = json.load(f)

    return {"model": "Hybrid GMM", "accuracy": data["accuracy"],
            "macro_f1": data["macro_f1"], "auroc": "N/A",
            "ood_rate": data["ood_detection_rate"]}


def print_comparison_table(results):
    """Print formatted comparison table."""
    print("\n" + "=" * 75)
    print(f"{'Model':<20} {'Accuracy':>10} {'Macro F1':>10} {'AUROC':>10} {'OOD Rate':>10}")
    print("-" * 75)
    for r in results:
        acc = f"{r['accuracy']:.4f}" if isinstance(r['accuracy'], float) else r['accuracy']
        f1 = f"{r['macro_f1']:.4f}" if isinstance(r['macro_f1'], float) else r['macro_f1']
        auroc = f"{r['auroc']:.4f}" if isinstance(r['auroc'], float) else r['auroc']
        ood = f"{r['ood_rate']:.4f}" if isinstance(r['ood_rate'], float) else r['ood_rate']
        print(f"{r['model']:<20} {acc:>10} {f1:>10} {auroc:>10} {ood:>10}")
    print("=" * 75)


def save_results(results, output_dir=None):
    """Save comparison as JSON and CSV."""
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(os.path.join(output_dir, "comparison_results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "macro_f1",
                                                "auroc", "ood_rate"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved to {output_dir}/comparison_results.json and .csv")


def run_evaluation(output_dir=None):
    """Run unified evaluation across all models."""
    print("Running unified evaluation across all models...")
    results = []

    for eval_fn, name in [(evaluate_baseline, "Baseline"),
                          (evaluate_dl, "DL"),
                          (evaluate_hybrid, "Hybrid")]:
        print(f"\nEvaluating {name}...")
        r = eval_fn()
        if r:
            results.append(r)

    if results:
        print_comparison_table(results)
        save_results(results, output_dir=output_dir)

    return results


if __name__ == "__main__":
    run_evaluation()
    print("\nDone!")
