"""
Diagnostic ablation study for MedGuard.

Analyzes the contribution of each pipeline component:
1. Per-class accuracy and F1 breakdown
2. Confusion matrix (counts + normalized)
3. Confidence calibration (reliability diagram + ECE)
4. Component contribution summary
5. LaTeX tables for the report
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import CLASS_NAMES, load_pathmnist, compute_class_weights
from src.dl_model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")


@torch.no_grad()
def _evaluate_with_probs(model, loader):
    """Evaluate and return predictions, labels, and softmax probabilities."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.squeeze().numpy())

    return (np.array(all_preds), np.array(all_labels),
            np.vstack(all_probs))


def per_class_analysis(preds, labels, output_dir):
    """Per-class accuracy and F1 bar chart."""
    report = classification_report(labels, preds, target_names=CLASS_NAMES,
                                   output_dict=True)

    per_acc, per_f1 = [], []
    for c in range(9):
        mask = labels == c
        per_acc.append((preds[mask] == labels[mask]).mean())
        per_f1.append(report[CLASS_NAMES[c]]["f1-score"])

    x = np.arange(9)
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, per_acc, width, label="Accuracy",
                   color="#4CAF50", alpha=0.85)
    bars2 = ax.bar(x + width / 2, per_f1, width, label="F1 Score",
                   color="#2196F3", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance Breakdown")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_performance.png"), dpi=150)
    plt.close()

    data = {CLASS_NAMES[i]: {"accuracy": float(per_acc[i]),
                              "f1": float(per_f1[i])}
            for i in range(9)}
    with open(os.path.join(output_dir, "per_class_metrics.json"), "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Saved per-class analysis to {output_dir}/per_class_performance.png")


def plot_confusion_matrix(preds, labels, output_dir):
    """Confusion matrix heatmaps (counts + normalized)."""
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax1.set_title("Confusion Matrix (Counts)")
    ax1.set_ylabel("True Label")
    ax1.set_xlabel("Predicted Label")
    ax1.tick_params(axis="x", rotation=45)
    ax1.tick_params(axis="y", rotation=0)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax2,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax2.set_title("Confusion Matrix (Normalized)")
    ax2.set_ylabel("True Label")
    ax2.set_xlabel("Predicted Label")
    ax2.tick_params(axis="x", rotation=45)
    ax2.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix to {output_dir}/confusion_matrix.png")


def confidence_calibration(preds, labels, probs, output_dir):
    """Reliability diagram, confidence histograms, and ECE."""
    confidences = probs.max(axis=1)
    correct = preds == labels

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(int(mask.sum()))
        else:
            bin_accs.append(0)
            bin_confs.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2
                   for i in range(n_bins)]
    ax1.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color="#2196F3",
            label="Model accuracy")
    ax1.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Reliability Diagram")
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(alpha=0.3)

    ax2.hist(confidences[correct], bins=30, alpha=0.7, color="#4CAF50",
             label="Correct", density=True)
    ax2.hist(confidences[~correct], bins=30, alpha=0.7, color="#F44336",
             label="Incorrect", density=True)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Density")
    ax2.set_title("Confidence Distribution")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_calibration.png"), dpi=150)
    plt.close()

    total = sum(bin_counts)
    ece = (sum(abs(a - c) * n for a, c, n in
               zip(bin_accs, bin_confs, bin_counts)) / total
           if total > 0 else 0)

    cal = {
        "ece": float(ece),
        "mean_confidence_correct": float(confidences[correct].mean()),
        "mean_confidence_incorrect": (float(confidences[~correct].mean())
                                      if (~correct).any() else 0),
        "accuracy": float(correct.mean()),
    }
    with open(os.path.join(output_dir, "calibration_metrics.json"), "w") as f:
        json.dump(cal, f, indent=2)

    print(f"  Saved calibration analysis to {output_dir}/confidence_calibration.png")
    print(f"  ECE: {ece:.4f}")
    return cal


def component_contribution(output_dir):
    """Bar chart comparing pipeline components using saved results."""
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "results")

    rows = []
    bl_path = os.path.join(results_dir, "phase1", "baseline",
                           "baseline_results.json")
    dl_path = os.path.join(results_dir, "phase1", "dl", "dl_results.json")
    hyb_path = os.path.join(results_dir, "phase2", "hybrid_results.json")

    if os.path.exists(bl_path):
        with open(bl_path) as f:
            bl = json.load(f)
        rows.append(("HOG + SVM\n(Baseline)", bl["accuracy"], bl["macro_f1"]))

    if os.path.exists(dl_path):
        with open(dl_path) as f:
            dl = json.load(f)
        no_tta_acc = dl.get("accuracy_no_tta", dl["accuracy"])
        no_tta_f1 = dl.get("macro_f1_no_tta", dl["macro_f1"])
        rows.append(("DenseNet121\n(no TTA)", no_tta_acc, no_tta_f1))
        rows.append(("DenseNet121\n(with TTA)", dl["accuracy"], dl["macro_f1"]))

    if os.path.exists(hyb_path):
        with open(hyb_path) as f:
            hyb = json.load(f)
        rows.append(("Hybrid GMM\n(in-dist only)",
                      hyb["in_dist_accuracy"], hyb["in_dist_f1"]))

    if not rows:
        print("  No results found. Run Phase 1 and 2 first.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r[0] for r in rows]
    accs = [r[1] for r in rows]
    f1s = [r[2] for r in rows]
    x = np.arange(len(rows))
    width = 0.35

    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy",
                   color="#4CAF50", alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1s, width, label="Macro F1",
                   color="#2196F3", alpha=0.85)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Component Contribution: Ablation Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ablation_summary.png"), dpi=150)
    plt.close()
    print(f"  Saved ablation summary to {output_dir}/ablation_summary.png")

    _generate_latex_tables(rows, output_dir)


def _generate_latex_tables(rows, output_dir):
    """Generate LaTeX ablation + augmentation tables."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Diagnostic Ablation: Component Contributions}",
        r"\label{tab:diagnostic_ablation}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Configuration & Accuracy & Macro F1 \\",
        r"\midrule",
    ]
    for name, acc, f1 in rows:
        clean = name.replace("\n", " ")
        lines.append(f"{clean} & {acc:.4f} & {f1:.4f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    with open(os.path.join(output_dir, "ablation_table.tex"), "w") as f:
        f.write("\n".join(lines))

    aug_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Training Augmentation Techniques}",
        r"\label{tab:augmentation}",
        r"\begin{tabular}{lp{7cm}}",
        r"\toprule",
        r"Technique & Purpose \\",
        r"\midrule",
        r"Mixup ($\alpha\!=\!0.2$) & Regularization via convex combinations of training pairs; reduces overconfidence \\",
        r"Label Smoothing (0.1) & Prevents hard-label overfitting; improves calibration \\",
        r"Stain Augmentation & ColorJitter + RandomErasing to simulate H\&E staining variation across labs \\",
        r"Test-Time Augmentation & Averages predictions over 10 augmented views for robust inference \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(os.path.join(output_dir, "augmentation_table.tex"), "w") as f:
        f.write("\n".join(aug_lines))

    print(f"  Saved LaTeX tables to {output_dir}/")


def run_diagnostic_ablation(output_dir=None):
    """Run all diagnostic ablation analyses."""
    from config import MODELS_DIR
    output_dir = output_dir or "results/phase3/diagnostic"
    os.makedirs(output_dir, exist_ok=True)

    print("  Loading DenseNet121 model...")
    model = build_model().to(DEVICE)
    model_path = os.path.join(MODELS_DIR, "densenet121_pathmnist.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                     weights_only=True))
    model.eval()

    _, _, test_loader, _ = load_pathmnist(mode="dl", batch_size=64)

    print("\n  [1/4] Per-class performance analysis...")
    preds, labels, probs = _evaluate_with_probs(model, test_loader)
    per_class_analysis(preds, labels, output_dir)

    print("\n  [2/4] Confusion matrix...")
    plot_confusion_matrix(preds, labels, output_dir)

    print("\n  [3/4] Confidence calibration...")
    confidence_calibration(preds, labels, probs, output_dir)

    print("\n  [4/4] Component contribution summary...")
    component_contribution(output_dir)

    print(f"\n  Diagnostic ablation complete. Results in {output_dir}/")


if __name__ == "__main__":
    run_diagnostic_ablation()
    print("\nDone!")
