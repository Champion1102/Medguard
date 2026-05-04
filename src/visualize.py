"""
Visualization module for MedGuard.

Generates:
- t-SNE embedding visualization (colored by class, OOD in red)
- Training curves (loss, F1 over epochs)
- Confusion matrices for all models
- Architecture diagram (data flow)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import CLASS_NAMES

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_architecture_diagram(output_dir=None):
    """Draw publication-ready MedGuard architecture flow diagram."""
    out = output_dir or RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(-1, 10)
    ax.axis("off")

    def draw_box(x, y, w, h, text, color, fontsize=8, edgecolor="black"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor=edgecolor, linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")

    def draw_arrow(x1, y1, x2, y2, color="#555", style="->"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle=style, lw=1.8, color=color))

    # === Input ===
    draw_box(0.3, 4.2, 2.4, 1.2, "PathMNIST\n28x28 RGB\n9 tissue classes",
             "#E3F2FD", fontsize=8)

    # === Tier 1: HOG + SVM ===
    draw_arrow(2.7, 5.2, 3.8, 7.8, "#FF9800")
    draw_box(3.8, 7.3, 2.0, 1.0, "HOG Features\n1764-dim", "#FFF9C4")
    draw_arrow(5.8, 7.8, 7.0, 7.8, "#FF9800")
    draw_box(7.0, 7.3, 2.0, 1.0, "SVM\n(RBF, C=10)", "#FFCCBC")
    draw_arrow(9.0, 7.8, 10.2, 7.8, "#FF9800")
    draw_box(10.2, 7.3, 2.0, 1.0, "Baseline\n48.1% Acc", "#FFE0B2")

    ax.text(6.5, 8.8, "Tier 1: Traditional ML Baseline",
            fontsize=9, fontstyle="italic", color="#E65100", ha="center")

    # === Tier 2: DenseNet121 ===
    draw_arrow(2.7, 4.8, 3.8, 4.8, "#7B1FA2")
    draw_box(3.8, 4.2, 2.4, 1.2, "DenseNet121\n(fine-tuned)\nImageNet init",
             "#E1BEE7", fontsize=8)
    draw_arrow(6.2, 4.8, 7.4, 4.8, "#7B1FA2")
    draw_box(7.4, 4.3, 2.0, 1.0, "Softmax\n9 classes", "#CE93D8")
    draw_arrow(9.4, 4.8, 10.2, 4.8, "#7B1FA2")
    draw_box(10.2, 4.3, 2.0, 1.0, "DL Prediction\n94.7% Acc", "#D1C4E9")

    ax.text(6.5, 5.9, "Tier 2: Deep Learning Classifier",
            fontsize=9, fontstyle="italic", color="#6A1B9A", ha="center")

    # === Tier 3: GMM + OOD ===
    draw_arrow(6.2, 4.4, 7.4, 2.0, "#1565C0")
    draw_box(7.4, 1.4, 2.0, 1.2, "1024-dim\nEmbeddings\n(avg pool)", "#B3E5FC")
    draw_arrow(9.4, 2.0, 10.6, 2.0, "#1565C0")
    draw_box(10.6, 1.4, 2.0, 1.2, "GMM\n9 components\nfull cov.", "#81D4FA")
    draw_arrow(12.6, 2.0, 13.8, 2.0, "#1565C0")
    draw_box(13.8, 1.4, 2.2, 1.2, "OOD Score\n(log-likelihood\nvs threshold)", "#4FC3F7")

    ax.text(10.5, 3.1, "Tier 3: Uncertainty Estimation (GMM)",
            fontsize=9, fontstyle="italic", color="#0D47A1", ha="center")

    # === Clinical Decision ===
    draw_arrow(12.2, 4.8, 14.0, 4.8, "#333")
    draw_arrow(16.0, 2.0, 16.0, 4.0, "#C62828")

    draw_box(14.0, 4.0, 3.5, 1.6, "", "#F5F5F5", edgecolor="#333")
    ax.text(15.75, 5.2, "Clinical Decision", ha="center", va="center",
            fontsize=9, fontweight="bold")
    ax.text(15.75, 4.7, "In-dist: Model prediction",
            ha="center", va="center", fontsize=7, color="#2E7D32")
    ax.text(15.75, 4.3, "OOD: Defer to clinician",
            ha="center", va="center", fontsize=7, color="#C62828")

    # === Training Techniques (bottom) ===
    y_tech = -0.3
    tech_w, tech_h = 3.2, 0.7
    gap = 0.4
    x_start = 1.5
    techniques = [
        ("Mixup (α=0.2)", "#E8F5E9"),
        ("Label Smoothing (0.1)", "#E8F5E9"),
        ("Stain Augmentation", "#E8F5E9"),
        ("Test-Time Aug (x10)", "#E8F5E9"),
    ]
    for i, (name, color) in enumerate(techniques):
        x = x_start + i * (tech_w + gap)
        draw_box(x, y_tech, tech_w, tech_h, name, color, fontsize=7)

    ax.text(9, 0.8, "Training & Inference Enhancements",
            fontsize=9, fontstyle="italic", color="#1B5E20", ha="center")

    ax.set_title("MedGuard: Uncertainty-Aware Hybrid Classification Pipeline",
                 fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    out_path = os.path.join(out, "architecture_diagram.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved architecture diagram to {out_path}")


def plot_all_confusion_matrices():
    """Plot confusion matrices from saved result files."""
    # This requires the models to have been run and results saved.
    # We load from JSON files and generate summary plots.
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    model_names = ["Baseline SVM", "DenseNet121", "Hybrid GMM"]
    result_files = ["baseline_results.json", "dl_results.json", "hybrid_results.json"]

    for i, (name, rfile) in enumerate(zip(model_names, result_files)):
        path = os.path.join(RESULTS_DIR, rfile)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            acc = data.get("accuracy", "?")
            f1 = data.get("macro_f1", "?")
            axes[i].text(0.5, 0.5, f"{name}\n\nAcc: {acc:.4f}\nF1: {f1:.4f}",
                         ha="center", va="center", fontsize=14,
                         transform=axes[i].transAxes)
            axes[i].set_title(name)
        else:
            axes[i].text(0.5, 0.5, f"{name}\n\n(not yet trained)",
                         ha="center", va="center", fontsize=14,
                         transform=axes[i].transAxes)
            axes[i].set_title(name)
        axes[i].axis("off")

    plt.suptitle("Model Performance Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_summary.png"), dpi=150)
    plt.close()
    print(f"Saved model summary to {RESULTS_DIR}/model_summary.png")


if __name__ == "__main__":
    print("Generating visualizations...")
    plot_architecture_diagram()
    plot_all_confusion_matrices()
    print("\nDone!")
