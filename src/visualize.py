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


def plot_architecture_diagram():
    """Draw MedGuard architecture flow diagram."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Helper to draw boxes
    def draw_box(x, y, w, h, text, color="#4CAF50", fontsize=9):
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="black",
                                        linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", wrap=True)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#333"))

    # Input
    draw_box(0.5, 3.5, 2.5, 1, "PathMNIST\n28x28, 9 classes", "#E3F2FD")

    # Branch 1: HOG + SVM
    draw_arrow(3, 4, 4, 6.5)
    draw_box(4, 6, 2.2, 1, "HOG Features\n(64x64)", "#FFF9C4")
    draw_arrow(6.2, 6.5, 7.5, 6.5)
    draw_box(7.5, 6, 2.2, 1, "SVM\n(RBF Kernel)", "#FFCCBC")
    draw_arrow(9.7, 6.5, 11, 6.5)
    draw_box(11, 6, 2.5, 1, "Baseline\nPredictions", "#C8E6C9")

    # Branch 2: ResNet18
    draw_arrow(3, 4, 4, 4)
    draw_box(4, 3.5, 2.2, 1, "ResNet18\n(Fine-tuned)", "#E1BEE7")
    draw_arrow(6.2, 4, 7.5, 4)
    draw_box(7.5, 3.5, 2.2, 1, "DL\nPredictions", "#C8E6C9")

    # Branch 3: Embeddings -> GMM
    draw_arrow(6.2, 3.8, 7.5, 1.5)
    draw_box(7.5, 1, 2.2, 1, "512-dim\nEmbeddings", "#B3E5FC")
    draw_arrow(9.7, 1.5, 11, 1.5)
    draw_box(11, 1, 2.2, 1, "GMM\n(9 comp.)", "#FFE0B2")
    draw_arrow(13.2, 1.5, 14, 2.5)
    draw_box(14, 2.2, 1.5, 0.7, "OOD\nDetection", "#EF9A9A")
    draw_arrow(13.2, 1.5, 14, 1)
    draw_box(14, 0.5, 1.5, 0.7, "Uncertainty\nScore", "#FFAB91")

    ax.set_title("MedGuard Architecture: Uncertainty-Aware Hybrid Classification",
                 fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "architecture_diagram.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"Saved architecture diagram to {RESULTS_DIR}/architecture_diagram.png")


def plot_all_confusion_matrices():
    """Plot confusion matrices from saved result files."""
    # This requires the models to have been run and results saved.
    # We load from JSON files and generate summary plots.
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    model_names = ["Baseline SVM", "ResNet18", "Hybrid GMM"]
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
