"""
Exploratory Data Analysis for PathMNIST.

Generates class distribution plots, sample image grids, and imbalance statistics.
All figures are saved to results/.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import get_raw_data, CLASS_NAMES

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def plot_class_distribution(output_dir=None):
    """Plot and save class distribution bar chart."""
    out = output_dir or RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    _, labels = get_raw_data("train")
    class_counts = np.bincount(labels, minlength=9)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(9), class_counts, color=sns.color_palette("husl", 9))
    ax.set_xticks(range(9))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Samples")
    ax.set_title("PathMNIST Training Set Class Distribution")

    for bar, count in zip(bars, class_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "class_distribution.png"), dpi=150)
    plt.close()
    print(f"  Saved class distribution plot to {out}/class_distribution.png")
    return class_counts


def plot_sample_images(output_dir=None):
    """Show 3 sample images per class in a grid."""
    out = output_dir or RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    images, labels = get_raw_data("train")

    fig, axes = plt.subplots(9, 3, figsize=(6, 18))
    for cls in range(9):
        idx = np.where(labels == cls)[0][:3]
        for j, i in enumerate(idx):
            axes[cls, j].imshow(images[i])
            axes[cls, j].axis("off")
            if j == 0:
                axes[cls, j].set_title(CLASS_NAMES[cls], fontsize=8, loc="left")

    plt.suptitle("Sample Images per Class (3 each)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "sample_images.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved sample images to {out}/sample_images.png")


def print_imbalance_stats():
    """Print class imbalance statistics."""
    _, labels = get_raw_data("train")
    class_counts = np.bincount(labels, minlength=9)

    print("\n--- Class Imbalance Statistics ---")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:35s}: {class_counts[i]:6d}")

    ratio = class_counts.max() / class_counts.min()
    print(f"\n  Max/Min class ratio: {ratio:.2f}")
    print(f"  Largest class:  {CLASS_NAMES[class_counts.argmax()]} ({class_counts.max()})")
    print(f"  Smallest class: {CLASS_NAMES[class_counts.argmin()]} ({class_counts.min()})")


if __name__ == "__main__":
    print("Running PathMNIST Data Exploration...")
    plot_class_distribution()
    plot_sample_images()
    print_imbalance_stats()
    print("\nDone!")
