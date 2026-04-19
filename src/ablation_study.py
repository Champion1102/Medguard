"""
Ablation study: comparison table across all 3 models.

Loads results from saved JSON files and generates:
- LaTeX-formatted comparison table
- Architecture comparison figure
"""

import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_all_results():
    """Load results from all models, checking phase subdirs and root."""
    results = {}
    files = {
        "HOG + SVM": ["phase1/baseline/baseline_results.json", "baseline_results.json"],
        "ResNet18": ["phase1/dl/dl_results.json", "dl_results.json"],
        "Hybrid GMM": ["phase2/hybrid_results.json", "hybrid_results.json"],
    }
    for name, candidates in files.items():
        found = False
        for fname in candidates:
            path = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(path):
                with open(path) as f:
                    results[name] = json.load(f)
                found = True
                break
        if not found:
            print(f"Warning: {name} results not found. Run the corresponding model first.")
    return results


def generate_latex_table(results, output_dir=None):
    """Generate LaTeX-formatted comparison table."""
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    latex = []
    latex.append(r"\begin{table}[h]")
    latex.append(r"\centering")
    latex.append(r"\caption{Model Performance Comparison on PathMNIST}")
    latex.append(r"\label{tab:comparison}")
    latex.append(r"\begin{tabular}{lccc}")
    latex.append(r"\toprule")
    latex.append(r"Model & Accuracy & Macro F1 & OOD Detection \\")
    latex.append(r"\midrule")

    for name, data in results.items():
        acc = f"{data['accuracy']:.4f}"
        f1 = f"{data['macro_f1']:.4f}"
        ood = f"{data.get('ood_detection_rate', 'N/A')}"
        if isinstance(ood, float):
            ood = f"{ood:.4f}"
        latex.append(f"{name} & {acc} & {f1} & {ood} \\\\")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")

    table_str = "\n".join(latex)
    print("\n--- LaTeX Table ---")
    print(table_str)

    out_path = os.path.join(output_dir, "comparison_table.tex")
    with open(out_path, "w") as f:
        f.write(table_str)
    print(f"\nSaved to {out_path}")
    return table_str


def plot_comparison_bar_chart(results, output_dir=None):
    """Generate bar chart comparing model metrics."""
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in models]
    f1_scores = [results[m]["macro_f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, accuracies, width, label="Accuracy",
                   color="#4CAF50", alpha=0.85)
    bars2 = ax.bar(x + width / 2, f1_scores, width, label="Macro F1",
                   color="#2196F3", alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Accuracy vs Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "ablation_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved comparison chart to {out_path}")


def print_summary_table(results):
    """Print formatted summary table to console."""
    print("\n" + "=" * 65)
    print(f"{'Model':<18} {'Accuracy':>10} {'Macro F1':>10} {'OOD Rate':>12}")
    print("-" * 65)
    for name, data in results.items():
        acc = f"{data['accuracy']:.4f}"
        f1 = f"{data['macro_f1']:.4f}"
        ood = data.get("ood_detection_rate", "N/A")
        if isinstance(ood, float):
            ood = f"{ood:.4f}"
        print(f"{name:<18} {acc:>10} {f1:>10} {ood:>12}")
    print("=" * 65)


def run_ablation(output_dir=None):
    """Run full ablation study."""
    print("Running Ablation Study...")
    results = load_all_results()

    if not results:
        print("No results found. Run the models first.")
        return None

    print_summary_table(results)
    generate_latex_table(results, output_dir=output_dir)
    plot_comparison_bar_chart(results, output_dir=output_dir)
    return results


if __name__ == "__main__":
    run_ablation()
    print("\nDone!")
