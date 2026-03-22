"""
Hybrid GMM model: Gaussian Mixture Model on ResNet18 embeddings for
uncertainty-aware classification and OOD detection.

Pipeline:
1. Load fine-tuned ResNet18 and extract 512-dim penultimate layer embeddings
2. Fit GMM (n_components=9) on training embeddings
3. Score test samples via log-likelihood
4. OOD threshold = 5th percentile of training log-likelihoods
5. Flag samples below threshold as OOD/uncertain

GMM log-likelihood:
    log p(x) = log sum_k [ pi_k * N(x | mu_k, Sigma_k) ]
where pi_k are mixture weights, N is the multivariate Gaussian density.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_pathmnist, CLASS_NAMES

RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")


def load_resnet_backbone():
    """Load fine-tuned ResNet18 and create embedding extractor."""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 9)
    )

    state_dict_path = os.path.join(MODELS_DIR, "resnet18_pathmnist.pth")
    if os.path.exists(state_dict_path):
        model.load_state_dict(torch.load(state_dict_path, map_location=DEVICE,
                                         weights_only=True))
        print("Loaded fine-tuned ResNet18 weights.")
    else:
        print("WARNING: No fine-tuned weights found. Using random initialization.")
        print("Run dl_model.py first to train the ResNet18.")

    # Remove the FC layer to get embeddings
    embedding_model = nn.Sequential(*list(model.children())[:-1])
    embedding_model = embedding_model.to(DEVICE)
    embedding_model.eval()
    return embedding_model, model


@torch.no_grad()
def extract_embeddings(model, loader):
    """Extract 512-dim embeddings from the penultimate layer."""
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        embeddings = model(images).squeeze(-1).squeeze(-1)  # (B, 512)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend(labels.squeeze().numpy())

    return np.vstack(all_embeddings), np.array(all_labels)


def fit_gmm(train_embeddings, n_components=9):
    """Fit Gaussian Mixture Model on training embeddings."""
    print(f"Fitting GMM with {n_components} components on {len(train_embeddings)} samples...")
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          random_state=42, max_iter=200, n_init=3)
    gmm.fit(train_embeddings)
    print(f"GMM converged: {gmm.converged_}")
    return gmm


def detect_ood(gmm, train_embeddings, test_embeddings, percentile=5):
    """
    Detect OOD samples using GMM log-likelihood scores.

    OOD threshold = percentile-th value of training scores.
    Samples with score < threshold are flagged as OOD.
    """
    train_scores = gmm.score_samples(train_embeddings)
    test_scores = gmm.score_samples(test_embeddings)

    threshold = np.percentile(train_scores, percentile)
    is_ood = test_scores < threshold

    print(f"\nOOD Detection (threshold = {percentile}th percentile):")
    print(f"  Threshold: {threshold:.2f}")
    print(f"  Train score range: [{train_scores.min():.2f}, {train_scores.max():.2f}]")
    print(f"  Test score range:  [{test_scores.min():.2f}, {test_scores.max():.2f}]")
    print(f"  OOD samples: {is_ood.sum()} / {len(test_scores)} "
          f"({100 * is_ood.mean():.1f}%)")

    return test_scores, is_ood, threshold


def plot_tsne_ood(embeddings, labels, is_ood, title="t-SNE: In-Distribution vs OOD"):
    """Generate t-SNE plot with OOD points highlighted."""
    print("Computing t-SNE (this may take a minute)...")
    # Subsample for speed if needed
    n = len(embeddings)
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        is_ood = is_ood[idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot in-distribution points colored by class
    in_dist_mask = ~is_ood
    scatter = ax.scatter(coords[in_dist_mask, 0], coords[in_dist_mask, 1],
                         c=labels[in_dist_mask], cmap="tab10", s=8, alpha=0.6)

    # Plot OOD points in red
    if is_ood.any():
        ax.scatter(coords[is_ood, 0], coords[is_ood, 1],
                   c="red", marker="x", s=30, alpha=0.8, label="OOD")

    cbar = plt.colorbar(scatter, ax=ax, ticks=range(9))
    cbar.set_ticklabels(CLASS_NAMES)
    ax.legend(fontsize=10)
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tsne_ood.png"), dpi=150)
    plt.close()
    print(f"Saved t-SNE OOD plot to {RESULTS_DIR}/tsne_ood.png")


def run_hybrid_gmm():
    """Full hybrid GMM pipeline."""
    # Load data
    train_loader, val_loader, test_loader, _ = load_pathmnist(
        mode="dl", batch_size=128)

    # Extract embeddings
    embedding_model, full_model = load_resnet_backbone()
    print("Extracting training embeddings...")
    train_emb, train_labels = extract_embeddings(embedding_model, train_loader)
    print(f"Training embeddings shape: {train_emb.shape}")

    print("Extracting test embeddings...")
    test_emb, test_labels = extract_embeddings(embedding_model, test_loader)
    print(f"Test embeddings shape: {test_emb.shape}")

    # Fit GMM
    gmm = fit_gmm(train_emb, n_components=9)
    joblib.dump(gmm, os.path.join(MODELS_DIR, "hybrid_gmm.pkl"))

    # GMM classification (assign to closest component)
    gmm_preds = gmm.predict(test_emb)

    # OOD detection
    test_scores, is_ood, threshold = detect_ood(gmm, train_emb, test_emb)

    # For in-distribution samples, use the full model's predictions
    full_model = full_model.to(DEVICE)
    full_model.eval()
    _, _, dl_preds, _ = _get_dl_predictions(full_model, test_loader)

    # Hybrid: use DL predictions for in-dist, flag OOD
    hybrid_preds = dl_preds.copy()
    in_dist_acc = accuracy_score(test_labels[~is_ood], hybrid_preds[~is_ood])
    in_dist_f1 = f1_score(test_labels[~is_ood], hybrid_preds[~is_ood], average="macro")
    overall_acc = accuracy_score(test_labels, hybrid_preds)
    overall_f1 = f1_score(test_labels, hybrid_preds, average="macro")

    print(f"\nHybrid GMM Results:")
    print(f"  Overall - Acc: {overall_acc:.4f}, F1: {overall_f1:.4f}")
    print(f"  In-dist - Acc: {in_dist_acc:.4f}, F1: {in_dist_f1:.4f}")
    print(f"  OOD detection rate: {is_ood.mean():.4f}")

    # Save results
    results = {
        "model": "Hybrid GMM (ResNet18 embeddings)",
        "accuracy": float(overall_acc),
        "macro_f1": float(overall_f1),
        "in_dist_accuracy": float(in_dist_acc),
        "in_dist_f1": float(in_dist_f1),
        "ood_detection_rate": float(is_ood.mean()),
        "ood_threshold": float(threshold),
        "n_ood_samples": int(is_ood.sum()),
        "n_total_samples": int(len(is_ood)),
    }
    with open(os.path.join(RESULTS_DIR, "hybrid_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # t-SNE visualization
    plot_tsne_ood(test_emb, test_labels, is_ood)

    return results


@torch.no_grad()
def _get_dl_predictions(model, loader):
    """Get predictions from the full DL model."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.squeeze().numpy())
    return None, None, np.array(all_preds), np.array(all_labels)


if __name__ == "__main__":
    run_hybrid_gmm()
    print("\nDone!")
