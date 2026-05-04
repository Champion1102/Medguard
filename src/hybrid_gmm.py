"""
Hybrid GMM model: Gaussian Mixture Model on DenseNet121 embeddings for
uncertainty-aware classification and OOD detection.

Pipeline:
1. Load fine-tuned DenseNet121 and extract 1024-dim penultimate layer embeddings
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
import time
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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_pathmnist, CLASS_NAMES

from config import RESULTS_DIR, MODELS_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")


class _DenseNetEmbedding(nn.Module):
    """Extract embeddings from DenseNet121 features (before classifier)."""
    def __init__(self, features):
        super().__init__()
        self.features = features

    def forward(self, x):
        f = self.features(x)
        f = nn.functional.relu(f, inplace=True)
        f = nn.functional.adaptive_avg_pool2d(f, (1, 1))
        return f


def load_densenet_backbone():
    """Load fine-tuned DenseNet121 and create embedding extractor."""
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1024, 9)
    )

    state_dict_path = os.path.join(MODELS_DIR, "densenet121_pathmnist.pth")
    if os.path.exists(state_dict_path):
        model.load_state_dict(torch.load(state_dict_path, map_location=DEVICE,
                                         weights_only=True))
        print("Loaded fine-tuned DenseNet121 weights.")
    else:
        print("WARNING: No fine-tuned weights found. Using random initialization.")
        print("Run dl_model.py first to train the DenseNet121.")

    embedding_model = _DenseNetEmbedding(model.features)
    embedding_model = embedding_model.to(DEVICE)
    embedding_model.eval()
    return embedding_model, model


@torch.no_grad()
def extract_embeddings(model, loader):
    """Extract 1024-dim embeddings from the penultimate layer."""
    model.eval()
    all_embeddings = []
    all_labels = []

    for images, labels in tqdm(loader, desc="  Batches", leave=False):
        images = images.to(DEVICE)
        embeddings = model(images).squeeze(-1).squeeze(-1)  # (B, 1024)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend(labels.squeeze().numpy())

    return np.vstack(all_embeddings), np.array(all_labels)


def fit_gmm(train_embeddings, n_components=9):
    """Fit Gaussian Mixture Model on training embeddings."""
    print(f"  Fitting GMM with {n_components} full-covariance components on "
          f"{len(train_embeddings)} x {train_embeddings.shape[1]}-dim embeddings...")
    print(f"  Running 3 random initializations, max 200 EM iterations each...")
    print(f"  (This is CPU-bound and may take 10-20 minutes)")
    t0 = time.time()
    gmm = GaussianMixture(n_components=n_components, covariance_type="full",
                          random_state=42, max_iter=200, n_init=3, verbose=1)
    gmm.fit(train_embeddings)
    elapsed = time.time() - t0
    print(f"  GMM converged: {gmm.converged_} | Iterations: {gmm.n_iter_} | "
          f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
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


def plot_tsne_ood(embeddings, labels, is_ood, output_dir=None,
                  title="t-SNE: In-Distribution vs OOD"):
    """Generate t-SNE plot with OOD points highlighted."""
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    n = len(embeddings)
    if n > 5000:
        idx = np.random.RandomState(42).choice(n, 5000, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]
        is_ood = is_ood[idx]

    print(f"  Computing t-SNE on {len(embeddings)} samples...")
    t0 = time.time()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=1)
    coords = tsne.fit_transform(embeddings)
    print(f"  t-SNE done in {time.time() - t0:.1f}s")

    fig, ax = plt.subplots(figsize=(12, 10))

    in_dist_mask = ~is_ood
    scatter = ax.scatter(coords[in_dist_mask, 0], coords[in_dist_mask, 1],
                         c=labels[in_dist_mask], cmap="tab10", s=8, alpha=0.6)

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
    out_path = os.path.join(output_dir, "tsne_ood.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved t-SNE OOD plot to {out_path}")


def run_hybrid_gmm(output_dir=None, tsne_dir=None):
    """Full hybrid GMM pipeline."""
    output_dir = output_dir or RESULTS_DIR
    tsne_dir = tsne_dir or output_dir
    os.makedirs(output_dir, exist_ok=True)
    pipeline_start = time.time()

    train_loader, val_loader, test_loader, _ = load_pathmnist(
        mode="dl", batch_size=128)

    # Step 1: Load model
    print("\n[Step 1/6] Loading DenseNet121 backbone...")
    embedding_model, full_model = load_densenet_backbone()

    # Step 2: Extract training embeddings
    print("\n[Step 2/6] Extracting training embeddings...")
    t0 = time.time()
    train_emb, train_labels = extract_embeddings(embedding_model, train_loader)
    print(f"  Shape: {train_emb.shape} | Time: {time.time() - t0:.1f}s")

    # Step 3: Extract test embeddings
    print("\n[Step 3/6] Extracting test embeddings...")
    t0 = time.time()
    test_emb, test_labels = extract_embeddings(embedding_model, test_loader)
    print(f"  Shape: {test_emb.shape} | Time: {time.time() - t0:.1f}s")

    # Step 4: Fit GMM
    print("\n[Step 4/6] Fitting Gaussian Mixture Model...")
    gmm = fit_gmm(train_emb, n_components=9)
    joblib.dump(gmm, os.path.join(MODELS_DIR, "hybrid_gmm.pkl"))
    print(f"  Saved GMM to {MODELS_DIR}/hybrid_gmm.pkl")

    # Step 5: OOD detection + classification
    print("\n[Step 5/6] Running OOD detection & classification...")
    t0 = time.time()
    gmm_preds = gmm.predict(test_emb)
    test_scores, is_ood, threshold = detect_ood(gmm, train_emb, test_emb)

    full_model = full_model.to(DEVICE)
    full_model.eval()
    print("  Getting DenseNet121 predictions...")
    _, _, dl_preds, _ = _get_dl_predictions(full_model, test_loader)

    hybrid_preds = dl_preds.copy()
    in_dist_acc = accuracy_score(test_labels[~is_ood], hybrid_preds[~is_ood])
    in_dist_f1 = f1_score(test_labels[~is_ood], hybrid_preds[~is_ood], average="macro")
    overall_acc = accuracy_score(test_labels, hybrid_preds)
    overall_f1 = f1_score(test_labels, hybrid_preds, average="macro")
    print(f"  Time: {time.time() - t0:.1f}s")

    print(f"\n  Hybrid GMM Results:")
    print(f"    Overall  - Acc: {overall_acc:.4f}, F1: {overall_f1:.4f}")
    print(f"    In-dist  - Acc: {in_dist_acc:.4f}, F1: {in_dist_f1:.4f}")
    print(f"    OOD rate - {is_ood.sum()}/{len(is_ood)} ({is_ood.mean():.1%})")

    results = {
        "model": "Hybrid GMM (DenseNet121 embeddings)",
        "accuracy": float(overall_acc),
        "macro_f1": float(overall_f1),
        "in_dist_accuracy": float(in_dist_acc),
        "in_dist_f1": float(in_dist_f1),
        "ood_detection_rate": float(is_ood.mean()),
        "ood_threshold": float(threshold),
        "n_ood_samples": int(is_ood.sum()),
        "n_total_samples": int(len(is_ood)),
    }
    with open(os.path.join(output_dir, "hybrid_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Step 6: t-SNE visualization
    print("\n[Step 6/6] Generating t-SNE visualization...")
    plot_tsne_ood(test_emb, test_labels, is_ood, output_dir=tsne_dir)

    total = time.time() - pipeline_start
    print(f"\n  Phase 2 pipeline complete! Total time: {total:.1f}s ({total/60:.1f} min)")

    return results


@torch.no_grad()
def _get_dl_predictions(model, loader):
    """Get predictions from the full DL model."""
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in tqdm(loader, desc="  Predicting", leave=False):
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.squeeze().numpy())
    return None, None, np.array(all_preds), np.array(all_labels)


if __name__ == "__main__":
    run_hybrid_gmm()
    print("\nDone!")
