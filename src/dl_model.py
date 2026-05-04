"""
Deep Learning model: Fine-tuned DenseNet121 for PathMNIST classification.

Uses pretrained ImageNet weights, replaces the final classifier with a
dropout + linear head for 9 classes. Training uses class-weighted
CrossEntropyLoss, Adam optimizer, and ReduceLROnPlateau scheduler with
early stopping (patience=5).
"""

import os
import sys
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_pathmnist, compute_class_weights

RESULTS_DIR = "results"
MODELS_DIR = "models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")


def build_model(num_classes=9, dropout=0.3):
    """Build DenseNet121 with custom classifier head."""
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features  # 1024
    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


def mixup_data(x, y, alpha=0.2):
    """Apply Mixup: blend pairs of images and create soft labels."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute Mixup loss as weighted combination."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, mixup_alpha=0.2):
    """Train for one epoch with Mixup augmentation."""
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.squeeze().long().to(DEVICE)

        mixed_images, targets_a, targets_b, lam = mixup_data(
            images, labels, alpha=mixup_alpha)

        optimizer.zero_grad()
        outputs = model(mixed_images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(targets_a.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Evaluate model, return average loss, F1, and predictions."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.squeeze().long().to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, f1, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def evaluate_tta(model, test_dataset, num_augments=10, batch_size=64):
    """Test Time Augmentation: average predictions over multiple augmented views."""
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from medmnist import PathMNIST

    model.eval()

    tta_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15,
                               saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    base_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Original predictions
    base_dataset = PathMNIST(split="test", transform=base_transform,
                             download=True, root="data")
    base_loader = DataLoader(base_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    all_probs = []
    all_labels = []
    for images, labels in base_loader:
        images = images.to(DEVICE)
        probs = torch.softmax(model(images), dim=1).cpu()
        all_probs.append(probs)
        all_labels.extend(labels.squeeze().numpy())
    avg_probs = torch.cat(all_probs, dim=0)

    # Augmented predictions
    for i in range(num_augments):
        aug_dataset = PathMNIST(split="test", transform=tta_transforms,
                                download=True, root="data")
        aug_loader = DataLoader(aug_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=2)
        aug_probs = []
        for images, _ in aug_loader:
            images = images.to(DEVICE)
            probs = torch.softmax(model(images), dim=1).cpu()
            aug_probs.append(probs)
        avg_probs += torch.cat(aug_probs, dim=0)

    avg_probs /= (num_augments + 1)
    preds = avg_probs.argmax(dim=1).numpy()
    labels_arr = np.array(all_labels)
    acc = (preds == labels_arr).mean()
    f1 = f1_score(labels_arr, preds, average="macro")
    return acc, f1, preds, labels_arr


def plot_training_curves(history, output_dir=None):
    """Plot and save training/validation loss and F1 curves."""
    out = output_dir or RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_f1"], "b-", label="Train F1")
    ax2.plot(epochs, history["val_f1"], "r-", label="Val F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Training & Validation F1 Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "dl_training_curves.png"), dpi=150)
    plt.close()
    print(f"  Saved training curves to {out}/dl_training_curves.png")


def train(num_epochs=30, patience=5, lr=1e-4, batch_size=64, output_dir=None):
    """Full training loop with early stopping."""
    out = output_dir or RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    print(f"  Using device: {DEVICE}")

    train_loader, val_loader, test_loader, info = load_pathmnist(
        mode="dl", batch_size=batch_size)
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    # Class-weighted loss with label smoothing
    class_weights = compute_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    model = build_model().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=2)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        train_loss, train_f1 = train_one_epoch(model, train_loader,
                                                criterion, optimizer)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"  Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best model
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, os.path.join(MODELS_DIR, "densenet121_pathmnist.pth"))
    print(f"  Saved best model to {MODELS_DIR}/densenet121_pathmnist.pth")

    # Test evaluation (standard)
    test_loss, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, criterion)
    test_acc = (test_preds == test_labels).mean()
    print(f"\n  Test Results (standard): Acc={test_acc:.4f}, F1={test_f1:.4f}, Loss={test_loss:.4f}")

    # Test evaluation (TTA)
    print("  Running Test Time Augmentation (10 augmented views)...")
    tta_acc, tta_f1, tta_preds, _ = evaluate_tta(model, None, num_augments=10,
                                                  batch_size=batch_size)
    print(f"  Test Results (TTA):      Acc={tta_acc:.4f}, F1={tta_f1:.4f}")

    # Use TTA results as primary
    results = {
        "model": "DenseNet121 (fine-tuned)",
        "accuracy": float(tta_acc),
        "macro_f1": float(tta_f1),
        "accuracy_no_tta": float(test_acc),
        "macro_f1_no_tta": float(test_f1),
        "test_loss": float(test_loss),
        "best_epoch": len(history["train_loss"]) - epochs_no_improve,
        "total_epochs": len(history["train_loss"]),
    }
    with open(os.path.join(out, "dl_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plot_training_curves(history, output_dir=out)
    return model, history


if __name__ == "__main__":
    train()
    print("\nDone!")
