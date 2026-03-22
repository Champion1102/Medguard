"""
Data loader for PathMNIST dataset.

Loads PathMNIST (28x28, 9-class colon pathology) using the medmnist API.
Provides train/val/test DataLoaders with configurable transforms,
and computes class weights for handling class imbalance.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import PathMNIST

# PathMNIST class names
CLASS_NAMES = [
    "Adipose", "Background", "Debris", "Lymphocytes",
    "Mucus", "Smooth Muscle", "Normal Colon Mucosa",
    "Cancer-associated Stroma", "Colorectal Adenocarcinoma"
]

DATA_DIR = "data"


def get_transforms(mode="dl"):
    """Return transforms for ML (resize to 64x64 grayscale) or DL (augmentation)."""
    if mode == "ml":
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return train_transform, test_transform


def load_pathmnist(mode="dl", batch_size=64):
    """
    Load PathMNIST dataset and return DataLoaders.

    Args:
        mode: "dl" for deep learning transforms, "ml" for ML (grayscale 64x64)
        batch_size: batch size for DataLoaders

    Returns:
        train_loader, val_loader, test_loader, info dict
    """
    if mode == "ml":
        transform = get_transforms("ml")
        train_dataset = PathMNIST(split="train", transform=transform,
                                  download=True, root=DATA_DIR)
        val_dataset = PathMNIST(split="val", transform=transform,
                                download=True, root=DATA_DIR)
        test_dataset = PathMNIST(split="test", transform=transform,
                                 download=True, root=DATA_DIR)
    else:
        train_transform, test_transform = get_transforms("dl")
        train_dataset = PathMNIST(split="train", transform=train_transform,
                                  download=True, root=DATA_DIR)
        val_dataset = PathMNIST(split="val", transform=test_transform,
                                download=True, root=DATA_DIR)
        test_dataset = PathMNIST(split="test", transform=test_transform,
                                 download=True, root=DATA_DIR)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    info = medmnist.INFO["pathmnist"]
    return train_loader, val_loader, test_loader, info


def compute_class_weights(split="train"):
    """
    Compute inverse-frequency class weights for imbalanced data.

    Returns:
        torch.Tensor of shape (9,) with class weights
    """
    dataset = PathMNIST(split=split, download=True, root=DATA_DIR)
    labels = dataset.labels.flatten()
    class_counts = np.bincount(labels, minlength=9)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts.astype(float))
    return torch.FloatTensor(weights)


def get_raw_data(split="train"):
    """Return raw images and labels as numpy arrays."""
    dataset = PathMNIST(split=split, download=True, root=DATA_DIR)
    return dataset.imgs, dataset.labels.flatten()


if __name__ == "__main__":
    train_loader, val_loader, test_loader, info = load_pathmnist()
    print(f"Dataset: {info['description']}")
    print(f"Number of classes: {info['n_channels']}")
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")
    weights = compute_class_weights()
    print(f"Class weights: {weights}")
