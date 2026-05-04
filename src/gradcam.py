"""
Grad-CAM visualization for DenseNet121.

Generates class activation maps showing which image regions the model
focuses on for each prediction. In medical imaging, this verifies
the model uses clinically relevant tissue features.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from
Deep Networks via Gradient-based Localization", ICCV 2017.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import CLASS_NAMES, load_pathmnist
from src.dl_model import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """Return (heatmap_64x64, predicted_class, softmax_probs)."""
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = F.interpolate(cam, size=(64, 64), mode="bilinear",
                            align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()

        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam, target_class, probs


def _denormalize(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def generate_gradcam_grid(model, test_loader, output_dir, num_per_class=2):
    """Grid of original + Grad-CAM overlay for each tissue class."""
    os.makedirs(output_dir, exist_ok=True)
    gradcam = GradCAM(model, model.features.denseblock4)

    samples = {i: [] for i in range(9)}
    for images, labels in test_loader:
        for img, lbl in zip(images, labels):
            c = lbl.item()
            if len(samples[c]) < num_per_class:
                samples[c].append(img)
        if all(len(v) >= num_per_class for v in samples.values()):
            break

    cols = num_per_class * 2
    fig, axes = plt.subplots(9, cols, figsize=(3.5 * cols, 3 * 9))

    for cls in range(9):
        for s in range(num_per_class):
            img_tensor = samples[cls][s].unsqueeze(0).to(DEVICE)
            img_display = _denormalize(samples[cls][s])

            cam, pred, probs = gradcam.generate(img_tensor, target_class=cls)
            conf = probs[cls]

            ax_orig = axes[cls, s * 2]
            ax_cam = axes[cls, s * 2 + 1]

            ax_orig.imshow(img_display)
            ax_orig.set_title(CLASS_NAMES[cls], fontsize=7, fontweight="bold")
            ax_orig.axis("off")

            ax_cam.imshow(img_display)
            ax_cam.imshow(cam, cmap="jet", alpha=0.5)
            ax_cam.set_title(f"conf: {conf:.2f}", fontsize=7)
            ax_cam.axis("off")

    plt.suptitle("Grad-CAM: DenseNet121 Attention per Tissue Class",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "gradcam_grid.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved Grad-CAM grid to {out_path}")


def run_gradcam(output_dir=None):
    """Main entry point for Grad-CAM generation."""
    from config import MODELS_DIR
    output_dir = output_dir or "results/phase3/gradcam"
    os.makedirs(output_dir, exist_ok=True)

    print("  Loading DenseNet121...")
    model = build_model().to(DEVICE)
    model_path = os.path.join(MODELS_DIR, "densenet121_pathmnist.pth")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE,
                                     weights_only=True))

    _, _, test_loader, _ = load_pathmnist(mode="dl", batch_size=32)

    print("  Generating Grad-CAM visualizations...")
    generate_gradcam_grid(model, test_loader, output_dir)


if __name__ == "__main__":
    run_gradcam()
    print("\nDone!")
