"""
MedGuard Streamlit Dashboard — Phase 1 & Phase 2
Run: streamlit run app.py
"""

import os
import sys
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_loader import get_raw_data, CLASS_NAMES, compute_class_weights
from config import (
    MODELS_DIR, PHASE1_EDA, PHASE1_BASELINE, PHASE1_DL,
    PHASE2_RESULTS, PHASE2_TSNE, PHASE2_ABLATION,
    PHASE3_RESULTS, PHASE3_GRADCAM, PHASE3_DIAGNOSTIC,
    SVM_MODEL_PATH, DENSENET_MODEL_PATH, GMM_MODEL_PATH,
)

st.set_page_config(
    page_title="MedGuard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-top: -10px;
        margin-bottom: 25px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card h2 {
        margin: 0;
        font-size: 2rem;
    }
    .metric-card p {
        margin: 4px 0 0 0;
        font-size: 0.9rem;
        opacity: 0.85;
    }
    .metric-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-orange {
        background: linear-gradient(135deg, #f2994a 0%, #f2c94c 100%);
    }
    .metric-blue {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
    }
    .metric-purple {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .ood-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 12px 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: 600;
    }
    .ood-flagged {
        background: linear-gradient(135deg, #e53935 0%, #ff7043 100%);
        padding: 12px 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Auto-download models on first run ──────────────────────
@st.cache_resource
def _ensure_models():
    from download_models import ensure_assets
    return ensure_assets()

with st.spinner("Checking model files..."):
    _ensure_models()


# ── Cached data loading ────────────────────────────────────
@st.cache_data
def load_dataset():
    train_imgs, train_labels = get_raw_data("train")
    test_imgs, test_labels = get_raw_data("test")
    return train_imgs, train_labels, test_imgs, test_labels


@st.cache_data
def load_results():
    results = {}
    sources = [
        ("baseline", os.path.join(PHASE1_BASELINE, "baseline_results.json"),
         os.path.join("results", "baseline_results.json")),
        ("dl", os.path.join(PHASE1_DL, "dl_results.json"),
         os.path.join("results", "dl_results.json")),
        ("hybrid", os.path.join(PHASE2_RESULTS, "hybrid_results.json"),
         os.path.join("results", "hybrid_results.json")),
        ("comparison", os.path.join(PHASE2_RESULTS, "comparison_results.json"),
         os.path.join("results", "comparison_results.json")),
    ]
    for name, primary, fallback in sources:
        for path in [primary, fallback]:
            if os.path.exists(path):
                with open(path) as f:
                    results[name] = json.load(f)
                break
    return results


@st.cache_resource
def load_svm_model():
    import joblib
    if os.path.exists(SVM_MODEL_PATH):
        data = joblib.load(SVM_MODEL_PATH)
        return data["svm"], data["scaler"]
    return None, None


@st.cache_resource
def load_densenet_model():
    import torch
    import torch.nn as nn
    from torchvision import models

    if not os.path.exists(DENSENET_MODEL_PATH):
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 9))
    model.load_state_dict(torch.load(DENSENET_MODEL_PATH, map_location=device,
                                     weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device


@st.cache_resource
def load_gmm_model():
    import joblib
    if os.path.exists(GMM_MODEL_PATH):
        return joblib.load(GMM_MODEL_PATH)
    return None


@st.cache_resource
def load_densenet_embedding_model():
    import torch
    import torch.nn as nn
    from torchvision import models

    if not os.path.exists(DENSENET_MODEL_PATH):
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 9))
    model.load_state_dict(torch.load(DENSENET_MODEL_PATH, map_location=device,
                                     weights_only=True))

    class _Embedder(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.features = features
        def forward(self, x):
            f = self.features(x)
            f = nn.functional.relu(f, inplace=True)
            f = nn.functional.adaptive_avg_pool2d(f, (1, 1))
            return f

    embedding_model = _Embedder(model.features)
    embedding_model = embedding_model.to(device)
    embedding_model.eval()
    return embedding_model, device


@st.cache_resource
def load_gradcam_model():
    import torch
    import torch.nn as nn
    from torchvision import models
    from src.gradcam import GradCAM

    if not os.path.exists(DENSENET_MODEL_PATH):
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    model = models.densenet121(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 9))
    model.load_state_dict(torch.load(DENSENET_MODEL_PATH, map_location=device,
                                     weights_only=True))
    model = model.to(device)
    model.eval()
    gradcam = GradCAM(model, model.features.denseblock4)
    return gradcam, model, device


def get_gradcam_heatmap(image):
    """Generate Grad-CAM heatmap for a single image."""
    import torch
    from torchvision import transforms

    gradcam, model, device = load_gradcam_model()
    if gradcam is None:
        return None, None

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    pil_img = Image.fromarray(image)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    cam, pred_class, probs = gradcam.generate(img_tensor)
    return cam, probs


def predict_single_svm(image):
    from src.baseline_ml import extract_hog_features
    svm, scaler = load_svm_model()
    if svm is None:
        return None, None
    features = extract_hog_features(image[np.newaxis, ...])
    features_scaled = scaler.transform(features)
    pred = svm.predict(features_scaled)[0]
    decision = svm.decision_function(features_scaled)[0]
    exp_scores = np.exp(decision - decision.max())
    probs = exp_scores / exp_scores.sum()
    return int(pred), probs


def predict_single_densenet(image):
    import torch
    from torchvision import transforms

    model, device = load_densenet_model()
    if model is None:
        return None, None

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    pil_img = Image.fromarray(image)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred = output.argmax(dim=1).item()

    return pred, probs


def get_ood_score(image):
    import torch
    from torchvision import transforms

    gmm = load_gmm_model()
    emb_model, device = load_densenet_embedding_model()
    if gmm is None or emb_model is None:
        return None, None

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    pil_img = Image.fromarray(image)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = emb_model(img_tensor).squeeze(-1).squeeze(-1).cpu().numpy()

    log_likelihood = gmm.score_samples(embedding)[0]
    hybrid_results = load_results().get("hybrid", {})
    threshold = hybrid_results.get("ood_threshold", -999)
    is_ood = log_likelihood < threshold

    return float(log_likelihood), bool(is_ood)


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 MedGuard")
    st.markdown("**Uncertainty-Aware Hybrid Classification for Robust Medical Vision**")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Overview", "Data Exploration", "Model Comparison",
         "OOD Detection", "Diagnostic Analysis",
         "Live Classification", "Technical Details"],
        index=0,
    )

    st.divider()
    st.markdown("**Phase 1** — EDA + SVM + DenseNet121")
    st.markdown("**Phase 2** — Hybrid GMM + OOD Detection")
    st.markdown("**Phase 3** — Ablation + Grad-CAM + Calibration")
    st.divider()
    st.caption("PathMNIST | 9 classes | 107K images")


# ── Load data ──────────────────────────────────────────────
train_imgs, train_labels, test_imgs, test_labels = load_dataset()
results = load_results()


# ═══════════════════════════════════════════════════════════
# PAGE: Overview
# ═══════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown('<p class="main-header">MedGuard Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Uncertainty-Aware Hybrid Classification for Robust Medical Vision</p>',
                unsafe_allow_html=True)

    baseline_acc = results.get("baseline", {}).get("accuracy", 0)
    baseline_f1 = results.get("baseline", {}).get("macro_f1", 0)
    dl_acc = results.get("dl", {}).get("accuracy", 0)
    dl_f1 = results.get("dl", {}).get("macro_f1", 0)
    hybrid_acc = results.get("hybrid", {}).get("accuracy", 0)
    hybrid_f1 = results.get("hybrid", {}).get("macro_f1", 0)
    hybrid_indist_acc = results.get("hybrid", {}).get("in_dist_accuracy", 0)
    ood_rate = results.get("hybrid", {}).get("ood_detection_rate", 0)

    st.markdown("#### Phase 1 — Baseline Models")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card metric-orange">
            <h2>{baseline_acc:.1%}</h2><p>SVM Accuracy</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card metric-orange">
            <h2>{baseline_f1:.1%}</h2><p>SVM Macro F1</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card metric-green">
            <h2>{dl_acc:.1%}</h2><p>DenseNet121 Accuracy</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card metric-green">
            <h2>{dl_f1:.1%}</h2><p>DenseNet121 Macro F1</p></div>""", unsafe_allow_html=True)

    st.markdown("#### Phase 2 — Hybrid GMM + OOD Detection")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card metric-blue">
            <h2>{hybrid_acc:.1%}</h2><p>Hybrid Overall Acc</p></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card metric-blue">
            <h2>{hybrid_f1:.1%}</h2><p>Hybrid Macro F1</p></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card metric-purple">
            <h2>{hybrid_indist_acc:.1%}</h2><p>In-Dist Accuracy</p></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card metric-purple">
            <h2>{ood_rate:.1%}</h2><p>OOD Detection Rate</p></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("### Problem Statement")
        st.markdown("""
        Medical image classification demands not only **high accuracy** but also
        **reliable uncertainty estimation**. A misclassified pathology image can lead
        to incorrect diagnosis with severe patient consequences.

        MedGuard addresses this through a three-model pipeline:
        1. **HOG + SVM** — traditional ML baseline using handcrafted features
        2. **DenseNet121** — fine-tuned deep learning with class-weighted loss
        3. **Hybrid GMM** — Gaussian Mixture Model on CNN embeddings for OOD detection
        """)

    with col_r:
        st.markdown("### Architecture")
        arch_path = os.path.join(PHASE3_RESULTS, "architecture_diagram.png")
        if not os.path.exists(arch_path):
            arch_path = os.path.join("results", "architecture_diagram.png")
        if os.path.exists(arch_path):
            st.image(arch_path, use_container_width=True)
        else:
            st.info("Run `python run.py --phase 3` to generate the architecture diagram.")

    st.markdown("### Dataset: PathMNIST")
    st.markdown(f"""
    | Property | Value |
    |----------|-------|
    | Source | MedMNIST Benchmark |
    | Task | 9-class colon pathology classification |
    | Image size | 28 x 28 pixels |
    | Training samples | {len(train_labels):,} |
    | Test samples | {len(test_labels):,} |
    | Imbalance ratio | {np.bincount(train_labels, minlength=9).max() / np.bincount(train_labels, minlength=9).min():.2f}x |
    """)


# ═══════════════════════════════════════════════════════════
# PAGE: Data Exploration
# ═══════════════════════════════════════════════════════════
elif page == "Data Exploration":
    st.markdown('<p class="main-header">Data Exploration</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">PathMNIST — 9-class colon pathology dataset</p>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Class Distribution", "Sample Images", "Class Details"])

    with tab1:
        class_counts = np.bincount(train_labels, minlength=9)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = sns.color_palette("husl", 9)
        bars = ax.bar(range(9), class_counts, color=colors)
        ax.set_xticks(range(9))
        ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("Number of Samples")
        ax.set_title("Training Set Class Distribution")
        for bar, count in zip(bars, class_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
                    f"{count:,}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        ratio = class_counts.max() / class_counts.min()
        st.info(f"**Imbalance ratio (max/min):** {ratio:.2f}x — "
                f"Largest: {CLASS_NAMES[class_counts.argmax()]} ({class_counts.max():,}) | "
                f"Smallest: {CLASS_NAMES[class_counts.argmin()]} ({class_counts.min():,})")

    with tab2:
        st.markdown("#### 4 random samples per class")
        for cls in range(9):
            st.markdown(f"**{cls}: {CLASS_NAMES[cls]}**")
            idx = np.where(train_labels == cls)[0]
            samples = np.random.RandomState(42).choice(idx, 4, replace=False)
            cols = st.columns(4)
            for j, i in enumerate(samples):
                with cols[j]:
                    st.image(train_imgs[i], width=100, caption=f"#{i}")

    with tab3:
        st.markdown("#### Per-class statistics")
        rows = []
        class_counts = np.bincount(train_labels, minlength=9)
        test_counts = np.bincount(test_labels, minlength=9)
        for i in range(9):
            rows.append({
                "Class ID": i,
                "Name": CLASS_NAMES[i],
                "Train Samples": int(class_counts[i]),
                "Test Samples": int(test_counts[i]),
                "Train %": f"{100 * class_counts[i] / class_counts.sum():.1f}%",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# PAGE: Model Comparison
# ═══════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.markdown('<p class="main-header">Model Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">HOG + SVM vs DenseNet121 vs Hybrid GMM</p>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "SVM Details", "DenseNet121 Details", "Hybrid GMM Details"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            models_list = ["HOG + SVM", "DenseNet121", "Hybrid GMM"]
            accs = [results.get("baseline", {}).get("accuracy", 0),
                    results.get("dl", {}).get("accuracy", 0),
                    results.get("hybrid", {}).get("accuracy", 0)]
            f1s = [results.get("baseline", {}).get("macro_f1", 0),
                   results.get("dl", {}).get("macro_f1", 0),
                   results.get("hybrid", {}).get("macro_f1", 0)]

            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(3)
            w = 0.35
            b1 = ax.bar(x - w / 2, accs, w, label="Accuracy", color="#4CAF50", alpha=0.85)
            b2 = ax.bar(x + w / 2, f1s, w, label="Macro F1", color="#2196F3", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(models_list)
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.set_title("Accuracy vs Macro F1 — All Models")
            ax.grid(axis="y", alpha=0.3)
            for b in b1:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                        f"{b.get_height():.3f}", ha="center", fontsize=10)
            for b in b2:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                        f"{b.get_height():.3f}", ha="center", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### Key Findings")
            if accs[0] > 0 and accs[1] > 0:
                improvement = (accs[1] - accs[0]) / accs[0] * 100
                st.markdown(f"""
                - DenseNet121 achieves **{accs[1]:.1%}** accuracy vs SVM's **{accs[0]:.1%}**
                  (**{improvement:.0f}% relative improvement**)
                - The gap confirms handcrafted features (HOG) cannot capture
                  fine-grained pathology textures
                - Class-weighted loss helps DenseNet121 handle the imbalance
                """)
            if accs[2] > 0:
                hybrid_data = results.get("hybrid", {})
                indist_acc = hybrid_data.get("in_dist_accuracy", 0)
                ood_rate = hybrid_data.get("ood_detection_rate", 0)
                st.markdown(f"""
                **Hybrid GMM (Phase 2):**
                - Overall accuracy: **{accs[2]:.1%}** with OOD safety layer
                - In-distribution accuracy: **{indist_acc:.1%}**
                - Flags **{ood_rate:.1%}** of test samples as OOD/uncertain
                - Maintains classification power while adding uncertainty awareness
                """)

        ablation_path = os.path.join(PHASE2_ABLATION, "ablation_comparison.png")
        if not os.path.exists(ablation_path):
            ablation_path = os.path.join("results", "ablation_comparison.png")
        if os.path.exists(ablation_path):
            st.markdown("### Ablation Study")
            st.image(ablation_path, use_container_width=True)

        if "comparison" in results:
            st.markdown("### Full Comparison Table")
            comp = results["comparison"]
            if isinstance(comp, list):
                rows = []
                for r in comp:
                    auroc_val = r.get("auroc", "N/A")
                    ood_val = r.get("ood_rate", "N/A")
                    rows.append({
                        "Model": r["model"],
                        "Accuracy": f"{r['accuracy']:.4f}" if isinstance(r['accuracy'], float) else r['accuracy'],
                        "Macro F1": f"{r['macro_f1']:.4f}" if isinstance(r['macro_f1'], float) else r['macro_f1'],
                        "AUROC": f"{auroc_val:.4f}" if isinstance(auroc_val, float) else auroc_val,
                        "OOD Rate": f"{ood_val:.4f}" if isinstance(ood_val, float) else ood_val,
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("#### SVM Confusion Matrix")
        cm_path = os.path.join(PHASE1_BASELINE, "baseline_confusion_matrix.png")
        if not os.path.exists(cm_path):
            cm_path = os.path.join("results", "baseline_confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)

        if "baseline" in results and "per_class" in results["baseline"]:
            st.markdown("#### Per-class Performance")
            rows = []
            for name in CLASS_NAMES:
                pc = results["baseline"]["per_class"].get(name, {})
                rows.append({
                    "Class": name,
                    "Precision": f"{pc.get('precision', 0):.3f}",
                    "Recall": f"{pc.get('recall', 0):.3f}",
                    "F1": f"{pc.get('f1-score', 0):.3f}",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("#### DenseNet121 Training Curves")
        curves_path = os.path.join(PHASE1_DL, "dl_training_curves.png")
        if not os.path.exists(curves_path):
            curves_path = os.path.join("results", "dl_training_curves.png")
        if os.path.exists(curves_path):
            st.image(curves_path, use_container_width=True)
        else:
            st.info("Training curves not found. Run `python run.py --phase 1` first.")

        if "dl" in results:
            st.markdown("#### Training Summary")
            dl = results["dl"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Test Accuracy", f"{dl['accuracy']:.4f}")
            col2.metric("Test Macro F1", f"{dl['macro_f1']:.4f}")
            col3.metric("Test Loss", f"{dl.get('test_loss', 'N/A')}")

    with tab4:
        st.markdown("#### Hybrid GMM Results")
        if "hybrid" in results:
            hybrid = results["hybrid"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Overall Accuracy", f"{hybrid['accuracy']:.4f}")
            col2.metric("Overall Macro F1", f"{hybrid['macro_f1']:.4f}")
            col3.metric("In-Dist Accuracy", f"{hybrid['in_dist_accuracy']:.4f}")
            col4.metric("In-Dist F1", f"{hybrid['in_dist_f1']:.4f}")

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("OOD Threshold", f"{hybrid['ood_threshold']:.2f}")
            col2.metric("OOD Samples", f"{hybrid['n_ood_samples']} / {hybrid['n_total_samples']}")
            col3.metric("OOD Detection Rate", f"{hybrid['ood_detection_rate']:.2%}")

            st.markdown("---")
            st.markdown("#### How it works")
            st.markdown("""
            1. DenseNet121 extracts 1024-dim embeddings from the penultimate layer
            2. A 9-component GMM is fit on training embeddings
            3. Test samples are scored via log-likelihood
            4. Samples below the 5th-percentile threshold are flagged as OOD
            5. Flagged samples can be routed for expert review instead of auto-classification
            """)
        else:
            st.info("Hybrid GMM results not available. Run `python run.py --phase 2` first.")


# ═══════════════════════════════════════════════════════════
# PAGE: OOD Detection
# ═══════════════════════════════════════════════════════════
elif page == "OOD Detection":
    st.markdown('<p class="main-header">OOD Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Out-of-Distribution detection via GMM log-likelihood on DenseNet121 embeddings</p>',
                unsafe_allow_html=True)

    if "hybrid" not in results:
        st.warning("Hybrid GMM results not available. Run `python run.py --phase 2` first.")
    else:
        hybrid = results["hybrid"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card metric-blue">
                <h2>{hybrid['n_ood_samples']}</h2><p>OOD Samples Flagged</p></div>""",
                unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card metric-purple">
                <h2>{hybrid['ood_detection_rate']:.1%}</h2><p>OOD Detection Rate</p></div>""",
                unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card metric-green">
                <h2>{hybrid['in_dist_accuracy']:.1%}</h2><p>In-Dist Accuracy</p></div>""",
                unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card metric-green">
                <h2>{hybrid['in_dist_f1']:.1%}</h2><p>In-Dist Macro F1</p></div>""",
                unsafe_allow_html=True)

        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["t-SNE Visualization", "OOD Statistics", "Clinical Implications"])

        with tab1:
            tsne_path = os.path.join(PHASE2_TSNE, "tsne_ood.png")
            if not os.path.exists(tsne_path):
                tsne_path = os.path.join("results", "tsne_ood.png")
            if os.path.exists(tsne_path):
                st.image(tsne_path, use_container_width=True,
                         caption="t-SNE of test embeddings — red crosses mark OOD samples")
            else:
                st.info("t-SNE plot not found. Run `python run.py --phase 2` to generate it.")

        with tab2:
            st.markdown("#### OOD Detection Summary")

            n_total = hybrid["n_total_samples"]
            n_ood = hybrid["n_ood_samples"]
            n_indist = n_total - n_ood

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 6))
                sizes = [n_indist, n_ood]
                labels = [f"In-Distribution\n({n_indist:,})", f"OOD / Uncertain\n({n_ood:,})"]
                colors_pie = ["#38ef7d", "#ef5350"]
                explode = (0, 0.05)
                ax.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                       autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
                ax.set_title("Test Set: In-Distribution vs OOD")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col2:
                st.markdown(f"""
                | Metric | Value |
                |--------|-------|
                | Total test samples | {n_total:,} |
                | In-distribution | {n_indist:,} ({100 * n_indist / n_total:.1f}%) |
                | OOD / uncertain | {n_ood:,} ({100 * n_ood / n_total:.1f}%) |
                | OOD threshold | {hybrid['ood_threshold']:.2f} |
                | In-dist accuracy | {hybrid['in_dist_accuracy']:.4f} |
                | In-dist macro F1 | {hybrid['in_dist_f1']:.4f} |
                | Overall accuracy | {hybrid['accuracy']:.4f} |
                | Overall macro F1 | {hybrid['macro_f1']:.4f} |
                """)

            st.markdown("#### Accuracy Comparison: Overall vs In-Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ["Overall", "In-Distribution Only"]
            acc_vals = [hybrid["accuracy"], hybrid["in_dist_accuracy"]]
            f1_vals = [hybrid["macro_f1"], hybrid["in_dist_f1"]]
            x = np.arange(len(categories))
            w = 0.35
            b1 = ax.bar(x - w / 2, acc_vals, w, label="Accuracy", color="#4CAF50", alpha=0.85)
            b2 = ax.bar(x + w / 2, f1_vals, w, label="Macro F1", color="#2196F3", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.set_title("Model Performance: All Samples vs In-Distribution Only")
            ax.grid(axis="y", alpha=0.3)
            for b in b1:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                        f"{b.get_height():.4f}", ha="center", fontsize=10)
            for b in b2:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                        f"{b.get_height():.4f}", ha="center", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab3:
            st.markdown("#### Why OOD Detection Matters in Medical Imaging")
            st.markdown("""
            In clinical settings, a model that **silently misclassifies** an unusual sample
            is far more dangerous than one that **says "I don't know."**

            MedGuard's hybrid approach provides this safety layer:

            - **High-confidence predictions** (in-distribution) are passed through
              with **{:.1%} accuracy** on trusted samples
            - **Low-confidence predictions** (OOD) are flagged for **expert review**
            - The clinician sees which samples the model is uncertain about,
              enabling **human-in-the-loop** decision making

            This is especially critical for:
            - **Rare tissue types** not well-represented in training data
            - **Artifact-corrupted slides** (e.g., staining issues, tissue folds)
            - **Edge cases** between adjacent tissue classes
            """.format(hybrid["in_dist_accuracy"]))


# ═══════════════════════════════════════════════════════════
# PAGE: Live Classification
# ═══════════════════════════════════════════════════════════
elif page == "Live Classification":
    st.markdown('<p class="main-header">Live Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Pick a test image and see predictions from all models + OOD scoring</p>',
                unsafe_allow_html=True)

    col_ctrl, col_img = st.columns([1, 3])

    with col_ctrl:
        mode = st.radio("Select image", ["Random from test set", "Choose by class"])

        if mode == "Random from test set":
            if st.button("🎲 Random Sample", use_container_width=True):
                st.session_state["sample_idx"] = np.random.randint(0, len(test_labels))
            idx = st.session_state.get("sample_idx", 0)
        else:
            selected_class = st.selectbox("Class", range(9),
                                          format_func=lambda x: f"{x}: {CLASS_NAMES[x]}")
            class_indices = np.where(test_labels == selected_class)[0]
            sample_num = st.slider("Sample #", 0, min(len(class_indices) - 1, 49), 0)
            idx = class_indices[sample_num]

        true_label = test_labels[idx]
        image = test_imgs[idx]

        st.image(Image.fromarray(image).resize((150, 150), Image.NEAREST),
                 caption=f"Test #{idx}", use_container_width=True)
        st.markdown(f"**True label:** {true_label} — {CLASS_NAMES[true_label]}")

        # OOD score panel
        st.markdown("---")
        st.markdown("### 🔍 OOD Assessment")
        ood_score, is_ood = get_ood_score(image)
        if ood_score is not None:
            threshold = results.get("hybrid", {}).get("ood_threshold", 0)
            if is_ood:
                st.markdown(f'<div class="ood-flagged">⚠ OOD / Uncertain</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ood-safe">✓ In-Distribution</div>',
                            unsafe_allow_html=True)
            st.markdown(f"**Log-likelihood:** {ood_score:.2f}")
            st.markdown(f"**Threshold:** {threshold:.2f}")
        else:
            st.info("GMM model not loaded.")

    with col_img:
        col_svm, col_resnet, col_gradcam = st.columns(3)

        with col_svm:
            st.markdown("### SVM Prediction")
            svm_pred, svm_probs = predict_single_svm(image)
            if svm_pred is not None:
                is_correct = svm_pred == true_label
                st.markdown(f"**Predicted:** {svm_pred} — {CLASS_NAMES[svm_pred]}")
                if is_correct:
                    st.success("Correct!")
                else:
                    st.error(f"Wrong! (True: {CLASS_NAMES[true_label]})")

                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#4CAF50" if i == true_label else "#ef5350" if i == svm_pred and not is_correct
                          else "#90CAF9" for i in range(9)]
                ax.barh(range(9), svm_probs, color=colors)
                ax.set_yticks(range(9))
                ax.set_yticklabels([f"{i}: {n[:15]}" for i, n in enumerate(CLASS_NAMES)], fontsize=8)
                ax.set_xlabel("Confidence")
                ax.set_title("SVM Decision Scores")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("SVM model not loaded. Run baseline_ml.py first.")

        with col_resnet:
            st.markdown("### DenseNet121 Prediction")
            dl_pred, dl_probs = predict_single_densenet(image)
            if dl_pred is not None:
                is_correct = dl_pred == true_label
                st.markdown(f"**Predicted:** {dl_pred} — {CLASS_NAMES[dl_pred]}")
                if is_correct:
                    st.success("Correct!")
                else:
                    st.error(f"Wrong! (True: {CLASS_NAMES[true_label]})")

                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#4CAF50" if i == true_label else "#ef5350" if i == dl_pred and not is_correct
                          else "#90CAF9" for i in range(9)]
                ax.barh(range(9), dl_probs, color=colors)
                ax.set_yticks(range(9))
                ax.set_yticklabels([f"{i}: {n[:15]}" for i, n in enumerate(CLASS_NAMES)], fontsize=8)
                ax.set_xlabel("Confidence")
                ax.set_title("DenseNet121 Softmax Probabilities")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown(f"**Top confidence:** {dl_probs.max():.1%}")

                if ood_score is not None and is_ood:
                    st.warning("⚠ This sample is flagged as **OOD** by the Hybrid GMM. "
                               "In a clinical setting, this prediction would be routed for expert review.")
            else:
                st.warning("DenseNet model not loaded. Run dl_model.py first.")

        with col_gradcam:
            st.markdown("### Grad-CAM")
            cam, cam_probs = get_gradcam_heatmap(image)
            if cam is not None:
                img_resized = np.array(Image.fromarray(image).resize((64, 64)))
                img_float = img_resized.astype(float) / 255.0

                import matplotlib.cm as mpl_cm
                heatmap = mpl_cm.jet(cam)[:, :, :3]
                overlay = 0.5 * img_float + 0.5 * heatmap
                overlay = np.clip(overlay, 0, 1)

                fig_cam, ax_cam = plt.subplots(figsize=(4, 4))
                ax_cam.imshow(overlay)
                ax_cam.axis("off")
                pred_cls = int(cam_probs.argmax())
                ax_cam.set_title(f"Attention: {CLASS_NAMES[pred_cls]}\n"
                                 f"(conf: {cam_probs[pred_cls]:.1%})", fontsize=10)
                plt.tight_layout()
                st.pyplot(fig_cam)
                plt.close()

                st.caption("Warm regions = high model attention. "
                           "Verifies the model uses clinically relevant tissue features.")
            else:
                st.info("Grad-CAM not available. Ensure DenseNet121 model exists.")


# ═══════════════════════════════════════════════════════════
# PAGE: Diagnostic Analysis
# ═══════════════════════════════════════════════════════════
elif page == "Diagnostic Analysis":
    st.markdown('<p class="main-header">Diagnostic Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Phase 3 — Ablation study, calibration, and model interpretability</p>',
                unsafe_allow_html=True)

    # Calibration metrics cards
    cal_path = os.path.join(PHASE3_DIAGNOSTIC, "calibration_metrics.json")
    if os.path.exists(cal_path):
        with open(cal_path) as f:
            cal = json.load(f)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card metric-blue">
                <h2>{cal['ece']:.4f}</h2><p>Expected Calibration Error</p></div>""",
                unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card metric-green">
                <h2>{cal['mean_confidence_correct']:.1%}</h2><p>Avg Conf (Correct)</p></div>""",
                unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card metric-orange">
                <h2>{cal['mean_confidence_incorrect']:.1%}</h2><p>Avg Conf (Incorrect)</p></div>""",
                unsafe_allow_html=True)
        with col4:
            st.markdown(f"""<div class="metric-card metric-purple">
                <h2>{cal['accuracy']:.1%}</h2><p>Overall Accuracy</p></div>""",
                unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Per-Class Performance", "Confusion Matrix",
        "Confidence Calibration", "Ablation Summary", "Grad-CAM"
    ])

    with tab1:
        img_path = os.path.join(PHASE3_DIAGNOSTIC, "per_class_performance.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
            pc_path = os.path.join(PHASE3_DIAGNOSTIC, "per_class_metrics.json")
            if os.path.exists(pc_path):
                with open(pc_path) as f:
                    pc = json.load(f)
                rows = [{"Class": name, "Accuracy": f"{m['accuracy']:.4f}",
                         "F1 Score": f"{m['f1']:.4f}"}
                        for name, m in pc.items()]
                st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("Run `python run.py --phase 3` to generate diagnostic results.")

    with tab2:
        img_path = os.path.join(PHASE3_DIAGNOSTIC, "confusion_matrix.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
            st.caption("Left: raw counts. Right: normalized by true class (row-wise). "
                       "Diagonal = correct predictions.")
        else:
            st.info("Run `python run.py --phase 3` to generate the confusion matrix.")

    with tab3:
        img_path = os.path.join(PHASE3_DIAGNOSTIC, "confidence_calibration.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
            st.markdown("""
            **Reliability Diagram** (left): Bars should follow the diagonal for
            a perfectly calibrated model. Bars above the line = underconfident,
            below = overconfident.

            **Confidence Distribution** (right): A well-calibrated model shows high
            confidence for correct predictions and lower confidence for incorrect ones.
            """)
        else:
            st.info("Run `python run.py --phase 3` to generate calibration analysis.")

    with tab4:
        img_path = os.path.join(PHASE3_DIAGNOSTIC, "ablation_summary.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        st.markdown("""
        #### Augmentation Techniques Used

        | Technique | Purpose |
        |-----------|---------|
        | **Mixup** (alpha=0.2) | Regularization via convex combinations of training pairs |
        | **Label Smoothing** (0.1) | Prevents hard-label overfitting, improves calibration |
        | **Stain Augmentation** | ColorJitter + RandomErasing simulates H&E staining variation |
        | **Test-Time Augmentation** | Averages predictions over 10 augmented views |
        """)

    with tab5:
        img_path = os.path.join(PHASE3_GRADCAM, "gradcam_grid.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True,
                     caption="Grad-CAM heatmaps: warm regions indicate where DenseNet121 "
                             "focuses for each tissue class")
            st.markdown("""
            **Why this matters in medical imaging:**
            Grad-CAM verifies that the model attends to clinically relevant tissue
            structures rather than background artifacts. This builds trust in the model's
            predictions and supports clinical adoption.
            """)
        else:
            st.info("Run `python run.py --phase 3` to generate Grad-CAM visualizations.")

        arch_path = os.path.join(PHASE3_RESULTS, "architecture_diagram.png")
        if os.path.exists(arch_path):
            st.markdown("---")
            st.markdown("#### Pipeline Architecture")
            st.image(arch_path, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# PAGE: Technical Details
# ═══════════════════════════════════════════════════════════
elif page == "Technical Details":
    st.markdown('<p class="main-header">Technical Details</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Model architectures, hyperparameters, and methodology</p>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["HOG + SVM", "DenseNet121", "Hybrid GMM", "Project Structure"])

    with tab1:
        st.markdown("""
        ### Histogram of Oriented Gradients (HOG) + SVM

        **Feature Extraction:**
        - Images resized to 64x64 grayscale
        - HOG parameters: 9 orientations, 8x8 pixels/cell, 2x2 cells/block
        - Feature dimension: 1,764

        **Classifier:**
        - Support Vector Machine with RBF (Radial Basis Function) kernel
        - Hyperparameters: C=10, gamma=scale, class_weight=balanced
        - Training on balanced subsample (2,000 per class = 18,000 total)

        **Mathematical Formulation:**

        The RBF kernel:
        """)
        st.latex(r"K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)")
        st.markdown("SVM decision function:")
        st.latex(r"f(\mathbf{x}) = \sum_{i=1}^{N_s} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b")

    with tab2:
        st.markdown("""
        ### Fine-tuned DenseNet121

        **Architecture:**
        - Pretrained on ImageNet (transfer learning)
        - Dense connectivity: each layer receives features from all preceding layers
        - Modified classifier head: Dropout(0.3) → Linear(1024, 9)
        - Input: 64x64 RGB with standard ImageNet normalization

        **Training:**
        - Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
        - Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
        - Early stopping: patience=5 epochs
        - Augmentation: random flip, rotation(15°), color jitter, random erasing

        **Advanced Techniques:**
        - **Mixup** (alpha=0.2): blends image pairs with soft labels for regularization
        - **Label Smoothing** (0.1): softens target distribution to improve calibration
        - **Stain Augmentation**: aggressive ColorJitter simulating H&E variation across labs
        - **Test-Time Augmentation**: averages 10 augmented + 1 clean prediction at inference

        **Class-weighted Cross-Entropy Loss:**
        """)
        st.latex(r"\mathcal{L} = -\sum_{c=1}^{C} w_c \cdot y_c \log(\hat{y}_c), \quad w_c = \frac{N}{C \cdot N_c}")

    with tab3:
        st.markdown("""
        ### Hybrid GMM — Uncertainty-Aware OOD Detection

        **Pipeline:**
        1. Load fine-tuned DenseNet121 and extract 1024-dim penultimate layer embeddings
        2. Fit a 9-component full-covariance GMM on training embeddings
        3. Score test samples via log-likelihood
        4. OOD threshold = 5th percentile of training log-likelihoods
        5. Flag samples below threshold as out-of-distribution

        **GMM Parameters:**
        - Components: 9 (one per class)
        - Covariance type: full
        - Max iterations: 200, n_init: 3

        **t-SNE Visualization:**
        - Perplexity: 30, max iterations: 1,000
        - Max samples: 5,000 (subsampled for visualization)

        **GMM Log-Likelihood:**
        """)
        st.latex(r"\log p(\mathbf{x}) = \log \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)")
        st.markdown("Where $\\pi_k$ are mixture weights, $\\boldsymbol{\\mu}_k$ are component means, "
                    "and $\\boldsymbol{\\Sigma}_k$ are covariance matrices.")
        st.markdown("**OOD Decision Rule:**")
        st.latex(r"\text{is\_ood}(\mathbf{x}) = \mathbb{1}\left[\log p(\mathbf{x}) < \tau\right], \quad \tau = \text{percentile}_5(\{\log p(\mathbf{x}_i)\}_{i=1}^{N_{\text{train}}})")

    with tab4:
        st.markdown("### Project Structure")
        st.code("""
MedGuard/
├── run.py                  # Phased CLI entry point
├── app.py                  # Streamlit dashboard (this!)
├── config.py               # Centralized configuration
├── setup.sh                # One-command project setup
├── requirements.txt
├── src/
│   ├── data_loader.py          # PathMNIST loading + class weights
│   ├── data_exploration.py     # EDA plots
│   ├── baseline_ml.py          # HOG + SVM (Phase 1)
│   ├── dl_model.py             # DenseNet121 fine-tuning (Phase 1)
│   ├── hybrid_gmm.py           # GMM + OOD detection (Phase 2)
│   ├── evaluate.py             # Unified evaluation
│   ├── visualize.py            # Architecture diagrams
│   ├── ablation_study.py       # Model comparison
│   ├── diagnostic_ablation.py  # Per-class + calibration analysis (Phase 3)
│   └── gradcam.py              # Grad-CAM visualizations (Phase 3)
├── models/                 # Saved weights (SVM, DenseNet121, GMM)
├── results/
│   ├── phase1/             # EDA, baseline, DL results
│   ├── phase2/             # Hybrid GMM, t-SNE, ablation
│   └── phase3/             # Diagnostics, Grad-CAM, architecture
└── report/
    └── main.tex            # LaTeX paper
        """, language="text")
