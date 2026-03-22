"""
MedGuard Streamlit Dashboard — Phase 1 Demo
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
    SVM_MODEL_PATH, RESNET_MODEL_PATH,
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached data loading ────────────────────────────────────
@st.cache_data
def load_dataset():
    train_imgs, train_labels = get_raw_data("train")
    test_imgs, test_labels = get_raw_data("test")
    return train_imgs, train_labels, test_imgs, test_labels


@st.cache_data
def load_results():
    results = {}
    for name, path in [("baseline", os.path.join(PHASE1_BASELINE, "baseline_results.json")),
                       ("dl", os.path.join(PHASE1_DL, "dl_results.json"))]:
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
    # Fallback to old results dir
    if "baseline" not in results:
        alt = os.path.join("results", "baseline_results.json")
        if os.path.exists(alt):
            with open(alt) as f:
                results["baseline"] = json.load(f)
    if "dl" not in results:
        alt = os.path.join("results", "dl_results.json")
        if os.path.exists(alt):
            with open(alt) as f:
                results["dl"] = json.load(f)
    return results


@st.cache_resource
def load_svm_model():
    import joblib
    if os.path.exists(SVM_MODEL_PATH):
        data = joblib.load(SVM_MODEL_PATH)
        return data["svm"], data["scaler"]
    return None, None


@st.cache_resource
def load_resnet_model():
    import torch
    import torch.nn as nn
    from torchvision import models

    if not os.path.exists(RESNET_MODEL_PATH):
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 9))
    model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device,
                                     weights_only=True))
    model = model.to(device)
    model.eval()
    return model, device


def predict_single_svm(image):
    """Predict class for a single image using SVM."""
    from src.baseline_ml import extract_hog_features
    svm, scaler = load_svm_model()
    if svm is None:
        return None, None
    features = extract_hog_features(image[np.newaxis, ...])
    features_scaled = scaler.transform(features)
    pred = svm.predict(features_scaled)[0]
    # Get decision function scores as proxy for confidence
    decision = svm.decision_function(features_scaled)[0]
    # Convert to pseudo-probabilities via softmax
    exp_scores = np.exp(decision - decision.max())
    probs = exp_scores / exp_scores.sum()
    return int(pred), probs


def predict_single_resnet(image):
    """Predict class for a single image using ResNet18."""
    import torch
    from torchvision import transforms

    model, device = load_resnet_model()
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


# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 MedGuard")
    st.markdown("**Uncertainty-Aware Hybrid Classification for Robust Medical Vision**")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Overview", "Data Exploration", "Model Comparison",
         "Live Classification", "Technical Details"],
        index=0,
    )

    st.divider()
    st.markdown("**Phase 1** — March 2026")
    st.markdown("EDA + SVM + ResNet18")
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

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card metric-orange">
            <h2>48.2%</h2><p>SVM Accuracy</p></div>""", unsafe_allow_html=True)
    with col2:
        baseline_f1 = results.get("baseline", {}).get("macro_f1", 0.4246)
        st.markdown(f"""<div class="metric-card metric-orange">
            <h2>{baseline_f1:.1%}</h2><p>SVM Macro F1</p></div>""", unsafe_allow_html=True)
    with col3:
        dl_acc = results.get("dl", {}).get("accuracy", 0.9195)
        st.markdown(f"""<div class="metric-card metric-green">
            <h2>{dl_acc:.1%}</h2><p>ResNet18 Accuracy</p></div>""", unsafe_allow_html=True)
    with col4:
        dl_f1 = results.get("dl", {}).get("macro_f1", 0.8949)
        st.markdown(f"""<div class="metric-card metric-green">
            <h2>{dl_f1:.1%}</h2><p>ResNet18 Macro F1</p></div>""", unsafe_allow_html=True)

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
        2. **ResNet18** — fine-tuned deep learning with class-weighted loss
        3. **Hybrid GMM** — Gaussian Mixture Model on CNN embeddings for OOD detection *(Phase 2)*
        """)

    with col_r:
        st.markdown("### Architecture")
        arch_path = os.path.join("results", "architecture_diagram.png")
        if os.path.exists(arch_path):
            st.image(arch_path, use_container_width=True)
        else:
            st.info("Architecture diagram not yet generated. Run `python src/visualize.py`")

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
    st.markdown('<p class="sub-header">HOG + SVM vs Fine-tuned ResNet18</p>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Summary", "SVM Details", "ResNet18 Details"])

    with tab1:
        col1, col2 = st.columns(2)

        # Comparison bar chart
        with col1:
            models_list = ["HOG + SVM", "ResNet18"]
            accs = [results.get("baseline", {}).get("accuracy", 0),
                    results.get("dl", {}).get("accuracy", 0)]
            f1s = [results.get("baseline", {}).get("macro_f1", 0),
                   results.get("dl", {}).get("macro_f1", 0)]

            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(2)
            w = 0.35
            b1 = ax.bar(x - w / 2, accs, w, label="Accuracy", color="#4CAF50", alpha=0.85)
            b2 = ax.bar(x + w / 2, f1s, w, label="Macro F1", color="#2196F3", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(models_list)
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.set_title("Accuracy vs Macro F1")
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
                - ResNet18 achieves **{accs[1]:.1%}** accuracy vs SVM's **{accs[0]:.1%}**
                - That's a **{improvement:.0f}% relative improvement**
                - The gap confirms handcrafted features (HOG) cannot capture
                  fine-grained pathology textures
                - Class-weighted loss helps ResNet handle the 1.63x imbalance
                """)
            st.markdown("### Phase 2 Preview")
            st.markdown("""
            - GMM on ResNet embeddings for **OOD detection**
            - Flag uncertain samples before they reach a clinician
            - Hybrid model maintains ResNet accuracy while adding safety
            """)

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
        st.markdown("#### ResNet18 Training Curves")
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


# ═══════════════════════════════════════════════════════════
# PAGE: Live Classification
# ═══════════════════════════════════════════════════════════
elif page == "Live Classification":
    st.markdown('<p class="main-header">Live Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Pick a test image and see predictions from both models</p>',
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

    with col_img:
        col_svm, col_resnet = st.columns(2)

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

                # Confidence bar chart
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
            st.markdown("### ResNet18 Prediction")
            dl_pred, dl_probs = predict_single_resnet(image)
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
                ax.set_title("ResNet18 Softmax Probabilities")
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.markdown(f"**Top confidence:** {dl_probs.max():.1%}")
            else:
                st.warning("ResNet model not loaded. Run dl_model.py first.")


# ═══════════════════════════════════════════════════════════
# PAGE: Technical Details
# ═══════════════════════════════════════════════════════════
elif page == "Technical Details":
    st.markdown('<p class="main-header">Technical Details</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Model architectures, hyperparameters, and methodology</p>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["HOG + SVM", "ResNet18", "Project Structure"])

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
        ### Fine-tuned ResNet18

        **Architecture:**
        - Pretrained on ImageNet (transfer learning)
        - Modified classifier head: Dropout(0.3) → Linear(512, 9)
        - Input: 64x64 RGB with standard ImageNet normalization

        **Training:**
        - Optimizer: Adam (lr=1e-4, weight_decay=1e-5)
        - Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)
        - Early stopping: patience=5 epochs
        - Augmentation: random flip, rotation(15°), color jitter

        **Class-weighted Cross-Entropy Loss:**
        """)
        st.latex(r"\mathcal{L} = -\sum_{c=1}^{C} w_c \cdot y_c \log(\hat{y}_c), \quad w_c = \frac{N}{C \cdot N_c}")

    with tab3:
        st.markdown("### Project Structure")
        st.code("""
MedGuard/
├── run.py              # Phased CLI entry point
├── app.py              # Streamlit dashboard (this!)
├── config.py           # Centralized configuration
├── requirements.txt
├── src/
│   ├── data_loader.py      # PathMNIST loading + class weights
│   ├── data_exploration.py  # EDA plots
│   ├── baseline_ml.py       # HOG + SVM
│   ├── dl_model.py          # ResNet18 fine-tuning
│   ├── hybrid_gmm.py        # GMM + OOD (Phase 2)
│   ├── evaluate.py          # Unified evaluation
│   ├── visualize.py         # Visualization utils
│   └── ablation_study.py    # Model comparison
├── models/             # Saved weights
├── results/            # Figures + JSON reports
└── report/
    └── main.tex        # LaTeX paper
        """, language="text")
