"""
MedGuard Entry Point — Phased Execution Controller.

Usage:
    python run.py --phase 1              # Full Phase 1 with training
    python run.py --phase 1 --demo       # Phase 1 demo (loads saved weights, no training)
    python run.py --phase 1 --force      # Force rerun everything
    python run.py --phase 2              # Phase 1 + Phase 2 (hybrid GMM)
"""

import argparse
import os
import sys
import time

from config import (
    PHASE1_EDA, PHASE1_BASELINE, PHASE1_DL,
    PHASE2_TSNE, PHASE2_ABLATION, PHASE2_RESULTS,
    SVM_MODEL_PATH, RESNET_MODEL_PATH, GMM_MODEL_PATH,
    RESULTS_DIR,
)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def phase_has_results(phase_dir):
    """Check if a phase directory has any output files."""
    if not os.path.exists(phase_dir):
        return False
    files = [f for f in os.listdir(phase_dir) if not f.startswith(".")]
    return len(files) > 0


def run_phase1(demo=False, force=False):
    """Phase 1: EDA + SVM Baseline + ResNet18."""
    print("\n" + "=" * 60)
    print("  PHASE 1: BASELINE & EDA")
    print("=" * 60)

    ensure_dirs(PHASE1_EDA, PHASE1_BASELINE, PHASE1_DL)

    # --- Step 1: EDA ---
    if not force and phase_has_results(PHASE1_EDA):
        print("\n[1/3] [CACHED] EDA results found, skipping...")
    else:
        print("\n[1/3] Running EDA...")
        from src.data_exploration import (
            plot_class_distribution, plot_sample_images, print_imbalance_stats
        )
        plot_class_distribution(output_dir=PHASE1_EDA)
        plot_sample_images(output_dir=PHASE1_EDA)
        print_imbalance_stats()

    # --- Step 2: SVM Baseline ---
    if not force and phase_has_results(PHASE1_BASELINE):
        print("\n[2/3] [CACHED] Baseline SVM results found, skipping...")
    elif demo:
        print("\n[2/3] [DEMO] Loading saved SVM model...")
        if not os.path.exists(SVM_MODEL_PATH):
            print("  ERROR: No saved SVM model found. Run without --demo first.")
            return False
        from src.baseline_ml import evaluate_and_save, extract_hog_features
        from src.data_loader import get_raw_data
        import joblib
        data = joblib.load(SVM_MODEL_PATH)
        test_images, test_labels = get_raw_data("test")
        print("  Extracting HOG features for evaluation...")
        X_test = extract_hog_features(test_images)
        evaluate_and_save(data["svm"], data["scaler"], X_test, test_labels,
                          output_dir=PHASE1_BASELINE)
    else:
        print("\n[2/3] Training SVM baseline...")
        from src.baseline_ml import (
            extract_hog_features, train_svm, evaluate_and_save,
            subsample_balanced
        )
        from src.data_loader import get_raw_data
        import joblib

        train_images, train_labels = get_raw_data("train")
        test_images, test_labels = get_raw_data("test")

        print("  Extracting HOG features (training set)...")
        X_train = extract_hog_features(train_images)
        print("  Extracting HOG features (test set)...")
        X_test = extract_hog_features(test_images)

        X_train_sub, y_train_sub = subsample_balanced(X_train, train_labels)
        print(f"  Subsampled: {len(X_train_sub)} from {len(X_train)}")

        svm, scaler = train_svm(X_train_sub, y_train_sub)
        joblib.dump({"svm": svm, "scaler": scaler}, SVM_MODEL_PATH)
        evaluate_and_save(svm, scaler, X_test, test_labels,
                          output_dir=PHASE1_BASELINE)

    # --- Step 3: ResNet18 ---
    if not force and phase_has_results(PHASE1_DL):
        print("\n[3/3] [CACHED] ResNet18 results found, skipping...")
    elif demo:
        print("\n[3/3] [DEMO] Loading saved ResNet18 model...")
        if not os.path.exists(RESNET_MODEL_PATH):
            print("  ERROR: No saved ResNet18 model found. Run without --demo first.")
            return False
        from src.dl_model import evaluate, build_model, plot_training_curves
        from src.data_loader import load_pathmnist, compute_class_weights
        import torch
        import torch.nn as nn
        import json
        from sklearn.metrics import f1_score

        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if torch.backends.mps.is_available()
                              else "cpu")
        model = build_model().to(device)
        model.load_state_dict(torch.load(RESNET_MODEL_PATH, map_location=device,
                                         weights_only=True))

        _, _, test_loader, _ = load_pathmnist(mode="dl", batch_size=64)
        class_weights = compute_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        test_loss, test_f1, test_preds, test_labels = evaluate(
            model, test_loader, criterion)
        test_acc = (test_preds == test_labels).mean()
        print(f"  Test Acc: {test_acc:.4f}, F1: {test_f1:.4f}")

        results = {
            "model": "ResNet18 (fine-tuned)",
            "accuracy": float(test_acc),
            "macro_f1": float(test_f1),
            "test_loss": float(test_loss),
        }
        with open(os.path.join(PHASE1_DL, "dl_results.json"), "w") as f:
            json.dump(results, f, indent=2)
    else:
        print("\n[3/3] Training ResNet18...")
        from src.dl_model import train
        train(output_dir=PHASE1_DL)

    # --- Summary ---
    print_phase1_summary()
    return True


def print_phase1_summary():
    """Print Phase 1 results summary table."""
    import json

    print("\n" + "=" * 60)
    print("  PHASE 1 RESULTS SUMMARY")
    print("=" * 60)

    svm_path = os.path.join(PHASE1_BASELINE, "baseline_results.json")
    dl_path = os.path.join(PHASE1_DL, "dl_results.json")

    print(f"\n  {'Model':<20} {'Accuracy':>10} {'Macro F1':>10}")
    print("  " + "-" * 42)

    for name, path in [("HOG + SVM", svm_path), ("ResNet18", dl_path)]:
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            acc = f"{data['accuracy']:.4f}"
            f1 = f"{data['macro_f1']:.4f}"
            print(f"  {name:<20} {acc:>10} {f1:>10}")
        else:
            print(f"  {name:<20} {'N/A':>10} {'N/A':>10}")

    print()

    # List generated files
    print("  Generated files:")
    for phase_dir in [PHASE1_EDA, PHASE1_BASELINE, PHASE1_DL]:
        if os.path.exists(phase_dir):
            for f in sorted(os.listdir(phase_dir)):
                if not f.startswith("."):
                    print(f"    {os.path.join(phase_dir, f)}")
    print()


def run_phase2(demo=False, force=False):
    """Phase 2: Hybrid GMM + OOD Detection + Ablation."""
    if not run_phase1(demo=demo, force=force):
        return False

    print("\n" + "=" * 60)
    print("  PHASE 2: HYBRID GMM & OOD DETECTION")
    print("=" * 60)

    ensure_dirs(PHASE2_RESULTS, PHASE2_TSNE, PHASE2_ABLATION)

    # --- Step 1: Hybrid GMM ---
    if not force and os.path.exists(os.path.join(PHASE2_RESULTS, "hybrid_results.json")) and os.path.exists(os.path.join(PHASE2_TSNE, "tsne_ood.png")):
        print("\n[1/3] [CACHED] Hybrid GMM results found, skipping...")
    else:
        print("\n[1/3] Running Hybrid GMM pipeline...")
        from src.hybrid_gmm import run_hybrid_gmm
        run_hybrid_gmm(output_dir=PHASE2_RESULTS, tsne_dir=PHASE2_TSNE)

    # --- Step 2: Unified evaluation ---
    if not force and os.path.exists(os.path.join(PHASE2_RESULTS, "comparison_results.json")):
        print("\n[2/3] [CACHED] Evaluation results found, skipping...")
    else:
        print("\n[2/3] Running unified evaluation...")
        from src.evaluate import run_evaluation
        run_evaluation(output_dir=PHASE2_RESULTS)

    # --- Step 3: Ablation study ---
    if not force and phase_has_results(PHASE2_ABLATION):
        print("\n[3/3] [CACHED] Ablation results found, skipping...")
    else:
        print("\n[3/3] Running ablation study...")
        from src.ablation_study import run_ablation
        run_ablation(output_dir=PHASE2_ABLATION)

    # --- Summary ---
    print_phase2_summary()
    return True


def print_phase2_summary():
    """Print Phase 2 results summary."""
    import json

    print("\n" + "=" * 60)
    print("  PHASE 2 RESULTS SUMMARY")
    print("=" * 60)

    hybrid_path = os.path.join(PHASE2_RESULTS, "hybrid_results.json")
    if os.path.exists(hybrid_path):
        with open(hybrid_path) as f:
            data = json.load(f)
        print(f"\n  Hybrid GMM:")
        print(f"    Overall Acc: {data['accuracy']:.4f}, F1: {data['macro_f1']:.4f}")
        print(f"    In-dist Acc: {data['in_dist_accuracy']:.4f}, F1: {data['in_dist_f1']:.4f}")
        print(f"    OOD samples: {data['n_ood_samples']} / {data['n_total_samples']} "
              f"({data['ood_detection_rate']:.1%})")
    else:
        print("\n  Hybrid GMM results not found.")

    print("\n  Generated files:")
    for phase_dir in [PHASE2_RESULTS, PHASE2_TSNE, PHASE2_ABLATION]:
        if os.path.exists(phase_dir):
            for f in sorted(os.listdir(phase_dir)):
                if not f.startswith("."):
                    full = os.path.join(phase_dir, f)
                    if os.path.isfile(full):
                        print(f"    {full}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="MedGuard: Uncertainty-Aware Medical Image Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --phase 1              # Full Phase 1 with training
  python run.py --phase 1 --demo       # Phase 1 demo (no training)
  python run.py --phase 1 --force      # Force rerun everything
  python run.py --phase 2              # Phase 1 + Phase 2
        """
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True,
                        help="Phase to run (1, 2, or 3)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: load saved weights, skip training")
    parser.add_argument("--force", action="store_true",
                        help="Force rerun even if results exist")
    args = parser.parse_args()

    start = time.time()
    print("\n" + "*" * 60)
    print("  MedGuard: Uncertainty-Aware Hybrid Classification")
    print("  for Robust Medical Vision")
    print("*" * 60)

    if args.demo:
        print("  Mode: DEMO (loading saved models, no training)")
    elif args.force:
        print("  Mode: FORCE (rerunning all steps)")
    else:
        print("  Mode: FULL (training if needed)")

    if args.phase == 1:
        run_phase1(demo=args.demo, force=args.force)
    elif args.phase == 2:
        run_phase2(demo=args.demo, force=args.force)
    elif args.phase == 3:
        print("\n  Phase 3 not yet implemented.")

    elapsed = time.time() - start
    print(f"  Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
