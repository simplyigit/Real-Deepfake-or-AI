import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle

import config

def load_data(dataset_path):
    """Loads training and test embeddings and labels."""
    dataset_path = Path(dataset_path)
    print(f"Loading data from {dataset_path}...")
    
    # Train set (augmented preferred if available)
    if (dataset_path / 'X_train_aug.npy').exists() and (dataset_path / 'y_train_aug.npy').exists():
        print("Using augmented training data (X_train_aug.npy)...")
        X_train = np.load(dataset_path / 'X_train_aug.npy')
        y_train = np.load(dataset_path / 'y_train_aug.npy')
    elif (dataset_path / 'X_train_ConvNeXt.npy').exists() and (dataset_path / 'y_train_ConvNeXt.npy').exists():
        print("Using standard training data (X_train_ConvNeXt.npy)...")
        X_train = np.load(dataset_path / 'X_train_ConvNeXt.npy')
        y_train = np.load(dataset_path / 'y_train_ConvNeXt.npy')
    else:
        raise FileNotFoundError(f"No training data found in {dataset_path}")

    # Test set
    if (dataset_path / 'X_test_ConvNeXt.npy').exists() and (dataset_path / 'y_test_ConvNeXt.npy').exists():
        X_test = np.load(dataset_path / 'X_test_ConvNeXt.npy')
        y_test = np.load(dataset_path / 'y_test_ConvNeXt.npy')
    else:
        raise FileNotFoundError(f"No test data found in {dataset_path}")

    # Optional nano-banana augmentation test set
    nano_x_path = dataset_path / 'nano-banana_X_test.npy'
    nano_y_path = dataset_path / 'nano-banana_y_test.npy'
    if nano_x_path.exists() and nano_y_path.exists():
        print("Including nano-banana test set...")
        nano_x = np.load(nano_x_path)
        nano_y = np.load(nano_y_path)
        X_test = np.concatenate([X_test, nano_x], axis=0)
        y_test = np.concatenate([y_test, nano_y], axis=0)

    # Atomic shuffle to keep features and labels synchronized
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    print(f"Train features: {X_train.shape}, labels: {y_train.shape}")
    print(f"Test features:  {X_test.shape}, labels: {y_test.shape}")
    return X_train, y_train, X_test, y_test

def build_classifier(classifier_type="sgd"):
    """Instantiates a classifier according to type."""
    if classifier_type == "sgd":
        return SGDClassifier(loss='hinge', class_weight='balanced', max_iter=1000, random_state=42, tol=1e-3)
    elif classifier_type == "rbf":
        return svm.SVC(kernel='rbf', gamma='scale', random_state=42, C=1.0)
    elif classifier_type == "linear_svm":
        return svm.LinearSVC(random_state=42, C=1.0, max_iter=2000)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

def evaluate_and_plot(model, X_test, y_test, class_names, output_dir, model_name="model"):
    """Computes evaluation metrics and generates plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    y_pred = model.predict(X_test)
    classes = sorted(np.unique(y_test))
    target_names = [class_names.get(c, f"Class {c}") for c in classes]

    print("\n" + "=" * 50)
    print(f"Evaluation Report: {model_name}")
    print("=" * 50)
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print(report)

    # Save text report
    with open(output_dir / f"{model_name}_Classification_Report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix ({model_name})")
    cm_path = output_dir / f"{model_name}_Confusion_Matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix: {cm_path}")

    # ROC Curves (if decision_function or predict_proba exists)
    if hasattr(model, 'decision_function') or hasattr(model, 'predict_proba'):
        if hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_test)
        else:
            y_score = model.predict_proba(X_test)

        y_bin = label_binarize(y_test, classes=classes)
        plt.figure(figsize=(8, 6))
        for i, c in enumerate(classes):
            c_name = class_names.get(c, f"Class {c}")
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{c_name} (AUC = {roc_auc:.3f})")

        plt.plot([0, 1], [0, 1], 'k--', label="Random Chance (AUC = 0.50)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves per Class ({model_name})")
        plt.legend(loc="lower right")
        roc_path = output_dir / f"{model_name}_ROC_Curves.png"
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved ROC curves: {roc_path}")

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Real vs Deepfake vs AI classifiers.")
    parser.add_argument("--classifier", choices=["sgd", "rbf", "linear_svm"], default="sgd",
                        help="Classifier architecture to train or evaluate.")
    parser.add_argument("--train", action="store_true",
                        help="Train a new model instead of loading an existing checkpoint.")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to load or save the model checkpoint.")
    parser.add_argument("--dataset-dir", type=str, default=str(config.DATASET_PATH),
                        help="Path to directory containing precomputed .npy embeddings.")
    parser.add_argument("--output-dir", type=str, default=str(config.OUTPUTS_DIR),
                        help="Directory to save evaluation plots and reports.")
    args = parser.parse_args()

    # Determine model checkpoint path
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        if not args.train:
            # Look for default pretrained model
            model_path = config.resolve_model_path("SGD_ConvNeXt_nano.pkl")
            if not model_path.exists():
                model_path = config.MODELS_DIR / f"{args.classifier}_ConvNeXt.pkl"
        else:
            model_path = config.MODELS_DIR / f"{args.classifier}_ConvNeXt.pkl"

    try:
        X_train, y_train, X_test, y_test = load_data(args.dataset_dir)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    if args.train:
        print(f"\nTraining {args.classifier.upper()} classifier...")
        model = build_classifier(args.classifier)
        model.fit(X_train, y_train)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"Model successfully saved to {model_path}")
    else:
        if not model_path.exists():
            print(f"Checkpoint not found at {model_path}. Training a new {args.classifier.upper()} model...")
            model = build_classifier(args.classifier)
            model.fit(X_train, y_train)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
        else:
            print(f"Loading trained model from {model_path}...")
            model = joblib.load(model_path)

    evaluate_and_plot(model, X_test, y_test, config.CLASS_NAMES, args.output_dir, model_name=model_path.stem)

if __name__ == "__main__":
    main()