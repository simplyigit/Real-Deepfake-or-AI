import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm

import config

def load_embeddings(dataset_path):
    """Loads training embeddings for dimensionality reduction."""
    dataset_path = Path(dataset_path)
    if (dataset_path / 'X_train_aug.npy').exists() and (dataset_path / 'y_train_aug.npy').exists():
        print("Loading X_train_aug.npy...")
        X = np.load(dataset_path / 'X_train_aug.npy')
        y = np.load(dataset_path / 'y_train_aug.npy')
    elif (dataset_path / 'X_train_ConvNeXt.npy').exists() and (dataset_path / 'y_train_ConvNeXt.npy').exists():
        print("Loading X_train_ConvNeXt.npy...")
        X = np.load(dataset_path / 'X_train_ConvNeXt.npy')
        y = np.load(dataset_path / 'y_train_ConvNeXt.npy')
    else:
        raise FileNotFoundError(f"No training embeddings found in {dataset_path}")
    return X, y

def visualize_lda_boundaries(X_train, y_train, output_dir, kernel="rbf", num_samples=1000):
    """
    Projects 768-D ConvNeXt embeddings into 2D using Linear Discriminant Analysis (LDA),
    then fits a 2D proxy SVM to visualize class separability and decision boundaries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset shape: {X_train.shape}")
    
    # 1. Reduce dimensionality using LDA (3 classes -> 2 components)
    print("Computing 2D Linear Discriminant Analysis (LDA) projection...")
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_2d = lda.fit_transform(X_train, y_train)

    # 2. Fit a proxy model on 2D coordinates
    print(f"Fitting proxy 2D {kernel.upper()} SVM on LDA features...")
    proxy_model = svm.SVC(kernel=kernel, C=1.0, gamma='scale', decision_function_shape='ovr')
    proxy_model.fit(X_2d, y_train)
    acc = proxy_model.score(X_2d, y_train)
    print(f"Proxy 2D Model Accuracy: {acc*100:.2f}%")

    # 3. Create meshgrid for decision contours
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = proxy_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 4. Subsample points for a clean visualization
    if len(X_2d) > num_samples:
        np.random.seed(42)
        indices = np.random.choice(len(X_2d), num_samples, replace=False)
        X_subset = X_2d[indices]
        y_subset = y_train[indices]
    else:
        X_subset, y_subset = X_2d, y_train

    # 5. Plot
    print("Generating plot...")
    plt.figure(figsize=(11, 9))
    cmap = plt.cm.RdYlBu

    # Contours & boundaries
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.7, alpha=0.6)

    # Scatter plot
    scatter = plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y_subset, 
                          cmap=cmap, edgecolors='k', s=35, alpha=0.8)

    # Legend
    unique_classes = sorted(np.unique(y_train))
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i / max(1, len(unique_classes) - 1)), 
                          markersize=10, markeredgecolor='k', label=config.CLASS_NAMES.get(i, f"Class {i}")) 
               for i in unique_classes]
    plt.legend(handles=handles, loc='upper right', title="Classes")

    plt.xlabel('LDA Component 1 (Primary Discriminative Axis)')
    plt.ylabel('LDA Component 2 (Secondary Discriminative Axis)')
    plt.title(f'Decision Boundaries (Proxy {kernel.upper()} SVM on 2D LDA Projection)')

    # Explanatory subtitle for academic integrity
    plt.figtext(0.5, 0.01, 
                "Note: 768-D ConvNeXt embeddings projected via LDA for qualitative 2D separability analysis.", 
                ha="center", fontsize=9, style='italic')

    out_file = output_dir / f"{kernel.upper()}_Decision_Boundary.png"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved successfully to: {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Visualize class decision boundaries via 2D LDA projection.")
    parser.add_argument("--kernel", choices=["rbf", "linear"], default="rbf",
                        help="Kernel type for the 2D proxy SVM.")
    parser.add_argument("--dataset-dir", type=str, default=str(config.DATASET_PATH),
                        help="Directory containing training embeddings.")
    parser.add_argument("--output-dir", type=str, default=str(config.OUTPUTS_DIR),
                        help="Directory to save the visualization plot.")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Number of data points to plot in scatter visualization.")
    args = parser.parse_args()

    try:
        X_train, y_train = load_embeddings(args.dataset_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    visualize_lda_boundaries(X_train, y_train, args.output_dir, kernel=args.kernel, num_samples=args.samples)

if __name__ == "__main__":
    main()
