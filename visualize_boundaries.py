import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from pathlib import Path
import config

def visualize_proxy():
    print("Loading data...")
    try:
        X_train = np.load(config.DATASET_PATH / 'X_train_aug.npy')
        y_train = np.load(config.DATASET_PATH / 'y_train_aug.npy')
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    print(f"Data shape: {X_train.shape}")
    
    # 1. Reduce dimensionality using LDA
    print("Computing LDA projection...")
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_2d = lda.fit_transform(X_train, y_train)
    
    # 2. Train a PROXY RBF SVM on the 2D data
    print("Training proxy RBF SVM on 2D data...")
    # Use RBF kernel to show curvature
    proxy_model = svm.SVC(kernel='rbf', C=1.0, gamma='scale', decision_function_shape='ovr')
    proxy_model.fit(X_2d, y_train)
    
    # Calculate accuracy of the proxy model
    acc = proxy_model.score(X_2d, y_train)
    print(f"Proxy Model (2D) Accuracy: {acc*100:.2f}%")
    
    # 3. Create a meshgrid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # 4. Predict using the PROXY model
    print("Computing decision function...")
    Z = proxy_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Subsample data for cleaner plot
    print("Subsampling data...")
    num_samples = 1000
    if len(X_2d) > num_samples:
        indices = np.random.choice(len(X_2d), num_samples, replace=False)
        X_subset = X_2d[indices]
        y_subset = y_train[indices]
    else:
        X_subset = X_2d
        y_subset = y_train

    # 5. Plot
    print("Plotting...")
    plt.figure(figsize=(12, 10))
    
    cmap = plt.cm.RdYlBu
    
    # Plot contours
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    # Add boundary lines
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)

    # Scatter plot
    scatter = plt.scatter(X_subset[:, 0], X_subset[:, 1], c=y_subset, 
                          cmap=cmap, edgecolors='k', s=30, alpha=0.8)
    
    # Legend
    unique_classes = np.unique(y_train)
    # Using config.CLASS_NAMES for consistency, assuming keys are 0, 1, 2
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i/2), 
                          markersize=10, markeredgecolor='k', label=config.CLASS_NAMES[i]) 
               for i in sorted(unique_classes)]
    plt.legend(handles=handles, loc='upper right', title="Classes")

    plt.xlabel('LDA Component 1')
    plt.ylabel('LDA Component 2')
    plt.title('SVM Decision Boundary (Proxy Model on LDA Projection)')
    
    plt.savefig(config.DATASET_PATH / 'RBF_Decision_Boundary.png', dpi=300)
    print("Plot saved to RBF_Decision_Boundary.png")
    print("Done.")

if __name__ == "__main__":
    visualize_proxy()
