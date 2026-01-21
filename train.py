from sklearn import svm
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, mean_squared_error
)
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.utils import shuffle
import config

# Settings
print(f"Loading data from {config.DATASET_PATH}...")

# Load preprocessed-data
try:
    X_train = np.load(config.DATASET_PATH / 'X_train_aug.npy')
    y_train = np.load(config.DATASET_PATH / 'y_train_aug.npy')

    X_test = np.load(config.DATASET_PATH / 'X_test_ConvNeXt.npy')
    y_test = np.load(config.DATASET_PATH / 'y_test_ConvNeXt.npy')

    nano_x_test = np.load(config.DATASET_PATH / 'nano-banana_X_test.npy')
    nano_y_test = np.load(config.DATASET_PATH / 'nano-banana_y_test.npy')
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    exit(1)

print("Features loaded (Train, Test):", X_train.shape, X_test.dtype)
print("Labels loaded (Train, Test):", y_train.shape, y_test.shape)

# Initialize SVC (Support Vector Classifiers) with RBF Kernel & Save The Model

# model = svm.SVC(kernel='rbf', gamma='scale', random_state=42, C=1.0)
# model.fit(X_train, y_train)
# joblib.dump(model, config.DATASET_PATH / 'RBF_ConvNeXt.pkl')

model_path = config.DATASET_PATH / 'RBF_ConvNeXt.pkl'
if not model_path.exists():
    print(f"Error: Model not found at {model_path}")
    exit(1)

model = joblib.load(model_path)

# Join test datasets together
X_test_aug = np.concatenate([X_test, nano_x_test], axis=0)
X_test_aug = shuffle(X_test_aug, random_state=42)
y_test_aug = np.concatenate([y_test, nano_y_test], axis=0)
y_test_aug = shuffle(y_test_aug, random_state=42)

# -----------------
# Evaluate
# -----------------
y_pred = model.predict(X_test_aug)
classes = np.unique(y_train)

print(classification_report(y_test_aug, y_pred, digits=3))
print(confusion_matrix(y_test_aug, y_pred))

for c in classes:
    idx = y_test_aug == c
    acc = (y_pred[idx] == y_test_aug[idx]).mean()
    print(f"Class {c} accuracy: {acc*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test_aug, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test_aug),
            yticklabels=np.unique(y_test_aug))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(config.DATASET_PATH / 'RBF_Confusion_Matrix.png', dpi=300, bbox_inches='tight')
# Changed filename slightly to remove spaces: RBF Confidence Matrix -> RBF_Confusion_Matrix

# Accuracy
acc = np.mean(y_test_aug == y_pred)
print(f"Overall Accuracy: {acc:.4f}")

# ROC Curves
y_bin = label_binarize(y_test_aug, classes=classes)
y_score = model.decision_function(X_test_aug)

plt.figure(figsize=(8,6))
for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {c} (AUC = {roc_auc:.2f})")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves per Class")
plt.legend()
plt.savefig(config.DATASET_PATH / 'RBF_ROC_Curves.png', dpi=300, bbox_inches='tight')
# Changed filename slightly to remove spaces