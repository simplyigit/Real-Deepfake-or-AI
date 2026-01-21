# Real, Deepfake or AI

This project uses a ConvNeXt model to extract embeddings from human face images and an SGD Classifier (SVM) to classify them as Real, AI Fake, or Deepfake.

## Setup

1.  **Clone the repository** (if applicable).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare the dataset**:
    -   Place your dataset in a folder named `dataset` in the project root.
    -   Structure:
        ```text
        dataset/
        ├── real/
        ├── ai_fake/
        └── deepfake/
        ```
    -   Scripts assume images are split into train/test sets (containing `-train` and `-test` in their paths) or you can use `dataset_split.py` to help organize them.

## Usage

### 1. Preprocessing
Extract embeddings from your images using the pre-trained ConvNeXt model.
```bash
python preprocess.py
```
This will generate `.npy` files in the `dataset/` directory.

### 2. Training & Evaluation
Train the SVM classifier and evaluate its performance.
```bash
python train.py
```
This script will:
- Load the preprocessed embeddings.
- Train the SGD Classifier.
- Evaluate on the test set.
- Display Classification Report, Confusion Matrix, and ROC Curves.

### Other Scripts
- `dataset_split.py`: Helper script to split dataset into train/test subsets.
- `predict.py`: (If applicable) Script for making predictions on new images.

# Real, Deepfake or AI (ConvNeXt)

This project uses a ConvNeXt model to extract embeddings from images and an SGD Classifier (SVM) to classify them as Real, AI Fake, or Deepfake.

## Setup

1.  **Clone the repository** (if applicable).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare the dataset**:
    -   Place your dataset in a folder named `dataset` in the project root.
    -   Structure:
        ```text
        dataset/
        ├── real/
        ├── ai_fake/
        └── deepfake/
        ```
    -   Scripts assume images are split into train/test sets (containing `-train` and `-test` in their paths) or you can use `dataset_split.py` to help organize them.

## Usage

### 1. Preprocessing
Extract embeddings from your images using the pre-trained ConvNeXt model.
```bash
python preprocess.py
```
This will generate `.npy` files in the `dataset/` directory.

### 2. Training & Evaluation
Train the SVM classifier and evaluate its performance.
```bash
python train.py
```
This script will:
- Load the preprocessed embeddings.
- Train the SGD Classifier.
- Evaluate on the test set.
- Display Classification Report, Confusion Matrix, and ROC Curves.

### Other Scripts
- `dataset_split.py`: Helper script to split dataset into train/test subsets.
- `predict.py`: (If applicable) Script for making predictions on new images.
