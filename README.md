# Real, Deepfake or AI Classifier (ConvNeXt + SVM)

This project implements a hybrid machine learning pipeline to classify human faces as **Real**, **AI Generated**, or **Deepfake**.

## 🧠 Architecture
The system uses a **two-stage** approach:
1.  **Feature Extraction**: A pre-trained `ConvNeXt Tiny` model (classifier removed) extracts 768-dimensional embeddings from images.
2.  **Classification**: A Support Vector Machine (SVM) with an RBF kernel is trained on these embeddings to classify the images.

**Why this approach?**
-   **Speed**: Training an SVM on embeddings is much faster than fine-tuning a deep Neural Network.
-   **Accuracy**: ConvNeXt provides state-of-the-art feature representation.

---

## 📂 Project Structure

### 1. Core Logic
-   **`config.py`**: Central configuration file. Contains paths (`DATASET_PATH`), device settings (MPS/CUDA/CPU), and constants.
-   **`utils.py`**: Shared utilities. Handles loading the ConvNeXt model and defining standard image transformations (Resize, Normalize).

### 2. Main Pipeline
-   **`preprocess.py`**:
    -   Scans the `dataset/` folder for images.
    -   Passes them through ConvNeXt to generate embeddings.
    -   Saves the embeddings as `.npy` files (e.g., `X_train_ConvNeXt.npy`).
    -   **Run this first!**
-   **`train.py`**:
    -   Loads the precomputed embeddings (`.npy`).
    -   Trains the SVM (RBF Kernel).
    -   Evaluates performance (Accuracy, Confusion Matrix, ROC Curves).
    -   Saves the trained model to `dataset/RBF_ConvNeXt.pkl`.
-   **`predict.py`**:
    -   Takes a single image name as input.
    -   Loads the trained model (`.pkl`) and the ConvNeXt extractor.
    -   Predicts if the image is Real, AI, or Deepfake.

### 3. Data & Utilities
-   **`nano_banana.py`**:
    -   A script to generate synthetic "AI Fake" images using Google's Gemini Flash model.
    -   Used to augment the dataset with high-quality AI faces.
    -   *Requires `gemini.env` with `API_KEY`.*
-   **`low-weight_preprocess.py`**:
    -   A specialized preprocessing script for augmenting specific low-weight classes (like the `nano-banana` dataset) by oversampling.
-   **`dataset_split.py`**:
    -   Helper tool to organize raw images into `train` and `test` folders based on a split ratio (default 80/20).
-   **`crop.py`**:
    -   Helper tool to center-crop images to focus on the face, improving embedding quality.
-   **`visualize_boundaries.py`**:
    -   Uses Linear Discriminant Analysis (LDA) to project the 768-dim embeddings into 2D.
    -   Visualizes the SVM decision boundaries to show how the model separates classes.

---

## 🚀 Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place your images in `dataset/` structured by class (e.g., `real`, `deepfake`).
If you need to generate AI images:
```bash
python nano_banana.py
```

### 3. Preprocess
Convert images to embeddings:
```bash
python preprocess.py
```

### 4. Train
Train the classifier:
```bash
python train.py
```

### 5. Predict
Test on a specific image (place it in `dataset/t_images/`):
```bash
python predict.py
```

---

## 📊 Results
The `train.py` script automatically saves plots to the dataset folder:
-   `RBF_Confusion_Matrix.png`
-   `RBF_ROC_Curves.png`
