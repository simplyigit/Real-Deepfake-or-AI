# Real, Deepfake or AI Classifier (ConvNeXt + SVM)

This project implements a hybrid machine learning pipeline to classify human faces as **Real**, **AI Generated**, or **Deepfake**.

## 🧠 Architecture
The system uses a **two-stage** approach:
1.  **Feature Extraction**: A pre-trained `ConvNeXt Tiny` model (classifier removed) extracts 768-dimensional embeddings from images.
2.  **Classification**: A classifier trained on these embeddings to classify the images.

**Why this approach?**
-   **Speed**: Training on embeddings is much faster than fine-tuning a deep Neural Network.
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
    -   Trains the classifier.
    -   Evaluates performance (Accuracy, Confusion Matrix, ROC Curves).
    -   Saves the trained model.
-   **`predict.py`**:
    -   Takes a single image name as input.
    -   Loads the trained model (`.pkl`) and the ConvNeXt extractor.
    -   Predicts if the image is Real, AI, or Deepfake.

### 3. Data & Utilities
-   **`nano_banana.py`**:
    -   A script to generate synthetic "AI Fake" images using Google's `nano-banana-pro-preview` model.
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
-   `Confusion_Matrix.png`
-   `ROC_Curves.png`

---

## 📦 Pre-trained Model
The repository includes a pre-trained model: **`SGD_ConvNeXt_nano.pkl`**.

If you want to use this model directly without training:
1.  Ensure you have the `dataset/` folder structure.
2.  Run `predict.py`.

### 📚 Datasets Used
This model was trained on the following datasets:
-   CelebA - Total 23.948 Images: 19.158/4790
-   [Selfies](https://www.kaggle.com/datasets/jkanthony/selfie-image-faces) - Total 1.676 Images: 1.340/336
-   [Face Coverage](https://www.kaggle.com/datasets/mantasu/glasses-and-coverings) - Total 2.032 Images: 1.625/407
-   [Stable Diffusion](https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset) - Total 1.000 Images: 800/200
-   [GAN](https://www.kaggle.com/datasets/shavaizbutt/ai-face-dataset-3000-images?select=seed1000163.png), [GAN](https://www.kaggle.com/datasets/hamzaboulahia/hardfakevsrealfaces) - Total 3.697 Images: 2.957/740
-   Nano Banana Pro - Total 143 Images: 114/29
-   [Deepfake](https://www.kaggle.com/datasets/fatimahirshad/faceforensics-c32-frames-cropped-aligned) - Total 24.951 Images: 19.961/4.990

> **Note**: If you have access to these datasets, you can reproduce the training by running `python preprocess.py` followed by `python train.py`.
