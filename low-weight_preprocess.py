import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
import time
from sklearn.utils import resample, shuffle

# Settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"

# 1. Setup device

device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 2. Load pretrained ConvNeXt
cnn = models.convnext_tiny(weights="DEFAULT")

# Remove classifier head → get 768-dim embedding
cnn.classifier = nn.Identity()
cnn.to(device)
cnn.eval()

# 3. Preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ROOT_NAME = input("Root folder name: ").lower()
SUB_NAME = input("Sub folder name: ").lower()
folders = [DATASET_PATH / ROOT_NAME]
labels = [1]

def extract_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = cnn(x).cpu().squeeze().numpy()
    return emb

def progress_print(folder_paths):
    all_files = []

    # First pass: collect all valid image files
    for folder in folder_paths:
        for p in Path(folder).rglob('*'):
            if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                if "-train" in str(p) or "-test" in str(p):
                    all_files.append(p)

    return len(all_files)

def build_dataset(folder_paths, labels):
    total = progress_print(folder_paths)
    processed = 0
    last_print = time.time()

    features_train, features_test = [], []
    labels_train, labels_test = [], []

    for folder, label in zip(folder_paths, labels):
        # Use rglob to recursively search through subfolders
        folder_path = Path(folder)

        for image_path in folder_path.rglob('*'):
            if SUB_NAME not in str(image_path):
                continue
            if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                if "-train" in str(image_path):
                    emb = extract_embedding(str(image_path))
                    features_train.append(emb)
                    labels_train.append(label)
                    processed += 1

                elif "-test" in str(image_path):
                    emb = extract_embedding(str(image_path))
                    features_test.append(emb)
                    labels_test.append(label)
                    processed += 1
                    
                try:
                    if time.time() - last_print >= 1:
                        percent = (processed / total) * 100
                        print(f"Progress: {percent:.2f}% ({processed}/{total})")
                        last_print = time.time()
                except Exception:
                    print("Error in progress printing.")

    X_train = np.stack(features_train)
    y_train = np.array(labels_train)

    X_test = np.stack(features_test)
    y_test = np.array(labels_test)

    return X_train, y_train, X_test, y_test

print("Preprocessing...")
nano_X_train, nano_y_train, nano_X_test, nano_y_test = build_dataset(folders, labels)

print("Train dataset shape:", nano_X_train.shape, nano_y_train.shape)
print("Test dataset shape:", nano_X_test.shape, nano_y_test.shape)

nano_X_train_os, nano_y_train_os = resample(
    nano_X_train, nano_y_train,
    replace=True,
    n_samples=798, # ~7× oversample
    random_state=42
)

X_train = np.load(DATASET_PATH / 'X_train_ConvNeXt.npy')
y_train = np.load(DATASET_PATH / 'y_train_ConvNeXt.npy')

X_train_aug = np.concatenate([X_train, nano_X_train_os], axis=0)
y_train_aug = np.concatenate([y_train, nano_y_train_os], axis=0)

X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=42)

np.save(DATASET_PATH / f'X_train_aug.npy', X_train_aug)
np.save(DATASET_PATH / f'y_train_aug.npy', y_train_aug)

np.save(DATASET_PATH / f'{SUB_NAME}_X_test.npy', nano_X_test)
np.save(DATASET_PATH / f'{SUB_NAME}_y_test.npy', nano_y_test)