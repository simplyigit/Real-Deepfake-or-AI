import torch
import numpy as np
import time
from pathlib import Path

import config
import utils

# 1. Setup device & model
print("Using device:", config.DEVICE)
cnn = utils.get_model(config.DEVICE)
transform = utils.get_transform()

def progress_print(folder_paths):
    all_files = []
    # Collect all valid image files
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
            if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                
                is_train = "-train" in str(image_path)
                is_test = "-test" in str(image_path)

                if is_train or is_test:
                    emb = utils.extract_embedding(cnn, str(image_path), transform, config.DEVICE)
                    
                    if emb is not None:
                        if is_train:
                            features_train.append(emb)
                            labels_train.append(label)
                        else:
                            features_test.append(emb)
                            labels_test.append(label)
                        processed += 1
                        
                # Progress print (avoid division by zero)
                if total > 0 and (time.time() - last_print >= 2):
                    percent = (processed / total) * 100
                    print(f"Progress: {percent:.2f}% ({processed}/{total})")
                    last_print = time.time()

    X_train = np.stack(features_train) if features_train else np.array([])
    y_train = np.array(labels_train)

    X_test = np.stack(features_test) if features_test else np.array([])
    y_test = np.array(labels_test)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    folders = [config.DATASET_PATH / 'real', config.DATASET_PATH / 'ai_fake', config.DATASET_PATH / 'deepfake']
    labels = [0, 1, 2] # Matched with config.CLASS_NAMES but kept as ints for array

    print("Preprocessing...")
    X_train, y_train, X_test, y_test = build_dataset(folders, labels)

    print("Train dataset shape:", X_train.shape, y_train.shape)
    print("Test dataset shape:", X_test.shape, y_test.shape)

    np.save(config.DATASET_PATH / 'X_train_ConvNeXt.npy', X_train)
    np.save(config.DATASET_PATH / 'y_train_ConvNeXt.npy', y_train)

    np.save(config.DATASET_PATH / 'X_test_ConvNeXt.npy', X_test)
    np.save(config.DATASET_PATH / 'y_test_ConvNeXt.npy', y_test)