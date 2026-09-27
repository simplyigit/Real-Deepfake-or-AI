import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch

import config
import utils

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def collect_image_files(dataset_path, folders_and_labels):
    """
    Collects train and test image paths and their corresponding labels in a single pass.
    """
    train_paths, train_labels = [], []
    test_paths, test_labels = [], []
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    
    for folder_name, label in folders_and_labels:
        folder = Path(dataset_path) / folder_name
        if not folder.exists():
            print(f"Warning: Folder not found: {folder}")
            continue
            
        print(f"Scanning {folder.name}...")
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix.lower() in valid_extensions:
                p_str = str(p)
                if "-train" in p_str:
                    train_paths.append(p)
                    train_labels.append(label)
                elif "-test" in p_str:
                    test_paths.append(p)
                    test_labels.append(label)
                    
    return train_paths, train_labels, test_paths, test_labels

def main():
    parser = argparse.ArgumentParser(description="Extract ConvNeXt embeddings for dataset images.")
    parser.add_argument("--dataset-dir", type=str, default=str(config.DATASET_PATH),
                        help="Path to dataset directory containing real/, ai_fake/, and deepfake/ subfolders.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for feature extraction.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers.")
    parser.add_argument("--output-dir", type=str, default=str(config.DATASET_PATH),
                        help="Directory where output .npy files will be saved.")
    parser.add_argument("--keep-layernorm", action="store_true",
                        help="Preserve LayerNorm2d in ConvNeXt (recommended for new trainings).")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {config.DEVICE}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path:  {output_path}")

    folders_and_labels = [
        ('real', 0),
        ('ai_fake', 1),
        ('deepfake', 2)
    ]

    print("\n1. Scanning dataset folders...")
    train_paths, train_labels, test_paths, test_labels = collect_image_files(dataset_path, folders_and_labels)

    print(f"Found {len(train_paths)} training images, {len(test_paths)} test images.")
    if len(train_paths) == 0 and len(test_paths) == 0:
        print("Error: No valid images found with '-train' or '-test' in paths.")
        sys.exit(1)

    print("\n2. Initializing ConvNeXt model...")
    cnn = utils.get_model(device=config.DEVICE, keep_layernorm=args.keep_layernorm)

    start_time = time.time()

    if train_paths:
        print(f"\n3. Extracting training features (batch_size={args.batch_size})...")
        X_train, y_train = utils.extract_batch_features(
            cnn, train_paths, train_labels, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            device=config.DEVICE
        )
        print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
        np.save(output_path / 'X_train_ConvNeXt.npy', X_train)
        np.save(output_path / 'y_train_ConvNeXt.npy', y_train)
        print(f"Saved training embeddings to {output_path}")

    if test_paths:
        print(f"\n4. Extracting test features (batch_size={args.batch_size})...")
        X_test, y_test = utils.extract_batch_features(
            cnn, test_paths, test_labels,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=config.DEVICE
        )
        print("X_test shape:", X_test.shape, "y_test shape:", y_test.shape)
        np.save(output_path / 'X_test_ConvNeXt.npy', X_test)
        np.save(output_path / 'y_test_ConvNeXt.npy', y_test)
        print(f"Saved test embeddings to {output_path}")

    elapsed = time.time() - start_time
    print(f"\nExtraction complete in {elapsed/60:.2f} minutes.")

if __name__ == "__main__":
    main()