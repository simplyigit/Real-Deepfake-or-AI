import argparse
from pathlib import Path
import numpy as np
from sklearn.utils import resample, shuffle

import config
import utils

def augment_with_jitter(X, y, target_samples=798, jitter_std=0.01, random_state=42):
    """
    Oversamples minority embeddings and adds slight Gaussian noise (feature jitter)
    to prevent exact duplicate vectors and reduce overfitting.
    """
    rng = np.random.RandomState(random_state)
    
    # 1. Resample with replacement
    X_os, y_os = resample(X, y, replace=True, n_samples=target_samples, random_state=random_state)
    
    # 2. Add subtle Gaussian noise to synthetic replicates (excluding original points)
    if jitter_std > 0 and len(X) > 0:
        std_per_feature = np.std(X, axis=0, keepdims=True) + 1e-6
        noise = rng.normal(0, jitter_std, size=X_os.shape) * std_per_feature
        # Keep original copies intact, jitter the rest
        X_os = X_os + noise

    return X_os, y_os

def main():
    parser = argparse.ArgumentParser(description="Extract and augment low-weight class embeddings (e.g. nano-banana).")
    parser.add_argument("--subfolder", type=str, default="nano-banana",
                        help="Subfolder name inside class folder (default: 'nano-banana').")
    parser.add_argument("--class-label", type=int, default=1,
                        help="Class label index for these images (default: 1 for AI Fake).")
    parser.add_argument("--class-folder", type=str, default="ai_fake",
                        help="Parent class directory (default: 'ai_fake').")
    parser.add_argument("--target-samples", type=int, default=798,
                        help="Target sample count for augmented training set.")
    parser.add_argument("--jitter-std", type=float, default=0.01,
                        help="Standard deviation for Gaussian feature jitter on replicates.")
    parser.add_argument("--dataset-dir", type=str, default=str(config.DATASET_PATH),
                        help="Base dataset directory.")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    target_dir = dataset_path / args.class_folder

    print(f"Scanning {target_dir} for '{args.subfolder}'...")
    train_paths, test_paths = [], []
    valid_exts = {'.jpg', '.jpeg', '.png'}

    for p in target_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in valid_exts and args.subfolder in str(p):
            if "-train" in str(p):
                train_paths.append(p)
            elif "-test" in str(p):
                test_paths.append(p)

    print(f"Found {len(train_paths)} train images, {len(test_paths)} test images for {args.subfolder}.")
    if not train_paths and not test_paths:
        print("Error: No matching images found.")
        return

    # Extract embeddings using shared utils
    cnn = utils.get_model(config.DEVICE, keep_layernorm=False)

    if train_paths:
        print("Extracting training embeddings...")
        nano_X_train, nano_y_train = utils.extract_batch_features(
            cnn, train_paths, [args.class_label] * len(train_paths),
            batch_size=32, device=config.DEVICE
        )
    else:
        nano_X_train, nano_y_train = np.empty((0, config.EMBEDDING_DIM)), np.empty((0,), dtype=int)

    if test_paths:
        print("Extracting test embeddings...")
        nano_X_test, nano_y_test = utils.extract_batch_features(
            cnn, test_paths, [args.class_label] * len(test_paths),
            batch_size=32, device=config.DEVICE
        )
    else:
        nano_X_test, nano_y_test = np.empty((0, config.EMBEDDING_DIM)), np.empty((0,), dtype=int)

    print(f"Extracted train shape: {nano_X_train.shape}, test shape: {nano_X_test.shape}")

    # Save test set
    np.save(dataset_path / f'{args.subfolder}_X_test.npy', nano_X_test)
    np.save(dataset_path / f'{args.subfolder}_y_test.npy', nano_y_test)
    print(f"Saved {args.subfolder} test arrays.")

    # Augment training set
    if len(nano_X_train) > 0:
        print(f"Augmenting training embeddings to {args.target_samples} samples with jitter...")
        nano_X_train_os, nano_y_train_os = augment_with_jitter(
            nano_X_train, nano_y_train, 
            target_samples=args.target_samples, 
            jitter_std=args.jitter_std
        )

        base_x_path = dataset_path / 'X_train_ConvNeXt.npy'
        base_y_path = dataset_path / 'y_train_ConvNeXt.npy'

        if base_x_path.exists() and base_y_path.exists():
            print("Combining with existing X_train_ConvNeXt.npy...")
            X_train_base = np.load(base_x_path)
            y_train_base = np.load(base_y_path)

            X_train_aug = np.concatenate([X_train_base, nano_X_train_os], axis=0)
            y_train_aug = np.concatenate([y_train_base, nano_y_train_os], axis=0)
            X_train_aug, y_train_aug = shuffle(X_train_aug, y_train_aug, random_state=42)

            np.save(dataset_path / 'X_train_aug.npy', X_train_aug)
            np.save(dataset_path / 'y_train_aug.npy', y_train_aug)
            print(f"Saved X_train_aug.npy ({X_train_aug.shape}) and y_train_aug.npy ({y_train_aug.shape})")
        else:
            print(f"Warning: Base embeddings {base_x_path} not found. Saving isolated augmented set.")
            np.save(dataset_path / f'{args.subfolder}_X_train_aug.npy', nano_X_train_os)
            np.save(dataset_path / f'{args.subfolder}_y_train_aug.npy', nano_y_train_os)

if __name__ == "__main__":
    main()