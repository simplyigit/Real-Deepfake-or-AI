import argparse
import os
import shutil
import random
import re
from pathlib import Path
from collections import defaultdict

import config

def get_group_id(filename):
    """
    Extracts video ID / subject ID from filenames to prevent frame-level data leakage.
    Examples:
      - '000_003_f2.jpg' -> '000_003'
      - 'frame_0012_v04.png' -> 'v04' or base prefix
    If no known video frame pattern matches, falls back to the full filename stem.
    """
    stem = Path(filename).stem
    # Match FaceForensics++ pattern like 000_003_f2 or video_123_f04
    match = re.match(r"^(.*)_f\d+$", stem, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Match frame_XXXX patterns
    match2 = re.match(r"^.*?(video_\d+|subj_\d+|id_\d+)", stem, re.IGNORECASE)
    if match2:
        return match2.group(0)

    return stem

def split_directory(target_path, train_ratio=0.8, seed=42, group_by_prefix=True):
    """
    Splits images in target_path into {target_path}-train and {target_path}-test,
    ensuring frames from the same video/subject stay together to prevent data leakage.
    """
    target_path = Path(target_path).resolve()
    if not target_path.exists() or not target_path.is_dir():
        print(f"Error: Target path does not exist or is not a directory: {target_path}")
        return

    # Prevent recursive splitting
    if target_path.name.endswith("-train") or target_path.name.endswith("-test"):
        print(f"Skipping already-split folder: {target_path.name}")
        return

    train_path = target_path.parent / f"{target_path.name}-train"
    test_path = target_path.parent / f"{target_path.name}-test"

    valid_extensions = {'.jpg', '.png', '.jpeg'}
    files = [f for f in os.listdir(target_path) if Path(f).suffix.lower() in valid_extensions]

    if not files:
        print(f"No image files found in {target_path}")
        return

    rng = random.Random(seed)

    if group_by_prefix:
        # Group files by video/subject ID
        groups = defaultdict(list)
        for f in files:
            gid = get_group_id(f)
            groups[gid].append(f)

        group_keys = sorted(groups.keys())
        rng.shuffle(group_keys)
        split_idx = int(len(group_keys) * train_ratio)

        train_groups = set(group_keys[:split_idx])
        test_groups = set(group_keys[split_idx:])

        train_files = [f for gid in train_groups for f in groups[gid]]
        test_files = [f for gid in test_groups for f in groups[gid]]
        print(f"Grouped {len(files)} files into {len(groups)} distinct video/subject entities.")
        print(f"Train: {len(train_groups)} groups ({len(train_files)} files) | Test: {len(test_groups)} groups ({len(test_files)} files)")
    else:
        rng.shuffle(files)
        split_idx = int(len(files) * train_ratio)
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        print(f"Shuffled {len(files)} files randomly. Train: {len(train_files)} | Test: {len(test_files)}")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)

    print(f"Copying files to:\n  Train: {train_path}\n  Test:  {test_path}")
    for fname in train_files:
        shutil.copy2(target_path / fname, train_path / fname)
    for fname in test_files:
        shutil.copy2(target_path / fname, test_path / fname)

    print(f"Done splitting {target_path.name}.")

def main():
    parser = argparse.ArgumentParser(description="Group-aware, leakage-free dataset split tool.")
    parser.add_argument("--folder", "-f", type=str, required=True,
                        help="Path to the directory to split (e.g. dataset/deepfake/face2face).")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                        help="Proportion of data to allocate to training (default: 0.8).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic reproducibility.")
    parser.add_argument("--no-grouping", action="store_true",
                        help="Disable group-aware splitting and perform naive random file shuffling.")
    args = parser.parse_args()

    split_directory(
        args.folder, 
        train_ratio=args.train_ratio, 
        seed=args.seed, 
        group_by_prefix=not args.no_grouping
    )

if __name__ == "__main__":
    main()