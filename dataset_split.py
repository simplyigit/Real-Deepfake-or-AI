# split_per_subfolder_files.py
import os, shutil, random
from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_PATH / "dataset"

ROOT_NAME = str(input("Root folder name: "))
SUB_NAME = str(input("Sub folder name: "))
PATH = DATASET_PATH / ROOT_NAME   # dataset/<class>/<subfolder>/*.jpg  OR dataset/<class>/*.jpg
train_ratio = 0.8

for cls in os.listdir(PATH):
    if SUB_NAME not in cls: 
        continue
    
    cls_path = os.path.join(PATH, cls)
    if not os.path.isdir(cls_path): 
        continue

    TRAIN_PATH = PATH / f"{cls}-train"
    TEST_PATH  = PATH / f"{cls}-test"

    os.makedirs(TRAIN_PATH, exist_ok=True)
    os.makedirs(TEST_PATH, exist_ok=True)

    if os.path.abspath(cls_path) in {str(TRAIN_PATH), str(TEST_PATH)}:
        continue

    all_files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    random.shuffle(all_files)
    split = int(len(all_files) * train_ratio)
    train_files = all_files[:split]
    test_files  = all_files[split:]

    for fname in train_files:
        src = os.path.join(cls_path, fname)
        os.makedirs(TRAIN_PATH, exist_ok=True)
        shutil.copy2(src, os.path.join(TRAIN_PATH, fname))
    for fname in test_files:
        src = os.path.join(cls_path, fname)
        os.makedirs(TEST_PATH, exist_ok=True)
        shutil.copy2(src, os.path.join(TEST_PATH, fname))

print("Done splitting per-class files.")