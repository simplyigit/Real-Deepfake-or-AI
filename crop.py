import os
from PIL import Image
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset"

ROOT_NAME = str(input("Root folder name: "))
SUB_NAME = str(input("Sub folder name: "))
PATH = DATASET_PATH / ROOT_NAME   # dataset/<class>/<subfolder>/*.jpg  OR dataset/<class>/*.jpg

for cls in os.listdir(PATH):
    if SUB_NAME not in cls: 
        continue
    
    cls_path = os.path.join(PATH, cls)
    if not os.path.isdir(cls_path): 
        continue

    for fname in os.listdir(cls_path):
        if not fname.lower().endswith(('.jpg','.png','.jpeg')):
            continue
        
        src = os.path.join(cls_path, fname)
        img = Image.open(src)
        original_width, original_height = img.size
        
        # Calculate new dimensions
        new_width = original_width - 100
        
        if new_width <= 0:
            # Skip if image is too small
            continue

        # Calculate scale factor to preserve aspect ratio
        scale_factor = new_width / original_width
        new_height = int(original_height * scale_factor)
        
        # Calculate coordinates for center crop
        left = (original_width - new_width) // 2
        top = (original_height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        
        # Perform crop
        cropped_img = img.crop((left, top, right, bottom))
        
        # Save image
        if fname.lower().endswith(('.jpg', '.jpeg')):
            cropped_img.save(src, quality=100, subsampling=0)
        else:
            cropped_img.save(src)
            