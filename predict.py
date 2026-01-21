import torch
import numpy as np
import joblib
from pathlib import Path
from PIL import Image

import config
import utils

# 1. User Input
image_name = str(input("Image name: "))

# Flexible image searching (handling extensions)
possible_extensions = [".jpg", ".jpeg", ".png"]
image_path = None

for ext in possible_extensions:
    p = config.DATASET_PATH / "t_images" / f"{image_name}{ext}"
    if p.exists():
        image_path = p
        break

if not image_path:
    print(f"Error: Image '{image_name}' not found in {config.DATASET_PATH / 't_images'}")
    exit(1)

# 2. Setup (Model & Transform)
print("Using device:", config.DEVICE)
cnn = utils.get_model(config.DEVICE)
transform = utils.get_transform()

# 3. Load Trained Classifier
# Assuming the user wants to use the 'SGD_ConvNeXt_nano.pkl' model mentioned in original file
model_path = config.DATASET_PATH / 'SGD_ConvNeXt_nano.pkl'
if not model_path.exists():
    print(f"Error: Model not found at {model_path}")
    exit(1)

model = joblib.load(model_path)

# 4. Predict
embedding = utils.extract_embedding(cnn, str(image_path), transform, config.DEVICE)

if embedding is not None:
    embedding = embedding.reshape(1, -1)  # 2D for sklearn
    pred = model.predict(embedding)
    
    # Probabilities
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(embedding)
        print("Decision scores:", scores)
        
        def softmax_row(scores_):
            e = np.exp(scores_ - np.max(scores_))
            return e / e.sum()

        s = np.array(scores)
        print(f"Softmax: {softmax_row(s) * 100}")
        print(f"Predicted class: {config.CLASS_NAMES[int(pred[0])]} {scores[0][int(pred[0])]}")
    else:
        print(f"Predicted class: {config.CLASS_NAMES[int(pred[0])]}")