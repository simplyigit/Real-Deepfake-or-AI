import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import config

def get_transform():
    """Returns the standard image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_model(device=config.DEVICE):
    """Loads the ConvNeXt model with Identity classifier."""
    cnn = models.convnext_tiny(weights=config.MODEL_WEIGHTS)
    cnn.classifier = nn.Identity()
    cnn.to(device)
    cnn.eval()
    return cnn

def extract_embedding(model, image_path, transform, device=config.DEVICE):
    """Extracts embedding from a single image path."""
    try:
        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = model(x).cpu().squeeze().numpy()
        return emb
    except Exception as e:
        print(f"Error extracting embedding for {image_path}: {e}")
        return None
