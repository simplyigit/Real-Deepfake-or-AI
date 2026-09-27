import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config

def get_transform():
    """Returns the standard image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_model(device=config.DEVICE, keep_layernorm=False):
    """
    Loads the ConvNeXt-Tiny model for feature extraction.
    
    Args:
        device: torch.device to place the model on.
        keep_layernorm: If True, preserves LayerNorm2d and Flatten, replacing only
                        the final Linear projection head (cnn.classifier[2] = nn.Identity()).
                        If False, replaces the entire classifier with nn.Identity(),
                        maintaining exact backward compatibility with SGD_ConvNeXt_nano.pkl.
    """
    cnn = models.convnext_tiny(weights=config.MODEL_WEIGHTS)
    if keep_layernorm:
        cnn.classifier[2] = nn.Identity()
    else:
        cnn.classifier = nn.Identity()
    cnn.to(device)
    cnn.eval()
    return cnn

def extract_embedding(model, image_path, transform, device=config.DEVICE):
    """Extracts a 1D embedding vector (768,) from a single image path."""
    try:
        img = Image.open(image_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = model(x).cpu()
            emb = out.reshape(out.shape[0], -1).squeeze(0).numpy()
        return emb
    except Exception as e:
        print(f"Error extracting embedding for {image_path}: {e}")
        return None

class ImageListDataset(Dataset):
    """PyTorch Dataset for efficient batch extraction from a list of image paths."""
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform if transform is not None else get_transform()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # Fallback to blank image if corrupted
            image = torch.zeros(3, 224, 224)
        
        if self.labels is not None:
            return image, self.labels[idx], str(path)
        return image, str(path)

def extract_batch_features(model, file_paths, labels=None, batch_size=64, num_workers=2, device=config.DEVICE):
    """
    Extracts embeddings for a list of file paths in batches using DataLoader.
    Significantly faster than single-image extraction.
    """
    dataset = ImageListDataset(file_paths, labels=labels, transform=get_transform())
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    features = []
    out_labels = []
    
    with torch.no_grad():
        for batch in loader:
            if labels is not None:
                imgs, lbls, _ = batch
                out_labels.append(lbls.numpy())
            else:
                imgs, _ = batch
            imgs = imgs.to(device)
            out = model(imgs).cpu()
            emb = out.reshape(out.shape[0], -1).numpy()
            features.append(emb)
            
    all_features = np.concatenate(features, axis=0) if features else np.empty((0, config.EMBEDDING_DIM))
    if labels is not None:
        all_labels = np.concatenate(out_labels, axis=0) if out_labels else np.empty((0,), dtype=int)
        return all_features, all_labels
    return all_features
