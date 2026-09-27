import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
import config

class RandomJPEGCompression(object):
    """
    Simulates varied JPEG compression artifacts (quality 50-95) to destroy
    camera/video sensor quantization shortcuts and enforce semantic robustness.
    """
    def __init__(self, quality_range=(50, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img
        quality = int(np.random.randint(self.quality_range[0], self.quality_range[1]))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")

def get_transform(augment=False):
    """
    Returns image transformation pipeline.
    
    Args:
        augment: If True, adds JPEG compression perturbation, horizontal flips,
                 subtle color jitter, and Gaussian blur to destroy domain/compression bias.
    """
    transform_list = [transforms.Resize((224, 224))]
    if augment:
        transform_list.extend([
            RandomJPEGCompression(quality_range=(50, 95), p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.5)),
        ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transforms.Compose(transform_list)

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

def extract_embedding(model, image_path, transform=None, device=config.DEVICE):
    """Extracts a 1D embedding vector (768,) from a single image path."""
    if transform is None:
        transform = get_transform(augment=False)
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
        self.transform = transform if transform is not None else get_transform(augment=False)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            image = torch.zeros(3, 224, 224)
        
        if self.labels is not None:
            return image, self.labels[idx], str(path)
        return image, str(path)

import time

def extract_batch_features(model, file_paths, labels=None, batch_size=128, num_workers=0, 
                           device=config.DEVICE, augment=False):
    """
    Extracts embeddings for a list of file paths in batches using DataLoader.
    Optionally applies anti-shortcut perturbation augmentations.
    """
    transform = get_transform(augment=augment)
    dataset = ImageListDataset(file_paths, labels=labels, transform=transform)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    features = []
    out_labels = []
    total_batches = len(loader)
    total_imgs = len(file_paths)
    processed_imgs = 0
    t_start = time.time()
    last_log = t_start
    
    with torch.no_grad():
        for b_idx, batch in enumerate(loader):
            if labels is not None:
                imgs, lbls, _ = batch
                out_labels.append(lbls.numpy())
            else:
                imgs, _ = batch
            imgs = imgs.to(device)
            out = model(imgs).cpu()
            emb = out.reshape(out.shape[0], -1).numpy()
            features.append(emb)
            processed_imgs += imgs.size(0)

            if time.time() - last_log >= 3 or (b_idx + 1) == total_batches:
                elapsed = time.time() - t_start
                rate = processed_imgs / max(1e-5, elapsed)
                eta = (total_imgs - processed_imgs) / max(1e-5, rate)
                print(f"Batch [{b_idx+1}/{total_batches}] - {processed_imgs}/{total_imgs} images ({processed_imgs/total_imgs*100:.1f}%) - {rate:.1f} imgs/s - ETA: {eta:.0f}s")
                last_log = time.time()
            
    all_features = np.concatenate(features, axis=0) if features else np.empty((0, config.EMBEDDING_DIM))
    if labels is not None:
        all_labels = np.concatenate(out_labels, axis=0) if out_labels else np.empty((0,), dtype=int)
        return all_features, all_labels
    return all_features
