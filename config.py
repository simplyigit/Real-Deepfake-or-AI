import os
from pathlib import Path
import torch

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset"

# Device
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# Model
MODEL_WEIGHTS = "DEFAULT"
EMBEDDING_DIM = 768

# Classes
CLASS_NAMES = {0: 'Real', 1: 'AI Fake', 2: 'Deepfake'}
