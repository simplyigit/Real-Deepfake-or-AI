import os
from pathlib import Path
import torch

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Dynamic Dataset Path Resolution
# Checks environment variable -> project root dataset -> parent dataset
env_dataset = os.environ.get("DATASET_DIR")
if env_dataset and Path(env_dataset).exists():
    DATASET_PATH = Path(env_dataset).resolve()
elif (PROJECT_ROOT / "dataset").exists() and (
    (PROJECT_ROOT / "dataset" / "real").exists()
    or (PROJECT_ROOT / "dataset" / "X_train_ConvNeXt.npy").exists()
    or (PROJECT_ROOT / "dataset" / "X_train_aug.npy").exists()
):
    DATASET_PATH = PROJECT_ROOT / "dataset"
elif (PROJECT_ROOT.parent / "dataset").exists():
    DATASET_PATH = PROJECT_ROOT.parent / "dataset"
else:
    DATASET_PATH = PROJECT_ROOT / "dataset"

# Model Checkpoint Resolution
def resolve_model_path(model_name="SGD_ConvNeXt_nano.pkl"):
    """Finds a model file by checking project root, models directory, and dataset directory."""
    candidates = [
        PROJECT_ROOT / model_name,
        MODELS_DIR / model_name,
        DATASET_PATH / model_name,
        Path(model_name),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return PROJECT_ROOT / model_name

DEFAULT_MODEL_PATH = resolve_model_path("SGD_ConvNeXt_nano.pkl")

# Device Configuration
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

# Model Parameters
MODEL_WEIGHTS = "DEFAULT"
EMBEDDING_DIM = 768

# Classes
CLASS_NAMES = {0: 'Real', 1: 'AI Fake', 2: 'Deepfake'}
