import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import joblib

import config
import utils

def find_image(image_input, dataset_path=config.DATASET_PATH):
    """
    Resolves an image input string into a valid Path.
    Supports direct file paths (absolute/relative) and bare names searched in t_images/.
    """
    image_input = image_input.strip().strip("'\"")
    raw_path = Path(image_input)
    
    # 1. Direct path check
    if raw_path.exists() and raw_path.is_file():
        return raw_path.resolve()

    # 2. Check relative to PROJECT_ROOT
    root_path = config.PROJECT_ROOT / image_input
    if root_path.exists() and root_path.is_file():
        return root_path.resolve()

    # 3. Check inside dataset t_images directory
    search_dirs = [
        dataset_path / "t_images",
        config.PROJECT_ROOT / "t_images",
        dataset_path,
    ]
    extensions = ["", ".jpg", ".jpeg", ".png", ".JPG", ".PNG", ".JPEG"]

    for directory in search_dirs:
        if directory.exists():
            for ext in extensions:
                candidate = directory / f"{raw_path.stem if ext else image_input}{ext}"
                if candidate.exists() and candidate.is_file():
                    return candidate.resolve()

    return None

def predict_single_image(image_path, model, cnn, transform, device=config.DEVICE):
    """Runs ConvNeXt feature extraction and classifier inference on an image."""
    embedding = utils.extract_embedding(cnn, str(image_path), transform, device)
    if embedding is None:
        raise RuntimeError(f"Could not extract embeddings from {image_path}")

    embedding = embedding.reshape(1, -1)
    pred_idx = int(model.predict(embedding)[0])
    class_name = config.CLASS_NAMES.get(pred_idx, f"Class {pred_idx}")

    probs = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(embedding)[0]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(embedding)[0]
        # Softmax over decision scores for normalized relative confidence
        e = np.exp(scores - np.max(scores))
        probs = e / e.sum()

    return class_name, pred_idx, probs

def main():
    parser = argparse.ArgumentParser(description="Classify an image as Real, AI Fake, or Deepfake.")
    parser.add_argument("--image", "-i", type=str, default=None,
                        help="Path to image or image name (searched in t_images/).")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to trained classifier checkpoint (.pkl).")
    parser.add_argument("--keep-layernorm", action="store_true",
                        help="Preserve LayerNorm2d in ConvNeXt (use if model was trained with it).")
    args = parser.parse_args()

    # Get image input (CLI argument or interactive prompt)
    image_query = args.image
    if not image_query:
        try:
            image_query = input("Enter image path or name: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)

    if not image_query:
        print("Error: No image specified.")
        sys.exit(1)

    image_path = find_image(image_query)
    if not image_path:
        print(f"Error: Image '{image_query}' could not be found.")
        print(f"Searched direct paths and inside: {config.DATASET_PATH / 't_images'}")
        sys.exit(1)

    # Resolve model checkpoint
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = config.resolve_model_path("SGD_ConvNeXt_nano.pkl")

    if not model_path.exists():
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train a model using `python train.py` or specify `--model <path>`.")
        sys.exit(1)

    # Load model and feature extractor
    print(f"Loading classifier: {model_path.name}")
    model = joblib.load(model_path)
    cnn = utils.get_model(config.DEVICE, keep_layernorm=args.keep_layernorm)
    transform = utils.get_transform()

    print(f"Processing image:   {image_path.name}")
    class_name, pred_idx, probs = predict_single_image(image_path, model, cnn, transform, config.DEVICE)

    # Display results
    print("\n" + "=" * 45)
    print("           PREDICTION RESULTS")
    print("=" * 45)
    print(f"File:       {image_path}")
    print(f"Prediction: {class_name.upper()}")
    
    if probs is not None:
        confidence = probs[pred_idx] * 100
        print(f"Confidence: {confidence:.2f}%\n")
        print("Class Probabilities:")
        for idx, p in sorted(config.CLASS_NAMES.items()):
            prob_pct = probs[idx] * 100
            bar = "#" * int(prob_pct / 5)
            print(f"  [{idx}] {p:<10s} : {prob_pct:5.1f}%  {bar}")
    print("=" * 45 + "\n")

if __name__ == "__main__":
    main()