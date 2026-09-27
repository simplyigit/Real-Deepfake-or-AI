import argparse
import os
from pathlib import Path
from PIL import Image

import config

def center_crop_image(img, crop_margin_pct=0.10):
    """
    Center-crops an image by trimming a percentage from the borders,
    preserving resolution scaling across varied dimensions.
    """
    width, height = img.size
    
    # Margin to remove from each side
    x_margin = int(width * (crop_margin_pct / 2.0))
    y_margin = int(height * (crop_margin_pct / 2.0))
    
    left = x_margin
    top = y_margin
    right = width - x_margin
    bottom = height - y_margin

    if right <= left or bottom <= top:
        return img

    return img.crop((left, top, right, bottom))

def process_folder(input_folder, output_folder=None, crop_margin_pct=0.10, in_place=False):
    """Processes all images in a folder with center-cropping."""
    input_path = Path(input_folder).resolve()
    if not input_path.exists():
        print(f"Error: Folder does not exist: {input_path}")
        return

    if not in_place and output_folder is None:
        output_path = input_path.parent / f"{input_path.name}_cropped"
    elif in_place:
        output_path = input_path
    else:
        output_path = Path(output_folder).resolve()

    output_path.mkdir(parents=True, exist_ok=True)
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}

    files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
    print(f"Found {len(files)} images to crop in {input_path.name}...")

    processed = 0
    for f in files:
        try:
            with Image.open(f) as img:
                cropped = center_crop_image(img, crop_margin_pct=crop_margin_pct)
                target_file = output_path / f.name
                if f.suffix.lower() in {'.jpg', '.jpeg'}:
                    cropped.save(target_file, quality=95, subsampling=0)
                else:
                    cropped.save(target_file)
                processed += 1
        except Exception as e:
            print(f"Skipping {f.name} due to error: {e}")

    print(f"Successfully cropped {processed}/{len(files)} images to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Proportional, non-destructive center-crop utility for face datasets.")
    parser.add_argument("--folder", "-f", type=str, required=True,
                        help="Path to folder of images to crop.")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (defaults to <folder>_cropped).")
    parser.add_argument("--crop-margin", type=float, default=0.10,
                        help="Total percentage margin to trim (default: 0.10 = 5% per side).")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite original images in-place (use with caution).")
    args = parser.parse_args()

    process_folder(
        args.folder,
        output_folder=args.output_dir,
        crop_margin_pct=args.crop_margin,
        in_place=args.in_place
    )

if __name__ == "__main__":
    main()