import argparse
import os
import time
import sys
import re
from pathlib import Path
from dotenv import load_dotenv

import config

def get_next_image_index(output_dir):
    """Finds the next index number based on existing nano_fake_*.png files."""
    existing_indices = []
    pattern = re.compile(r"nano_fake_(\d+)\.png")
    for f in output_dir.glob("nano_fake_*.png"):
        m = pattern.search(f.name)
        if m:
            existing_indices.append(int(m.group(1)))
    return max(existing_indices) + 1 if existing_indices else 1

def generate_nano_images(total_images=50, output_dir=None, image_model="nano-banana-pro-preview"):
    # Load environment variables
    for env_file in [config.PROJECT_ROOT / "gemini.env", config.PROJECT_ROOT / ".env"]:
        if env_file.exists():
            load_dotenv(env_file)

    api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: API_KEY not found in gemini.env or environment.")
        print("Please copy gemini.env.example to gemini.env and set your Gemini API key.")
        sys.exit(1)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: `google-genai` is not installed. Please run `pip install -r requirements.txt`.")
        sys.exit(1)

    target_dir = Path(output_dir) if output_dir else config.DATASET_PATH / "ai_fake" / "nano-banana"
    target_dir.mkdir(parents=True, exist_ok=True)

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=25000)
    )

    def get_batch_prompts(n=10):
        print(f"Requesting {n} synthetic portrait prompts from Gemini...")
        sys_prompt = (
            f"Generate {n} highly detailed distinct real face image generation prompts. "
            "Vary the faces: include photorealistic portraits, model agency shots, everyday shots, "
            "above-shoulder shots, close-up shots, different genders and ages. No illustrations, no cartoons, no sci-fi. "
            "Return ONLY the prompts, one per line, no numbering. "
            "Every image should center the face in the frame. No borders, no text."
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=sys_prompt
        )
        prompts = [line.strip() for line in response.text.split('\n') if line.strip()]
        return prompts[:n]

    next_idx = get_next_image_index(target_dir)
    generated_count = 0
    skipped = 0

    print(f"\n--- STARTING GENERATION OF {total_images} IMAGES ---")
    print(f"Target Model:  {image_model}")
    print(f"Output Target: {target_dir}")
    print(f"Starting at Index: {next_idx:04d}")

    while generated_count < total_images:
        prompts = get_batch_prompts(n=10)
        for prompt in prompts:
            if generated_count >= total_images:
                break
            try:
                print(f"[{generated_count+1}/{total_images}] Generating: {prompt[:40]}...")
                aspect_ratios = ["1:1", "3:4", "2:3", "4:3", "3:2"]
                aspect_ratio = aspect_ratios[generated_count % len(aspect_ratios)]

                response = client.models.generate_content(
                    model=image_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        image_config=types.ImageConfig(
                            aspect_ratio=aspect_ratio,
                            image_size="1K"
                        )
                    )
                )

                for part in response.parts:
                    if part.inline_data is not None:
                        image = part.as_image()
                        filename = target_dir / f"nano_fake_{next_idx:04d}.png"
                        image.save(filename)
                        print(f"Saved: {filename.name}")
                        generated_count += 1
                        next_idx += 1

                time.sleep(3)
            except Exception as e:
                print(f"Skipping prompt due to error: {e}")
                skipped += 1
                time.sleep(1)

    print(f"\nGeneration completed: {generated_count} images saved, {skipped} skipped.")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic portrait faces using Google Gemini models.")
    parser.add_argument("--count", type=int, default=50, help="Number of images to generate.")
    parser.add_argument("--output-dir", type=str, default=None, help="Destination directory.")
    parser.add_argument("--model", type=str, default="nano-banana-pro-preview", help="Image generator model name.")
    args = parser.parse_args()

    generate_nano_images(total_images=args.count, output_dir=args.output_dir, image_model=args.model)

if __name__ == "__main__":
    main()