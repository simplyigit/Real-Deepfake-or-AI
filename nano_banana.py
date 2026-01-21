import os
import time
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv

import config

# --- CONFIGURATION ---
load_dotenv(config.PROJECT_ROOT / "gemini.env")
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("Error: API_KEY not found in gemini.env")
    sys.exit(1)

image_model = "nano-banana-pro-preview"

# Paths
OUTPUT_DIR = config.DATASET_PATH / "ai_fake" / "nano-banana"
TOTAL_IMAGES = 170

# V2 Client
client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=25000)
)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Prompt Generator
def get_batch_prompts(n=10):
    print(f"Creating {n} unique prompts...")
    sys_prompt = (
        f"Generate {n} highly detailed distinct real face image generation prompts. "
        "Vary the faces: include photorealistic portraits, model agency shots, every-day shots, above-shoulder shots, close-up shots, different genders and ages. No illustrations, no cartoons and no sci-fi. "
        "Return ONLY the prompts, one per line, no numbering. "
        "Every image should center the face in the frame. No borders, no text."
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=sys_prompt
    )
    
    # Clean up the response into a list
    prompts = [line.strip() for line in response.text.split('\n') if line.strip()]
    return prompts[:n]

# --- PART 2: THE IMAGE GENERATOR ---
def generate_nano_images():
    generated_count = 0
    skipped = 0
    
    print(f"--- STARTING GENERATION OF {TOTAL_IMAGES} IMAGES ---")
    print(f"Target Model: {image_model}")
    
    while generated_count < TOTAL_IMAGES:
        # 1. Get a batch of fresh prompts
        prompts = get_batch_prompts(n=10)
        
        for prompt in prompts:
            if generated_count >= TOTAL_IMAGES:
                break
                
            try:
                print(f"[{generated_count+1}/{TOTAL_IMAGES}] Generating: {prompt[:40]}...")
                # Cycle aspect ratios
                aspect_ratios = ["1:1", "3:4", "2:3", "4:3", "3:2"]
                aspect_ratio = aspect_ratios[generated_count % len(aspect_ratios)]
                
                # 2. Generate Image
                response = client.models.generate_content(
                    model=image_model, 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        image_config = types.ImageConfig(
                            aspect_ratio = aspect_ratio,
                            image_size = "1K"
                        )
                    )
                )
                
                # 3. Save Image
                for part in response.parts:
                    if part.inline_data is not None:
                        image = part.as_image()
                        filename = OUTPUT_DIR / f"nano_fake_{generated_count+151:04d}.png"
                        image.save(filename)
                        generated_count += 1
                        print(f"Saved {filename}")
                        
                # Sleep to avoid rate limits
                time.sleep(4)
                    
            except Exception as e:
                print(f"Skipping due to error: {e}")
                skipped += 1
                time.sleep(1)

    print(f"--- DONE. Saved {generated_count} images to {OUTPUT_DIR} ---\n")
    print(f"Skipped {skipped} images in total.")

if __name__ == "__main__":
    start = time.perf_counter()
    generate_nano_images()
    end = time.perf_counter()
    print(f"Total Time: {(end - start) / 60:.2f} Minutes")