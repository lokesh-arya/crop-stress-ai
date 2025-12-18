import cv2
import os
import glob
from tqdm import tqdm
import numpy as np

# --- Configuration ---
INPUT_DIR = "selected_images"
OUTPUT_DIR = "assets"
# ---------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Starting Phase 1: Creating RGBA Leaf Assets...")

# Use the current working directory for paths
image_paths = glob.glob(os.path.join(INPUT_DIR, "**/*.JPG"), recursive=True) + \
              glob.glob(os.path.join(INPUT_DIR, "**/*.PNG"), recursive=True)

for img_path in tqdm(image_paths, desc="Masking Assets"):
    try:
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Use your threshold (adjust if necessary, but keep it high)
        threshold_value, mask = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU )

        kernel = np.ones((5, 5), np.uint8)

        # Fill Holes Inside the Leaf (CLOSING)
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Remove Speckles/Dots from the Background (OPENING)
        mask_cleaned = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

        image_rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        image_rgba[:, :, 3] = mask_cleaned

        # --- Preserve subfolder structure ---
        rel_path = os.path.relpath(img_path, INPUT_DIR)
        rel_dir = os.path.dirname(rel_path)
        save_dir = os.path.join(OUTPUT_DIR, rel_dir)

        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        cv2.imwrite(os.path.join(save_dir, filename), image_rgba)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print(f"\nPhase 1 Complete. Assets saved to {OUTPUT_DIR}.")