import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO
from tqdm import tqdm
import re



# --- Input/Output Paths ---
ORIGINAL_ROOT_DIR = 'selected_images'
STAGE1_MODEL_PATH = 'runs/detect/stage1_yolov8n/weights/best.pt'

OUTPUT_SPECIES_DIR = 'cropped_for_species_id'
OUTPUT_STRESS_DIR = 'cropped_for_stress_class'

PADDING_PERCENTAGE = 0.10


def get_class_names(path):
    """
    Example: original_leaves/ash_gourd__healthy/img1.png
    species → ash_gourd
    subtype → ash_gourd__healthy
    """
    full_subtype = os.path.basename(os.path.dirname(path))
    species = full_subtype.split('__')[0]
    return species, full_subtype

def ensure_directory_structure(class_names, base_dir):
    """Creates the output folders for classification training."""
    for class_name in class_names:
        os.makedirs(os.path.join(base_dir, class_name), exist_ok=True)

def crop_and_save_leaf(detector, image_path, species_name, subtype_name):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Skipping (load error): {image_path}")
        return False

    H, W, _ = img.shape

    # --- YOLO DETECT ---
    results = detector(image_path, conf=0.25, iou=0.5, verbose=False)
    if not results or not results[0].boxes:
        return False

    x_min, y_min, x_max, y_max = results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)

    # --- Padding ---
    pad_w = int((x_max - x_min) * PADDING_PERCENTAGE)
    pad_h = int((y_max - y_min) * PADDING_PERCENTAGE)

    x1 = max(0, x_min - pad_w)
    y1 = max(0, y_min - pad_h)
    x2 = min(W, x_max + pad_w)
    y2 = min(H, y_max + pad_h)

    cropped_leaf = img[y1:y2, x1:x2]

    # --- Create unique filename ---
    fname = os.path.basename(image_path)
    unique_name = f"{fname}"

    # ---- Save to Stage 2 (species) ----
    species_out = os.path.join(OUTPUT_SPECIES_DIR, species_name, unique_name)
    if not cv2.imwrite(species_out, cropped_leaf):
        print(f"Failed to save: {species_out}")

    # ---- Save to Stage 3 (subtype) ----
    stress_out = os.path.join(OUTPUT_STRESS_DIR, subtype_name, unique_name)
    if not cv2.imwrite(stress_out, cropped_leaf):
        print(f"Failed to save: {stress_out}")

    return True



def main():

    if not os.path.exists(STAGE1_MODEL_PATH):
        print(f"ERROR: YOLO model not found at: {STAGE1_MODEL_PATH}")
        return

    # Collect images
    image_paths = glob.glob(os.path.join(ORIGINAL_ROOT_DIR, "**/*.JPG"), recursive=True)
    image_paths += glob.glob(os.path.join(ORIGINAL_ROOT_DIR, "**/*.jpg"), recursive=True)
    image_paths += glob.glob(os.path.join(ORIGINAL_ROOT_DIR, "**/*.png"), recursive=True)

    if not image_paths:
        print(f"ERROR: No images found in {ORIGINAL_ROOT_DIR}")
        return

    # Load model
    detector = YOLO(STAGE1_MODEL_PATH)

    # Extract all class names
    species_set = set()
    subtype_set = set()

    for path in image_paths:
        sp, sub = get_class_names(path)
        species_set.add(sp)
        subtype_set.add(sub)

    print(f"Found {len(species_set)} species, {len(subtype_set)} subtypes.")

    # Ensure output folders exist
    ensure_directory_structure(species_set, OUTPUT_SPECIES_DIR)
    ensure_directory_structure(subtype_set, OUTPUT_STRESS_DIR)

    print("Directory structure created.\n")

    # Process images
    count = 0
    for path in tqdm(image_paths, desc="Cropping leaves"):
        sp, sub = get_class_names(path)
        if crop_and_save_leaf(detector, path, sp, sub):
            count += 1

    print(f"\nDone! Saved {count} cropped leaves.")
    print("Outputs:")
    print(f" - Species-level crops → {OUTPUT_SPECIES_DIR}")
    print(f" - Stress-level crops →  {OUTPUT_STRESS_DIR}")


if __name__ == "__main__":
    main()