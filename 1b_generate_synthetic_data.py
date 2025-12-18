import cv2
import numpy as np
import os
import random
import glob
from tqdm import tqdm

# --- Configuration ---
ASSET_DIR = "assets"
BACKGROUND_DIR = "background"
OUTPUT_DIR = "synthetic_dataset"

NUM_TRAIN_IMAGES = 18000  # Total images for training split
NUM_VAL_IMAGES = 2000     # Total images for validation split
LEAF_CLASS_ID = 0        # YOLO Class ID for 'leaf'
# ---------------------

# Load leaf image
asset_paths = glob.glob(os.path.join(ASSET_DIR, "**/*.png"), recursive=True)
print(len(asset_paths))

# Load environment image
bg_paths = glob.glob(os.path.join(BACKGROUND_DIR, "*.jpg")) + \
             glob.glob(os.path.join(BACKGROUND_DIR, "*.jpeg")) + \
             glob.glob(os.path.join(BACKGROUND_DIR, "*.png"))
print(len(bg_paths))

if not asset_paths or not bg_paths:
    raise FileNotFoundError("Check your 'assets/' and 'backgrounds/' folders. Paths are empty.")

# Pastes an RGBA foreground image onto an RGB background image at (x, y).
def paste_rgba_on_bg(rgba_img, bg_img, x, y):
    h, w, _ = rgba_img.shape
    leaf_rgb = rgba_img[:, :, :3]
    mask = rgba_img[:, :, 3] / 255.0
    mask_3d = np.stack([mask, mask, mask], axis=2)
    
    # Define the ROI slice using the leaf's position and size
    y_end = y + h
    x_end = x + w
    
    # Get the background slice that matches the leaf's size
    roi = bg_img[y:y_end, x:x_end]
    
    # Safety check (should not be necessary with correct placement logic)
    if roi.shape != (h, w, 3):
        return bg_img
    
    # Blending formula: (Foreground * Mask) + (Background * Inverse Mask)
    blended_roi = (leaf_rgb * mask_3d) + (roi * (1.0 - mask_3d))
    
    # Place the blended ROI back into the background image
    bg_img[y:y_end, x:x_end] = blended_roi.astype(np.uint8)
    return bg_img

# Converts a bounding box (x_min, y_min, w, h) to YOLO format.
def bbox_to_yolo(box, img_size):
    x_min, y_min, w, h = box
    img_h, img_w = img_size
    
    x_center = (x_min + w / 2) / img_w
    y_center = (y_min + h / 2) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    
    return f"{LEAF_CLASS_ID} {x_center} {y_center} {norm_w} {norm_h}"

# --- Main Generation Loop ---
def generate_data(num_images, split_name):
    img_dir = os.path.join(OUTPUT_DIR, "images", split_name)
    lbl_dir = os.path.join(OUTPUT_DIR, "labels", split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for i in tqdm(range(num_images), desc=f"Generating {split_name} data"):
        bg_path = random.choice(bg_paths)
        bg = cv2.imread(bg_path)
        
        # --- Background Augmentation (Brightness/Contrast) ---
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2) # Contrast
            beta = random.randint(-20, 20)   # Brightness
            bg = cv2.convertScaleAbs(bg, alpha=alpha, beta=beta)
        
        bg_h, bg_w, _ = bg.shape

        asset_path = random.choice(asset_paths)
        leaf_rgba = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
        h_orig, w_orig, _ = leaf_rgba.shape # Original size
        
        # --- Scale Leaf to Background's Smallest Side (30%-80%) ---
        
        # 1. Determine the target size for the leaf based on the background's smallest side
        smallest_bg_dim = min(bg_h, bg_w)
        
        # Target scale range: 30% to 80% of the smallest background dimension
        target_scale_factor = random.uniform(0.30, 0.80)
        target_size_px = int(smallest_bg_dim * target_scale_factor)
        
        # 2. Calculate the factor needed to resize the original leaf down to the target size
        max_leaf_dim = max(h_orig, w_orig)
        if max_leaf_dim == 0: continue
            
        resize_factor = target_size_px / max_leaf_dim
        
        w_scaled = int(w_orig * resize_factor)
        h_scaled = int(h_orig * resize_factor)

        # --- Leaf Augmentation: Rotation (Applied before final scale/flip) ---
        angle = random.randint(-180, 180) 
        M = cv2.getRotationMatrix2D((w_orig / 2, h_orig / 2), angle, 1) 
        leaf_rotated = cv2.warpAffine(leaf_rgba, M, (w_orig, h_orig), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        
        # Final Resize of the rotated leaf
        leaf_aug = cv2.resize(leaf_rotated, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)

        if random.random() > 0.5:
            # Horizontal flip
            leaf_aug = cv2.flip(leaf_aug, 1) 
            
        # --- GET FINAL DIMENSIONS ---
        final_h, final_w, _ = leaf_aug.shape
        
        # --- Placement ---
        
        # Since the leaf size is guaranteed to be less than 80% of the smallest background side, 
        # it should always fit, but we keep the checks for robustness.
        
        # Calculate max placement coordinates
        max_x = bg_w - final_w
        max_y = bg_h - final_h
        
        # If the background is unexpectedly small or placement is impossible
        if max_x <= 0 or max_y <= 0: continue 
        
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)

        # Paste and Label
        bg_with_leaf = paste_rgba_on_bg(leaf_aug, bg.copy(), x, y) 
        
        # Bounding Box and YOLO Label Generation
        bbox = (x, y, final_w, final_h) 
        yolo_label = bbox_to_yolo(bbox, (bg_h, bg_w))

        img_filename = f"leaf_{split_name}_{i:05d}.jpg"
        lbl_filename = f"leaf_{split_name}_{i:05d}.txt"
        
        cv2.imwrite(os.path.join(img_dir, img_filename), bg_with_leaf)
        with open(os.path.join(lbl_dir, lbl_filename), 'w') as f:
            f.write(yolo_label)

# Generate the datasets
generate_data(NUM_TRAIN_IMAGES, "train")
generate_data(NUM_VAL_IMAGES, "val")
print("\nSynthetic dataset generated successfully!")