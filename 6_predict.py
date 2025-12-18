import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time

# Import the vein enhancement utility
from data_utils import get_vein_channel 

# --- Model Paths and Config ---
YOLO_MODEL_PATH = 'runs/detect/stage1_yolov8n/weights/best.pt'
SPECIES_MODEL_PATH = 'runs/classification/stage2_species_id_best.pth'
STRESS_MODEL_PATH = 'runs/classification/stage3_stress_class_best.pth'

# Define the number of classes used during training
NUM_SPECIES_CLASSES = 8
NUM_STRESS_CLASSES = 57 

# Define the list of class names (MUST match training order)
# NOTE: You must populate these lists based on the folder names in your training data
SPECIES_NAMES = sorted(os.listdir('cropped_for_species_id')) 
STRESS_NAMES = sorted(os.listdir('cropped_for_stress_class')) 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -----------------------------


# --- 1. Model Structure Re-creation (Needed to load state_dict) ---

def modify_input_layer(model, in_channels):
    """Recreates the 4-channel input layer."""
    original_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels, 
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias
    )
    model.conv1 = new_conv
    return model

def load_classifier_model(path, num_classes):
    """Loads a trained ResNet model with the correct 4-channel and output structure."""
    model = resnet50()
    model = modify_input_layer(model, in_channels=4) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load the saved weights
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval() # Set to evaluation mode
    return model

# --- 2. Pre-processing Pipeline ---

# Defines the necessary transformations for classification input (MUST match training)
CLF_TRANSFORMS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406, 0.0], 
        std=[0.229, 0.224, 0.225, 1.0]
    )
])

def preprocess_for_clf(cropped_img_bgr):
    """
    Applies vein enhancement and classification transforms to the cropped leaf.
    Returns a 4-channel tensor ready for the CNN.
    """
    # 1. Get Vein Channel
    vein_channel = get_vein_channel(cropped_img_bgr)
    
    # 2. Convert RGB image to float array (0-1)
    img_rgb = cv2.cvtColor(cropped_img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_float = img_rgb.astype(np.float32) / 255.0

    # 3. Stack the channels (HxWx4)
    img_4channel_numpy = np.concatenate([img_rgb_float, vein_channel], axis=-1)
    
    # 4. Apply PyTorch transforms
    # Note: ToPILImage expects HxWxC, then subsequent transforms apply
    img_tensor = CLF_TRANSFORMS(img_4channel_numpy)
    
    # Add batch dimension (1, C, H, W)
    return img_tensor.unsqueeze(0).to(DEVICE)

# --- 3. Main Inference Function ---

def predict_leaf_status(input_image_path):
    # 1. Load All Models
    yolo_detector = YOLO(YOLO_MODEL_PATH)
    species_classifier = load_classifier_model(SPECIES_MODEL_PATH, NUM_SPECIES_CLASSES)
    stress_classifier = load_classifier_model(STRESS_MODEL_PATH, NUM_STRESS_CLASSES)

    # --- Stage 1: Detection and Cropping ---
    
    # Run YOLO prediction (outputs BGR image array)
    img_bgr = cv2.imread(input_image_path)
    if img_bgr is None:
        return "Error: Could not load image."
        
    yolo_results = yolo_detector(input_image_path, conf=0.25, iou=0.5, verbose=False)
    
    if not yolo_results or not yolo_results[0].boxes:
        return "Detection Failed: No leaf found in the image."
        
    # Get the bounding box (xyxy)
    box = yolo_results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
    x1, y1, x2, y2 = box
    
    # Simple crop (no padding added here for simplicity, relying on training data padding)
    cropped_leaf = img_bgr[y1:y2, x1:x2] 
    
    # --- Stage 2: Species Identification ---
    
    # Preprocess the cropped leaf for classification
    clf_input_tensor = preprocess_for_clf(cropped_leaf)
    
    with torch.no_grad():
        # Species Prediction
        species_output = species_classifier(clf_input_tensor)
        _, species_pred_idx = torch.max(species_output, 1)
        species_confidence = torch.softmax(species_output, dim=1)[0][species_pred_idx.item()].item()
        
        # Stress Prediction
        stress_output = stress_classifier(clf_input_tensor)
        _, stress_pred_idx = torch.max(stress_output, 1)
        stress_confidence = torch.softmax(stress_output, dim=1)[0][stress_pred_idx.item()].item()

    # --- Final Result ---
    
    predicted_species = SPECIES_NAMES[species_pred_idx.item()]
    predicted_stress = STRESS_NAMES[stress_pred_idx.item()]
    
    # Optional: Display the prediction result on the image
    # Note: cv2 uses BGR, so you'd need to convert back for complex display
    
    return {
        "Species_ID": predicted_species,
        "Species_Confidence": f"{species_confidence:.2f}",
        "Stress_Classification": predicted_stress,
        "Stress_Confidence": f"{stress_confidence:.2f}",
        "Bounding_Box": (x1, y1, x2, y2)
    }

# --- Example Usage ---
if __name__ == '__main__':
    # Define a test image path here
    TEST_IMAGE = 'path/to/your/test_image.jpg' 
    
    if not os.path.exists(TEST_IMAGE):
        print("Please replace 'path/to/your/test_image.jpg' with a real test image path.")
    elif not os.path.exists(YOLO_MODEL_PATH):
        print("Error: YOLO detector weights not found. Ensure training completed.")
    else:
        print(f"--- Running 3-Stage Inference on: {TEST_IMAGE} ---")
        start_time = time.time()
        
        result = predict_leaf_status(TEST_IMAGE)
        
        end_time = time.time()
        
        if isinstance(result, dict):
            print("\n✅ Final Diagnosis:")
            print(f"  Species Identified: {result['Species_ID']} (Conf: {result['Species_Confidence']})")
            print(f"  Diagnosis:          {result['Stress_Classification']} (Conf: {result['Stress_Confidence']})")
            print(f"  Processing Time:    {(end_time - start_time):.3f} seconds")
        else:
            print(result)