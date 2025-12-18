import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os

def get_vein_channel(image_bgr):
    """
    Applies Gaussian Blur and Laplacian filter to highlight vein structure.
    Returns a normalized 1-channel numpy array (0 to 1.0).
    """
    # 1. Convert BGR (loaded by cv2) to Grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur (Denoise)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Laplacian Filter (Edge/Vein Detection)
    laplacian = cv2.Laplacian(blurred, cv2.CV_8U)
    
    # 4. Normalize (Scales the laplacian result back to 0-255, then converts to float 0-1)
    laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to float and normalize to 0-1 for PyTorch compatibility
    vein_channel = laplacian_norm.astype(np.float32) / 255.0
    
    # Expand dimension for stacking (from HxW to HxWx1)
    return np.expand_dims(vein_channel, axis=-1)


class FourChannelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        
        # Collect all image paths and map class names to labels (0 to 8)
        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.image_paths = []
        for cls in self.classes:
            cls_path = os.path.join(data_dir, cls)
            for file in os.listdir(cls_path):
                if file.lower().endswith(('.jpg', '.png')):
                    self.image_paths.append(os.path.join(cls_path, file))

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load the image
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB for standard processing

        # 1. Get Vein Channel (Structural input)
        vein_channel = get_vein_channel(img_bgr)
        
        # 2. Convert RGB image to float array (0-1)
        img_rgb_float = img_rgb.astype(np.float32) / 255.0

        # 3. Stack the channels (HxWx4)
        img_4channel_numpy = np.concatenate([img_rgb_float, vein_channel], axis=-1)

        # Apply transformations 
        if self.transform:
            img_tensor = self.transform(img_4channel_numpy) 
        else:
             # Manually convert HxWxC to CxHxW tensor
            img_tensor = torch.from_numpy(img_4channel_numpy).permute(2, 0, 1)

        # Get the label ID
        class_name = os.path.basename(os.path.dirname(img_path))
        label = self.class_to_idx[class_name]
        
        return img_tensor, label