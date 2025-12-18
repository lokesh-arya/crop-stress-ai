import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import os
import time
from tqdm import tqdm

# Import the custom components from data_utils.py
from data_utils import FourChannelDataset

# --- Configuration ---
DATA_ROOT = 'cropped_for_species_id'
NUM_CLASSES = 8
BATCH_SIZE = 32
NUM_EPOCHS = 30 # Start with 30 epochs for fine-tuning
LEARNING_RATE = 1e-4 # Low LR for fine-tuning
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Output path to save the best model weights
MODEL_SAVE_PATH = 'runs/classification/stage2_species_id_best.pth'
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
# ---------------------

def modify_input_layer(model, in_channels):
    """
    Modifies the first convolutional layer (conv1) of the ResNet model 
    to accept 4 input channels instead of 3.
    """
    original_conv = model.conv1
    
    # Create the new 4-channel convolutional layer
    new_conv = nn.Conv2d(
        in_channels=in_channels, 
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias
    )
    
    # Copy weights: first 3 channels get pretrained weights, 4th channel gets zeros.
    with torch.no_grad():
        original_weights = original_conv.weight.data
        new_weights = new_conv.weight.data.clone().zero_()
        
        # Copy original RGB weights (3 channels)
        new_weights[:, :3, :, :] = original_weights
        
        # Assign the new weights
        new_conv.weight.data = new_weights
    
    model.conv1 = new_conv
    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    """The main training and validation loop."""
    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            # Use tqdm for a nice progress bar
            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} Phase'):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"Model saved with new best accuracy: {best_acc:.4f}")

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load the best model weights after training finishes
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    return model

def main():
    # --- 1. Data Preparation and Split ---
    
    # Note: We must define 4 channels in the Normalize transform (RGB mean/std + Vein mean/std)
    data_transforms = transforms.Compose([
        # The Custom Dataset output is a HxWxC numpy array (4 channels)
        transforms.ToPILImage(), # Convert to PIL for transforms (Note: may drop 4th channel if not customized)
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # Converts HxWxC to CxHxW tensor
        
        # Mean/Std for 4 channels (standard ImageNet for RGB, 0/1 for Vein)
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.0], 
            std=[0.229, 0.224, 0.225, 1.0]
        )
    ])
    
    # Load the full dataset (all 1934 real images)
    full_dataset = FourChannelDataset(data_dir=DATA_ROOT, transform=data_transforms)
    
    # Split the dataset: 80% train, 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    
    # --- 2. Model Initialization and Modification ---
    
    # Load ResNet-50 with default pretrained weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Modify the input layer to accept 4 channels
    model = modify_input_layer(model, in_channels=4)
    
    # Modify the final classification head for 9 species classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    model = model.to(DEVICE)
    
    # --- 3. Training Setup ---
    
    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Use a low learning rate (1e-4) because we are fine-tuning a pre-trained model
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    # --- 4. Start Training ---
    
    print(f"Starting Stage 2 training on {NUM_CLASSES} species classes...")
    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=NUM_EPOCHS)
    
    print(f"\nStage 2 Species ID Model trained and saved to: {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()