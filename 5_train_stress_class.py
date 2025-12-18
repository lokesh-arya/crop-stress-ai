import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import os
import time
from tqdm import tqdm
from data_utils import FourChannelDataset
# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------
DATA_ROOT = 'cropped_for_stress_class'
NUM_CLASSES = 57
BATCH_SIZE = 16          # smaller batch for better generalization
NUM_EPOCHS = 40          # more epochs (with scheduler)
WARMUP_EPOCHS = 5
LEARNING_RATE = 3e-5     # improved LR
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_SAVE_PATH = 'runs/classification/stage3_stress_best.pth'
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# -------------------------------------------------------
# MODIFY INPUT LAYER FOR 4-CHANNEL IMAGES
# -------------------------------------------------------
def modify_input_layer(model, in_channels=4):
    original_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = original_conv.weight
        new_conv.weight[:, 3] = 0.0
    model.conv1 = new_conv
    return model

# -------------------------------------------------------
# TRAINING LOOP (WITH AMP + SCHEDULER + WARMUP)
# -------------------------------------------------------
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    best_acc = 0.0
    scaler = torch.cuda.amp.GradScaler()   # for mixed precision

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase}"):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # AMP forward pass
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # Backprop (only in training)
                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = (running_corrects.double() / len(dataloaders[phase].dataset)).item()

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Scheduler step (based on validation loss)
            if phase == 'val':
                scheduler.step(epoch_loss)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    print(f"✓ New Best Model Saved (Acc: {best_acc:.4f})")

    print(f"\nTraining complete. Best val Acc: {best_acc:.4f}")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    return model

# MAIN
def main():

    # DATA AUGMENTATION (STRONG)
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.0],
            std=[0.229, 0.224, 0.225, 1.0],
        ),
    ])

    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406, 0.0],
            std=[0.229, 0.224, 0.225, 1.0],
        ),
    ])

    # Dataset
    full_dataset = FourChannelDataset(DATA_ROOT, transform=train_transforms)

    # Train/Val Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply val transforms
    val_dataset.dataset.transform = val_transforms

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4),
    }

    # MODEL SETUP
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model = modify_input_layer(model, in_channels=4)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model = model.to(DEVICE)

    # CLASS WEIGHTS (optional)
    try:
        class_counts = full_dataset.get_class_counts()
        weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
        weights = weights / weights.sum() * NUM_CLASSES
        criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))
        print("Using class-balanced loss.")
    except:
        print("Class counts not found — using standard loss.")
        criterion = nn.CrossEntropyLoss()

    # OPTIMIZER + SCHEDULER
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.3
    )

    # WARMUP: freeze backbone
    print("\nFreezing backbone for warmup...")
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    train_model(model, dataloaders, criterion, optimizer, scheduler, WARMUP_EPOCHS)

    print("\nUnfreezing all layers...")
    for param in model.parameters():
        param.requires_grad = True

    train_model(model, dataloaders, criterion, optimizer, scheduler, NUM_EPOCHS)

    print(f"\nTraining finished. Best model saved at {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()