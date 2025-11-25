# Building-an-AI-Classifier-Identifying-Cats-Dogs-Pandas-with-PyTorch
# Aim:
Your goal is to build an image classification model using transfer learning in PyTorch to predict whether an image belongs to a cat, dog, or panda.
# Program:
```python
import os
import torch
import random
import numpy as np
from pathlib import Path
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# -------------------- CONFIG --------------------
DATA_DIR = Path(r"C:\Users\admin\Desktop\w2\Cat-Dog_Pandas")  # ← your data
BATCH_SIZE = 32
EPOCHS = 12
IMG_SIZE = 224
LR = 0.001
NUM_CLASSES = 3
CHECKPOINT_PATH = "model_best.pth"

# -------------------- SEEDS --------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# -------------------- DEVICE --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# -------------------- TRANSFORMS --------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_t = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

test_t = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

# -------------------- DATA LOADING --------------------
train_path = DATA_DIR / "train"
test_path = DATA_DIR / "test"

train_ds = datasets.ImageFolder(train_path, transform=train_t)
test_ds = datasets.ImageFolder(test_path, transform=test_t)

# make validation set (15%)
val_size = int(0.15 * len(train_ds))
train_size = len(train_ds) - val_size
train_ds, val_ds = random_split(train_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

print("Train:", len(train_ds), "Val:", len(val_ds), "Test:", len(test_ds))

# -------------------- MODEL --------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# freeze feature extractor
for param in model.parameters():
    param.requires_grad = False

# replace classifier head
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, NUM_CLASSES)
)

model.to(device)

# -------------------- TRAINING SETUP --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

best_val_acc = 0

# -------------------- TRAINING LOOP --------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ------ validation ------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, preds = torch.max(output, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ------ save best model ------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "val_acc": best_val_acc
        }, CHECKPOINT_PATH)
        print("Saved best model!")

# -------------------- TEST EVALUATION --------------------
model.load_state_dict(torch.load(CHECKPOINT_PATH)["model_state_dict"])
model.eval()

correct = 0
total = 0
test_loss = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = criterion(output, labels)
        test_loss += loss.item()
        _, preds = torch.max(output, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

print("\nTest Accuracy:", correct / total)
print("Test Loss:", test_loss / len(test_loader))
print("Model saved as:", CHECKPOINT_PATH)
```
# Output:
<img width="679" height="699" alt="image" src="https://github.com/user-attachments/assets/e9fb95a7-0aaa-405f-b649-6b099040620c" />

# Result:
Thus image classification model using transfer learning in PyTorch to predict whether an image belongs to a cat, dog, or panda is implemented successfully.
