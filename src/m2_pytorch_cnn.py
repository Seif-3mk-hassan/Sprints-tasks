import numpy as np
from datasets import load_dataset
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


dataset = load_dataset("blanchon/EuroSAT_RGB")

# Inspect the structure
print(dataset)
print("Train samples:", len(dataset["train"]))
print("Validation samples:", len(dataset["validation"]))
print("Test samples:", len(dataset["test"]))

# Preview a single sample
sample = dataset["train"][0]
print("Image:", sample["image"])        # PIL Image
print("Label:", sample["label"])        # Integer 0-9
print("Image size:", sample["image"].size)  # (64, 64)



def extract_images_labels(split):
    """
    Convert a Hugging Face split into NumPy arrays.
    - Images: float32 arrays normalized to [0, 1]
    - Labels: int arrays
    """
    images = []
    labels = []

    for sample in split:
        # Convert PIL Image → NumPy array (H, W, C) uint8
        img_array = np.array(sample["image"], dtype=np.float32)

        # Normalize pixel values from [0, 255] → [0, 1]
        img_array /= 255.0

        images.append(img_array)
        labels.append(sample["label"])

    images = np.array(images)   # Shape: (N, 64, 64, 3)
    labels = np.array(labels)   # Shape: (N,)
    return images, labels


EPOCHS = 10
BATCH_SIZE = 512

class EuroSATDataset(Dataset):
    """
    Wraps NumPy arrays into a PyTorch Dataset.
    Converts channels-last (H,W,C) → channels-first (C,H,W).
    """
    def __init__(self, images, labels):
        # images: (N, 64, 64, 3) float32 NumPy
        # Transpose to (N, 3, 64, 64) for PyTorch
        self.images = torch.tensor(images).permute(0, 3, 1, 2)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=32, kernel_size=3)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)

        # Classifier head
        # After conv1 (62×62) → pool (31×31)
        # After conv2 (29×29) → pool (14×14)
        # Flattened: 64 * 14 * 14 = 12544
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # → (B, 32, 31, 31)
        x = self.pool(F.relu(self.conv2(x)))   # → (B, 64, 14, 14)
        x = x.view(x.size(0), -1)              # Flatten → (B, 12544)
        x = F.relu(self.fc1(x))                # → (B, 128)
        x = self.fc2(x)                        # → (B, 10) raw logits
        return x

def pytorch_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):

        # ── Training phase ──
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        # ── Validation phase ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:2d}/{EPOCHS}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

    return model

if '__main__' == __name__:
    # Extract all three splits
    print("Processing train split...")
    X_train, y_train = extract_images_labels(dataset["train"])

    print("Processing validation split...")
    X_val, y_val = extract_images_labels(dataset["validation"])

    print("Processing test split...")
    X_test, y_test = extract_images_labels(dataset["test"])

    print(f"Train:      {X_train.shape}, Labels: {y_train.shape}")
    print(f"Validation: {X_val.shape},  Labels: {y_val.shape}")
    print(f"Test:       {X_test.shape},  Labels: {y_test.shape}")

    # Create Dataset objects
    train_ds = EuroSATDataset(X_train, y_train)
    val_ds = EuroSATDataset(X_val, y_val)
    test_ds = EuroSATDataset(X_test, y_test)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Quick sanity check
    imgs, lbls = next(iter(train_loader))
    print(f"Batch images shape: {imgs.shape}")  # (32, 3, 64, 64)
    print(f"Batch labels shape: {lbls.shape}")  # (32,)

    model=pytorch_train()

    torch.save(model.state_dict(), "../models/pytorch_cnn.pth")
    print("PyTorch weights saved → pytorch_cnn.pth")

