import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights

# === Auto-correct path from script location ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "../dataset")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "../models/mobilenet_model.pth")

# === Use GPU if available ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# === Image transformations ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Load dataset ===
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# === Class count check ===
num_classes = len(dataset.classes)
print(f"🧠 Classes found: {dataset.classes} → Total: {num_classes}")

# === Load pretrained MobileNetV2 ===
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# === Freeze feature extractor layers (optional) ===
for param in model.features.parameters():
    param.requires_grad = False

# === Replace final layer with our classifier ===
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# === Loss and optimizer ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Training ===
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"📘 Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}")

# === Save model ===
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved at: {MODEL_SAVE_PATH}")
