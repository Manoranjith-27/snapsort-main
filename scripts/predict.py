import os
import torch
from torchvision import models, transforms
from PIL import Image
from torchvision.models import MobileNet_V2_Weights

# === Setup paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/mobilenet_model.pth")
IMAGE_PATH = os.path.join(BASE_DIR, "../test_images/test1.jpg")
DATASET_PATH = os.path.join(BASE_DIR, "../dataset")

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Transformation must match training ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Load class names from dataset ===
from torchvision.datasets import ImageFolder
dummy_dataset = ImageFolder(DATASET_PATH)
class_names = dummy_dataset.classes
print(f"🧠 Class names: {class_names}")

# === Load model and update final layer ===
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === Load image and predict ===
image = Image.open(IMAGE_PATH).convert('RGB')
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image_tensor)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    print(f"✅ Predicted Class: {predicted_class}")
