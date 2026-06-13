import os
import torch
from torchvision import models, transforms
from PIL import Image
import shutil
from torchvision.datasets import ImageFolder
from torchvision.models import MobileNet_V2_Weights

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/mobilenet_model.pth")
INPUT_DIR = os.path.join(BASE_DIR, "../test_images")  # Mixed images here
DATASET_PATH = os.path.join(BASE_DIR, "../dataset")   # For class names

# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD CLASSES ===
dummy_dataset = ImageFolder(DATASET_PATH)
class_names = dummy_dataset.classes  # ['handwritten', 'normal']

# === LOAD MODEL ===
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === PROCESS IMAGES ===
for fname in os.listdir(INPUT_DIR):
    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
        src_path = os.path.join(INPUT_DIR, fname)
        image = Image.open(src_path).convert("RGB")
        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)
            _, pred = torch.max(output, 1)
            predicted_class = class_names[pred.item()]

        # Destination folder (create if needed)
        dest_dir = os.path.join(INPUT_DIR, predicted_class)
        os.makedirs(dest_dir, exist_ok=True)

        # Copy to classified folder
        dst_path = os.path.join(dest_dir, fname)
        shutil.copy(src_path, dst_path)

        print(f"{fname} → {predicted_class}")

print("\n✅ Sorting completed.")
