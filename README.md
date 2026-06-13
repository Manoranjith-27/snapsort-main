# Deep Image Classifier: Handwritten vs Normal Photos

This project is a simple deep learning-based image classifier that separates handwritten photos from normal photos. It uses a pretrained MobileNetV2 model fine-tuned on your dataset.

## Project Structure
deep_classifier/
├── dataset/ # Training images organized in class folders
│ ├── handwritten/
│ └── normal/
├── models/ # Saved trained model weights
│ └── mobilenet_model.pth
├── scripts/ # Python scripts for training and prediction
│ ├── train.py
│ ├── predict.py
│ └── utils.py (optional helper functions)
├── test_images/ # Folder containing images to classify
├── requirements.txt # Required Python packages
└── README.md # This file


## Setup

1. Create a Python virtual environment:

python -m venv venv
source venv/bin/activate    # On Windows: venv\\Scripts\\activate


pip install -r requirements.txt

Install dependencies:

pip install -r requirements.txt

## Training
Run the training script to train MobileNetV2 on your dataset:

python scripts/train.py

Make sure your dataset is organized as shown above.

## Prediction and Sorting
To classify images in test_images folder and automatically sort them into subfolders by predicted class, run:

python scripts/predict.py
This will copy each image into either test_images/handwritten/ or 
test_images/normal/.

## Notes
Input images are resized to 224x224 for compatibility with MobileNetV2.

You can modify the scripts to move images instead of copying to save space.

The model uses GPU if available.