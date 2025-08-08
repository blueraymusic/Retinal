import sys
import os

# Add project root to sys.path for imports
sys.path.append('/Users/bla/Desktop/RetinalAnomalyDetection')

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd

from glcm.resnet_glcm import ResNetWithInternalGLCM  


# Define the same normalization transform used during training
img_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    # Assuming label columns start from 2nd column onwards after 'filename'
    return df.columns[1:].tolist()


def load_model(model_path, num_classes, device):
    model = ResNetWithInternalGLCM(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict(model, image_path, device, disease_labels):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image '{image_path}' not found")

    image = Image.open(image_path).convert('RGB')
    input_tensor = img_transforms(image).unsqueeze(0).to(device)  # add batch dim

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs).cpu().squeeze(0).numpy()

    results = list(zip(disease_labels, probs))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def explain_prediction(results):
    top_label, top_prob = results[0]
    explanation = f"The image is most likely to show: {top_label} with confidence {top_prob:.2f}."
    if top_prob < 0.5:
        explanation += " However, confidence is low, so it may be normal or uncertain."
    return explanation


if __name__ == "__main__":
    MODEL_PATH = 'models/v4_.pth'  # adjust if different
    LABELS_CSV = 'data/train/train.csv'  # adjust if different
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    labels = load_labels(LABELS_CSV)
    model = load_model(MODEL_PATH, num_classes=len(labels), device=DEVICE)

    # Hardcoded image path here:
    image_path = 'data/test/03e7e29071a5.jpg'  # <- your image path here

    predictions = predict(model, image_path, DEVICE, labels)
    explanation = explain_prediction(predictions)

    print("Predictions (disease: confidence):")
    for label, prob in predictions:
        print(f"{label}: {prob:.3f}")

    print("\nExplanation:")
    print(explanation)
