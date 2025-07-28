import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# Set paths
img_dir = 'data/train/train/'
label_csv = 'data/train/train.csv'
df = pd.read_csv(label_csv)

# Disease columns
diseases = [
    'opacity',
    'diabetic retinopathy',
    'glaucoma',
    'macular edema',
    'macular degeneration',
    'retinal vascular occlusion',
    'normal'
]

# Texture features to extract
features = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'correlation']

# Initialize dictionary to hold feature values by disease
feature_data = {feat: {d: [] for d in diseases} for feat in features}
brightness_data = {d: [] for d in diseases}

print("Extracting features from images...")

for i in tqdm(df.index):
    img_path = os.path.join(img_dir, df.loc[i, 'filename'])
    # Read grayscale image and resize for consistency
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Warning: Failed to load image {img_path}")
        continue
    gray = cv2.resize(gray, (128, 128))

    # Calculate GLCM matrix and properties
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    # Extract all features for this image
    img_features = {feat: graycoprops(glcm, feat)[0, 0] for feat in features}

    # Calculate brightness (mean pixel intensity)
    brightness = gray.mean()

    # Assign features and brightness to diseases
    for disease in diseases:
        if df.loc[i, disease] == 1:
            for feat in features:
                feature_data[feat][disease].append(img_features[feat])
            brightness_data[disease].append(brightness)

# Plot mean GLCM features per disease
for feat in features:
    means = {disease: np.mean(feature_data[feat][disease]) if feature_data[feat][disease] else 0 for disease in diseases}
    plt.figure(figsize=(10, 5))
    plt.bar(means.keys(), means.values(), color='teal')
    plt.title(f'Mean GLCM {feat.title()} per Disease')
    plt.ylabel(feat.title())
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot mean brightness per disease
brightness_means = {disease: np.mean(brightness_data[disease]) if brightness_data[disease] else 0 for disease in diseases}
plt.figure(figsize=(10, 5))
plt.bar(brightness_means.keys(), brightness_means.values(), color='purple')
plt.title('Mean Brightness (Grayscale Intensity) per Disease')
plt.ylabel('Brightness')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter plot: brightness vs contrast by disease
plt.figure(figsize=(10, 6))
for disease in diseases:
    x = brightness_data[disease]
    y = feature_data['contrast'][disease]
    if x and y:
        plt.scatter(x, y, label=disease, alpha=0.6, edgecolors='w', s=50)

plt.xlabel('Brightness (Mean Grayscale Intensity)')
plt.ylabel('GLCM Contrast')
plt.title('Brightness vs Contrast by Disease')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()