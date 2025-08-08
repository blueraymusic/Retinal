# Experiment Log

A detailed log tracking model architecture changes, feature engineering (e.g., GLCM), regularization, and performance metrics.

---

## Baseline
- **Model**: ResNet50 (pretrained)
- **Fine-tuned layers**: `layer4`, `fc`
- **FC Layer**: `Linear(2048, num_classes)`
- **Augmentations**: (random flip, rotation, etc.)
- **Dropout**: None
- **GLCM features**: None

**Train Loss**: 0.4883
**Train Accuracy**: 42.22% 
**Validation Loss**: ~0.519 
**Validation Accuracy**: 40.99% 
**Observation**: Severe underfitting, possible regularization imbalance.

---

## Experiment 1 — Image Random Flip
- **Change**: Change the Image Random from 360 to 15 degrees:
  ```python
  img_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  #360 to 15
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1), 
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])}

**Observation**: Model overfitting slightly reduced, but overall accuracy plateaued after ~15 epochs. Large rotations likely introduced noise.

---

## Experiment 2 — Baseline (360 flips) + Dropout (p=0.2)
- **Change**: Replaced final FC layer with Dropout + Linear:
  ```python
  model.fc = nn.Sequential(
      nn.Dropout(0.2),
      nn.Linear(2048, num_classes)
  )

**Training Accuracy**: 0,6247
**Validation Accuracy**:0,558
**Observation**: Improved the generalization but still overfits


---

## Experiment 3 — Image Augmentations + Dropout (p=0.2 && p=0.35 && p=0.5)
- **Change**: Added a dropout layer to the "Experiment 1" model,  in order to reduce the overfitting
  ```python
  model.fc = nn.Sequential(
      nn.Dropout(0.2), #tried for 0.35 && 0.5
      nn.Linear(2048, num_classes)
  )

**Training Accuracies** && **Validation Accuracies** (order of (p=0.2 && p=0.35 && p=0.5)): 
    0,823	  0,6423
    0,8221	0,6561
    0,8065	0,657
    
**Observation**: Stagnent generalization and still overfits


---

## Experiment 4 — V2 Tranformations + GlCM features (Contrast + Dissimilarity)
- **Change**: Implemented GLCM as an additional feature coupled with the V2 model tranformations
  ```python
  import torch
  import numpy as np
  from skimage.feature import graycomatrix, graycoprops
  
  def compute_glcm_features(image_tensor: torch.Tensor) -> torch.Tensor:
      if image_tensor.dim() != 3:
          raise ValueError(f"Expected a 3D tensor (C x H x W), got {image_tensor.shape}")
  
      if image_tensor.shape[0] == 3:
          grayscale = 0.2989 * image_tensor[0] + 0.5870 * image_tensor[1] + 0.1140 * image_tensor[2]
      else:
          grayscale = image_tensor.squeeze(0)
  
      img = (grayscale * 255).clamp(0, 255).detach().cpu().numpy().astype(np.uint8)
  
      if img.ndim != 2:
          raise ValueError(f"Expected 2D grayscale image. Got shape: {img.shape}")
  
      glcm = graycomatrix(img, distances=[1], angles=[0], symmetric=True, normed=True)
      props = ['contrast', 'dissimilarity']
      features = [graycoprops(glcm, prop)[0, 0] for prop in props]
  
      return torch.tensor(features, dtype=torch.float32)

**Training Accuracies** && **Validation Accuracies** (order of (p=0.2 && p=0.35 && p=0.5)): 
    0,823	  0,6423
    0,8221	0,6561
    0,8065	0,657
    
**Observation**: Overfitting but still stagnent values


---

## Experiment 5 — GlCM features (Contrast + Dissimilarity) + Dropout (p=0.2 && p=0.5)
- **Change**: Added a dropout layer to the "Experiment 1" model,  in order to reduce the overfitting
  ```python
  model.fc = nn.Sequential(
      nn.Dropout(0.2), #tried for 0.5
      nn.Linear(2048, num_classes)
  )

**Training Accuracies** && **Validation Accuracies** (order of (p=0.5 && p=0.2)): 
    0.6658 	0.6453
    
**Observation**: Similar training & validation
