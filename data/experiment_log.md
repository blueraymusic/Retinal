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


---

# Model Experiment Summary (Cleaned)

| Model                                            | Dropout Rate          | GLCM Features                  | Training Accuracy | Validation Accuracy | Fine-tuned Layers / Modification   | Change Description                                                 | Notes                                                                  | Early Stop                             | Type                         | Activation Function |
| ------------------------------------------------ | --------------------- | ------------------------------ | ----------------- | ------------------- | ---------------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------- | -------------------------------------- | ---------------------------- | ------------------- |
| model\_v1.pth                                    | —                     | —                              | 0.4238            | 0.4099              | Layer 4 (last block)               | Baseline model                                                     | —                                                                      | —                                      | Convolutional Neural Network | ReLU                |
| model\_v1\_dropout.pth                           | 20%                   | —                              | 0.6247            | 0.5581              | Same as baseline                   | Added dropout                                                      | Improved generalization                                                | —                                      | —                            | —                   |
| model\_v2.pth                                    | —                     | —                              | 0.8363            | 0.6599              | Modified normalization             | Limited augmentation (±15°)                                        | Higher accuracy in training and validation                             | —                                      | —                            | —                   |
| v2\_20\_dropout.pth                              | 20%                   | —                              | 0.8230            | 0.6423              | Same as v2                         | Added dropout                                                      | Slight dip in performance                                              | —                                      | —                            | —                   |
| v2\_35\_dropout.pth                              | 35%                   | —                              | 0.8221            | 0.6561              | —                                  | Increased dropout                                                  | Slight recovery                                                        | —                                      | —                            | —                   |
| v2\_50\_dropout.pth                              | 50%                   | —                              | 0.8065            | 0.6570              | —                                  | High dropout                                                       | Plateaued                                                              | —                                      | —                            | —                   |
| model\_v3\_based\_v2.pth                         | —                     | \['contrast', 'dissimilarity'] | 0.8020            | 0.6570              | v2 augmentation                    | Still overfitting but slight improvement                           | "patience=5, delta=0.001"                                              | —                                      | —                            | —                   |
| v3\_v1\_l2.pth                                   | 50%                   | —                              | 0.6868            | 0.6424              | \["layer3", "layer4", "fc"]        | Baseline augmentations                                             | Slight overfitting reduction                                           | —                                      | —                            | —                   |
| v3\_20\_v1.pth                                   | 30%                   | —                              | 0.6723            | 0.6308              | —                                  | Limited augmentation (±180°)                                       | —                                                                      | "patience=3, delta=0.001"              | —                            | —                   |
| v3\_v2\_.pth                                     | 30%                   | —                              | 0.8531            | 0.6628              | —                                  | Limited augmentation (±15°)                                        | Still overfitting                                                      | —                                      | —                            | —                   |
| v4.pth                                           | 10%                   | —                              | 0.9434            | 0.7180              | —                                  | Limited augmentation (±5°), lr 0.0001 → 0.00003                    | Severe overfitting                                                     | "patience=5, delta=0.001, epoch += 10" | —                            | —                   |
| v4\_.pth                                         | 30% - 50% - 30%       | —                              | 0.6619            | 0.6337              | \["layer2","layer3","layer4","fc"] | Limited augmentation (±5°), SiLU, GeLU & ReLU, lr 0.0001 → 0.00003 | Learning, generalizing, improving well; penalty 0.3 (any) & 0.5 normal | —                                      | —                            | SiLU, ReLU, GELU    |
| v4\_less\_strict.pth                             | —                     | —                              | 0.6493            | 0.6279              | —                                  | —                                                                  | 0.7 for normal penalty                                                 | —                                      | —                            | —                   |
| v4\_d\_changed.pth                               | 30% - 30% - 30% - 30% | —                              | 0.6386            | 0.6130              | —                                  | —                                                                  | 0.8 for normal penalty                                                 | —                                      | —                            | —                   |
| Bagging\_ensemble (4 models)                     | —                     | —                              | 0.591 (Avg)       | —                   | —                                  | Ensemble Test Accuracy: 0.3171                                     | Correct predictions: 111/350                                           | —                                      | —                            | —                   |
| Ensemble\_v4\_except\_mainv4                     | —                     | —                              | 0.6499 (Avg)      | —                   | All v4 except main 'v4'            | —                                                                  | "accuracy": 0.3343, "correct\_predictions": 117                        | —                                      | —                            | —                   |
| Ensemble\_v4\_except\_mainv4 + Bagging\_ensemble | —                     | —                              | 0.6203 (Avg)      | —                   | Bagging Models + V4 except v4.pth  | —                                                                  | "accuracy": 0.3229, "correct\_predictions": 113                        | —                                      | —                            | —                   |
| v4\_best.pth                                     | 30% - 50% - 30%       | —                              | 0.6157            | 0.5814              | —                                  | —                                                                  | —                                                                      | —                                      | —                            | —                   |
| v4\_best\_b.pth                                  | 30% - 50% - 30%       | —                              | 0.9342            | 0.9132              | —                                  | —                                                                  | 0.3 for normal penalty                                                 | —                                      | —                            | —                   |

---

## Model Comparison Sheet
- Excel that sums up to overall achievements and progress of each trials
- Path: https://docs.google.com/spreadsheets/d/1m7Sh9IqNyRXR4kUhNtJr5ysHhbIO8t5taDWMtYt0KWE/edit?usp=sharing

