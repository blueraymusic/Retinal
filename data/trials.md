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


## Experiment 2 — Baseline (360 flips) + Dropout (p=0.2)
- **Change**: Replaced final FC layer with Dropout + Linear:
  ```python
  model.fc = nn.Sequential(
      nn.Dropout(0.2),
      nn.Linear(2048, num_classes)
  )

**Training Accuracy**: 0,6247
**Validation Accuracy**:0,558
**Observation**: Improved the generalization but induced a pronounced overfitting
