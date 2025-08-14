# Retinal Disease Classification – Updates and Advancements

Tracking experiments and model performance over time.

---

## Versions

Training logs stored in the "trainOuputs/"
### **v1**
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1     | 0.8652     | 0.1436    | 0.7408   | 0.1890  |
| 2     | 0.7168     | 0.2255    | 0.6760   | 0.2442  |
| ...   | ...        | ...       | ...      | ...     |
| 30    | 0.4883     | 0.4222    | 0.5229   | 0.3924  |
<img width="632" height="265" alt="Capture d’écran 2025-07-22 à 16 39 21" src="https://github.com/user-attachments/assets/48200e20-0cbe-458d-829f-fcc250652567" />


---

### **v2**
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 1     | 0.7343     | 0.2171    | 0.5629   | 0.3721  |
| 2     | 0.5123     | 0.4141    | 0.5019   | 0.4244  |
| ...   | ...        | ...       | ...      | ...     |
| 29    | 0.1142     | 0.8373    | 0.4264   | 0.6599  |
| 30    | 0.1152     | 0.8363    | 0.4347   | 0.6599  |

 **Training complete in** `276m 10s`  
 **Best Validation Loss:** `0.4156`

<img width="1426" height="783" alt="Capture d’écran 2025-07-14 à 15 07 39" src="https://github.com/user-attachments/assets/7c3603db-d9f9-417d-a23e-1ddd2b365e26" />

---

## Comparison Summary

| Metric              | v1       | v2       |
|---------------------|----------|----------|
| Initial Train Acc   | 0.1436   | 0.2171   |
| Initial Val Acc     | 0.1890   | 0.3721   |
| Final Train Acc     | ~0.4222  | 0.8363   |
| Final Val Acc       | ~0.3924  | 0.6628   |
| Best Val Loss       | ~0.5229  | 0.4156   |

 **Validation accuracy improved from 39% → 66%**
 [!] Disrepencies in training and validation loss
 Although v2 shows significantly improved validation accuracy, the gap between training and validation performance suggests potential overfitting.

---

## Current Transformations

### v1 vs v2 Transforms 

```python
#v1 Transformations
img_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5, hue=0.3),
        transforms.RandomRotation(degrees=360),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}

#v2 Transformations

img_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
}
```
---
## Data Analysis 
For in-depth metrics, validation trends, and visualization of training dynamics, see the full analysis here:

-> [View Full Data Analysis](data/data_analysis.md)

This includes:
- GLCM texture statistics
  - Dissimilarity
  - Contrast
  - Homogeneity
- RGB intensity spectrum
- Sharpness distribution
- Brightness distribution


---

## v3

- **Incorporate GLCM (Gray-Level Co-occurrence Matrix)** for texture feature extraction.
- Consider **color spectrum features** as additional input dimensions.
- Analyze impact of:
  - Dropout layers during training
  - Alternative activation functions
- Use statistical metrics like **Shannon Entropy** to quantify image disorder and variation.


---

---
## Planned Analysis (Checklist)

- [x] Implemented GLCM as a feature (contrast & dissimilarity -> Texture)
- [x] Implemented penalty loss for co-occurence with "normal" predicitons (.8 confidence)
- [x] Fined tuned more layers to reduce overfitting & augment accuracy -> ["layer2","layer3", "layer4", "fc"]
- [x] Tried Ensemble Bagging to improve generalization and reduce overfitting

- [x] Created a general ensemble model trials (Bagging + Previous Models)



- Targets
  - [x] Achieved 63% without overfitting
  - [] Achieved 80% without overfitting

## Check Progress & Trials
-> [View the Experiment Logs](data/experiment_log.md)

---

