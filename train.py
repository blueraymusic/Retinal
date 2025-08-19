import os
import time
import copy
import numpy as np
import json

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import f1_score


# utils.py for visuals

#from utils import load_data, category_percentage, correlation_between_labels, venn_diagram
from chart.utils import load_data, category_percentage, correlation_between_labels, venn_diagram
from glcm.resnet_glcm import ResNetWithInternalGLCM  # custom GLCM-enhanced model




"""
- Transfer Learning: ResNet-50 pretrained on ImageNet is fine-tuned 
  for retinal disease classification.

- Data Augmentation: Aggressive augmentations (flips, jitter, rotations) 
  are applied during training to increase robustness.

- Access the resnet_glcm to see the "ResNetWithInternalGLCM" 
"""

def constrained_bce_loss(preds, targets, pos_weight=None, normal_idx=-1):
    """
    BCE loss with a differentiable conflict penalty for normal+disease conflicts.
    
    Args:
        preds: [batch_size, num_classes] raw logits
        targets: same shape, float labels (0/1)
        pos_weight: tensor for class imbalance
        normal_idx: index of the 'normal' class
    """
    # Split disease vs normal
    disease_preds = preds[:, :normal_idx]
    normal_preds = preds[:, normal_idx]

    disease_targets = targets[:, :normal_idx]
    normal_targets = targets[:, normal_idx]

    # Base BCE losses
    disease_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight[:normal_idx])(disease_preds, disease_targets)
    normal_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight[normal_idx:])(normal_preds, normal_targets)

    # Sigmoid probabilities
    disease_probs = torch.sigmoid(disease_preds)
    normal_probs = torch.sigmoid(normal_preds)

    # Differentiable conflict score
    # Penalizes high normal + high disease prediction
    # Soft margin: values > 1.8 (0.9+0.9) start to get penalized
    max_disease = disease_probs.max(dim=1).values
    conflict_score = torch.clamp(max_disease + normal_probs - 1.8, min=0)
    penalty = conflict_score.mean() * 0.3  # Weight of penalty

    # Total weighted loss
    total_loss = 0.7 * disease_loss + 0.3 * normal_loss + penalty

    return total_loss



class RetinalDisorderDataset(Dataset):
    def __init__(self, data_file, img_dir, transform=None):
        self.img_data = data_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_data.iloc[idx]['filename'])
        image = read_image(img_path) 
        image = self.transform(image)
        
        # Ensure label values are float type (fix conversion error)
        label = self.img_data.iloc[idx, 1:].astype(float).values
        label = torch.tensor(label, dtype=torch.float32)
        return image, label


def plot_loss_history(train_loss, val_loss):
    plt.figure(figsize=(20, 8))
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, lw=3, color='red', label='Training Loss')
    plt.plot(epochs, val_loss, lw=3, color='green', label='Validation Loss')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Training and Validation Loss', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('loss_plot.png')
    plt.show()


def get_pos_weight(df):
    pos_weight = []
    for c in range(df.shape[1]):
        denom = (df.iloc[:, c] == 1).sum()
        weight = (df.iloc[:, c] == 0).sum() / denom if denom != 0 else 1.0
        pos_weight.append(weight)
    return pos_weight

"""
less stricter validations
"""

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25, model_name=None,
                early_stopping_patience=5, early_stopping_delta=0.0, use_f1_early_stop=False):
    model_name = model_name if model_name else model.__class__.__name__

    if not os.path.exists('models'):
        os.mkdir('models')

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = float('-inf')  # Track best F1 or loss
    epochs_no_improve = 0

    loss_history = {'train': [], 'val': []}

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            all_labels = []
            all_preds = []

            for inputs, labels in tqdm(dataloaders[phase], leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    preds = torch.sigmoid(outputs)
                    preds_rounded = torch.round(preds)   #after the sigmoid transformation

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_labels.append(labels.detach().cpu())
                all_preds.append(preds_rounded.detach().cpu())

                #------------------ Saving for Analysis ------------------
                """data = {
                    pre_rounded: 
                }
                try:
                    with open("pred_data.json", 'r') as file:
                        json.load(file)
                
                except FileNotFoundError:
                    data = []
                

                """

            all_labels = torch.cat(all_labels, dim=0).numpy()
            all_preds = torch.cat(all_preds, dim=0).numpy()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = (all_preds == all_labels).mean()  # Per-label accuracy
            epoch_f1_micro = f1_score(all_labels, all_preds, average='micro')
            epoch_f1_macro = f1_score(all_labels, all_preds, average='macro')

            loss_history[phase].append(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} '
                  f'F1_micro: {epoch_f1_micro:.4f} F1_macro: {epoch_f1_macro:.4f}')

            # Scheduler step
            if phase == 'train':
                scheduler.step()
            else:
                current_score = epoch_f1_micro if use_f1_early_stop else -epoch_loss  # maximize F1 or minimize loss
                if current_score > best_score + early_stopping_delta:
                    print(f'Validation improved: {best_score:.4f} → {current_score:.4f}')
                    best_score = current_score
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), f'models/{model_name}.pth')
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f'No improvement for {epochs_no_improve} epoch(s)')

                if epochs_no_improve >= early_stopping_patience:
                    print(f'\nEarly stopping triggered after {epoch} epochs.')
                    model.load_state_dict(best_model_wts)
                    return model, loss_history

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    print(f'Best val Score: {best_score:.4f}')

    model.load_state_dict(best_model_wts)
    return model, loss_history


if __name__ == '__main__':
    # Load CSV data - adjust path as necessary
    train_df = load_data('data/train/train.csv', ',')

    # Labels excluding filename column (Because the excel starts with 'filename')
    disease_labels = train_df.columns[1:]

    # Data splits (90% to 10%)
    train_data, val_data = train_test_split(train_df, train_size=0.9, random_state=42)

    # Image transformations for augmentation and normalization
    """
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
    """
    img_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.2), 
        transforms.RandomVerticalFlip(p=0.2),           
        transforms.RandomRotation(degrees=5),        
        transforms.ColorJitter(brightness=0.15, contrast=0.25, saturation=0.25), 
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.1)),
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




    data_df = {'train': train_data, 'val': val_data}

    image_datasets = {x: RetinalDisorderDataset(data_file=data_df[x],
                                               img_dir='data/train/train/',
                                               transform=img_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=48,
                                 shuffle=True, num_workers=4)
                   for x in ['train', 'val']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    """model = models.resnet50(pretrained=True)

    # Freeze layers except last block & fc
    for name, param in model.named_parameters():
        #if not any(layer in name for layer in ['layer3', 'layer4', 'fc']):
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(disease_labels))
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        #nn.Dropout(p=0.35),  # 20% dropout before final layer
        nn.Linear(num_ftrs, len(disease_labels))
    )
    model = model.to(device)"""
    
    model = ResNetWithInternalGLCM(num_classes=len(disease_labels)).to(device)


    pos_weight = get_pos_weight(train_df.iloc[:, 1:])
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)

    #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = lambda outputs, targets: constrained_bce_loss(
    outputs, targets, pos_weight=pos_weight, normal_idx=len(disease_labels) - 1
)

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00003, weight_decay=1e-4) #try lr = 0.00005 

    #scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    model, loss_history = train_model(
        model, criterion, optimizer, scheduler, dataloaders, device,
        num_epochs=40,
        model_name='v4_best_b',
        early_stopping_patience=5,
        early_stopping_delta=0.001
    )

    plot_loss_history(loss_history['train'], loss_history['val'])
