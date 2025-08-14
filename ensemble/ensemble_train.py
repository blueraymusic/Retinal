import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm

# Custom imports
from chart.utils import load_data, category_percentage, correlation_between_labels, venn_diagram
from glcm.resnet_glcm import ResNetWithInternalGLCM

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def constrained_bce_loss(preds, targets, pos_weight=None, normal_idx=-1):
    """Enhanced loss function with conflict penalty"""
    # Split predictions and targets
    disease_preds = preds[:, :normal_idx]
    normal_preds = preds[:, normal_idx]
    disease_targets = targets[:, :normal_idx]
    normal_targets = targets[:, normal_idx]
    
    # Calculate base losses
    disease_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight[:normal_idx])(disease_preds, disease_targets)
    normal_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight[normal_idx:])(normal_preds, normal_targets)
    
    # Calculate conflict penalty
    disease_probs = torch.sigmoid(disease_preds.detach())
    normal_probs = torch.sigmoid(normal_preds.detach())
    conflict = (disease_probs.max(dim=1).values > 0.8) & (normal_probs > 0.85)
    penalty = conflict.float() * 0.3
    
    # Weighted combination
    total_loss = 0.7*disease_loss + 0.3*normal_loss + penalty.mean()
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
        label = self.img_data.iloc[idx, 1:].astype(float).values
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

def get_pos_weight(df):
    """Calculate positive class weights for loss function"""
    pos_weight = []
    for c in range(df.shape[1]):
        denom = (df.iloc[:, c] == 1).sum()
        weight = (df.iloc[:, c] == 0).sum() / denom if denom != 0 else 1.0
        pos_weight.append(weight)
    return pos_weight

def train_single_model(model, train_df, val_df, img_dir, img_transforms, device, 
                      model_idx, num_epochs=25, early_stopping_patience=5):
    """Train a single model in the ensemble"""
    # Create data loaders
    image_datasets = {
        'train': RetinalDisorderDataset(
            data_file=train_df,
            img_dir=img_dir,
            transform=img_transforms['train']
        ),
        'val': RetinalDisorderDataset(
            data_file=val_df,
            img_dir=img_dir,
            transform=img_transforms['val']
        )
    }
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=48,
                          shuffle=True, num_workers=4),
        'val': DataLoader(image_datasets['val'], batch_size=48,
                        shuffle=False, num_workers=4)
    }
    
    # Training setup
    pos_weight = get_pos_weight(train_df.iloc[:, 1:])
    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
    
    criterion = lambda outputs, targets: constrained_bce_loss(
        outputs, targets, pos_weight=pos_weight, 
        normal_idx=len(train_df.columns[1:]) - 1
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Train model
    best_loss = float('inf')
    epochs_no_improve = 0
    loss_history = {'train': [], 'val': []}
    
    for epoch in range(1, num_epochs + 1):
        print(f'\nModel {model_idx} - Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in tqdm(dataloaders[phase], leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = torch.sigmoid(outputs)
                    preds_rounded = torch.round(preds)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum((preds_rounded == labels).all(dim=1)).item()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            loss_history[phase].append(epoch_loss)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                scheduler.step()
            else:
                if epoch_loss < best_loss - 0.001:  # Early stopping delta
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), f'models/bagging_model_{model_idx}.pth')
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    print(f'No improvement for {epochs_no_improve} epoch(s)')
                
                if epochs_no_improve >= early_stopping_patience:
                    print(f'\nEarly stopping triggered after {epoch} epochs.')
                    model.load_state_dict(best_model_wts)
                    return model, loss_history
    
    return model, loss_history

class BaggingEnsemble:
    def __init__(self, n_estimators=5, num_classes=5):
        self.n_estimators = n_estimators
        self.num_classes = num_classes
        self.models = []
    
    def fit(self, full_train_df, img_dir, img_transforms, device, 
            num_epochs=25, early_stopping_patience=5):
        """Train ensemble of models on bootstrap samples"""
        # Create validation set (same for all models)
        train_df, val_df = train_test_split(full_train_df, train_size=0.9, random_state=42)
        
        for i in range(self.n_estimators):
            print(f"\n=== Training Model {i+1}/{self.n_estimators} ===")
            
            # Create bootstrap sample
            bootstrap_df = resample(train_df, replace=True, n_samples=len(train_df))
            
            # Initialize model
            model = ResNetWithInternalGLCM(num_classes=self.num_classes).to(device)
            
            # Train model
            model, _ = train_single_model(
                model, bootstrap_df, val_df, img_dir, img_transforms, device,
                model_idx=i+1, num_epochs=num_epochs,
                early_stopping_patience=early_stopping_patience
            )
            
            self.models.append(model)
    
    def predict(self, dataloader, device):
        """Get averaged predictions from all models"""
        all_preds = []
        
        with torch.no_grad():
            for model in self.models:
                model.eval()
                model_preds = []
                
                for inputs, _ in dataloader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs)
                    model_preds.append(probs.cpu())
                
                model_preds = torch.cat(model_preds, dim=0)
                all_preds.append(model_preds)
        
        return torch.mean(torch.stack(all_preds), dim=0)

def plot_loss_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(20, 8))
    epochs = range(1, len(history['train']) + 1)
    plt.plot(epochs, history['train'], lw=3, color='red', label='Training Loss')
    plt.plot(epochs, history['val'], lw=3, color='green', label='Validation Loss')
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.title('Training and Validation Loss', fontsize=20)
    plt.legend(loc='best', fontsize=16)
    plt.grid()
    plt.savefig('loss_plot.png')
    plt.show()

def main():
    # Data preparation
    train_df = load_data('data/train/train.csv', ',')
    disease_labels = train_df.columns[1:]
    
    # Image transformations
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
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # Create output directory
    if not os.path.exists('models'):
        os.mkdir('models')
    
    # Initialize and train ensemble
    ensemble = BaggingEnsemble(
        n_estimators=5,  # Number of models in ensemble
        num_classes=len(disease_labels)
    )
    
    ensemble.fit(
        full_train_df=train_df,
        img_dir='data/train/train/',
        img_transforms=img_transforms,
        device=device,
        num_epochs=40,
        early_stopping_patience=5
    )
    
    # Evaluate on validation set
    val_df = train_test_split(train_df, train_size=0.9, random_state=42)[1]
    val_dataset = RetinalDisorderDataset(
        data_file=val_df,
        img_dir='data/train/train/',
        transform=img_transforms['val']
    )
    val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False)
    
    # Get predictions   
    val_preds = ensemble.predict(val_loader, device)
    val_preds_binary = (val_preds > 0.5).float()
    
    # Calculate accuracy
    val_labels = torch.tensor(val_df.iloc[:, 1:].values, dtype=torch.float32)
    correct = (val_preds_binary == val_labels).all(dim=1).sum().item()
    accuracy = correct / len(val_df)
    print(f"\nEnsemble Validation Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()