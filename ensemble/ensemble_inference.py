import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from glcm.resnet_glcm import ResNetWithInternalGLCM

class RetinalDisorderDataset(Dataset):
    def __init__(self, data_file, img_dir, transform=None):
        self.img_data = pd.read_csv(data_file, index_col=0)
        self.img_dir = img_dir
        self.transform = transform
        self.label_columns = self.img_data.columns[1:]

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_data.iloc[idx]['filename'])
        image = read_image(img_path)
        image = self.transform(image)
        label = self.img_data.iloc[idx, 1:].astype(float).values
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

class ConfidenceEnsemble:
    def __init__(self, model_paths, num_classes, device, confidence_thresh=0.85):
        self.models = []
        self.num_classes = num_classes
        self.device = device
        self.confidence_thresh = confidence_thresh
        
        missing_models = [p for p in model_paths if not os.path.exists(p)]
        if missing_models:
            raise FileNotFoundError(f"Model files not found: {missing_models}")
        
        for path in model_paths:
            try:
                model = ResNetWithInternalGLCM(num_classes=7).to(device)
                state_dict = torch.load(path, map_location=device)
                
                if num_classes != 7:
                    model = self._adapt_model_for_classes(model, state_dict, num_classes)
                else:
                    model.load_state_dict(state_dict)
                
                model.eval()
                self.models.append(model)
            except Exception as e:
                raise RuntimeError(f"Error loading model {path}: {str(e)}")

    def _adapt_model_for_classes(self, model, state_dict, num_classes):
        """Handle class number mismatch between saved model and current needs"""
        old_weight = state_dict['classifier.10.weight']
        old_bias = state_dict['classifier.10.bias']
        
        new_model = ResNetWithInternalGLCM(num_classes=num_classes).to(self.device)
        new_state_dict = new_model.state_dict()
        
        # Copy all weights except final layer
        for key in state_dict:
            if key not in ['classifier.10.weight', 'classifier.10.bias']:
                new_state_dict[key] = state_dict[key]
        
        # Initialize final layer
        new_state_dict['classifier.10.weight'][:7] = old_weight
        new_state_dict['classifier.10.bias'][:7] = old_bias
        
        if num_classes > 7:
            new_state_dict['classifier.10.weight'][7:] = nn.init.xavier_normal_(
                torch.empty(num_classes-7, old_weight.shape[1]))
            new_state_dict['classifier.10.bias'][7:] = 0.0
        
        new_model.load_state_dict(new_state_dict)
        return new_model

    def predict(self, dataloader):
        all_results = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Processing batches"):
                inputs = inputs.to(self.device)
                batch_probs = []
                
                for model in self.models:
                    outputs = model(inputs)
                    probs = torch.sigmoid(outputs)
                    batch_probs.append(probs.cpu())
                
                avg_probs = torch.mean(torch.stack(batch_probs), dim=0)
                final_preds = self._apply_confidence_threshold(avg_probs)
                
                batch_results = {
                    'filenames': [dataloader.dataset.img_data.iloc[i]['filename'] 
                                for i in range(len(labels))],
                    'true_labels': labels.numpy().tolist(),
                    'avg_probs': avg_probs.numpy().tolist(),
                    'final_preds': final_preds.numpy().tolist()
                }
                all_results.append(batch_results)
        
        return all_results

    def _apply_confidence_threshold(self, probs):
        final_preds = probs.clone()
        max_probs, max_indices = torch.max(probs, dim=1)
        high_conf_mask = max_probs > self.confidence_thresh
        
        if high_conf_mask.any():
            final_preds[high_conf_mask] = 0
            final_preds[high_conf_mask, max_indices[high_conf_mask]] = 1
        
        final_preds[~high_conf_mask] = (final_preds[~high_conf_mask] > 0.5).float()
        return final_preds

def test_ensemble():
    # Configuration 

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_df_path = os.path.join('data', 'test', 'test.csv')
    img_dir = os.path.join('data', 'test')
    
    model_paths = [
        os.path.join('models', 'v4_.pth'),
        os.path.join('models', 'v4_d_changed.pth'),
        os.path.join('models', 'v4_less_strict.pth'),
        os.path.join('models_bagging', 'bagging_model_1.pth'),
        os.path.join('models_bagging', 'bagging_model_2.pth'),
        os.path.join('models_bagging', 'bagging_model_3.pth'),
        os.path.join('models_bagging', 'bagging_model_4.pth'),
    ]
    
    output_file = 'ensemble_baggings&v4_var.json'
    
    # Verify test data exists
    if not os.path.exists(test_df_path):
        raise FileNotFoundError(f"Test CSV not found at {test_df_path}")
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found at {img_dir}")
    
    # Load test data
    test_df = pd.read_csv(test_df_path, index_col=0)
    num_classes = len(test_df.columns[1:])
    
    # Image transformations
    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create dataset and loader
    test_dataset = RetinalDisorderDataset(test_df_path, img_dir, img_transforms)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize and run ensemble
    ensemble = ConfidenceEnsemble(model_paths, num_classes, device)
    results = ensemble.predict(test_loader)
    
    # Calculate and save results
    accuracy = _save_results(results, output_file, test_dataset.label_columns)
    return accuracy

def _save_results(results, output_file, label_columns):
    all_true = []
    all_preds = []
    
    for batch in results:
        all_true.extend(batch['true_labels'])
        all_preds.extend(batch['final_preds'])
    
    true_labels = np.array(all_true)
    pred_labels = np.array(all_preds)
    correct = np.all(true_labels == pred_labels, axis=1).sum()
    total = len(true_labels)
    accuracy = correct / total
    
    print(f"\nEnsemble Test Accuracy: {accuracy:.4f}")
    print(f"Correct predictions: {correct}/{total}")
    
    output_data = {
        'accuracy': accuracy,
        'correct_predictions': int(correct),
        'total_samples': int(total),
        'label_columns': label_columns.tolist(),
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_file}")
    
    return accuracy

if __name__ == '__main__':
    try:
        test_accuracy = test_ensemble()
    except Exception as e:
        print(f"Error: {str(e)}")