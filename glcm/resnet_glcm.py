import torch
import torch.nn as nn
from torchvision import models
from glcm.glcm import compute_glcm_features

class ResNetWithInternalGLCM(nn.Module):
    def __init__(self, num_classes: int):
        super(ResNetWithInternalGLCM, self).__init__()
        self.base_model = models.resnet50(pretrained=True)

        # More strategic layer freezing - unfreeze gradually
        for name, param in self.base_model.named_parameters():
            if 'layer2' in name:
                param.requires_grad = True  # Allow some mid-level feature adaptation
            elif 'layer3' in name or 'layer4' in name:
                param.requires_grad = True  # Definitely train higher layers
            elif 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  # Freeze early layers

        # Replace the ResNet FC head with identity
        num_resnet_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        # GLCM feature processing - give it more capacity
        self.glc_processor = nn.Sequential(
            nn.Linear(2, 32),  # Expand GLCM features
            nn.SiLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3)
        )

        # Enhanced classifier with attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(num_resnet_features + 32, 256),
            nn.ReLU(),
            nn.Linear(256, num_resnet_features + 32),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_resnet_features + 32),
            nn.Dropout(0.1), #5 -> 3 -> 1
            nn.Linear(num_resnet_features + 32, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.1), #4 -> 3 -> 1
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1), #3 -> 3 -> 1
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)
        device = x.device

        # Extract ResNet features
        features = self.base_model(x)

        # Extract and process GLCM features
        glcm_features = []
        for i in range(batch_size):
            glcm_feat = compute_glcm_features(x[i])
            glcm_features.append(glcm_feat)
        
        glcm_tensor = torch.stack(glcm_features).to(device)
        
        # Normalize only if batch_size > 1
        if batch_size > 1:
            glcm_tensor = (glcm_tensor - glcm_tensor.mean(dim=0)) / (glcm_tensor.std(dim=0) + 1e-6)
        
        # Process GLCM features
        processed_glcm = self.glc_processor(glcm_tensor)
        
        # Combine features
        combined = torch.cat([features, processed_glcm], dim=1)
        
        # Apply attention mechanism
        attention_weights = self.attention(combined)
        attended_features = combined * attention_weights
        
        return self.classifier(attended_features)