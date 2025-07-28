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