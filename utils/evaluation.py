"""
Evaluation metrics for GAN: FID and Inception Score.
"""
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
import logging

logger = logging.getLogger(__name__)

class InceptionV3FeatureExtractor(nn.Module):
    """Wrapper for Inception V3 to extract features for FID."""
    def __init__(self):
        super().__init__()
        inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.blocks = nn.ModuleList([
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2), inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2), inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a, inception.Mixed_6b,
            inception.Mixed_6c, inception.Mixed_6d, inception.Mixed_6e,
            inception.Mixed_7a, inception.Mixed_7b, inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        ])
        
    def forward(self, x):
        # Inception v3 expects inputs to be 299x299. For simplicity, we assume
        # input is resized before passing here or we use standard extraction
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        for block in self.blocks:
            x = block(x)
        return x.view(x.size(0), -1)

def calculate_activation_statistics(images, model, batch_size=32, device='cuda'):
    """Calculates mean and covariance of Inception activations."""
    model.eval()
    act = np.empty((len(images), 2048))
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            # Normalize to [0, 1] as expected by inception
            if batch.min() < 0:
                batch = (batch + 1) / 2.0
            pred = model(batch).cpu().numpy()
            act[i:i + batch_size] = pred
            
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Computes the Frechet Distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

def calculate_fid(real_images, fake_images, device='cuda'):
    """Calculates FID score between real and fake images."""
    model = InceptionV3FeatureExtractor().to(device)
    
    mu1, sigma1 = calculate_activation_statistics(real_images, model, device=device)
    mu2, sigma2 = calculate_activation_statistics(fake_images, model, device=device)
    
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def calculate_inception_score(images, batch_size=32, splits=10, device='cuda'):
    """Calculates Inception Score."""
    inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).to(device)
    inception_model.eval()
    
    preds = np.zeros((len(images), 1000))
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device)
            if batch.min() < 0:
                batch = (batch + 1) / 2.0
            batch = torch.nn.functional.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            pred = torch.nn.functional.softmax(inception_model(batch), dim=1).cpu().numpy()
            preds[i:i + batch_size] = pred
            
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
        
    return np.mean(scores), np.std(scores)
