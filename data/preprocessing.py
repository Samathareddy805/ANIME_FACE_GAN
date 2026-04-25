"""
Module for data preprocessing and PyTorch DataLoader creation.
"""
import os
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def get_dataloader(data_dir: str, batch_size: int = 64, image_size: int = 64) -> DataLoader:
    """
    Creates a PyTorch DataLoader with standard preprocessing transforms.
    
    Args:
        data_dir (str): Path to the root directory containing images.
        batch_size (int): Number of images per batch.
        image_size (int): Target size for resizing images (image_size x image_size).
        
    Returns:
        DataLoader: PyTorch DataLoader ready for training.
    """
    logger.info(f"Setting up DataLoader for dataset at {data_dir}")
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    try:
        dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        logger.info(f"DataLoader setup complete. Total batches: {len(dataloader)}")
        return dataloader
    except Exception as e:
        logger.error(f"Error creating DataLoader: {e}")
        raise
