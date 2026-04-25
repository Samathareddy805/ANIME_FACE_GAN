"""
Utility functions for GAN training: logging, checkpointing, and visualization.
"""
import os
import torch
import logging
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from typing import Tuple, List

logger = logging.getLogger(__name__)

def save_sample_images(images: torch.Tensor, epoch: int, out_dir: str = "samples") -> None:
    """
    Saves a grid of sample images.
    
    Args:
        images (torch.Tensor): Batch of images to save in range [-1, 1].
        epoch (int): Current epoch number.
        out_dir (str): Directory to save images.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Normalize from [-1, 1] to [0, 1] for saving
    vutils.save_image(images.detach(), f"{out_dir}/epoch_{epoch:03d}.png", normalize=True, value_range=(-1, 1), nrow=4)

def plot_loss_curves(g_losses: List[float], d_losses: List[float], out_dir: str = "results") -> None:
    """
    Plots and saves Generator and Discriminator loss curves.
    
    Args:
        g_losses (List[float]): List of generator losses over time.
        d_losses (List[float]): List of discriminator losses over time.
        out_dir (str): Directory to save the plot.
    """
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G Loss")
    plt.plot(d_losses, label="D Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "loss_curves.png"))
    plt.close()

def save_checkpoint(epoch: int, generator: torch.nn.Module, discriminator: torch.nn.Module, 
                    g_optimizer: torch.optim.Optimizer, d_optimizer: torch.optim.Optimizer,
                    out_dir: str = "checkpoints") -> None:
    """
    Saves model weights and optimizer states.
    
    Args:
        epoch (int): Current epoch number.
        generator, discriminator (nn.Module): Models to save.
        g_optimizer, d_optimizer (Optimizer): Optimizers to save.
        out_dir (str): Directory to save checkpoints.
    """
    os.makedirs(out_dir, exist_ok=True)
    g_path = os.path.join(out_dir, f"generator_epoch_{epoch}.pth")
    d_path = os.path.join(out_dir, f"discriminator_epoch_{epoch}.pth")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': g_optimizer.state_dict(),
    }, g_path)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': discriminator.state_dict(),
        'optimizer_state_dict': d_optimizer.state_dict(),
    }, d_path)
    logger.info(f"Checkpoints saved to {out_dir} at epoch {epoch}")

def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> int:
    """
    Loads model weights and (optionally) optimizer state.
    
    Args:
        filepath (str): Path to the .pth file.
        model (nn.Module): The PyTorch model to populate.
        optimizer (Optimizer, optional): The optimizer to populate.
        
    Returns:
        int: The epoch at which the checkpoint was saved.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
        
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    epoch = checkpoint.get('epoch', 0)
    logger.info(f"Loaded checkpoint from {filepath} (Epoch {epoch})")
    return epoch
