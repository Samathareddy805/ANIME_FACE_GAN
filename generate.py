"""
Script for generating anime faces using a trained GAN model.
"""
import torch
import torchvision.utils as vutils
import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from models.generator import Generator
from utils.training_utils import load_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100

def generate_faces(generator: torch.nn.Module, num_images: int, out_dir: str = "generated_samples") -> None:
    """Generates a batch of images and saves them as a grid."""
    os.makedirs(out_dir, exist_ok=True)
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(num_images, LATENT_DIM, device=DEVICE)
        fake_images = generator(noise)
        
    vutils.save_image(fake_images, f"{out_dir}/generated_grid.png", normalize=True, value_range=(-1, 1), nrow=int(num_images**0.5))
    print(f"Saved {num_images} generated images to {out_dir}/generated_grid.png")

def interpolate(generator: torch.nn.Module, steps: int = 10, out_dir: str = "generated_samples") -> None:
    """Interpolates between two random latent vectors and saves the result."""
    os.makedirs(out_dir, exist_ok=True)
    generator.eval()
    
    with torch.no_grad():
        z1 = torch.randn(1, LATENT_DIM, device=DEVICE)
        z2 = torch.randn(1, LATENT_DIM, device=DEVICE)
        
        alphas = torch.linspace(0, 1, steps).unsqueeze(1).to(DEVICE)
        # Spherical interpolation could be used, but linear is simpler for demonstration
        z_interp = z1 * (1 - alphas) + z2 * alphas 
        
        generated = generator(z_interp)
        
    vutils.save_image(generated, f"{out_dir}/interpolation.png", normalize=True, value_range=(-1, 1), nrow=steps)
    print(f"Saved interpolation sequence to {out_dir}/interpolation.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Anime Faces using DCGAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint (.pth)")
    parser.add_argument("--num_images", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--interpolate", action="store_true", help="Generate latent space interpolation demo")
    
    args = parser.parse_args()
    
    netG = Generator(LATENT_DIM).to(DEVICE)
    try:
        load_checkpoint(args.checkpoint, netG)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)
        
    generate_faces(netG, args.num_images)
    
    if args.interpolate:
        interpolate(netG)
