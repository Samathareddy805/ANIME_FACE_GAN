"""
Main training loop for Anime Face GAN.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

from data.preprocessing import get_dataloader
from data.download_dataset import download_dataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils.training_utils import save_sample_images, plot_loss_curves, save_checkpoint

# Hyperparameters
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LATENT_DIM = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def train():
    """Main training function."""
    logger.info(f"Using device: {DEVICE}")
    
    # Setup data
    data_path = download_dataset()
    dataloader = get_dataloader(data_path, batch_size=BATCH_SIZE)
    
    # Initialize models
    netG = Generator(LATENT_DIM).to(DEVICE)
    netD = Discriminator().to(DEVICE)
    
    # Loss function and Optimizers
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(16, LATENT_DIM, device=DEVICE) # For consistent generation
    
    optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))
    optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
    
    # TensorBoard setup
    writer = SummaryWriter(log_dir="runs/anime_gan")
    
    G_losses = []
    D_losses = []
    
    logger.info("Starting Training Loop...")
    for epoch in range(1, EPOCHS + 1):
        # For tracking loss per epoch
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
        for i, (real_images, _) in enumerate(progress_bar):
            b_size = real_images.size(0)
            real_images = real_images.to(DEVICE)
            
            # Label Smoothing: Real=0.9, Fake=0.1
            real_label = torch.full((b_size, 1), 0.9, dtype=torch.float, device=DEVICE)
            fake_label = torch.full((b_size, 1), 0.1, dtype=torch.float, device=DEVICE)
            
            # ==========================================
            # 1. Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
            # ==========================================
            netD.zero_grad()
            
            # Train with real batch
            output_real = netD(real_images)
            errD_real = criterion(output_real, real_label)
            errD_real.backward()
            D_x = output_real.mean().item()
            
            # Train with fake batch
            noise = torch.randn(b_size, LATENT_DIM, device=DEVICE)
            fake_images = netG(noise)
            output_fake = netD(fake_images.detach())
            errD_fake = criterion(output_fake, fake_label)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()
            
            errD = errD_real + errD_fake
            optimizerD.step()
            
            # ==========================================
            # 2. Train Generator: max log(D(G(z)))
            # ==========================================
            netG.zero_grad()
            
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake_images)
            # Generator wants discriminator to output 1 (real) for its fakes
            errG = criterion(output, torch.ones_like(output, device=DEVICE)) 
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()
            
            # Accumulate metrics
            epoch_g_loss += errG.item()
            epoch_d_loss += errD.item()
            
            progress_bar.set_postfix({'Loss_D': f'{errD.item():.4f}', 'Loss_G': f'{errG.item():.4f}'})
        
        # Save metrics
        G_losses.append(epoch_g_loss / len(dataloader))
        D_losses.append(epoch_d_loss / len(dataloader))
        
        writer.add_scalar('Loss/Generator', G_losses[-1], epoch)
        writer.add_scalar('Loss/Discriminator', D_losses[-1], epoch)
        
        # Generate samples and save checkpoint every 10 epochs (or epoch 1)
        if epoch == 1 or epoch % 10 == 0:
            with torch.no_grad():
                fake = netG(fixed_noise)
            save_sample_images(fake, epoch)
            save_checkpoint(epoch, netG, netD, optimizerG, optimizerD)
            plot_loss_curves(G_losses, D_losses)

    writer.close()
    logger.info("Training Finished.")

if __name__ == '__main__':
    train()
