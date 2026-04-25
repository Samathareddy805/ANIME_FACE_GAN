"""
Script to evaluate a trained GAN model using FID and Inception Scores.
"""
import torch
import argparse
import os
import json
from tqdm import tqdm

from models.generator import Generator
from data.download_dataset import download_dataset
from data.preprocessing import get_dataloader
from utils.training_utils import load_checkpoint
from utils.evaluation import calculate_fid, calculate_inception_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100

def evaluate(generator_path: str, num_samples: int = 1000):
    """Evaluates the model by calculating FID and IS."""
    print(f"Evaluating model on {DEVICE} with {num_samples} samples...")
    
    # 1. Load Generator
    netG = Generator(LATENT_DIM).to(DEVICE)
    load_checkpoint(generator_path, netG)
    netG.eval()
    
    # 2. Get Real Images
    data_path = download_dataset()
    # Use larger batch size for faster evaluation dataloading
    dataloader = get_dataloader(data_path, batch_size=256)
    
    real_images = []
    print("Collecting real images...")
    for imgs, _ in dataloader:
        real_images.append(imgs)
        if sum([x.size(0) for x in real_images]) >= num_samples:
            break
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    # 3. Generate Fake Images
    print("Generating fake images...")
    fake_images = []
    with torch.no_grad():
        for _ in tqdm(range(0, num_samples, 256)):
            b_size = min(256, num_samples - sum([x.size(0) for x in fake_images]))
            if b_size <= 0: break
            noise = torch.randn(b_size, LATENT_DIM, device=DEVICE)
            fake_images.append(netG(noise).cpu())
    fake_images = torch.cat(fake_images, dim=0)
    
    # 4. Calculate Metrics
    print("Calculating Inception Score...")
    is_mean, is_std = calculate_inception_score(fake_images, device=DEVICE)
    print(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}")
    
    print("Calculating FID...")
    fid_score = calculate_fid(real_images, fake_images, device=DEVICE)
    print(f"FID Score: {fid_score:.4f}")
    
    # 5. Save Report
    os.makedirs("results", exist_ok=True)
    report = {
        "num_samples": num_samples,
        "inception_score_mean": float(is_mean),
        "inception_score_std": float(is_std),
        "fid_score": float(fid_score)
    }
    
    with open("results/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=4)
    print("Evaluation report saved to results/evaluation_report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to evaluate on")
    args = parser.parse_args()
    
    evaluate(args.checkpoint, args.num_samples)
