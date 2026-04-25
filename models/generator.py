"""
Generator Architecture for Anime Face GAN.
"""
import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator network for DCGAN.
    
    Takes a latent vector (noise) and generates a 64x64 RGB image.
    """
    def __init__(self, latent_dim: int = 100):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        
        # Initial dense layer to project 100-dim vector to 8x8x256 feature map
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * 256, bias=False)
        )
        
        # Transposed convolutional layers for upsampling
        self.conv_blocks = nn.Sequential(
            # Input: (256, 8, 8)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (128, 16, 16)
            
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # Output: (64, 32, 32)
            
            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # Output: (3, 64, 64) in range [-1, 1]
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights from a normal distribution N(0, 0.02)"""
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear)):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Generator.
        
        Args:
            x (torch.Tensor): Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            torch.Tensor: Generated image of shape (batch_size, 3, 64, 64)
        """
        assert x.dim() == 2 and x.size(1) == self.latent_dim, "Input must be (batch_size, latent_dim)"
        
        # Project and reshape
        out = self.fc(x)
        out = out.view(-1, 256, 8, 8)
        
        # Upsample
        out = self.conv_blocks(out)
        return out
