"""
Discriminator Architecture for Anime Face GAN.
"""
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator network for DCGAN.
    
    Classifies 64x64 RGB images as real (from dataset) or fake (from Generator).
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv_blocks = nn.Sequential(
            # Input: (3, 64, 64)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # Output: (64, 32, 32)
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # Output: (128, 16, 16)
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            # Output: (256, 8, 8)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes weights from a normal distribution N(0, 0.02)"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Discriminator.
        
        Args:
            x (torch.Tensor): Image tensor of shape (batch_size, 3, 64, 64)
            
        Returns:
            torch.Tensor: Probability of the image being real, shape (batch_size, 1)
        """
        assert x.dim() == 4 and x.size(1) == 3, "Input must be (batch_size, 3, 64, 64)"
        
        features = self.conv_blocks(x)
        prob = self.classifier(features)
        return prob
