import torch
import torch.nn as nn
from typing import List

class GarmentAdapter(nn.Module):
    """
    A more sophisticated GarmentAdapter that outputs feature maps at multiple resolutions,
    matching the structure of a U-Net's downsampling path.
    """
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 320,
        num_layers: int = 4, # We need 4 levels for Stable Diffusion 1.5
        num_groups: int = 32,
    ):
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList()
        
        # Build the downsampling blocks
        for _ in range(num_layers):
            block = nn.Sequential(
                nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, model_channels), nn.SiLU(),
                nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, model_channels), nn.SiLU(),
                nn.Conv2d(model_channels, model_channels, kernel_size=3, stride=2, padding=1)
            )
            self.blocks.append(block)

    def forward(self, cloth_image: torch.Tensor) -> List[torch.Tensor]:
        """
        Processes the cloth image and returns a list of feature maps from each downsampling level.
        """
        feature_maps = []
        x = self.conv_in(cloth_image)
        
        # The first feature map is from the initial convolution
        feature_maps.append(x)
        
        # Pass through the downsampling blocks and collect feature maps
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)
            
        return feature_maps