import torch
import torch.nn as nn
from typing import List

class GarmentAdapter(nn.Module):
    """
    The GarmentAdapter network.
    It takes a 4-channel latent representation of a garment and outputs a list of
    feature maps, one for each resolution level of the U-Net's downsampling path.
    """
    def __init__(
        self,
        in_channels: int = 4, # The adapter operates on 4-channel latents
        model_channels: int = 320,
        num_down_blocks: int = 4,
        num_groups: int = 32,
    ):
        super().__init__()
        
        self.down_blocks = nn.ModuleList([])
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # Build the downsampling blocks to match the U-Net/ControlNet structure
        for i in range(num_down_blocks):
            block_out_channels = model_channels * (2**i)
            block_in_channels = model_channels * (2**(i-1)) if i > 0 else model_channels

            block = nn.ModuleList([
                nn.Conv2d(block_in_channels, block_out_channels, kernel_size=3, padding=1, stride=2 if i > 0 else 1),
                nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, padding=1),
                nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, padding=1)
            ])
            self.down_blocks.append(block)

    def forward(self, latents: torch.Tensor) -> List[torch.Tensor]:
        """
        Processes the garment latents and returns a list of feature maps.
        """
        output = []
        x = self.conv_in(latents)

        for block in self.down_blocks:
            for layer in block:
                x = layer(x)
                output.append(x)
        
        # Returns 12 feature maps to match the ControlNet's output for down-blocks
        return output