import torch
import torch.nn as nn
from typing import List

class GarmentAdapter(nn.Module):
    """
    The GarmentAdapter network.
    It takes a garment image and outputs a list of feature maps, one for each
    resolution level of the U-Net's downsampling path. This allows for direct,
    element-wise addition with the ControlNet's residual outputs.
    """
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 320,
        # ControlNet has 4 down blocks + 1 mid block, we need to match this.
        # It outputs 12 feature maps in total. We will do the same.
        num_down_blocks: int = 4,
        num_groups: int = 32,
    ):
        super().__init__()
        
        # This will be a list of lists of layers.
        self.down_blocks = nn.ModuleList([])

        # The initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # Build the downsampling blocks
        for i in range(num_down_blocks):
            # Each down block in ControlNet has 3 outputs.
            # We will create 3 convolutional layers for each block.
            block_out_channels = model_channels * (2**i)
            block_in_channels = model_channels * (2**(i-1)) if i > 0 else model_channels

            block = nn.ModuleList([
                nn.Conv2d(block_in_channels, block_out_channels, kernel_size=3, padding=1, stride=2 if i > 0 else 1),
                nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, padding=1),
                nn.Conv2d(block_out_channels, block_out_channels, kernel_size=3, padding=1)
            ])
            self.down_blocks.append(block)

    def forward(self, cloth_image: torch.Tensor) -> List[torch.Tensor]:
        """
        Processes the cloth image and returns a list of feature maps.
        """
        output = []
        x = self.conv_in(cloth_image)

        for block in self.down_blocks:
            for layer in block:
                x = layer(x)
                output.append(x)
        
        # We need 12 feature maps to match the ControlNet output.
        # The last feature map is used for the mid block.
        return output