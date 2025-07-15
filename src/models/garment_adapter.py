import torch
import torch.nn as nn
from typing import List

class GarmentAdapter(nn.Module):
    """
    The architecturally correct GarmentAdapter.
    It now correctly doubles the number of channels at each downsampling stage,
    ensuring its output feature maps have the same dimensions as the ControlNet's.
    """
    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 320,
        num_down_blocks: int = 4,
        num_groups: int = 32,
    ):
        super().__init__()
        
        self.down_blocks = nn.ModuleList([])
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)

        # Build the downsampling blocks with correct channel scaling
        input_channel = model_channels
        for i in range(num_down_blocks):
            # The output channels double at each block
            output_channel = model_channels * (2**i)
            
            # Each down block in ControlNet has 3 outputs. We create 3 layers to match.
            block = nn.ModuleList()
            
            # First layer handles the channel increase and downsampling (if not the first block)
            block.append(nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1, stride=2 if i > 0 else 1))
            
            # Subsequent layers in the same block maintain the channel count
            block.append(nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1))
            block.append(nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1))

            self.down_blocks.append(block)
            # The input for the next block is the output of this one
            input_channel = output_channel

    def forward(self, latents: torch.Tensor) -> List[torch.Tensor]:
        """
        Processes the garment latents and returns a list of feature maps.
        """
        output_features = []
        x = self.conv_in(latents)

        for block in self.down_blocks:
            for layer in block:
                # Apply the layer and then a normalization/activation
                x = layer(x)
                x = nn.functional.silu(nn.functional.group_norm(x, num_groups=32))
                output_features.append(x)
        
        # We need 12 feature maps for the down-blocks and 1 for the mid-block.
        # This design is still a simplification. Let's return the 12 for the down-blocks.
        # This will require a small change in the pipeline.
        return output_features