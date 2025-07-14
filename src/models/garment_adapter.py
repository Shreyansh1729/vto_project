import torch
import torch.nn as nn

class GarmentAdapter(nn.Module):
    """
    A lightweight convolutional network designed to extract features from a garment image.
    These features are then used to condition the main diffusion model (U-Net).
    
    The architecture consists of a series of downsampling convolutional blocks.
    """
    def __init__(
        self,
        in_channels: int = 3,          # Input channels (3 for RGB cloth image)
        model_channels: int = 320,     # The number of channels in the U-Net we are injecting into
        num_layers: int = 3,           # Number of downsampling layers
        num_groups: int = 32,          # Number of groups for GroupNorm
    ):
        super().__init__()
        
        # The initial convolution layer to bring the input image into the feature space
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        self.blocks = nn.ModuleList()
        
        # Build the downsampling blocks
        for i in range(num_layers):
            # Each block consists of two convolutions and a downsampling layer
            block = nn.Sequential(
                nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, model_channels),
                nn.SiLU(), # Swish activation function
                nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, model_channels),
                nn.SiLU(),
                nn.Conv2d(model_channels, model_channels, kernel_size=3, stride=2, padding=1) # Downsampling
            )
            self.blocks.append(block)

        # A final convolution to refine the output
        self.conv_out = nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1)
        
    def forward(self, cloth_image: torch.Tensor) -> torch.Tensor:
        """
        Processes the cloth image to extract feature maps.
        
        Args:
            cloth_image (torch.Tensor): The input tensor of the cloth image.
        
        Returns:
            torch.Tensor: The extracted feature map.
        """
        # 1. Initial convolution
        x = self.conv_in(cloth_image)
        
        # 2. Pass through the downsampling blocks
        for block in self.blocks:
            x = block(x)
            
        # 3. Final convolution
        return self.conv_out(x)