import torch
import torch.nn as nn
from typing import List

class GarmentAdapter(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 320,
        num_layers: int = 4,
        num_groups: int = 32,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
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
        feature_maps = []
        x = self.conv_in(cloth_image)
        feature_maps.append(x)
        for block in self.blocks:
            x = block(x)
            feature_maps.append(x)
        return feature_maps