import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, ControlNetModel

from .garment_adapter import GarmentAdapter

class TryOnPipeline(nn.Module):
    """
    The main model that orchestrates the virtual try-on process.
    """
    def __init__(self, unet_path, unet_subfolder, controlnet_path):
        super().__init__()

        print("Loading pre-trained U-Net...")
        self.unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder=unet_subfolder)
        
        print("Loading pre-trained ControlNet...")
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path)

        print("Initializing Garment Adapter...")
        self.garment_adapter = GarmentAdapter()

        self.unet.eval()
        self.controlnet.eval()
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(False)
        self.garment_adapter.train()

    def forward(
        self,
        latents: torch.Tensor,
        timestep: int,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor, # This is the pose map
        cloth_image: torch.Tensor,
    ):
        # 1. Get pose guidance from the ControlNet
        controlnet_down_res, controlnet_mid_res = self.controlnet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond, # THE CORRECTED KEYWORD
            return_dict=False
        )
        
        # NOTE: Placeholder logic for garment features
        # We will properly implement this after we get the training loop running.
        # For now, we will not use the garment_adapter's output.

        # 2. Pass to the main U-Net
        noise_pred = self.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=controlnet_down_res,
            mid_block_additional_residual=controlnet_mid_res,
        ).sample

        return noise_pred