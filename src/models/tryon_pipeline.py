import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, ControlNetModel

from .garment_adapter import GarmentAdapter

class TryOnPipeline(nn.Module):
    """
    The main model with corrected feature fusion logic.
    """
    def __init__(self, unet_path, unet_subfolder, controlnet_path):
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder=unet_subfolder)
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path)
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
        controlnet_cond: torch.Tensor,
        cloth_image: torch.Tensor,
    ):
        # 1. Get pose guidance from ControlNet
        controlnet_down_res, controlnet_mid_res = self.controlnet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )

        # 2. Get clothing features from our Garment Adapter
        # Note: We need to resize the cloth image to match the latent space dimensions
        # before passing it to the adapter.
        cloth_latents = torch.nn.functional.interpolate(cloth_image, size=(64, 48), mode='bilinear')
        garment_features = self.garment_adapter(cloth_latents)
        
        # The adapter gives features for down blocks, we use the ControlNet's mid-block feature
        garment_down_res = garment_features
        garment_mid_res = torch.zeros_like(controlnet_mid_res) # Placeholder for now

        # 3. Combine the residuals
        # We iterate through the list of tensors and add them element-wise
        combined_down_residuals = [
            c_res + g_res for c_res, g_res in zip(controlnet_down_res, garment_down_res)
        ]
        combined_mid_residual = controlnet_mid_res + garment_mid_res
        
        # 4. Pass to the main U-Net
        noise_pred = self.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=combined_down_residuals,
            mid_block_additional_residual=combined_mid_residual,
        ).sample

        return noise_pred