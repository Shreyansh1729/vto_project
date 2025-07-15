import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, ControlNetModel

# from .garment_adapter import GarmentAdapter # Temporarily disabled

class TryOnPipeline(nn.Module):
    def __init__(self, unet_path, unet_subfolder, controlnet_path):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder=unet_subfolder)
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path)
        # self.garment_adapter = GarmentAdapter() # Disabled
        self.unet.eval()
        self.controlnet.eval()
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(False)

    def forward(
        self,
        latents: torch.Tensor,
        timestep: int,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        cloth_image: torch.Tensor,
    ):
        controlnet_down_res, controlnet_mid_res = self.controlnet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )
        noise_pred = self.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=controlnet_down_res,
            mid_block_additional_residual=controlnet_mid_res,
        ).sample
        return noise_pred