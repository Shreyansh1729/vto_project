import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, ControlNetModel, AutoencoderKL

from .garment_adapter import GarmentAdapter

class TryOnPipeline(nn.Module):
    def __init__(self, unet: UNet2DConditionModel, controlnet: ControlNetModel, vae: AutoencoderKL):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.vae = vae
        self.garment_adapter = GarmentAdapter()
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.garment_adapter.train()

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
        with torch.no_grad():
            cloth_latents = self.vae.encode(cloth_image).latent_dist.sample() * self.vae.config.scaling_factor
        
        garment_down_res = self.garment_adapter(cloth_latents)
        
        # Use a zero-tensor for the mid-block residual from the garment adapter for now
        garment_mid_res = torch.zeros_like(controlnet_mid_res)

        combined_down_residuals = [
            c_res + g_res
            for c_res, g_res in zip(controlnet_down_res, garment_down_res)
        ]
        combined_mid_residual = controlnet_mid_res + garment_mid_res
        
        noise_pred = self.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=combined_down_residuals,
            mid_block_additional_residual=combined_mid_residual,
        ).sample

        return noise_pred