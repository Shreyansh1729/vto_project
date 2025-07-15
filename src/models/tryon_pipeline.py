import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, ControlNetModel, AutoencoderKL

from .garment_adapter import GarmentAdapter

class TryOnPipeline(nn.Module):
    """
    The final, correct TryOnPipeline with proper latent-space feature fusion.
    """
    def __init__(self, unet: UNet2DConditionModel, controlnet: ControlNetModel, vae: AutoencoderKL):
        super().__init__()
        # Hold instances of the pre-trained models
        self.unet = unet
        self.controlnet = controlnet
        self.vae = vae
        # Initialize our custom, trainable adapter
        self.garment_adapter = GarmentAdapter()

        # Freeze pre-trained models to prevent training them
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(False)
        self.vae.requires_grad_(False)
        # Ensure our adapter is in training mode
        self.garment_adapter.train()

    def forward(
        self,
        latents: torch.Tensor,
        timestep: int,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        cloth_image: torch.Tensor,
    ):
        # 1. Get pose guidance from ControlNet (operates on person latents)
        controlnet_down_res, controlnet_mid_res = self.controlnet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )

        # 2. Encode the cloth image into the latent space using the VAE
        with torch.no_grad():
            # The VAE is frozen, so we don't need gradients here
            cloth_latents = self.vae.encode(cloth_image).latent_dist.sample() * self.vae.config.scaling_factor
        
        # 3. Get clothing features from our Garment Adapter using the cloth LATENTS
        garment_features = self.garment_adapter(cloth_latents)
        
        # The last feature from our adapter's list corresponds to the mid-block
        garment_mid_res = garment_features.pop()
        garment_down_res = garment_features # The rest are for the down-blocks

        # 4. Combine the residuals with element-wise addition
        combined_down_residuals = [
            c_res + g_res
            for c_res, g_res in zip(controlnet_down_res, garment_down_res)
        ]
        combined_mid_residual = controlnet_mid_res + garment_mid_res
        
        # 5. Pass to the main U-Net for the final denoising step
        noise_pred = self.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=combined_down_residuals,
            mid_block_additional_residual=combined_mid_residual,
        ).sample

        return noise_pred