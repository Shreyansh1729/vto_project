import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, ControlNetModel

from .garment_adapter import GarmentAdapter

class TryOnPipeline(nn.Module):
    """
    The final, correct TryOnPipeline with proper feature fusion.
    """
    def __init__(self, unet_path, unet_subfolder, controlnet_path):
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder=unet_subfolder)
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path)
        self.garment_adapter = GarmentAdapter()

        # Freeze pre-trained models, keep adapter trainable
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
        # The input cloth image needs to be resized to the same spatial dimensions
        # as the initial latents for the adapter to work correctly.
        latent_h, latent_w = latents.shape[2], latents.shape[3]
        cloth_image_resized = torch.nn.functional.interpolate(
            cloth_image, size=(latent_h * 8, latent_w * 8), mode='bilinear', align_corners=False
        )
        garment_features = self.garment_adapter(cloth_image_resized)
        
        # The last feature from our adapter corresponds to the mid-block
        garment_mid_res = garment_features.pop()

        # 3. Combine the residuals with element-wise addition
        # The controlnet_down_res and garment_features should now be lists of tensors with matching shapes.
        combined_down_residuals = [
            c_res + g_res
            for c_res, g_res in zip(controlnet_down_res, garment_features)
        ]
        combined_mid_residual = controlnet_mid_res + garment_mid_res
        
        # 4. Pass to the main U-Net for the final denoising step
        noise_pred = self.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=combined_down_residuals,
            mid_block_additional_residual=combined_mid_residual,
        ).sample

        return noise_pred