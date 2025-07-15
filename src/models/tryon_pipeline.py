import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, ControlNetModel, AutoencoderKL

# NOTE: GarmentAdapter is not used in this debug version.

class TryOnPipeline(nn.Module):
    """
    A simplified debug pipeline that ONLY uses the U-Net and ControlNet.
    This is to verify that the core training loop and data flow are correct.
    """
    def __init__(self, unet: UNet2DConditionModel, controlnet: ControlNetModel, vae: AutoencoderKL):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.vae = vae

        # Set all models to eval mode as nothing is being trained
        self.unet.eval()
        self.controlnet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def forward(
        self,
        latents: torch.Tensor,
        timestep: int,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        cloth_image: torch.Tensor, # This argument is unused but required by the script's call signature
    ):
        # 1. Get pose guidance ONLY from the ControlNet
        controlnet_down_res, controlnet_mid_res = self.controlnet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )

        # 2. Pass ONLY the ControlNet residuals to the main U-Net
        noise_pred = self.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=controlnet_down_res,
            mid_block_additional_residual=controlnet_mid_res,
        ).sample

        return noise_pred