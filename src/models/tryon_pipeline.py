import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, ControlNetModel

# We are temporarily not using the GarmentAdapter to isolate the bug.
# from .garment_adapter import GarmentAdapter 

class TryOnPipeline(nn.Module):
    """
    A simplified pipeline that ONLY uses the U-Net and ControlNet.
    This is to verify that the core training loop and data flow are correct.
    The GarmentAdapter logic has been temporarily disabled.
    """
    def __init__(self, unet_path, unet_subfolder, controlnet_path):
        super().__init__()

        self.unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder=unet_subfolder)
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path)
        
        # --- The Garment Adapter is disabled for now ---
        # self.garment_adapter = GarmentAdapter()

        # We set both pre-trained models to eval mode and freeze them.
        # Since we have no custom parts, nothing will be trained in this run.
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
        cloth_image: torch.Tensor, # This argument is now unused but kept for API consistency
    ):
        # 1. Get pose guidance from the ControlNet. This is the only guidance we use.
        controlnet_down_res, controlnet_mid_res = self.controlnet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            return_dict=False,
        )

        # 2. Pass ONLY the ControlNet residuals to the main U-Net.
        noise_pred = self.unet(
            sample=latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=controlnet_down_res,
            mid_block_additional_residual=controlnet_mid_res,
        ).sample

        return noise_pred