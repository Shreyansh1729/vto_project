import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, ControlNetModel

from .garment_adapter import GarmentAdapter

class TryOnPipeline(nn.Module):
    """
    The main model that orchestrates the virtual try-on process.
    It combines a pre-trained U-Net, a ControlNet for pose guidance,
    and our custom GarmentAdapter for clothing feature injection.
    """
    def __init__(self, unet_path, controlnet_path):
        super().__init__()

        # --- Load Pre-trained Models ---
        # These models are large and contain most of the image generation knowledge.
        # We will not train them; we will use them in inference mode.
        print("Loading pre-trained U-Net...")
        self.unet = UNet2DConditionModel.from_pretrained(unet_path)
        
        print("Loading pre-trained ControlNet...")
        self.controlnet = ControlNetModel.from_pretrained(controlnet_path)

        # --- Initialize Our Custom Model ---
        # This is the only part of the pipeline that we will train.
        print("Initializing Garment Adapter...")
        self.garment_adapter = GarmentAdapter()

        # --- Set Models to Correct Modes ---
        # We freeze the U-Net and ControlNet to prevent them from being trained.
        self.unet.eval()
        self.controlnet.eval()
        self.unet.requires_grad_(False)
        self.controlnet.requires_grad_(False)
        
        # We set our adapter to training mode.
        self.garment_adapter.train()

    def forward(
        self,
        latents: torch.Tensor,
        timestep: int,
        encoder_hidden_states: torch.Tensor, # Text embeddings (usually from CLIP)
        controlnet_cond: torch.Tensor,       # The pose map for ControlNet
        cloth_image: torch.Tensor,           # The clothing image for our adapter
    ):
        """
        Defines the forward pass for a single denoising step.
        """
        # 1. Get pose guidance from the ControlNet
        controlnet_residuals, _ = self.controlnet(
            latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            control_image=controlnet_cond,
            return_dict=False
        )

        # 2. Get clothing features from our Garment Adapter
        garment_features = self.garment_adapter(cloth_image)

        # 3. Inject the features into the U-Net
        # Here, we are simply adding the features from both control models.
        # More sophisticated fusion methods could be explored later.
        combined_residuals = {
            "down_block_res_samples": [
                c_res + g_res for c_res, g_res in zip(controlnet_residuals['down_block_res_samples'], garment_features['down_block_res_samples'])
            ],
            "mid_block_res_sample": controlnet_residuals['mid_block_res_sample'] + garment_features['mid_block_res_sample']
        }


        # 4. Pass everything to the main U-Net to predict the noise
        noise_pred = self.unet(
            latents,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=combined_residuals["down_block_res_samples"],
            mid_block_additional_residual=combined_residuals["mid_block_res_sample"],
        ).sample

        return noise_pred