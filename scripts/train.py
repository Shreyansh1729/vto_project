import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer

# Our custom modules
from src.data.viton_hd_dataset import VitonHDDataset
from src.models.tryon_pipeline import TryOnPipeline

def main(config):
    """
    The main training function.
    """
    print("--- Initializing Training ---")

    # --- 1. Setup Models and Tokenizers ---
    print("Loading base models (VAE, Tokenizer, Text Encoder)...")
    
    # The VAE is used to encode images into latents and decode back to images
    vae = AutoencoderKL.from_pretrained(config['model']['vae_path'])
    
    # The tokenizer and text encoder are for processing text prompts (though we may not use them heavily)
    tokenizer = CLIPTokenizer.from_pretrained(config['model']['tokenizer_path'])
    text_encoder = CLIPTextModel.from_pretrained(config['model']['text_encoder_path'])

    # --- 2. Initialize Our Custom Pipeline ---
    print("Initializing custom TryOnPipeline model...")
    tryon_model = TryOnPipeline(
        unet_path=config['model']['unet_path'],
        controlnet_path=config['model']['controlnet_path']
    )

    # --- 3. Freeze non-trainable parts ---
    print("Freezing VAE and Text Encoder...")
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # The TryOnPipeline __init__ already freezes the U-Net and ControlNet
    # We only need to train the GarmentAdapter
    trainable_params = list(tryon_model.garment_adapter.parameters())
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad):,}")

    # --- 4. Setup Optimizer ---
    print("Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['learning_rate']
    )

    # --- 5. Setup Data ---
    print("Setting up dataset and dataloader...")
    train_dataset = VitonHDDataset(
        data_root=config['data']['data_root'],
        mode='train',
        image_size=(config['data']['height'], config['data']['width'])
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers']
    )

    # --- 6. Setup Schedulers ---
    noise_scheduler = DDPMScheduler.from_pretrained(config['model']['scheduler_path'])

    # --- 7. Move models to device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    vae.to(device)
    text_encoder.to(device)
    tryon_model.to(device)

    # --- 8. Training Loop ---
    print("\n--- Starting Training Loop ---")
    for epoch in range(config['training']['num_epochs']):
        tryon_model.garment_adapter.train() # Ensure adapter is in training mode
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            person_image = batch['person_image'].to(device)
            cloth_image = batch['cloth_image'].to(device)
            pose_map = batch['pose_map'].to(device)
            
            # Note: We are not using text prompts for now, but the models require the input.
            # We create dummy text inputs.
            text_input = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
            text_embeddings = text_encoder(text_input.to(device))[0]

            # Encode person image into latents
            with torch.no_grad():
                latents = vae.encode(person_image).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample a random timestep for each image
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # --- Main Forward Pass ---
            noise_pred = tryon_model(
                latents=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=pose_map,
                cloth_image=cloth_image
            )
            
            # --- Calculate Loss ---
            # The loss is the mean squared error between our predicted noise and the actual noise
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # --- Backpropagation ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} completed. Loss: {loss.item():.4f}")
        # Add checkpoint saving logic here in the future
        
    print("\n--- Training Finished ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)