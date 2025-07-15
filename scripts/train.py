import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer

from src.data.viton_hd_dataset import VitonHDDataset
from src.models.tryon_pipeline import TryOnPipeline

def main(config):
    print("--- Initializing Training ---")

    # --- 1. Load all pre-trained models ---
    print("Loading base models...")
    vae = AutoencoderKL.from_pretrained(config['model']['vae_path'], subfolder=config['model']['subfolder_vae'])
    tokenizer = CLIPTokenizer.from_pretrained(config['model']['tokenizer_path'], subfolder=config['model']['subfolder_tokenizer'])
    text_encoder = CLIPTextModel.from_pretrained(config['model']['text_encoder_path'], subfolder=config['model']['subfolder_text_encoder'])
    unet = UNet2DConditionModel.from_pretrained(config['model']['unet_path'], subfolder=config['model']['subfolder'])
    controlnet = ControlNetModel.from_pretrained(config['model']['controlnet_path'])
    
    # --- 2. Initialize our custom pipeline with the pre-trained models ---
    print("Initializing custom TryOnPipeline model...")
    tryon_model = TryOnPipeline(
        unet=unet,
        controlnet=controlnet,
        vae=vae
    )
    
    # --- 3. Setup Optimizer for our adapter ---
    trainable_params = list(tryon_model.garment_adapter.parameters())
    print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params if p.requires_grad):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=config['training']['learning_rate'])

    # --- 4. Setup Data ---
    print("Setting up dataset and dataloader...")
    train_dataset = VitonHDDataset(
        data_root=config['data']['data_root'],
        mode='train',
        image_size=(config['data']['height'], config['data']['width'])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    
    # --- 5. Setup Scheduler and Device ---
    noise_scheduler = DDPMScheduler.from_pretrained(config['model']['scheduler_path'], subfolder=config['model']['subfolder_scheduler'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tryon_model.to(device)

    # --- 6. Training Loop ---
    print("\n--- Starting Training Loop ---")
    for epoch in range(config['training']['num_epochs']):
        tryon_model.garment_adapter.train()
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            with torch.no_grad():
                person_image = batch['person_image'].to(device)
                cloth_image = batch['cloth_image'].to(device)
                pose_map = batch['pose_map'].to(device)
                
                text_input = tokenizer([""] * person_image.shape[0], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").input_ids
                text_embeddings = text_encoder(text_input.to(device))[0]
                
                latents = tryon_model.vae.encode(person_image).latent_dist.sample() * tryon_model.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            noise_pred = tryon_model(
                latents=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=pose_map,
                cloth_image=cloth_image
            )
            
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} completed. Loss: {loss.item():.4f}")
        
    print("\n--- Training Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)