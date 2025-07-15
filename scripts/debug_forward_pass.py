import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from transformers import CLIPTextModel, CLIPTokenizer

# We use the simplified debug pipeline that has no trainable parts
from src.models.tryon_pipeline import TryOnPipeline
from src.data.viton_hd_dataset import VitonHDDataset

def main(config):
    print("--- Running Forward Pass Debug Script ---")

    # 1. Load Models
    print("Loading base models...")
    vae = AutoencoderKL.from_pretrained(config['model']['vae_path'], subfolder=config['model']['subfolder_vae'])
    tokenizer = CLIPTokenizer.from_pretrained(config['model']['tokenizer_path'], subfolder=config['model']['subfolder_tokenizer'])
    text_encoder = CLIPTextModel.from_pretrained(config['model']['text_encoder_path'], subfolder=config['model']['subfolder_text_encoder'])
    unet = UNet2DConditionModel.from_pretrained(config['model']['unet_path'], subfolder=config['model']['subfolder'])
    controlnet = ControlNetModel.from_pretrained(config['model']['controlnet_path'])
    
    # 2. Initialize the simplified (non-trainable) pipeline
    print("Initializing debug TryOnPipeline model...")
    tryon_model = TryOnPipeline(unet=unet, controlnet=controlnet, vae=vae)

    # 3. Setup Data
    print("Setting up dataset...")
    train_dataset = VitonHDDataset(
        data_root=config['data']['data_root'],
        mode='train',
        image_size=(config['data']['height'], config['data']['width'])
    )
    # We only need one batch for this test
    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # 4. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    text_encoder.to(device)
    tryon_model.to(device)

    # 5. The Forward Pass Test
    print("\n--- Attempting a single forward pass... ---")
    try:
        # Get one single batch of data
        batch = next(iter(train_dataloader))
        
        # Everything happens in no_grad, as nothing is being trained
        with torch.no_grad():
            person_image = batch['person_image'].to(device)
            cloth_image = batch['cloth_image'].to(device) # Unused, but for API consistency
            pose_map = batch['pose_map'].to(device)
            
            text_input = tokenizer([""] * person_image.shape[0], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids
            text_embeddings = text_encoder(text_input.to(device))[0]
            
            latents = tryon_model.vae.encode(person_image).latent_dist.sample() * tryon_model.vae.config.scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device=device).long()
            noisy_latents = torch.randn_like(latents) # Simplified noise for debug

            # The actual forward pass
            noise_pred = tryon_model(
                latents=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=pose_map,
                cloth_image=cloth_image
            )

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            print("\n" + "="*50)
            print(f"✅✅✅ SUCCESS! Forward pass completed. ✅✅✅")
            print(f"   - Input person tensor shape: {person_image.shape}")
            print(f"   - Latents tensor shape: {latents.shape}")
            print(f"   - Predicted noise tensor shape: {noise_pred.shape}")
            print(f"   - Calculated Loss: {loss.item():.4f}")
            print("="*50)

    except Exception as e:
        print("\n" + "!"*50)
        print(f"❌❌❌ ERROR during forward pass: {e}")
        print("!"*50)
        raise e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file.')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    main(config)