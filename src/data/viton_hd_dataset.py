import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VitonHDDataset(Dataset):
    """
    Custom PyTorch Dataset for the VITON-HD dataset.
    This class handles loading of image pairs (person, cloth) and all 
    necessary conditioning images for the ControlNet model.
    """
    def __init__(self, data_root, mode='train', image_size=(512, 384)):
        """
        Initializes the dataset.

        Args:
            data_root (str): The root directory of the VITON-HD dataset 
                             (e.g., '/kaggle/input/my-viton-hd-1/').
            mode (str): The dataset mode, 'train' or 'test'.
            image_size (tuple): The target size to resize images to (height, width).
        """
        if mode not in ['train', 'test']:
            raise ValueError("Mode must be 'train' or 'test'.")

        self.data_root = data_root
        self.mode = mode
        self.image_size = image_size
        
        # Construct the path to the pair list file
        pair_list_filename = f'{mode}_pairs.txt'
        pair_list_path = os.path.join(self.data_root, pair_list_filename)

        # Load the dataframe of image pairs
        self.data_frame = pd.read_csv(pair_list_path, sep=' ', header=None)
        self.data_frame.columns = ['person_id', 'cloth_id']

        # --- Define Image Transformations ---
        # Transform for color images (person, cloth, pose map)
        self.transform_rgb = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
        ])

        # Transform for single-channel masks
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor() # Normalize to [0, 1]
        ])

    def __len__(self):
        """Returns the total number of pairs in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, index):
        """
        Retrieves a single data item (person image, cloth image, etc.) at a given index.
        """
        # Get the person and cloth IDs from the dataframe
        person_id = self.data_frame.iloc[index]['person_id']
        cloth_id = self.data_frame.iloc[index]['cloth_id']

        # --- Define all file paths ---
        base_path = os.path.join(self.data_root, self.mode)
        
        # RGB images
        person_image_path = os.path.join(base_path, 'image', person_id)
        cloth_image_path = os.path.join(base_path, 'cloth', cloth_id)
        pose_map_path = os.path.join(base_path, 'openpose_img', person_id.replace('.jpg', '.png'))

        # Mask images (single channel)
        # Note: The dataset uses different extensions for masks, so we must replace them.
        person_parse_path = os.path.join(base_path, 'image-parse-v3', person_id.replace('.jpg', '.png'))
        cloth_mask_path = os.path.join(base_path, 'cloth-mask', cloth_id) # The cloth mask is often needed
        
        # --- Load and Transform all images ---
        
        # Load RGB images
        person_image = Image.open(person_image_path).convert('RGB')
        cloth_image = Image.open(cloth_image_path).convert('RGB')
        pose_map = Image.open(pose_map_path).convert('RGB')
        
        # Load single-channel masks
        person_parse_mask = Image.open(person_parse_path).convert('L') # 'L' for grayscale
        cloth_mask = Image.open(cloth_mask_path).convert('L')

        # Apply transformations
        person_image_tensor = self.transform_rgb(person_image)
        cloth_image_tensor = self.transform_rgb(cloth_image)
        pose_map_tensor = self.transform_rgb(pose_map)
        
        person_parse_tensor = self.transform_mask(person_parse_mask)
        cloth_mask_tensor = self.transform_mask(cloth_mask)
        
        # Return a dictionary with all the data
        return {
            'person_id': person_id,
            'cloth_id': cloth_id,
            'person_image': person_image_tensor,
            'cloth_image': cloth_image_tensor,
            'pose_map': pose_map_tensor,
            'person_parse_mask': person_parse_tensor,
            'cloth_mask': cloth_mask_tensor,
        }