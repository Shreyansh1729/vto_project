import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VitonHDDataset(Dataset):
    """
    Custom PyTorch Dataset for the VITON-HD dataset.
    This class handles loading of image pairs (person, cloth) and their
    corresponding metadata for the virtual try-on task.
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

        # Define the image transformations
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        """Returns the total number of pairs in the dataset."""
        return len(self.data_frame)

    def __getitem__(self, index):
        """
        Retrieves a single data item (person image, cloth image, etc.) at a given index.
        
        Args:
            index (int): The index of the data item to retrieve.
            
        Returns:
            dict: A dictionary containing all the necessary data for one training step.
        """
        # Get the person and cloth IDs from the dataframe
        person_id = self.data_frame.iloc[index]['person_id']
        cloth_id = self.data_frame.iloc[index]['cloth_id']

        # --- Load Person Image ---
        person_image_path = os.path.join(self.data_root, self.mode, 'image', person_id)
        person_image = Image.open(person_image_path).convert('RGB')

        # --- Load Cloth Image ---
        cloth_image_path = os.path.join(self.data_root, self.mode, 'cloth', cloth_id)
        cloth_image = Image.open(cloth_image_path).convert('RGB')
        
        # Apply transformations to images
        person_image_tensor = self.transform(person_image)
        cloth_image_tensor = self.transform(cloth_image)

        # We will add more data points (masks, pose maps) here in later steps.
        # For now, we return the essential items.
        
        return {
            'person_id': person_id,
            'cloth_id': cloth_id,
            'person_image': person_image_tensor,
            'cloth_image': cloth_image_tensor,
        }