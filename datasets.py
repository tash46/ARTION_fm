import os
import numpy as np
from PIL import Image
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor()
])

class CustomToTensor:
    def __call__(self, pic):
        return torch.tensor(np.array(pic), dtype=torch.float32)

mask_transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    CustomToTensor()  # Use custom transformation to keep masks in [0, 1]
])

class Datasets(Dataset):
    def __init__(self, image_dir, mask_dir=None, transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(image_dir))
        
        if mask_dir:
            self.masks = sorted(os.listdir(mask_dir))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("L").resize((1024, 1024))  # Convert to grayscale
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.masks[idx])
            mask = Image.open(mask_path).convert("L").resize((1024, 1024), resample=Image.NEAREST)
            
            # Convert and transform the mask
            mask = self.convert_mask(mask)
            mask = self.mask_transform(mask)
            
            # Add channel dimension to the mask
            mask = mask.unsqueeze(0) #when binary
            #mask = mask.long() #when multi-class
            
            # Transform the image
            image = self.transform(image)
            
            return image, mask
        else:
            # Transform the image
            image = self.transform(image)
            
            return image

    def convert_mask(self, mask):
        mask = np.array(mask)  # Convert the PIL image to a NumPy array
        binary_mask = np.where(mask > 0, 1, 0)  # Binarize the mask (0 or 1)
        return Image.fromarray(binary_mask.astype(np.uint8))
    
def get_loaders_from_config(config_path, batch_size=16):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)

    train_dataset = Datasets(
        config['train_image_dir'], config['train_mask_dir'],
        transform=image_transform, mask_transform=mask_transform
    )

    val_dataset = Datasets(
        config['val_image_dir'], config['val_mask_dir'],
        transform=image_transform, mask_transform=mask_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
