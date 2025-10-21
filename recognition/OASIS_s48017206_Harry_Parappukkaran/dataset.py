import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class OASISDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert('L')
        mask_img = Image.open(mask_path).convert('L')

        # Apply transforms to image only (keep mask as integer labels)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        # Convert mask to Long tensor of class indices
        mask_np = np.array(mask_img)
        if mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]
        # If mask has arbitrary label values, remap to 0..K-1
        vals = np.unique(mask_np)
        if not np.array_equal(vals, np.arange(vals.size)):
            remap = {v: i for i, v in enumerate(vals)}
            mask_np = np.vectorize(remap.get)(mask_np)
        mask = torch.from_numpy(mask_np).long()

        return image, mask

def get_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size=8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    train_dataset = OASISDataset(train_img_dir, train_mask_dir, transform)
    val_dataset = OASISDataset(val_img_dir, val_mask_dir, transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
