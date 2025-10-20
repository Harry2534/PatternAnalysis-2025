# dataset_oasis.py
import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

class OASISDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        # Support NIfTI and PNG images
        exts = (".nii", ".nii.gz", ".png")
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(exts)]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_filename = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_filename)
        label_path = os.path.join(self.label_dir, img_filename)

        if img_filename.lower().endswith((".nii", ".nii.gz")):
            img_vol = nib.load(img_path).get_fdata()
            label_vol = nib.load(label_path).get_fdata()
            # Take middle slice
            img_np = img_vol[:, :, img_vol.shape[2] // 2]
            label_np = label_vol[:, :, label_vol.shape[2] // 2]
            # Normalize image to [0,1] with epsilon for safety
            imin, imax = np.min(img_np), np.max(img_np)
            img_np = (img_np - imin) / (imax - imin + 1e-8)
        else:
            # PNG case
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label not found for {img_filename}: {label_path}")
            # Load as grayscale for image; label kept as-is then cast to long
            img_np = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
            label_np = np.array(Image.open(label_path))
            # If label is RGB, reduce to single channel (assuming indexed colors)
            if label_np.ndim == 3:
                # Take first channel; adjust if your masks are encoded differently
                label_np = label_np[:, :, 0]

        img = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(label_np, dtype=torch.long)              # (H, W)

        if self.transform:
            img = self.transform(img)

        return img, label
