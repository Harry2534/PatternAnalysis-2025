# evaluate.py
import torch
import numpy as np
from modules_unet2d import ImprovedUNet
from dataset_oasis import OASISDataset

def dice_coef(pred, target, eps=1e-6):
    pred = torch.argmax(pred, dim=1)
    intersection = (pred * target).sum()
    return (2. * intersection + eps) / (pred.sum() + target.sum() + eps)

model = ImprovedUNet(in_channels=1, out_channels=3)
model.load_state_dict(torch.load("unet_oasis.pth", map_location="cpu"))
model.eval()

# Update these paths as needed; keep raw strings on Windows to avoid unicode issues
dataset = OASISDataset(
    r"c:\Users\hpara\PatternAnalysis-2025\OASIS\keras_png_slices_test",
    r"c:\Users\hpara\PatternAnalysis-2025\OASIS\keras_png_slices_train"
)

total_dice = 0
for i in range(min(10, len(dataset))):  # sample 10 test images
    img, mask = dataset[i]
    with torch.no_grad():
        pred = model(img.unsqueeze(0))
    dice = dice_coef(pred, mask)
    total_dice += dice.item()
print(f"Average Dice coefficient: {total_dice/ max(1, min(10, len(dataset))):.3f}")
