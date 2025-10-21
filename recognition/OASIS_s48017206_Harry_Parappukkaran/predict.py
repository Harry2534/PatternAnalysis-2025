import torch
from modules import ImprovedUNet
from dataset import OASISDataset
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ImprovedUNet(n_channels=1, n_classes=3).to(device)
model.load_state_dict(torch.load("saved_models/improved_unet.pth", map_location=device))
model.eval()

# Example single image
img_path = "OASIS/-keras_png_slices_test/case_001_slice_0.nii.png"
mask_path = "OASIS/-keras_png_slices_seg_test/seg_001_slice_0.nii.png"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

image = Image.open(img_path).convert('L')
mask = Image.open(mask_path).convert('L')
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).cpu().squeeze().numpy()

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(image, cmap='gray')
plt.title("Input Image")
plt.subplot(1,3,2)
plt.imshow(mask, cmap='gray')
plt.title("Ground Truth")
plt.subplot(1,3,3)
plt.imshow(pred, cmap='gray')
plt.title("Predicted Mask")
plt.show()
