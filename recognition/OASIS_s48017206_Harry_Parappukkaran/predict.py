import torch
from modules import ImprovedUNet
from dataset import OASISDataset
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load checkpoint first and infer n_classes to avoid size mismatch
ckpt_path = "saved_models/improved_unet.pth"
ckpt = torch.load(ckpt_path, map_location=device)
# Handle common checkpoint formats
if isinstance(ckpt, dict) and any(isinstance(k, str) and k.endswith(".weight") for k in ckpt.keys()):
    state_dict = ckpt
else:
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    if state_dict is None:
        raise ValueError("Checkpoint does not contain a state_dict")

# Strip 'module.' prefix if saved with DataParallel
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def infer_n_classes(sd: dict, fallback: int = 3) -> int:
    for k in ("up4.conv.conv.0.weight", "outc.weight", "final_conv.weight"):
        if k in sd:
            return sd[k].shape[0]
    # Fallback heuristic: pick the smallest out_channels among 2D conv weights
    conv_weights = [v for k, v in sd.items() if k.endswith(".weight") and getattr(v, "dim", lambda: 0)() == 4]
    return min((w.shape[0] for w in conv_weights), default=fallback)

n_classes = infer_n_classes(state_dict, fallback=3)
model = ImprovedUNet(n_channels=1, n_classes=n_classes).to(device)
model.load_state_dict(state_dict)
model.eval()

# Example single image
img_path = "OASIS_test/keras_png_slices_test/case_441_slice_0.nii.png"
mask_path = "OASIS_test/keras_png_slices_seg_test/seg_441_slice_0.nii.png"

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
