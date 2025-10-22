import torch
import torch.nn as nn
import torch.optim as optim
from modules import ImprovedUNet
from dataset import get_loaders
import matplotlib.pyplot as plt
import os

# Paths (update with actual OASIS paths)
train_img_dir = "OASIS_test/keras_png_slices_train"
train_mask_dir = "OASIS_test/keras_png_slices_seg_train"
val_img_dir = "OASIS_test/keras_png_slices_validate"
val_mask_dir = "OASIS_test/keras_png_slices_seg_validate"

device = "cu:da" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4  # <- masks have classes {0,1,2,3}
model = ImprovedUNet(n_channels=1, n_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
train_loader, val_loader = get_loaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, batch_size=4)

def dice_coefficient(logits, targets, num_classes=4, epsilon=1e-6):
	# Multiclass Dice: average per-class Dice between argmax(logits) and targets
	preds = torch.argmax(logits, dim=1)
	dice = 0.0
	for c in range(num_classes):
		p = (preds == c).float()
		t = (targets == c).float()
		intersection = (p * t).sum()
		union = p.sum() + t.sum()
		dice += (2 * intersection + epsilon) / (union + epsilon)
	return dice / num_classes

num_epochs = 20
train_losses, val_losses, val_dices = [], [], []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))
    
    model.eval()
    val_loss = 0
    dice_score = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            dice_score += dice_coefficient(outputs, masks, num_classes=NUM_CLASSES).item()
    val_losses.append(val_loss / len(val_loader))
    val_dices.append(dice_score / len(val_loader))
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Dice: {val_dices[-1]:.4f}")

# Save model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/improved_unet.pth")

# Plot losses and Dice
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.subplot(1,2,2)
plt.plot(val_dices, label="Validation Dice")
plt.legend()
plt.show()