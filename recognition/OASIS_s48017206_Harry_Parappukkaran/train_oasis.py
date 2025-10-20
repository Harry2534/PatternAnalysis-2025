# train_oasis.py
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from modules_unet2d import ImprovedUNet
from dataset_oasis import OASISDataset
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
dataset = OASISDataset(
    r"c:\Users\hpara\PatternAnalysis-2025\OASIS\keras_png_slices_test",         # images
    r"c:\Users\hpara\PatternAnalysis-2025\OASIS\keras_png_slices_train"         # labels
)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8)

model = ImprovedUNet(in_channels=1, out_channels=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_losses, val_losses = [], []

for epoch in range(15):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in val_loader)
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_losses[-1]:.4f}")

torch.save(model.state_dict(), "unet_oasis.pth")

plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.legend()
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_curve.png")
