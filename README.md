# 3D Brain MRI Segmentation with Improved UNet
This project implements a 3D brain MRI segmentation pipeline using an enhanced UNet model. It is designed to process volumetric data from the OASIS dataset (and can be extended to HipMRI). The model performs end-to-end 3D segmentation and is evaluated using the Dice coefficient for overlap accuracy.

# Table of Contents
- Project Overview
- Installation
- Dataset
- Training
- Evaluation
- Results
- File Structure
- Usage

# Project Overview
The segmentation pipeline includes:
- Data Loading: Custom PyTorch Dataset for 3D MRI volumes.
- Model: ImprovedUNet3D with multi-level encoding/decoding and trilinear upsampling.
- Training: Uses CrossEntropyLoss with Dice coefficient evaluation.
- Prediction & Evaluation: Computes per-case Dice scores and mean Dice across test cases.
The goal is to achieve accurate and efficient 3D volumetric brain structure segmentation.

# Installation
```
Clone the repository
git clone <your-repo-url>
cd PatternAnalysis-2025/recognition

Install dependencies
pip install torch torchvision numpy pillow matplotlib
```

# Dataset
The project uses the OASIS MRI dataset, organized as 3D volumes reconstructed from 2D PNG slices:

```
OASIS/
  keras_png_slices_train/
  keras_png_slices_seg_train/
  keras_png_slices_validate/
  keras_png_slices_seg_validate/
  keras_png_slices_test/
  keras_png_slices_seg_test/
```
- Images: Grayscale MRI scans.
- Masks: Multi-class segmentation maps (classes 0–3).

# Training
To train the train_oasis.py
```
python train_oasis.py
```

Default Hyperparameters:
- Epochs: 20 (adjustable)
- Batch size: 4
- Learning rate: 1e-4
- Loss: CrossEntropyLoss
The trained model is saved automatically to:
```
saved_models/improved_unet.pth
```
Example Training Output
```
Epoch 001/100 | Train Loss 0.8695 | Val Loss 0.7435 | Val Dice 0.8894
Epoch 010/100 | Train Loss 0.0838 | Val Loss 0.0815 | Val Dice 0.9499
Epoch 100/100 | Train Loss 0.0124 | Val Loss 0.0319 | Val Dice 0.9694
```

# Evaluation
To evaluate the model on the test set:
```
python predict.py
```
This computes Dice coefficients for each case and the mean Dice across all test cases.
All images are normalized to [-1, 1], and mask classes are remapped to [0..K-1] if necessary.
Example Output:
```
Case 441: Dice=0.9322
Case 445: Dice=0.9838
Evaluated 17 cases.
Mean Dice: 0.9544
```













