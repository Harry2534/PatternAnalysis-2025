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



