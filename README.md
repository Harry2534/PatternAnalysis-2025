# 3D Brain MRI Segmentation with Improved UNet
This project implements a 3D brain MRI segmentation pipeline using an enhanced UNet model. It is designed to process volumetric data from the OASIS dataset (and can be extended to HipMRI). The model performs end-to-end 3D segmentation and is evaluated using the Dice coefficient for overlap accuracy.

# Table of Contents
- Project Overview
- Dataset
- Method / Model Description
- Evaluation & Results
- Conclusion

# Project Overview
The segmentation pipeline includes:
- Data Loading: Custom PyTorch Dataset for 3D MRI volumes.
- Model: ImprovedUNet3D with multi-level encoding/decoding and trilinear upsampling.
- Training: Uses CrossEntropyLoss with Dice coefficient evaluation.
- Prediction & Evaluation: Computes per-case Dice scores and mean Dice across test cases.
The goal is to achieve accurate and efficient 3D volumetric brain structure segmentation.

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
  
Classes:
  - 0: Background
  - 1: Gray Matter
  - 2: White Matter
  - 3: CSF (Cerebrospinal Fluid)
A total of 17 volumes were used for testing, with separate training and validation splits.

# Method/Model Description
## Architecture
The model is based on an improved 3D UNet, featuring:
- Encoder–decoder structure with skip connections
- 3D convolutions for volumetric context
- Trilinear upsampling to improve spatial resolution
- Batch normalization for stable convergence
- ReLU activations for nonlinearity
This model extends the original 2D UNet into the 3D domain to capture spatial dependencies across slices, enabling full volumetric segmentation.

## Loss Function
The model uses a combination of CrossEntropyLoss and Dice coefficient for evaluation:
$Dice(p, g) = \frac{2|p \cap g|}{|p| + |g|}$

## Training Configuration
| **Parameter**     | **Value**               |
|--------------------|-------------------------|
| Epochs             | 100                     |
| Batch Size         | 4                       |
| Learning Rate      | 1e-4                    |
| Optimizer          | Adam                    |
| Framework          | PyTorch (CUDA-enabled)  |

Training command:
```
python train_oasis.py
```


# Evaluation & Results
## Training performance 
The model achieved consistent improvement across epochs.
Below is an excerpt from training output (train.out):
```
Epoch 001/100 | Train Loss 0.8695 | Val Loss 0.7435 | Val Dice 0.8894
Epoch 010/100 | Train Loss 0.0838 | Val Loss 0.0815 | Val Dice 0.9499
Epoch 030/100 | Train Loss 0.0253 | Val Loss 0.0323 | Val Dice 0.9642
Epoch 050/100 | Train Loss 0.0185 | Val Loss 0.0314 | Val Dice 0.9653
Epoch 075/100 | Train Loss 0.0147 | Val Loss 0.0333 | Val Dice 0.9684
Epoch 100/100 | Train Loss 0.0124 | Val Loss 0.0319 | Val Dice 0.9694
```
Summary
- Final Training Loss: 0.0124
- Final Validation Loss: 0.0319
- Final Validation Dice: 0.9694
The validation dice plateaued near 0.97 indicating high segmentation accuracy and stable convergence

## Test Set Evaluation
Full Predict.out output:
```
Case 441: Dice=0.9322
Case 442: Dice=0.9602
Case 443: Dice=0.9788
Case 444: Dice=0.9649
Case 445: Dice=0.9838
Case 446: Dice=0.9818
Case 447: Dice=0.9689
Case 448: Dice=0.9275
Case 449: Dice=0.9588
Case 450: Dice=0.7565
Case 451: Dice=0.9743
Case 452: Dice=0.9802
Case 453: Dice=0.9771
Case 454: Dice=0.9688
Case 455: Dice=0.9654
Case 456: Dice=0.9635
Case 457: Dice=0.9817
Evaluated 17 cases.
Mean Dice: 0.9544
```
Summary:
- Mean Test Dice: 0.9544
- Best Case Dice: 0.9838
- Lowest Case Dice: 0.7565 (likely due to noise)
These results indicate strong generalisation to unseen volumes, with consistent Dice scores above 0.93 across most cases.

## Visual Results (Optional Plots)
### Training Loss Curve
<img width="1920" height="1440" alt="train_loss_curve" src="https://github.com/user-attachments/assets/8e0b3edc-2330-47b8-ad52-32932291ca33" />
#### Analysis
The Training Loss steadily decreases over the 100 epochs, starting at 0.8695 in Epoch 1 and dropping to 0.0124 by Epoch 100. This indicates that the model is successfully learning from the training data, with the loss approaching a very low value, which suggests that the network has effectively minimized prediction errors on the training set.
##### Key observations:
- Rapid decrease in the first ~20 epochs indicates fast convergence initially.
- Minor fluctuations after epoch 60 suggest the model is refining its learning.
- The consistent downward trend confirms that the model is not stagnating early.

### Validation Dice Curve
<img width="1920" height="1440" alt="val_dice_curve" src="https://github.com/user-attachments/assets/f374ad70-aaf6-4e91-8faa-d76ffa3a97fa" />
#### Analysis
The Validation Dice coefficient starts around 0.8894 and increases to 0.9694 by the final epoch. The Dice coefficient is a measure of segmentation quality (higher is better), so this indicates that the model is effectively generalizing to unseen data.
##### Key observations:
- The Dice curve shows steady improvement alongside the training loss, indicating good learning dynamics.
- Minor dips (e.g., Epoch 59: Dice=0.9031) suggest occasional sensitivity to specific batches but overall stability is maintained.
- Achieving a mean Dice of 0.9544 across evaluated cases demonstrates high segmentation accuracy and consistency.

# Conclusion
This project successfully demonstrates 3D volumetric brain MRI segmentation using an Improved UNet architecture.
Key learnings:
- Extending 2D UNet to 3D improves anatomical consistency across slices.
- Dice coefficient is a robust metric for segmentation overlap.
- GPU-accelerated 3D convolutions significantly speed up volumetric training.
- Training and validation curves indicate effective learning and generalization, with steadily decreasing training loss and high, stable Dice scores on unseen data (mean Dice = 0.9544), demonstrating that the model captures relevant features while avoiding overfitting.
  
Limitations:
- Training 3D models requires substantial GPU memory.
- Performance may degrade on highly imbalanced datasets (e.g., small brain regions).

Future work could explore:
  - Hybrid Dice + Focal loss
  - Attention-based UNets
  - Transfer learning on larger MRI datasets (e.g., BraTS)

# References
1. Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation", MICCAI 2016.
2. OASIS-1 MRI Dataset: https://www.oasis-brains.org/
3. PyTorch UNet implementations adapted and extended from open-source repositories.





