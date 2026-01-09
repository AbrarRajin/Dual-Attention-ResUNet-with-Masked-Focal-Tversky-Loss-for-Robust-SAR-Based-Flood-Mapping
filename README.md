# Dual-Attention ResUNet with Masked Focal-Tversky Loss for Robust SAR-Based Flood Mapping
Dual-Attention ResUNet with Masked Focal-Tversky Loss for Robust Flood Mapping on Sentinel-1 SAR imagery.




A deep learning pipeline for automated flood detection using Synthetic Aperture Radar (SAR) imagery. This project implements a ResUNet architecture with attention mechanisms, trained on a unified dataset combining S1F11-Otsu and Sen1Floods11 datasets.

## Overview

This repository provides a complete end-to-end pipeline for flood segmentation from Sentinel-1 SAR data. The pipeline handles dataset preprocessing, model training with validity-aware masking, and inference on test samples.

## Key Features

- **Unified Dataset Processing**: Combines weakly-labeled (Otsu) and hand-labeled (Sen1Floods11) SAR flood datasets
- **Validity-Aware Training**: Excludes invalid/uncertain pixels from loss calculation and metrics
- **Attention-Enhanced Architecture**: ResUNet with channel and spatial attention mechanisms
- **Advanced Learning Rate Scheduling**: Cosine annealing with warm restarts
- **Comprehensive Evaluation**: Per-dataset and per-image metrics with detailed visualizations

## Pipeline Architecture

### 1. Dataset Preprocessing
- The Hand labelled dataset has "-1" as invalid label. This requires the "-1" labels to be excluded from the training and also adjust the metrics calculations.

- The preprocessing stage unifies handlabelled and otsu labelled data:

**Band Engineering**:
The pipeline computes three bands from VH and VV polarizations:
- **Band 1**: VV (vertical transmit, vertical receive)
- **Band 2**: (VH - VV) / (VH + VV) - polarization ratio
- **Band 3**: sqrt((VH² + VV²) / 2) - intensity magnitude

<img src="/images/dataset_samples.png">

Sentinel-1 SAR imagery visualization showing: (a) VV band, (b) NewBand1, (c) NewBand2, (d) annotated ground truth with flood (blue), no-flood (gray), and excluded (red) pixels, and (e) ground truth overlay on VV band with excluded regions.

**Output Structure**:
- Images: Normalized 3-channel SAR data (256×256)
- Masks: Binary ground truth labels
- Validity Masks: Indicates which pixels should be included in training
- Unified train/val/test splits with balanced dataset representation

### 2. Model Architecture

**ResUNet with Attention**:
- Encoder-decoder structure with residual connections
- Channel attention (Squeeze-and-Excitation blocks) for feature recalibration
- Spatial attention for localization refinement
- Skip connections preserve spatial information across scales

**Architecture Highlights**:
- Input: 256×256×3 SAR imagery
- Encoder depths: 64 → 128 → 256 → 512 filters
- Symmetric decoder with upsampling and concatenation
- Output: Single-channel probability map (sigmoid activation)

### 3. Training Strategy

**Masked Loss Function**:
- Focal Tversky Loss with validity masking
- Only valid pixels (validity_mask = 1) contribute to gradients
- Invalid pixels are completely excluded from optimization
- Prevents model bias from uncertain regions

**Learning Rate Schedule**:
- Cosine annealing with warm restarts
- Periodic learning rate resets enable exploration
- Smooth transitions between cycles for stable convergence

**Data Augmentation**:
- Random horizontal/vertical flips
- 90°/180°/270° rotations
- Brightness and contrast adjustments
- Augmentations applied consistently to image, mask, and validity mask

**Training Configuration**:
- Mixed dataset batches (Otsu + Sen1Floods11)
- Early stopping based on validation IoU
- TensorBoard logging for monitoring
- Model checkpointing saves best weights

### 4. Evaluation & Metrics

**Masked Metrics**:
All evaluation metrics exclude invalid pixels to ensure fair comparison:
- IoU (Intersection over Union)
- Dice Coefficient
- Precision, Recall, F1 Score
- Binary Accuracy
- Per-class metrics

**Evaluation Strategy**:
- Per-image metrics for detailed analysis
- Aggregated confusion matrices
- Dataset-specific performance comparison (Otsu vs Sen1Floods11)
- Visualization with validity mask overlays

### 5. Inference Pipeline

The inference module:
- Loads trained model with custom objects
- Selects random test samples from each dataset
- Generates predictions with validity-aware evaluation
- Creates comprehensive visualizations showing:
  - Original SAR composite
  - Ground truth with invalid regions marked
  - Validity masks
  - Model predictions
  - Masked predictions (valid regions only)
  - Per-sample metrics

## Project Structure

```
├── Dataset Preprocessing
│   ├── Unified dataset loader
│   ├── Band computation
│   ├── Normalization parameter calculation
│   ├── Validity mask generation
│   └── Train/val/test splitting
│
├── ResUNet Training
│   ├── Model architecture definition
│   ├── Masked loss functions
│   ├── Cosine annealing scheduler
│   ├── Training loop with callbacks
│   └── Metrics calculation
│
└── Inference
    ├── Model loading with custom objects
    ├── Sample selection
    ├── Prediction generation
    ├── Metrics calculation
    └── Results visualization
```

## Methodology Highlights

**Validity Masking Approach**:
The key innovation is treating invalid pixels as missing data rather than background. During training:
1. Ground truth and validity masks are concatenated
2. Loss functions receive both masks
3. Only valid pixels contribute to gradient updates
4. Metrics are computed exclusively on valid regions

This prevents the model from learning spurious patterns in uncertain areas and ensures evaluation reflects true performance on confident labels.

**Dataset Fusion Strategy**:
By combining Otsu and hand-labeled data:
- Otsu provides scale and diversity
- Hand-labels provide accuracy and serve as anchors
- Unified normalization ensures consistent feature distributions
- Balanced sampling prevents dataset bias

**Attention Mechanisms**:
- Channel attention reweights feature maps based on global statistics
- Spatial attention highlights relevant spatial locations
- Combined attention improves flood boundary delineation
- Residual connections maintain gradient flow

## Output Artifacts

**Training Outputs**:
- Best model checkpoint
- Training history plots (loss, metrics, learning rate)
- Normalization parameters
- Class weights for imbalance handling
- TensorBoard logs
- CSV training logs

**Inference Outputs**:
- Prediction visualizations
- Per-sample metrics (CSV)
- Dataset-specific statistics (JSON)
- Confusion matrices  

Below are some sample inferences from our pipeline:  
<img src="/images/inference_and_gradcam.png">

Flood segmentation by the best-performing model (ResUNet+AttentionBlock+SE) with Grad-CAM++ interpretability. (a) Sentinel-1 SAR composite. (b) Ground truth. (c) Model prediction. (d) Grad-CAM++ overlay. (e) Grad-CAM++ heatmap.


## Usage Notes

The notebook is designed for Kaggle environments with GPU support. Key paths to configure:
- Dataset directories (S1F11-Otsu and Sen1Floods11)
- Output directories for preprocessed data
- Model checkpoint locations

The pipeline supports reproducibility through fixed random seeds and deterministic operations.

## Technical Requirements

- TensorFlow/Keras for deep learning
- Rasterio for geospatial data handling
- NumPy, pandas for data manipulation
- Matplotlib for visualization
- scikit-learn for metrics and splits
- scikit-image for image processing

## Performance Characteristics


### Table for Performance metrics of our models:  
<img src="/images/Table-our-models.png">

### Hyperparameters for best model:
<img src="/images/HyperParam_best_model.png">

### Performance analysis of different loss functions for ResUNet + AB + SE.
<img src="/images/variation_loss.png">
---