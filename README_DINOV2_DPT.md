# DINOv2+DPT Encoder for TiO-Depth

This document explains how to use DINOv2+DPT as an encoder backbone for the TiO-Depth architecture for depth estimation.

## Overview

This implementation integrates DINOv2+DPT as a feature extractor encoder to replace the original Swin Transformer in the TiO-Depth architecture. The DINOv2 model (with various sizes: ViT-S, ViT-B, ViT-L, ViT-G) combined with DPT-style feature extraction provides a powerful alternative encoder for depth estimation.

## Implementation Details

The implementation consists of:

1. **DINOv2+DPT Backbone**: A modified backbone that extracts features from DINOv2 and converts them to a feature pyramid compatible with TiO-Depth's expected input.
   - Located in: `/models/backbones/dinov2_dpt.py`

2. **Configuration Files**:
   - Base config: `/options/_base/networks/tio_depth_dinov2.yaml`
   - Training config: `/options/TiO-Depth/train/tio_depth-dinov2b_384crop256_kitti_stereo.yaml`

3. **Training Script**:
   - `/options/TiO-Depth/train/train_tio_depth_dinov2b.sh`

## Feature Extraction Mechanism

The DINOv2+DPT encoder works as follows:

1. The DINOv2 Vision Transformer processes an input image and extracts intermediate features
2. The DPT-style head takes these features and constructs a feature pyramid with compatible shapes
3. The feature scales match what the subsequent decoder expects:
   - For Swin-T with embed_stride=2, features are at 1/4, 1/8, 1/16, 1/32 resolution
   - DINOv2+DPT produces features at the same scales

## Available DINOv2 Variants

- `dinov2s` - DINOv2 ViT-Small
- `dinov2b` - DINOv2 ViT-Base (default)
- `dinov2l` - DINOv2 ViT-Large
- `dinov2g` - DINOv2 ViT-Giant

## How to Train

To train TiO-Depth with DINOv2-B as the encoder:

```bash
cd /home/ywan0794/TiO-Depth_pytorch
bash options/TiO-Depth/train/train_tio_depth_dinov2b.sh
```

The script sets up the appropriate configuration and runs the training process.

## Creating Your Own Configurations

To create a configuration using a different DINOv2 variant:

1. Create a new base config file (copy and modify `/options/_base/networks/tio_depth_dinov2.yaml`)
2. Change the `encoder_name` parameter to the desired variant (e.g., `dinov2l` for ViT-Large)
3. Create a new training config file (copy and modify `/options/TiO-Depth/train/tio_depth-dinov2b_384crop256_kitti_stereo.yaml`)
4. Create a training script (copy and modify `/options/TiO-Depth/train/train_tio_depth_dinov2b.sh`)

## Dependencies

This implementation requires:
- The Depth-Anything-V2 repository at `/home/ywan0794/Depth-Anything-V2`
- The DINOv2 and DPT implementations from the Depth-Anything-V2 repository

## Performance Considerations

- DINOv2-B and DINOv2-L provide a good balance between model size and performance
- DINOv2-G offers the highest quality but has significant memory requirements
- For smaller GPUs, DINOv2-S is a lightweight alternative

## Pretrained Weights

The implementation loads pretrained weights from the Depth-Anything-V2 repository. The loading process is designed to handle missing keys gracefully, allowing the model to function even if some weights don't match exactly. 