#!/usr/bin/env python3
"""
Comparison Test:
1. Swin Transformer feature pyramid (varying channel numbers)
2. DINOv2+DPT feature pyramid (original features with natural channels)

This script helps visualize and compare features from different encoder architectures
to better understand their characteristics for depth estimation tasks.
"""
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms

from models.backbones.swin import get_orgwintrans_backbone  
from models.backbones.dinov2_dpt import get_dinov2_dpt_backbone, check_image_size_for_patching

# 添加Depth-Anything-V2到路径
depth_anything_path = '/home/ywan0794/Depth-Anything-V2'
if depth_anything_path not in sys.path:
    sys.path.append(depth_anything_path)

# Import the exact same transforms as used in dpt.py
from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose


def load_and_preprocess_image(image_path, input_size=518):
    """
    Load and preprocess image using the exact same logic as in dpt.py's image2tensor method.
    
    Args:
        image_path: Path to the input image
        input_size: Target size for the image (default: 518, same as in Depth-Anything-V2)
    
    Returns:
        tensor_image: Preprocessed image tensor ready for the network
        original_image: The loaded image in RGB format (for visualization)
        original_size: Tuple of original height and width
        resized_size: Tuple of resized height and width after preprocessing
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Copy the exact transform from dpt.py
    transform = Compose([
        Resize(
            width=input_size,
            height=input_size,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    
    # Load the raw image with OpenCV (exactly like in dpt.py)
    raw_image = cv2.imread(image_path)
    
    if raw_image is None:
        raise ValueError(f"Failed to load image at {image_path}")
    
    # Get original dimensions
    h, w = raw_image.shape[:2]
    
    # Convert to RGB and normalize to 0-1 range (exactly like in dpt.py)
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    # Save a copy of the original RGB image for visualization
    original_img = Image.fromarray((image * 255).astype(np.uint8))
    
    # Apply the transform (exactly like in dpt.py)
    image = transform({'image': image})['image']
    
    # Convert to tensor (exactly like in dpt.py)
    tensor_image = torch.from_numpy(image).unsqueeze(0)
    
    # Get the new dimensions after transformation
    input_shape = tensor_image.shape
    resized_h, resized_w = input_shape[2], input_shape[3]
    
    print(f"Loaded image '{os.path.basename(image_path)}'")
    print(f"Original size: {h}x{w}")
    print(f"Transformed to: {resized_h}x{resized_w}")

    return tensor_image, original_img, (h, w), (resized_h, resized_w)


def visualize_features(features, name="feature", save_dir="./", original_img=None, tensor_size=None,resize=True):
    """
    Visualize feature maps following the DPT logic and save to specified directory.
    Focuses on visualizing individual channels rather than channel means.
    Uses English titles and includes resolution information.
    
    Args:
        features: List of feature tensors
        name: Name prefix for saved files
        save_dir: Directory to save visualizations
        original_img: Original PIL image (for reference)
        tensor_size: Tuple of (height, width) of input tensor used for network
    """
    # Create subdirectory for this feature set
    os.makedirs(save_dir, exist_ok=True)
    
    # Save original image if provided
    if original_img is not None:
        img_save_path = os.path.join(save_dir, f"{name}_original_image.png")
        original_img.save(img_save_path)
        print(f"Saved original image: {img_save_path}")
    
    # Use tensor dimensions for target size calculations if provided
    if tensor_size is not None:
        tensor_h, tensor_w = tensor_size
        print(f"Using tensor dimensions for scaling: {tensor_h}x{tensor_w}")
    else:
        # Fall back to original image size estimation
        if original_img is not None:
            tensor_h, tensor_w = original_img.height, original_img.width
        else:
            # Estimate from feature maps assuming standard downsampling
            sample_feat = features[0]
            tensor_h = sample_feat.shape[2] * 4  # Approximate
            tensor_w = sample_feat.shape[3] * 4
        print(f"No tensor size provided, estimated: {tensor_h}x{tensor_w}")
    
    # Define specific scale for each layer
    scales = [1/2, 1/4, 1/8, 1/16]
    
    # Process each feature layer
    for i, feat in enumerate(features):
        print(f"Visualizing {name} feature level {i+1}, shape: {feat.shape}")
        scale = scales[i]
        scale_name = f"1/{int(1/scale)}"  # English format: "1/4" instead of "1_4"
        
        # Calculate target size for specific scale based on tensor dimensions
        target_h = int(tensor_h * scale)
        target_w = int(tensor_w * scale)
        
        # Get current feature dimensions
        _, num_channels, feat_h, feat_w = feat.shape
        print(f"Feature dimensions: {num_channels} channels, {feat_h}x{feat_w} resolution")
        
        # Save feature layer information visualization
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 
                f"Feature Layer {i+1}\n"
                f"Shape: {feat.shape}\n"
                f"Channels: {num_channels}\n"
                f"Resolution: {feat_h}x{feat_w}\n"
                f"Scale: {scale_name}\n", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}_layer{i+1}_info.png"), dpi=300)
        plt.close()
        
        # Visualize select individual channels
        # Choose specific channels to visualize instead of mean
        channels_to_visualize = [0, 1, 2, 3]  # First 4 channels
        if num_channels > 8:
            # Add some middle and late channels if available
            mid_idx = num_channels // 2
            channels_to_visualize.extend([mid_idx-1, mid_idx])
            channels_to_visualize.extend([num_channels-2, num_channels-1])
        else:
            # Add all remaining channels if less than 8 total
            channels_to_visualize.extend(range(4, min(8, num_channels)))
        
        # Limit to available channels
        channels_to_visualize = channels_to_visualize[:min(len(channels_to_visualize), num_channels)]
        
        # Create a grid of selected channels
        n_channels = len(channels_to_visualize)
        rows = int(np.ceil(np.sqrt(n_channels)))
        cols = int(np.ceil(n_channels / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
        
        for j, channel_idx in enumerate(channels_to_visualize):
            if j < len(axes):
                channel_data = feat[0, channel_idx].detach().cpu().numpy()
                channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
                im = axes[j].imshow(channel_data, cmap='viridis')
                axes[j].set_title(f"Channel {channel_idx} ({feat_h}x{feat_w})")
                axes[j].axis('off')
                fig.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)
        
        # Hide unused axes
        for j in range(n_channels, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}_layer{i+1}_channels.png"), dpi=300)
        plt.close()
        
        # For each selected channel, create a comparison of original vs. resized
        for channel_idx in channels_to_visualize[:4]:  # Limit to first 4 for brevity
            channel_feature = feat[0, channel_idx].detach()
            
            plt.figure(figsize=(16, 8))
            
            # Original feature
            plt.subplot(1, 2, 1)
            channel_np = channel_feature.cpu().numpy()
            channel_np = (channel_np - channel_np.min()) / (channel_np.max() - channel_np.min() + 1e-8)
            plt.imshow(channel_np, cmap='viridis')
            plt.colorbar(label='Normalized Value')
            plt.title(f"Original Channel {channel_idx} ({feat_h}x{feat_w})")
            plt.axis('off')
            
            # Resized feature to target scale
            plt.subplot(1, 2, 2)
            feature_tensor = channel_feature.unsqueeze(0).unsqueeze(0)
            if resize:
                resized_feature = F.interpolate(
                    feature_tensor, 
                    size=(target_h, target_w), 
                    mode='bicubic', 
                    align_corners=True
                )[0, 0]
            else:
                resized_feature = feature_tensor[0, 0]
            
            resized_np = resized_feature.cpu().numpy()
            resized_np = (resized_np - resized_np.min()) / (resized_np.max() - resized_np.min() + 1e-8)
            
            plt.imshow(resized_np, cmap='viridis')
            plt.colorbar(label='Normalized Value')
            plt.title(f"Scaled to {scale_name} ({target_h}x{target_w})")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{name}_layer{i+1}_channel{channel_idx}_comparison.png"), dpi=300)
            plt.close()
        
        # Create a multi-channel composite visualization (RGB-like)
        if num_channels >= 3:
            plt.figure(figsize=(10, 8))
            
            # Take first 3 channels and combine them into RGB
            r_channel = feat[0, 0].detach().cpu().numpy()
            g_channel = feat[0, 1].detach().cpu().numpy()
            b_channel = feat[0, 2].detach().cpu().numpy()
            
            # Normalize each channel
            r_channel = (r_channel - r_channel.min()) / (r_channel.max() - r_channel.min() + 1e-8)
            g_channel = (g_channel - g_channel.min()) / (g_channel.max() - g_channel.min() + 1e-8)
            b_channel = (b_channel - b_channel.min()) / (b_channel.max() - b_channel.min() + 1e-8)
            
            # Stack into RGB
            rgb_like = np.stack([r_channel, g_channel, b_channel], axis=2)
            
            plt.imshow(rgb_like)
            plt.title(f"RGB Composite (Ch0=R, Ch1=G, Ch2=B) - {feat_h}x{feat_w}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{name}_layer{i+1}_rgb_composite.png"), dpi=300)
            plt.close()
    
    # Create a feature pyramid visualization
    plt.figure(figsize=(15, 10))
    
    # Plot each feature level
    for i, feat in enumerate(features):
        plt.subplot(2, 2, i+1)
        
        # Use the first channel for visualization
        channel_data = feat[0, 0].detach().cpu().numpy()
        channel_data = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min() + 1e-8)
        
        plt.imshow(channel_data, cmap='plasma')
        plt.title(f"Layer {i+1} - Channel 0 - {feat.shape[2]}x{feat.shape[3]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_feature_pyramid.png"), dpi=300)
    plt.close()
    
    print(f"Feature visualizations saved to: {save_dir}")


def create_feature_info(features, name, input_height):
    """Create information about feature pyramid from feature list"""
    info = []
    for i, feat in enumerate(features):
        level_info = {
            "level": i+1,
            "shape": list(feat.shape),
            "channels": feat.shape[1],
            "resolution": f"{feat.shape[2]}x{feat.shape[3]}",
            "scale_factor": input_height / feat.shape[2]
        }
        info.append(level_info)
    return info


def print_feature_info(feature_info, name):
    """Print feature pyramid information"""
    print(f"\n{name} Feature Pyramid:")
    print(f"Total layers: {len(feature_info)}")
    
    for info in feature_info:
        print(f"  Level {info['level']}: Shape={info['shape']}, Channels={info['channels']}, "
              f"Resolution={info['resolution']}, Downsampling Factor=1/{info['scale_factor']:.0f}")


def print_feature_table(feature_info, name):
    """Print feature pyramid information in table format"""
    print(f"\n{name} Feature Pyramid:")
    print("Level | Channels | Resolution | Downsampling | Shape")
    print("-" * 70)
    for info in feature_info:
        print(f"{info['level']} | {info['channels']} | {info['resolution']} | 1/{info['scale_factor']:.0f} | {info['shape']}")


def load_weights(model, weights_path):
    """
    最简单的方式加载预训练权重。直接使用PyTorch的load_state_dict并忽略不匹配的键。
    
    Args:
        model: 目标模型
        weights_path: 预训练权重文件路径
        
    Returns:
        loaded_count: 成功加载的参数数量
    """
    if not os.path.exists(weights_path):
        print(f"预训练权重文件不存在: {weights_path}")
        return 0
    
    print(f"正在加载预训练权重: {weights_path}")
    
    # 加载权重
    checkpoint = torch.load(weights_path, map_location='cpu')
    
    # 处理不同格式的权重
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 直接加载，strict=False允许跳过不匹配的参数
    incompatible = model.load_state_dict(state_dict, strict=False)
    
    # 打印加载结果
    if incompatible.missing_keys:
        print(f"未找到的参数: {len(incompatible.missing_keys)} 个")
        if len(incompatible.missing_keys) < 10:
            print("具体包括:", incompatible.missing_keys)
        else:
            print("前10个:", incompatible.missing_keys[:10])
    
    if incompatible.unexpected_keys:
        print(f"模型中未使用的权重: {len(incompatible.unexpected_keys)} 个")
    
    # 计算实际加载的参数数量
    total_params = len(model.state_dict().keys())
    loaded_params = total_params - len(incompatible.missing_keys)
    print(f"成功加载 {loaded_params}/{total_params} 个参数")
    
    return loaded_params


def main():
    parser = argparse.ArgumentParser(description='Feature extraction test for DINOv2+DPT and Swin Transformer')
    parser.add_argument('--save_dir', type=str, default='./encoder_feature_test', help='Directory to save results')
    parser.add_argument('--swin_weights', type=str, 
                       default='/home/ywan0794/TiO-Depth_pytorch/swin_tiny_patch4_window7_224_22k.pth', 
                       help='Swin Transformer pre-trained weights')
    parser.add_argument('--dino_weights', type=str,
                       default='/home/ywan0794/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth',
                       help='Depth Anything pre-trained weights')
    parser.add_argument('--image_path', type=str,
                       default='/home/ywan0794/TiO-Depth_pytorch/demo01.jpg',
                       help='Path to test image')
    parser.add_argument('--input_size', type=int, default=518, 
                       help='Input size for the longer dimension of the image (default: 518)')
    parser.add_argument('--encoder', type=str, default='vitb',
                       choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='DINOv2 encoder size to use (default: vitb)')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load and preprocess image using dpt.py-style preprocessing
    print(f"Loading image: {args.image_path}")
    x, original_img, orig_size, input_size = load_and_preprocess_image(args.image_path, args.input_size)
    orig_h, orig_w = orig_size
    resized_h, resized_w = input_size
    
    x = x.to(device)
    
    print("=" * 80)
    print(f"Input tensor shape: {x.shape}")
    print(f"Original image dimensions: {orig_h}x{orig_w}")
    print(f"Resized dimensions: {resized_h}x{resized_w}")
    print(f"ViT Patch grid: {resized_h//14}x{resized_w//14} = {(resized_h//14) * (resized_w//14)} patches")
    print("=" * 80)
    
    # Create results dictionary
    results = OrderedDict()
    
    # =========================================================================
    # 1. Swin Transformer
    # =========================================================================
    print("\n" + "=" * 30 + " Swin Transformer " + "=" * 30)
    
    # Create and load Swin Transformer
    swin_encoder, swin_channels = get_orgwintrans_backbone('orgSwin-T-s2', False)
    swin_encoder = swin_encoder.to(device)
    
    # 加载预训练权重（使用最简单的方式）
    swin_loaded_params = load_weights(
        swin_encoder, 
        args.swin_weights
    )
    
    # 提取特征
    with torch.no_grad():
        swin_features = swin_encoder(x)
    
    # 收集信息
    swin_info = create_feature_info(swin_features, "Swin Transformer", resized_h)
    print_feature_info(swin_info, "Swin Transformer")
    
    # 存储结果
    results["Swin Transformer"] = {
        "features": swin_features,
        "info": swin_info,
        "channels": swin_channels,
        "loaded_params": swin_loaded_params,
        "total_params": len(swin_encoder.state_dict())
    }
    
    # =========================================================================
    # 2. DINOv2+DPT Original features (with natural channel configuration)
    # =========================================================================
    print("\n" + "=" * 30 + f" DINOv2+DPT ({args.encoder}) Features " + "=" * 30)
    
    # Map the encoder argument to backbone name
    backbone_map = {
        'vits': 'dinov2s',
        'vitb': 'dinov2b',
        'vitl': 'dinov2l',
        'vitg': 'dinov2g'
    }
    backbone_name = backbone_map[args.encoder]
    
    # Create model and load pre-trained weights
    # use_scratch=False to keep natural channel configuration from DPT
    dino_encoder, dino_channels = get_dinov2_dpt_backbone(
        backbone_name, True)
    dino_encoder = dino_encoder.to(device)
    
    # 加载预训练权重（使用最简单的方式）
    dino_loaded_params = load_weights(
        dino_encoder, 
        args.dino_weights
    )
    
    # 提取特征
    with torch.no_grad():
        dino_features = dino_encoder(x)
    
    # 收集信息 
    dino_info = create_feature_info(dino_features, f"DINOv2+DPT {args.encoder}", resized_h)
    print_feature_info(dino_info, f"DINOv2+DPT {args.encoder}")
    
    # 存储结果
    results[f"DINOv2+DPT {args.encoder}"] = {
        "features": dino_features,
        "info": dino_info,
        "channels": dino_channels,
        "loaded_params": dino_loaded_params,
        "total_params": len(dino_encoder.state_dict())
    }

    # =========================================================================
    # Results Summary
    # =========================================================================
    print("\n" + "=" * 30 + " Feature Pyramid Comparison " + "=" * 30)
    
    # Print feature dimension tables for each method
    for name, result in results.items():
        print_feature_table(result["info"], name)
    
    # Channel comparison
    print("\nChannel comparison:")
    print("Layer | Swin Transformer | DINOv2+DPT")
    print("-" * 50)
    for i in range(4):
        swin_ch = results["Swin Transformer"]["channels"][i]
        dino_ch = results[f"DINOv2+DPT {args.encoder}"]["channels"][i]
        print(f"{i+1} | {swin_ch:16d} | {dino_ch:d}")
    
    # Visualize features
    print("\n" + "=" * 30 + " Visualizing Feature Maps " + "=" * 30)
    
    tensor_size = (resized_h, resized_w)  # Use tensor dimensions
    
    # Swin visualization
    swin_save_dir = os.path.join(args.save_dir, "swin_transformer")
    print(f"\nVisualizing Swin Transformer features...")
    visualize_features(
        swin_features,
        name="swin",
        save_dir=swin_save_dir,
        original_img=original_img,
        tensor_size=tensor_size,
        resize=False
    )
    
    # DINOv2+DPT visualization
    dino_save_dir = os.path.join(args.save_dir, f"dinov2_dpt_{args.encoder}")
    print(f"\nVisualizing DINOv2+DPT features...")
    visualize_features(
        dino_features,
        name=f"dinov2_{args.encoder}",
        save_dir=dino_save_dir,
        original_img=original_img,
        tensor_size=tensor_size
    )
    
    print("\n" + "=" * 30 + " Done " + "=" * 30)
    print(f"All results saved to {args.save_dir}")


if __name__ == "__main__":
    main() 