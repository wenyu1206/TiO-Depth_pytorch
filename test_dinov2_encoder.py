#!/usr/bin/env python3
"""
测试DINOv2+DPT与Swin Transformer编码器的特征金字塔输出差异
重点比较：
1. 两种编码器输出的特征维度
2. 特征通道数是否统一
3. 特征的空间分辨率
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
import copy

from models.backbones.swin import get_orgwintrans_backbone
from models.backbones.dinov2_dpt import get_dinov2_dpt_backbone, DINOv2_DPT_Backbone, check_image_size_for_patching
from models.backbones.dinov2_dpt import DPTHeadFeaturePyramid

# 添加Depth-Anything-V2到路径
depth_anything_path = '/home/ywan0794/Depth-Anything-V2'
if depth_anything_path not in sys.path:
    sys.path.append(depth_anything_path)

from depth_anything_v2.dinov2 import DINOv2


def visualize_features(features, name="feature", save_dir="./"):
    """可视化特征图并保存到指定目录"""
    for i, feat in enumerate(features):
        # 计算通道平均值
        feature_map = feat[0].mean(dim=0).detach().cpu().numpy()
        
        # 归一化以便可视化
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(feature_map, cmap='viridis')
        plt.colorbar(label='Normalized Feature Value')
        plt.title(f"{name} Feature Level {i+1}\nShape: {feat.shape} (B, C, H, W)")
        
        # 保存图像
        save_path = os.path.join(save_dir, f"{name}_level_{i+1}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存特征可视化: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='测试与比较DINOv2+DPT和Swin Transformer编码器')
    parser.add_argument('--save_dir', type=str, default='./encoder_test', help='保存可视化结果的目录')
    parser.add_argument('--swin_weights', type=str, 
                       default='/home/ywan0794/TiO-Depth_pytorch/swin_tiny_patch4_window7_224_22k.pth', 
                       help='Swin Transformer预训练权重路径')
    parser.add_argument('--dino_weights', type=str,
                       default='/home/ywan0794/Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth',
                       help='DepthAnything预训练权重路径')
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建随机输入，确保尺寸能被14整除
    batch_size = 1
    h, w = 378, 1246  # 图像尺寸确保被14整除
    
    # 验证尺寸是否适合ViT的patch大小
    check_image_size_for_patching(h, w, 14)
    
    x = torch.randn(batch_size, 3, h, w).to(device)
    
    print("=" * 80)
    print(f"输入图像形状: {x.shape}")
    print(f"ViT Patch网格大小: {h//14}x{w//14} (总计 {h//14 * w//14} 个patches)")
    print("=" * 80)
    
    # =========================================================================
    # 1. 测试Swin Transformer编码器
    # =========================================================================
    print("\n" + "=" * 30 + " Swin Transformer 编码器 " + "=" * 30)
    print(f"加载Swin Transformer预训练权重: {args.swin_weights}")
    print("注意: Swin Transformer的每个层级输出有不同的通道数")
    
    # 加载Swin编码器
    swin_encoder, swin_channels = get_orgwintrans_backbone('orgSwin-T-s2', False)  # 先不加载预训练权重
    
    # 手动加载预训练权重，以便更清晰地展示过程
    if os.path.exists(args.swin_weights):
        print(f"正在加载Swin预训练权重...")
        swin_weights = torch.load(args.swin_weights, map_location=torch.device('cpu'))
        
        # 如果是state_dict格式
        if 'model' in swin_weights:
            swin_weights = swin_weights['model']
        
        # 检查关键层的权重加载情况
        model_dict = swin_encoder.state_dict()
        pretrained_dict = {k: v for k, v in swin_weights.items() if k in model_dict and model_dict[k].shape == v.shape}
        
        print(f"总共找到 {len(pretrained_dict)}/{len(model_dict)} 个可加载的参数")
        model_dict.update(pretrained_dict)
        swin_encoder.load_state_dict(model_dict)
    else:
        print(f"警告: Swin预训练权重文件未找到 - {args.swin_weights}")
    
    swin_encoder = swin_encoder.to(device)
    
    # 提取Swin特征，记录维度
    with torch.no_grad():
        swin_features = swin_encoder(x)
    
    print(f"\nSwin Transformer特征金字塔输出:")
    print(f"总层数: {len(swin_features)}")
    
    swin_info = []
    for i, feat in enumerate(swin_features):
        level_info = {
            "level": i+1,
            "shape": list(feat.shape),
            "channels": feat.shape[1],
            "resolution": f"{feat.shape[2]}x{feat.shape[3]}"
        }
        swin_info.append(level_info)
        scale_factor = h / feat.shape[2]
        print(f"  层级 {i+1}: 形状={feat.shape}, 通道数={feat.shape[1]}, 分辨率降采样因子=1/{scale_factor:.0f}")
    
    # 可视化Swin特征
    swin_viz_dir = os.path.join(args.save_dir, "swin_features")
    os.makedirs(swin_viz_dir, exist_ok=True)
    visualize_features(swin_features, "swin", swin_viz_dir)
    
    # =========================================================================
    # 2. 测试DINOv2+DPT编码器
    # =========================================================================
    print("\n" + "=" * 30 + " DINOv2+DPT 编码器 " + "=" * 30)
    print(f"加载DINOv2+DPT预训练权重: {args.dino_weights}")
    print("注意: DPT会将不同层级的特征统一为相同的通道数")
    
    # 创建DINOv2+DPT编码器
    # 首先创建原始的DINOv2模型和DPT，然后加载预训练权重
    dinov2_model = DINOv2(model_name='vitb')
    
    # 提取模型的关键参数
    embed_dim = dinov2_model.embed_dim
    
    # 创建自定义的DINOv2+DPT骨干网络
    dinov2_encoder, dinov2_channels = get_dinov2_dpt_backbone('dinov2b', False)  # 先不加载预训练权重
    
    # 手动加载DepthAnything预训练权重（如果存在）
    if os.path.exists(args.dino_weights):
        print(f"正在加载DepthAnything预训练权重...")
        # 一般DepthAnything会自动加载权重，这里我们需要手动加载
        # 这是一个示范，实际上可能需要适配具体的权重格式
        depth_anything_weights = torch.load(args.dino_weights, map_location=torch.device('cpu'))
        
        # 如果是state_dict格式
        if isinstance(depth_anything_weights, dict) and 'model' in depth_anything_weights:
            pretrained_dict = depth_anything_weights['model']
        else:
            pretrained_dict = depth_anything_weights
            
        # 获取模型字典
        model_dict = dinov2_encoder.state_dict()
        
        # 找到匹配的键
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                         if k in model_dict and model_dict[k].shape == v.shape}
        
        print(f"总共找到 {len(pretrained_dict)}/{len(model_dict)} 个可加载的参数")
        
        # 更新模型字典并加载
        model_dict.update(pretrained_dict)
        dinov2_encoder.load_state_dict(model_dict, strict=False)
    else:
        print(f"警告: DepthAnything预训练权重文件未找到 - {args.dino_weights}")
    
    dinov2_encoder = dinov2_encoder.to(device)
    
    # 提取DINOv2+DPT特征
    with torch.no_grad():
        dinov2_features = dinov2_encoder(x)
    
    print(f"\nDINOv2+DPT特征金字塔输出:")
    print(f"总层数: {len(dinov2_features)}")
    
    dino_info = []
    for i, feat in enumerate(dinov2_features):
        level_info = {
            "level": i+1,
            "shape": list(feat.shape),
            "channels": feat.shape[1],
            "resolution": f"{feat.shape[2]}x{feat.shape[3]}"
        }
        dino_info.append(level_info)
        scale_factor = h / feat.shape[2]
        print(f"  层级 {i+1}: 形状={feat.shape}, 通道数={feat.shape[1]}, 分辨率降采样因子=1/{scale_factor:.0f}")
    
    # 可视化DINOv2+DPT特征
    dino_viz_dir = os.path.join(args.save_dir, "dinov2_dpt_features")
    os.makedirs(dino_viz_dir, exist_ok=True)
    visualize_features(dinov2_features, "dinov2_dpt", dino_viz_dir)
    
    # =========================================================================
    # 3. 特征对比总结
    # =========================================================================
    print("\n" + "=" * 30 + " 特征金字塔对比总结 " + "=" * 30)
    
    print("Swin Transformer 特征金字塔:")
    print("层级 | 通道数 | 特征分辨率 | 形状")
    print("-" * 60)
    for info in swin_info:
        print(f"{info['level']} | {info['channels']} | {info['resolution']} | {info['shape']}")
    
    print("\nDINOv2+DPT 特征金字塔:")
    print("层级 | 通道数 | 特征分辨率 | 形状")
    print("-" * 60)
    for info in dino_info:
        print(f"{info['level']} | {info['channels']} | {info['resolution']} | {info['shape']}")
    
    print("\n关键区别:")
    
    # 检查各层通道数是否相同
    dino_unified_channels = all(info['channels'] == dino_info[0]['channels'] for info in dino_info)
    swin_unified_channels = all(info['channels'] == swin_info[0]['channels'] for info in swin_info)
    
    print(f"1. Swin Transformer各层通道数{'相同' if swin_unified_channels else '不同'}")
    print(f"2. DINOv2+DPT各层通道数{'相同' if dino_unified_channels else '不同'}")
    
    if dino_unified_channels:
        print(f"   DPT将所有特征层统一为 {dino_info[0]['channels']} 通道")
    
    print(f"3. 两种编码器的特征分辨率降采样因子相似，但具体数值可能有差异")
    
    # 提示 
    if dino_unified_channels and not swin_unified_channels:
        print("\n注意:")
        print("DPT确实使用了_make_scratch函数将所有特征层的通道数统一为相同值，")
        print(f"而Swin Transformer保持了各层不同的通道数: {[info['channels'] for info in swin_info]}")
        print("在TiO-Depth中，为了保持与Swin兼容，您可能需要:")
        print("1. 修改DPT特征金字塔的通道数，使其与Swin匹配")
        print("2. 或者在TiO-Depth_pytorch/models/networks/tio_depth.py中适配统一通道数的特征金字塔")
    
    print("\n可视化结果已保存到:")
    print(f"- Swin特征: {swin_viz_dir}")
    print(f"- DINOv2+DPT特征: {dino_viz_dir}")


if __name__ == "__main__":
    main() 