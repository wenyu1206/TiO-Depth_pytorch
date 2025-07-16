#!/bin/bash
# 使用DINOv2+DPT作为编码器训练TiO-Depth模型
# 图像尺寸调整为378×1246，可被patch_size=14整除

CUDA_VISIBLE_DEVICES=0 python train_dist_2.py \
    --name TiO-Depth-DINOv2b-DPT_378x1246_KITTI_S_B8 \
    --exp_opts options/TiO-Depth/train/tio_depth-dinov2b-dpt_378x1246_kitti_stereo.yaml \
    --batch_size 8 \
    --metric_source rawdepth sdepth refdepth \
    --save_freq 5 \
    --visual_freq 1000 