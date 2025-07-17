#!/bin/bash

LOG_DIR="./logs"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Start debug"

python train_dist_2.py \
    --name TiO-Depth-DAv2-MeLo-flsea-all \
    --exp_opts options/TiO-Depth/train/tio_depth-dav2_melo-flsea.yaml \
    --batch_size 1 \
    --metric_source rawdepth sdepth refdepth \
    --save_freq 5 \
    --visual_freq 1000 \
    --start_epoch 22 \
    --freeze_lora True \
    2>&1 | tee "${LOG_DIR}/train_dav2_melo_${TIMESTAMP}.log"

rm -rf /date2/zhang_h/stereo/TiO-Depth_pytorch/train_log