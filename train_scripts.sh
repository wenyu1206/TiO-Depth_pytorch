# train TiO-Depth with Stereo in 256x832 for 50 epochs
# trained with KITTI Full
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256,expandable_segments:True"
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6"
CUDA_VISIBLE_DEVICES=0 python\
 train_dist_2.py\
 --name TiO-Depth-Swint-M_rc256_KITTIfull_S_B8_resume1\
 --exp_opts options/TiO-Depth/train/tio_depth-swint-m_384crop256_kittifull_stereo.yaml\
 --batch_size 4\
 --metric_source rawdepth sdepth refdepth\
 --metric_name depth_kitti_stereo2015\
 --save_freq 5\
 --visual_freq 1000\
 --start_epoch 21\
 --pretrained_path train_log/2025-02-27_03h01m15s_TiO-Depth-Swint-M_rc256_KITTIfull_S_B8/model/last_model20.pth
