#!/bin/bash

# GLOVER++ 自定义标注数据集训练脚本
# 使用包含affordance点坐标的数据集进行LoRA微调
# 基于预训练的GLOVER++权重进行微调
# 极速优化版本：最大化训练速度

# 设置CUDA内存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0,1,2

deepspeed --include localhost:0,1,2 --master_port=23914 train_glover_custom.py \
  --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus" \
  --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
  --sam_vit_path="/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth" \
  --dataset_dir="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/annotation/non_grasp_frames_json/images" \
  --annotation_dir="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/annotation/output" \
  --dataset="custom_annotation" \
  --sample_rates="1" \
  --exp_name="glover++_custom_annotation_with_points" \
  --log_base_dir="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/output" \
  --vis_save_path="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/output/vis_output" \
  --lr=0.001 \
  --epochs=10 \
  --batch_size=48 \
  --steps_per_epoch=64 \
  --grad_accumulation_steps=8 \
  --workers=4 \
  --ce_loss_weight=0.0 \
  --use_text_emb_in_suffix_sam \
  --kl_loss_weight=0.1 \
  --train_firs_mask_decoder \
  --use_diff_lr \
  --lr_ratio=0.1 \
  --lora_r=256 \
  --lora_alpha=512 \
  --lora_dropout=0.05 \
  --lora_target_modules="q_proj,v_proj" 