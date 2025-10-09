#!/bin/bash

# GLOVER-Qwen-VL训练脚本
# 基于Qwen-7B-VL模型进行GLOVER训练

deepspeed --include localhost:2 --master_port=23914 ../train_glover_qwen_simple.py \
  --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL" \
  --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14" \
  --sam_vit_path="/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth" \
  --dataset_dir="/mnt/data-oss/rap-prod-bak/GLOVER/dataset/HOVA-500K" \
  --dataset="3doi" \
  --sample_rates="1" \
  --exp_name="glover_qwen_vl" \
  --log_base_dir="/mnt/data-oss/data-cpfs/GLOVER/output" \
  --vis_save_path="/mnt/data-oss/data-cpfs/GLOVER/output/vis_output4_kl" \
  --model_arch="glover_qwen" \
    --lr=0.0001 \
    --epochs=5 \
    --batch_size=4 \
    --grad_accumulation_steps=2 \
    --steps_per_epoch=100 \
    --ce_loss_weight=0.01 \
    --use_text_emb_in_suffix_sam \
    --kl_loss_weight=0.1 \
  --train_firs_mask_decoder \
  --use_diff_lr \
  --lr_ratio=0.1 \
  --lora_r=8 \
  --lora_alpha=16 \
  --lora_dropout=0.05 \
  --lora_target_modules="q_proj,v_proj"
