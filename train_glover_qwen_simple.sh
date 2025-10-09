#!/bin/bash

# 使用高版本transformers环境
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate

# 启动训练 - 使用3张GPU (0-2号)
deepspeed --include localhost:0,1,2 --master_port=23914 train_glover_qwen_simple.py \
  --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL" \
  --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL" \
  --sam_vit_path="/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth" \
  --dataset_dir="/mnt/data-oss/rap-prod-bak/GLOVER/dataset/HOVA-500K" \
  --dataset="3doi" \
  --sample_rates="1" \
  --exp_name="glover_qwen" \
  --log_base_dir="/mnt/data-oss/data-cpfs/GLOVER/output" \
  --vis_save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/qwen4" \
  --model_arch="glover_qwen" \
  --lr=0.001 \
  --epochs=10 \
  --batch_size=8 \
  --steps_per_epoch=196 \
  --grad_accumulation_steps=2 \
         --precision="bf16" \
  --use_mm_start_end \
  --conv_type="llava_v1" \
  --ce_loss_weight=1.0 \
  --dice_loss_weight=0.5 \
  --bce_loss_weight=2.0 \
  --use_text_emb_in_suffix_sam \
  --kl_loss_weight=0.5 \
  --train_firs_mask_decoder \
  --use_diff_lr \
  --lr_ratio=0.1 \
  --lora_r=16 \
  --lora_alpha=32 \
  --lora_dropout=0.1 \
  --lora_target_modules="q_proj,v_proj,lm_head,gate_proj,up_proj,down_proj"