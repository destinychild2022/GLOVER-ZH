#!/bin/bash

# 使用高版本transformers环境
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate

# 启动训练
deepspeed --include localhost:1 --master_port=23914 train_glover_qwen.py \
  --model_arch="glover_qwen" \
  --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B" \
  --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14" \
  --sam_vit_path="/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth" \
  --dataset_dir='/mnt/data-oss/rap-prod-bak/GLOVER/dataset/HOVA-500K' \
  --dataset="3doi||ego4d||epic100||handal" \
  --sample_rates="1,1,1,1" \
  --exp_name="glover_qwen" \
  --log_base_dir="/mnt/data-oss/data-cpfs/GLOVER/output" \
  --vis_save_path="/mnt/data-oss/data-cpfs/GLOVER/output/vis_output" \
  --lr=0.0005 \
  --epochs=10 \
  --batch_size=32 \
  --steps_per_epoch=196 \
  --ce_loss_weight=0.0 \
  --use_text_emb_in_suffix_sam \
  --kl_loss_weight=0.1 \
  --train_firs_mask_decoder \
  --use_diff_lr \
  --lr_ratio=0.1
