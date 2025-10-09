#!/bin/bash

# 激活虚拟环境
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/exp_eval/.venv/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 设置bitsandbytes相关环境变量
export BITSANDBYTES_FUNCTIONAL=1
export BITSANDBYTES_CUDA_SETUP=0
export BITSANDBYTES_NO_CUDA=1

# 设置临时目录
export HOME=/tmp
export USERPROFILE=/tmp
export XDG_CONFIG_HOME=/tmp/.config
export XDG_CACHE_HOME=/tmp/.cache

echo "=========================================="
echo "开始合并LoRA权重到基础模型..."
echo "=========================================="

# 合并权重
python merge_weights.py \
  --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus" \
  --weight="/mnt/data-oss/rap-prod-bak/GLOVER/output/finetune/agi_sim/pytorch_model.bin" \
  --save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/finetune/agi_sim/merged_model" \
  --model_arch="GLOVER++" \
  --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
  --precision="bf16" \
  --lora_r=256 \
  --lora_alpha=512 \
  --lora_dropout=0.05 \
  --lora_target_modules="q_proj,v_proj"
echo "=========================================="
echo "权重合并完成"
echo "合并后的模型保存在: /mnt/data-oss/rap-prod-bak/GLOVER/output/finetune/agi_sim/merged_model"
echo "=========================================="
