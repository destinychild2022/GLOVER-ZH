#!/bin/bash

# 激活虚拟环境
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate

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
echo "开始合并Qwen LoRA权重到基础模型..."
echo "=========================================="

# 合并权重
python merge_weights_qwen.py \
  --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL" \
  --weight="/mnt/data-oss/rap-prod-bak/GLOVER/output/qwen_handal2/pytorch_model.bin" \
  --save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/qwen_handal2/merged_model" \
  --model_arch="glover_qwen" \
  --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL" \
  --precision="bf16" \
  --lora_r=16 \
  --lora_alpha=32 \
  --lora_dropout=0.1 \
  --lora_target_modules="q_proj,v_proj,lm_head,gate_proj,up_proj,down_proj" \
  --use_mm_start_end \
  --train_mask_decoder \
  --out_dim=256 \
  --conv_type="llava_v1" \
  --model_max_length=512 \
  --sam_vit_path="/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth"

echo "=========================================="
echo "权重合并完成"
echo "合并后的模型保存在: /mnt/data-oss/rap-prod-bak/GLOVER/output/qwen_handal2/merged_model"
echo "=========================================="

