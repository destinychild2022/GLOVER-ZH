#!/bin/bash

# 激活虚拟环境
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 设置离线模式，避免网络连接
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 设置临时目录
export HOME=/tmp
export USERPROFILE=/tmp
export XDG_CONFIG_HOME=/tmp/.config
export XDG_CACHE_HOME=/tmp/.cache

echo "=========================================="
echo "开始 GLOVER-Qwen 推理..."
echo "使用基础模型 + LoRA权重 + 训练后的GLOVER组件"
echo "=========================================="

# 设置 PYTHONPATH
export PYTHONPATH=/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER:$PYTHONPATH

# 运行推理 - 使用基础模型+LoRA权重+训练后的GLOVER组件
python infer_qwen_fixed.py \
    --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL" \
    --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
    --model_arch="glover_qwen" \
    --precision="bf16" \
    --image_size="1024" \
    --model_max_length="512" \
    --vis_save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/qwen/infer/vis_output_new" \
    --vis_argmax_save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/qwen/infer/vis_argmax_output_new" \
    --image_path="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/images/%s.png" \
    --objects="book" \
    --actions="open" \
    --prompt="Where should I interact with the %s to %s it? Please output segmentation mask." \
    --use_text_emb_in_suffix_sam \
    --local-rank=0

echo "=========================================="
echo "GLOVER-Qwen 推理完成"
echo "结果保存在: /mnt/data-oss/rap-prod-bak/GLOVER/output/qwen/infer/"
echo "=========================================="
