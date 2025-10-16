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
echo "开始GLOVER-Qwen推理..."
echo "使用合并后的模型权重"
echo "=========================================="

# 运行推理
python infer_qwen.py \
    --version="/mnt/data-oss/rap-prod-bak/GLOVER/output/qwen_handal2/merged_model" \
    --sam_vit_path="/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth" \
    --precision="bf16" \
    --image_size=1024 \
    --model_max_length=512 \
    --vis_save_path="./infer_results/vis_output" \
    --vis_argmax_save_path="./infer_results/vis_argmax_output" \
    --image_path="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/images/%s" \
    --objects="cup" \
    --actions="pick up" \
    --prompt="Where should I interact with the %s to %s it? Please output segmentation mask." \
    --use_text_emb_in_suffix_sam \
    --use_mm_start_end \
    --conv_type="llava_v1" \
    --local_rank=0

echo "=========================================="
echo "GLOVER-Qwen推理完成"
echo "结果保存在: /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/qwen_scripts/infer_results/"
echo "=========================================="