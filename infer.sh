#!/bin/bash

# Activate virtual environment
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate

# 禁用bitsandbytes以避免权限错误
export BITSANDBYTES_FUNCTIONAL=1
export BITSANDBYTES_CPU_ONLY=1

# 使用2号和3号GPU卡
export CUDA_VISIBLE_DEVICES=2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
export CUDA_LAUNCH_BLOCKING=1

echo "=========================================="
echo "开始多GPU推理..."
echo "使用GPU 2,3，处理对象: bread"
echo "内存优化设置: max_split_size_mb=128"
echo "=========================================="

# 单GPU运行
python infer.py \
    --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus" \
    --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
    --model_arch="glover++" \
    --precision="bf16" \
    --image_size="128" \
    --model_max_length="64" \
    --vis_save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/infer/vis_output" \
    --vis_argmax_save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/infer/vis_argmax_output" \
    --image_path="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/images/%s.png" \
    --objects="bread,ham" \
    --actions="grab,grab" \
    --prompt="Where should I interact with the %s to %s it? Please output segmentation mask." \
    --use_text_emb_in_suffix_sam \
    --local-rank=0

echo "=========================================="
echo "单GPU推理完成"
echo "==========================================" 