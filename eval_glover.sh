#!/bin/bash

# Activate virtual environment
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate

# 设置要使用的GPU ID
GPU_ID=0

# 使用指定的GPU，设置内存优化
export CUDA_VISIBLE_DEVICES=$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32

echo "开始单GPU推理..."
echo "使用GPU $GPU_ID，处理对象: cup, book"

# 单进程运行，处理所有对象
python infer.py \
    --version="/mnt/data-oss/rap-prod-bak/GLOVER/runs/glover/trained_model" \
    --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
    --model_arch="glover" \
    --precision="bf16" \
    --vis_save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/infer/vis_output" \
    --vis_argmax_save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/infer/vis_argmax_output" \
    --image_path="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/images/%s.png" \
    --objects="cup,book" \
    --actions="grab,open" \
    --prompt="Where should I interact with the %s to %s it? Please output segmentation mask." \
    --load_in_8bit \
    --local-rank=0 
    