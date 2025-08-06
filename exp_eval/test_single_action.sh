#!/bin/bash

# 激活uv虚拟环境
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate

# 设置环境变量
export BITSANDBYTES_FUNCTIONAL=1
export BITSANDBYTES_CPU_ONLY=1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=1

echo "=========================================="
echo "开始单个动作测试..."
echo "测试动作: carry"
echo "最大样本数: 3"
echo "使用GPU 0"
echo "图像尺寸: 1024"
echo "=========================================="

# 运行单个动作测试
python eval_agd20k_single_action.py \
    --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus" \
    --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
    --model_arch="glover++" \
    --precision="bf16" \
    --image_size="1024" \
    --model_max_length="512" \
    --use_text_emb_in_suffix_sam \
    --local_rank=0 \
    --dataset_dir="/mnt/data-oss/rap-prod-bak/GLOVER/dataset/AGD20K/AGD20K/AGD20K/Unseen/testset" \
    --output_dir="./agd20k_single_action_results" \
    --test_action="carry" \
    --max_samples=3

echo "=========================================="
echo "单个动作测试完成"
echo "结果保存在: ./agd20k_single_action_results"
echo "==========================================" 