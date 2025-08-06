#!/bin/bash

# 激活uv虚拟环境
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate

# 设置环境变量
export BITSANDBYTES_FUNCTIONAL=1
export BITSANDBYTES_CPU_ONLY=1
export CUDA_VISIBLE_DEVICES=2,3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
export CUDA_LAUNCH_BLOCKING=1

echo "=========================================="
echo "开始AGD20K数据集评估..."
echo "使用GPU 2,3"
echo "数据集: AGD20K Unseen testset"
echo "模型: GLOVER_plus"
echo "评估指标: KLD, SIM, NSS"
echo "每种物体最大样本数: 3"
echo "图像尺寸: 1024"
echo "=========================================="

# 运行评估 - 使用mask图片
echo "=========================================="
echo "测试1: 使用GT mask图片"
echo "=========================================="
python eval_agd20k.py \
    --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus" \
    --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
    --model_arch="glover++" \
    --precision="bf16" \
    --image_size="1024" \
    --model_max_length="64" \
    --use_text_emb_in_suffix_sam \
    --local_rank=0 \
    --dataset_dir="/mnt/data-oss/rap-prod-bak/GLOVER/dataset/AGD20K/AGD20K/AGD20K/Unseen/testset" \
    --output_dir="./agd20k_results_mask" \
    --max_samples_per_object=3 \
    --pred_threshold=0.01 \
    --gt_threshold=0.1

# echo "=========================================="
# echo "测试2: 使用JSON点坐标生成mask"
# echo "=========================================="
# # python eval_agd20k.py \
#     --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus" \
#     --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
#     --model_arch="glover++" \
#     --precision="bf16" \
#     --image_size="1024" \
#     --model_max_length="64" \
#     --use_text_emb_in_suffix_sam \
#     --local_rank=0 \
#     --dataset_dir="/mnt/data-oss/rap-prod-bak/GLOVER/dataset/AGD20K/AGD20K/AGD20K/Unseen/testset" \
#     --output_dir="./agd20k_results_json" \
#     --max_samples_per_object=3 \
#     --use_json_points \
#     --pred_threshold=0.01 \
#     --gt_threshold=0.1

echo "预测阈值: 0.01 (降低阈值以显示更多预测区域)"
echo "=========================================="
echo "AGD20K评估完成"
echo "Mask图片结果保存在: ./agd20k_results_mask"
echo "JSON点坐标结果保存在: ./agd20k_results_json"
echo "==========================================" 