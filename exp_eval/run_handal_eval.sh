#!/bin/bash

# 激活uv虚拟环境（使用exp_eval目录下的环境）
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/exp_eval/.venv/bin/activate

# 设置环境变量
export BITSANDBYTES_FUNCTIONAL=1
export BITSANDBYTES_CPU_ONLY=1
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8
export CUDA_LAUNCH_BLOCKING=1

# 设置Python路径
export PYTHONPATH="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER:$PYTHONPATH"

echo "=========================================="
echo "开始HANDAL数据集评估（GLOVER++真实预测）..."
echo "使用GPU 2,3"
echo "数据集: HOVA-500K HANDAL"
echo "模型: GLOVER_plus"
echo "评估指标: KLD, SIM, NSS"
echo "最大样本数: 15838 (全部数据)"
echo "图像尺寸: 1024"
echo "随机种子: 42"
echo "=========================================="

# 切换到项目根目录
cd /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER

# 运行评估
echo "=========================================="
echo "开始HANDAL数据集评估"
echo "=========================================="
python exp_eval/eval_handal.py \
    --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus" \
    --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
    --model_arch="glover++" \
    --precision="bf16" \
    --image_size="1024" \
    --model_max_length="512" \
    --use_text_emb_in_suffix_sam \
    --local-rank=0 \
    --dataset_dir="/mnt/data-oss/rap-prod-bak/GLOVER/dataset/HOVA-500K/HANDAL" \
    --max_samples=15838 \
    --seed=42

echo "=========================================="
echo "HANDAL评估完成"
echo "结果保存在: ./handal_eval_results.txt"
echo "可视化结果保存在: ./visualizations/"
echo "==========================================" 