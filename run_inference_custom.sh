#!/bin/bash

# 激活虚拟环境
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/exp_eval/.venv/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "=========================================="
echo "开始使用合并后的模型进行推理..."
echo "使用GPU 0"
echo "=========================================="

# 运行推理（使用合并后的模型）
python infer_custom_weights.py \
    --merged_model="/mnt/data-oss/rap-prod-bak/GLOVER/output/finetune/agi_sim/merged_model" \
    --use_merged_model \
    --vision_tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
    --precision="bf16" \
    --image_size="1024" \
    --model_max_length="512" \
    --max_new_tokens="32" \
    --image_path="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/test/image.png" \
    --objects="bottle" \
    --actions="grasp" \
    --prompt="Where should I interact with the %s to %s it? Please output segmentation mask." \
    --output_dir="./inference_results" \
    --use_text_emb_in_suffix_sam \
    --device="cuda:0"

echo ""
echo "=========================================="
echo "可选：使用原始权重文件进行推理"
echo "=========================================="
# echo "如果要使用原始权重文件，请运行："
# echo "python infer_custom_weights.py \\"
# echo "    --base_model=\"/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus\" \\"
# echo "    --custom_weights=\"/mnt/data-oss/rap-prod-bak/GLOVER/output/finetune/agi_sim/pytorch_model.bin\" \\"
# echo "    --use_merged_model=False \\"
# echo "    --objects=\"bread,ham\" \\"
# echo "    --actions=\"grab,grab\" \\"
# echo "    [其他参数...]"
echo ""

echo "=========================================="
echo "推理完成"
echo "结果保存在: ./inference_results"
echo "==========================================" 