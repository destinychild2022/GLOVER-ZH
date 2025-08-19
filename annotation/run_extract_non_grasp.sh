#!/bin/bash

# 非抓取帧提取脚本（基于JSON文件）
# 从机械臂操作数据中抽取左右夹爪都张开的时间帧

# 设置数据路径
DATA_ROOT="/mnt/data-oss/rap-prod-bak/AgibotWorld-Challenge/AgiBotWorldChallenge-2025/iros_agibot_sim"
OUTPUT_DIR="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/annotation/non_grasp_frames_json"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "开始提取非抓取帧数据（基于JSON文件）..."
echo "数据根目录: $DATA_ROOT"
echo "输出目录: $OUTPUT_DIR"

# 运行提取脚本
python extract_non_grasp_frames.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --max_tasks 10 \
    --frames_per_episode 20 \
    --episodes_per_task 6 \
    --grasp_threshold 10.0

echo "提取完成！"
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "生成的数据集结构:"
echo "- images/: 按task_name分成子文件夹的图像"
echo "- annotations/: 每个任务的标注文件"
echo "- extraction_stats.json: 提取统计信息" 