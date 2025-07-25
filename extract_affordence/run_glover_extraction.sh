#!/bin/bash

# 设置数据路径和输出目录
DATA_ROOT="/mnt/data-oss/rap-prod-bak/AgibotWorld-Challenge/AgiBotWorldChallenge-2025/iros_agibot_sim"
OUTPUT_DIR="./test_episode_output"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "开始测试单个episode的GLOVER格式affordance数据提取..."
echo "数据源: $DATA_ROOT"
echo "输出目录: $OUTPUT_DIR"
echo "测试episode: 2810130/3335440"

# 运行优化后的提取脚本
python3 extract_for_glover.py \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --test_episode "2810130/3335440"

echo "测试完成！"
echo "输出目录: $OUTPUT_DIR"

# 检查输出文件
echo "检查输出文件..."
ls -la $OUTPUT_DIR/
if [ -d "$OUTPUT_DIR/images" ]; then
    echo "图像文件数量: $(ls $OUTPUT_DIR/images/ | wc -l)"
fi
if [ -d "$OUTPUT_DIR/masks" ]; then
    echo "掩码文件数量: $(ls $OUTPUT_DIR/masks/ | wc -l)"
fi
if [ -d "$OUTPUT_DIR/annotations" ]; then
    echo "标注文件: $(ls $OUTPUT_DIR/annotations/)"
fi

echo "开始可视化验证..."
python3 visualize_glover_data.py \
    --data_dir $OUTPUT_DIR \
    --mode stats

echo "验证完成！" 