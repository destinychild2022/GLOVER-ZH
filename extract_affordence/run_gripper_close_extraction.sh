#!/bin/bash

# 设置数据路径和输出目录
DATA_ROOT="/mnt/data-oss/rap-prod-bak/AgibotWorld-Challenge/AgiBotWorldChallenge-2025/iros_agibot_sim"
OUTPUT_DIR="./gripper_close_output"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "开始提取夹爪闭合时刻数据..."
echo "数据源: $DATA_ROOT"
echo "输出目录: $OUTPUT_DIR"

# 运行夹爪闭合时刻提取脚本
python3 extract_grasp_moments.py \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --max_tasks 2 \
    --max_episodes_per_task 1

echo "提取完成！"
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
if [ -d "$OUTPUT_DIR/visualizations" ]; then
    echo "可视化文件: $(ls $OUTPUT_DIR/visualizations/)"
fi
if [ -d "$OUTPUT_DIR/annotations" ]; then
    echo "标注文件: $(ls $OUTPUT_DIR/annotations/)"
fi

echo "开始查看可视化结果..."
if [ -d "$OUTPUT_DIR/visualizations" ]; then
    echo "可视化文件列表:"
    ls -la $OUTPUT_DIR/visualizations/
    
    # 显示第一个可视化文件的信息
    first_vis=$(ls $OUTPUT_DIR/visualizations/*.png | head -1)
    if [ -n "$first_vis" ]; then
        echo "第一个可视化文件: $first_vis"
        echo "文件大小: $(du -h $first_vis | cut -f1)"
    fi
fi

echo "验证完成！" 