#!/bin/bash

# 准备训练好的模型
echo "开始准备训练好的模型..."

# 设置路径
PRETRAINED_MODEL="/mnt/data-oss/rap-prod-bak/GLOVER/model/LISA_Plus_7b"
CHECKPOINT_DIR="/mnt/data-oss/rap-prod-bak/GLOVER/runs/glover/ckpt_model"
OUTPUT_MODEL_DIR="/mnt/data-oss/rap-prod-bak/GLOVER/runs/glover/trained_model"

# 创建输出目录
mkdir -p $OUTPUT_MODEL_DIR

echo "步骤1: 转换DeepSpeed checkpoint为PyTorch格式..."
cd $CHECKPOINT_DIR
python zero_to_fp32.py . $OUTPUT_MODEL_DIR/pytorch_model.bin

echo "步骤2: 复制配置文件..."
cp $PRETRAINED_MODEL/config.json $OUTPUT_MODEL_DIR/
cp $PRETRAINED_MODEL/tokenizer_config.json $OUTPUT_MODEL_DIR/
cp $PRETRAINED_MODEL/special_tokens_map.json $OUTPUT_MODEL_DIR/
cp $PRETRAINED_MODEL/tokenizer.model $OUTPUT_MODEL_DIR/
cp $PRETRAINED_MODEL/added_tokens.json $OUTPUT_MODEL_DIR/
cp $PRETRAINED_MODEL/generation_config.json $OUTPUT_MODEL_DIR/

echo "步骤3: 检查生成的文件..."
ls -la $OUTPUT_MODEL_DIR/

echo "完成！训练好的模型已保存到: $OUTPUT_MODEL_DIR"
echo "现在可以使用以下参数进行评估和推理："
echo "--version=\"$OUTPUT_MODEL_DIR\"" 