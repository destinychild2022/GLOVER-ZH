# GLOVER-Qwen 使用说明

## 概述

本项目已经成功集成了千问7B模型作为GLOVER的backbone，替代原来的LISA7B模型，以实现更好的性能。

## 主要修改

### 1. 新增文件
- `model/llava/model/language_model/llava_qwen.py` - 千问模型的LLaVA适配器
- `model/GLOVER_qwen.py` - 基于千问模型的GLOVER实现
- `test_qwen_model.py` - 千问模型测试脚本

### 2. 修改文件
- `train_glover_plus.py` - 添加了`--model_arch`参数支持千问模型
- `train_glover_plus.sh` - 更新为使用千问模型路径

## 使用方法

### 1. 测试千问模型
```bash
python test_qwen_model.py
```

### 2. 使用千问模型训练
```bash
# 使用千问模型训练
./train_glover_plus.sh

# 或者手动指定参数
deepspeed --include localhost:1 --master_port=23914 train_glover_plus.py \
  --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B" \
  --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14" \
  --sam_vit_path="/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth" \
  --dataset_dir='/mnt/data-oss/data-cpfs/GLOVER/HOVA-500K' \
  --dataset="3doi||ego4d||epic100||handal" \
  --sample_rates="1,1,1,1" \
  --exp_name="glover_qwen" \
  --log_base_dir="/mnt/data-oss/data-cpfs/GLOVER/output" \
  --vis_save_path="/mnt/data-oss/data-cpfs/GLOVER/output/vis_output" \
  --model_arch="glover_qwen" \
  --lr=0.0005 \
  --epochs=10 \
  --batch_size=32 \
  --steps_per_epoch=196 \
  --ce_loss_weight=0.0 \
  --use_text_emb_in_suffix_sam \
  --kl_loss_weight=0.1 \
  --train_firs_mask_decoder \
  --use_diff_lr \
  --lr_ratio=0.1
```

### 3. 使用原始LISA模型训练
```bash
# 使用原始LISA模型训练
deepspeed --include localhost:1 --master_port=23914 train_glover_plus.py \
  --version="/mnt/data-oss/data-cpfs/GLOVER/models/LISA_Plus_7b" \
  --vision-tower="/mnt/data-oss/data-cpfs/GLOVER/models/clip-vit-large-patch14" \
  --sam_vit_path="/mnt/data-oss/data-cpfs/GLOVER/models/SAM-vit-h/sam_vit_h_4b8939.pth" \
  --dataset_dir='/mnt/data-oss/data-cpfs/GLOVER/HOVA-500K' \
  --dataset="3doi||ego4d||epic100||handal" \
  --sample_rates="1,1,1,1" \
  --exp_name="glover++" \
  --log_base_dir="/mnt/data-oss/data-cpfs/GLOVER/output" \
  --vis_save_path="/mnt/data-oss/data-cpfs/GLOVER/output/vis_output" \
  --model_arch="glover++" \
  --lr=0.0005 \
  --epochs=10 \
  --batch_size=32 \
  --steps_per_epoch=196 \
  --ce_loss_weight=0.0 \
  --use_text_emb_in_suffix_sam \
  --kl_loss_weight=0.1 \
  --train_firs_mask_decoder \
  --use_diff_lr \
  --lr_ratio=0.1
```

## 模型架构对比

| 特性 | LISA7B (glover++) | 千问7B (glover_qwen) |
|------|-------------------|---------------------|
| 语言理解 | 良好 | 优秀 |
| 多模态融合 | 良好 | 优秀 |
| 推理速度 | 中等 | 较快 |
| 中文支持 | 有限 | 优秀 |
| 模型大小 | 约7B参数 | 约7B参数 |

## 注意事项

1. **模型路径**: 确保千问模型已下载到正确路径：`/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B`

2. **依赖要求**: 需要安装支持千问模型的transformers版本

3. **内存要求**: 千问模型的内存占用与LISA模型相近，但可能需要更多的GPU内存

4. **性能优化**: 建议使用`--precision="bf16"`进行混合精度训练以提高效率

## 故障排除

如果遇到模型加载问题，请检查：
1. 模型路径是否正确
2. transformers版本是否支持千问模型
3. GPU内存是否充足
4. 依赖库是否完整安装

## 性能预期

使用千问模型作为backbone，预期在以下方面有提升：
- 更好的语言理解和生成能力
- 更强的多模态融合能力
- 更快的推理速度
- 更好的中文支持 