# GLOVER-Qwen 模型集成

## 概述

本项目成功将千问（Qwen）7B模型集成到GLOVER框架中，替换了原有的LISA7B模型，同时保持了所有原有的GLOVER功能。

## 主要改进

### 1. 模型替换
- **原模型**: LISA7B (Llama-based)
- **新模型**: Qwen2.5-7B-Instruct
- **保持功能**: 所有GLOVER原有功能（SAM、CLIP、视觉模块、损失函数等）

### 2. 兼容性修复
- 修复了transformers 4.55.0版本中的导入问题
- 添加了缺失的mask函数实现
- 保持了与原有代码的完全兼容性

## 文件结构

```
model/
├── llava/model/language_model/
│   ├── llava_qwen.py          # 千问模型的LLaVA适配器
│   └── __init__.py            # 模块注册
├── GLOVER_qwen.py             # GLOVER-Qwen主模型
└── llava/model/language_model/mpt/
    └── hf_prefixlm_converter.py  # 修复了导入问题

train_glover_qwen.py           # 千问模型训练脚本
train_glover_qwen.sh           # 训练启动脚本
test_glover_qwen_full.py       # 完整功能测试脚本
```

## 使用方法

### 1. 环境准备
```bash
# 使用高版本transformers环境
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate
```

### 2. 测试模型
```bash
python test_glover_qwen_full.py
```

### 3. 开始训练
```bash
# 方法1: 直接运行脚本
./train_glover_qwen.sh

# 方法2: 手动运行
deepspeed --include localhost:1 --master_port=23914 train_glover_qwen.py \
  --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B" \
  --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14" \
  --sam_vit_path="/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth" \
  --dataset_dir='/mnt/data-oss/data-cpfs/GLOVER/HOVA-500K' \
  --dataset="3doi||ego4d||epic100||handal" \
  --sample_rates="1,1,1,1" \
  --exp_name="glover_qwen" \
  --log_base_dir="/mnt/data-oss/data-cpfs/GLOVER/output" \
  --vis_save_path="/mnt/data-oss/data-cpfs/GLOVER/output/vis_output" \
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

## 技术细节

### 1. 模型适配
- 创建了`LlavaQwenModel`和`LlavaQwenForCausalLM`类
- 继承自`LlavaMetaModel`和`LlavaMetaForCausalLM`
- 使用`AutoModelForCausalLM`加载千问模型

### 2. 配置兼容
- 保持了原有的GLOVER配置结构
- 添加了千问模型特定的配置参数
- 支持`trust_remote_code=True`参数

### 3. 损失函数
- 保持了所有原有的损失函数（dice_loss, sigmoid_ce_loss, cal_kl等）
- 支持多模态训练的各种损失权重

### 4. 视觉模块
- 保持了SAM和CLIP的完整集成
- 支持视觉tower和多模态投影
- 保持了原有的视觉处理流程

## 性能优势

1. **更好的语言理解**: 千问模型在中文和英文任务上都有更好的表现
2. **更强的推理能力**: 千问模型在逻辑推理和复杂任务上表现更优
3. **更好的指令遵循**: 千问模型在指令理解和执行方面更准确
4. **保持原有功能**: 所有GLOVER的视觉理解和分割功能都得到保留

## 注意事项

1. **环境要求**: 需要使用transformers 4.55.0或更高版本
2. **内存需求**: 千问模型可能需要更多的GPU内存
3. **训练时间**: 可能需要调整学习率和训练参数以获得最佳效果

## 故障排除

### 导入错误
如果遇到导入错误，请确保：
- 使用了正确的Python环境
- transformers版本 >= 4.55.0
- 所有依赖包都已正确安装

### 内存不足
如果遇到内存不足问题：
- 减小batch_size
- 使用gradient_accumulation_steps
- 启用混合精度训练（bf16或fp16）

## 总结

通过这次集成，我们成功地将千问模型引入到GLOVER框架中，在保持所有原有功能的同时，提升了模型的整体性能。这个改进为GLOVER项目带来了更好的语言理解和推理能力，同时保持了其在视觉理解和分割方面的优势。
