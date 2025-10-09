# HANDAL数据集评估

## 概述

这个目录包含了用于在HANDAL数据集上评估GLOVER++模型affordance预测性能的代码。

## 文件说明

### 核心文件
- `eval_handal.py` - HANDAL数据集评估脚本
- `run_handal_eval.sh` - 启动脚本
- `handal_eval_results.txt` - 评估结果文件

### 环境
- `.venv/` - Python虚拟环境（包含所有必要的依赖）

### 其他文件
- `eval_agd20k.py` - AGD20K数据集评估脚本（参考）
- `eval_agd20k.sh` - AGD20K启动脚本（参考）
- `eval_agd20k_single_action.py` - 单动作评估脚本（参考）
- `test_single_action.sh` - 单动作测试脚本（参考）

## 使用方法

### 1. 运行HANDAL评估

```bash
# 方法1：使用启动脚本（推荐）
./run_handal_eval.sh

# 方法2：直接运行Python脚本
source .venv/bin/activate
export PYTHONPATH="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER:$PYTHONPATH"
python eval_handal.py --max_samples 5
```

### 2. 参数说明

- `--max_samples`: 最大测试样本数（默认5）
- `--seed`: 随机种子（默认42）
- `--dataset_dir`: 数据集路径
- `--version`: GLOVER++模型路径
- `--vision-tower`: CLIP视觉模型路径

### 3. 评估指标

- **mKLD**: Kullback-Leibler散度（越小越好）
- **mSIM**: 相似度（越大越好）
- **mNSS**: 归一化扫描路径显著性（越大越好）

## 数据集结构

HANDAL数据集包含以下物体类型：
- `handal_dataset_spatulas` - 铲子
- `handal_dataset_strainers` - 滤网
- `handal_dataset_utensils` - 餐具
- `handal_dataset_whisks` - 打蛋器

每个物体类型包含：
- `test/*/rgb/*.jpg` - 测试图像
- `test/*/mask_parts/*.png` - 可抓取部位标注

## 最新结果

```
model name: GLOVER_plus
mKLD: 0.9539180978153524
mSIM: 0.500845113037911
mNSS: 11.962352076406564
```

## 环境要求

- Python 3.11
- PyTorch 2.2.0
- Transformers 4.31.0
- CUDA 12.1

## 注意事项

1. 确保GPU可用（脚本使用GPU 2,3）
2. 确保数据集路径正确
3. 确保模型权重文件存在
4. 使用正确的虚拟环境（.venv） 