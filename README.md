# 半自动标注工具

基于SAM和LISA模型的半自动标注工具，用于生成GLOVER++训练数据集。

## 功能特点

- 🎯 **交互式点击标注**：GUI界面，鼠标点击标注前景点和背景点
- 🔍 **SAM自动分割**：根据点击点自动分割物体
- 🤖 **LISA智能识别**：自动识别物体类别和动作类别
- 📦 **批量处理**：支持批量处理多张图像
- ✏️ **人工校正**：支持手动修正识别结果
- 💾 **标准格式输出**：生成与3doi数据集相同格式的标注文件

## 快速开始

### 1. 安装依赖
```bash
pip install torch torchvision opencv-python matplotlib transformers
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 2. 运行标注工具

#### 交互式标注（推荐）
```bash
./run_annotation_tool.sh
```

#### 批量处理
```bash
python batch_annotation.py \
    --sam_checkpoint /mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth \
    --lisa_model_path /mnt/data-oss/rap-prod-bak/GLOVER/model/LISA_Plus_7b \
    --image_dir /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/images \
    --output_file annotations.json
```

## 文件说明

- `semi_auto_annotation.py` - 主程序（GUI交互式标注）
- `batch_annotation.py` - 批量处理脚本
- `run_annotation_tool.sh` - 启动脚本
- `README.md` - 说明文档

## 模型路径

工具已配置为使用本地模型：
- SAM: `/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth`
- LISA: `/mnt/data-oss/rap-prod-bak/GLOVER/model/LISA_Plus_7b`

## 使用方法

### 交互式标注流程

1. **加载图像**：点击"Load Image"按钮选择图像
2. **添加点击点**：
   - 左键点击：添加前景点（目标物体）
   - 右键点击：添加背景点（非目标区域）
3. **分割物体**：点击"Segment Object"按钮
4. **识别物体和动作**：点击"Recognize Object & Action"按钮
5. **校正结果**：在下拉菜单中选择正确的类别
6. **保存标注**：点击"Save Annotation"按钮

### 批量处理

```bash
# 创建点坐标模板
python batch_annotation.py --create_template --image_dir /path/to/images

# 批量处理（使用默认中心点）
python batch_annotation.py \
    --image_dir /path/to/images \
    --output_file annotations.json

# 批量处理（使用预定义点坐标）
python batch_annotation.py \
    --image_dir /path/to/images \
    --points_file points_template.json \
    --output_file annotations.json
```

## 输出格式

生成与3doi数据集相同格式的标注文件：
```json
{
    "img_name": "example.jpg",
    "bbox": [0.42109375, 0.3861111111111111, 0.584375, 1.0],
    "affordance": [0.43671875, 0.48055555555555557],
    "height": 720,
    "width": 1280,
    "object": "cup",
    "action": "grasp",
    "gt_path": "masks/example_mask.png"
}
```

## 预定义类别

### 物体类别（35种）
```
cup, bowl, plate, fork, spoon, knife, pan, pot, 
blender, microwave, refrigerator, dishwasher, toaster,
phone, laptop, book, pen, paper, chair, table,
door, window, light, switch, remote, keys, wallet,
clothes, shoes, bag, bottle, can, box, container
```

### 动作类别（30种）
```
grasp, lift, move, place, put, take, pick, hold,
open, close, push, pull, turn, rotate, press,
pour, fill, empty, clean, wash, wipe, sweep,
cut, slice, chop, mix, stir, cook, heat, cool
```

## 注意事项

1. **GPU内存**：建议使用至少8GB显存
2. **点击精度**：点击点越准确，分割效果越好
3. **识别准确性**：LISA模型的识别结果可能需要人工校正
4. **文件保存**：标注数据会自动保存到masks目录和annotations.json文件

## 测试示例

使用images文件夹中的图片进行测试：
```bash
# 交互式标注测试
./run_annotation_tool.sh

# 批量处理测试
python batch_annotation.py \
    --image_dir /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/images \
    --output_file test_annotations.json
```
