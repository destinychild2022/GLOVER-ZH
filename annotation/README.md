# 半自动标注工具

这是一个基于SAM（Segment Anything Model）的半自动标注工具，用于生成GLOVER++训练所需的数据集。

## 功能特性

- **SAM分割**: 使用SAM模型进行精确的物体分割
- **交互式标注**: 支持手动点击添加affordance点
- **批量处理**: 支持批量处理多张图像
- **Jupyter支持**: 提供Jupyter notebook交互式标注工具
- **输出格式**: 生成与3doi.json相同格式的标注数据
- **掩码可视化**: 生成的掩码包含affordance点标记

## 文件说明

### 核心脚本
- `simple_annotation.py`: 批量标注工具（命令行）
- `interactive_annotation.py`: Jupyter notebook交互式标注工具
- `visual_annotation.py`: 可视化交互式标注工具（推荐）

### 启动脚本
- `run_simple_annotation.sh`: 批量标注工具启动脚本
- `run_visual_annotation.sh`: 可视化标注工具启动脚本

### 配置文件
- `points_template.json`: 点击点坐标模板文件
- `README.md`: 说明文档

### 输出目录
- `output/`: 所有生成文件的输出目录
  - `output/annotations.json`: 标注数据文件
  - `output/masks/`: 分割掩码图像目录
  - `output/points_template.json`: 生成的点击点模板

## 模型路径

确保以下模型文件存在：
- SAM模型: `/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth`

## 使用方法

### 1. 可视化交互标注（推荐）

```bash
# 激活虚拟环境
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate

# 启动可视化标注工具
./run_visual_annotation.sh
```

然后在Jupyter notebook中运行：
```python
from visual_annotation import create_visual_annotator

# 创建标注器
gui = create_visual_annotator("/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth")

# 显示控件
gui.display_controls()
```

### 2. 批量标注

```bash
# 激活虚拟环境
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate

# 创建点击点模板
./run_simple_annotation.sh template

# 编辑模板文件（可选）
# 修改 output/points_template.json 中的点击坐标

# 运行批量标注
./run_simple_annotation.sh batch output/my_annotations.json
```

### 3. 交互式标注（Jupyter Notebook）

```python
# 在Jupyter notebook中运行
from interactive_annotation import create_interactive_annotator

# 创建标注器
gui = create_interactive_annotator("/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth")

# 显示控件
gui.display_controls()
```

## 输出格式

所有输出文件保存在`output/`目录中：

### 标注数据 (annotations.json)
```json
[
  {
    "img_name": "image.jpg",
    "bbox": [0.1, 0.2, 0.3, 0.4],  // [x, y, width, height] 归一化坐标
    "affordance": [0.5, 0.6],      // [x, y] 归一化交互点
    "height": 480,
    "width": 640,
    "object": "cup",
    "action": "grasp",
    "gt_path": "output/masks/image_mask.png"
  }
]
```

### 分割掩码
- 位置: `output/masks/`
- 格式: PNG图像
- 特点: 包含红色圆点标记affordance点位置

## 预定义类别

### 物体类别
```
cup, bowl, plate, fork, spoon, knife, pan, pot, 
blender, microwave, refrigerator, dishwasher, toaster,
phone, laptop, book, pen, paper, chair, table,
door, window, light, switch, remote, keys, wallet,
clothes, shoes, bag, bottle, can, box, container
```

### 动作类别
```
grasp, lift, move, place, put, take, pick, hold,
open, close, push, pull, turn, rotate, press,
pour, fill, empty, clean, wash, wipe, sweep,
cut, slice, chop, mix, stir, cook, heat, cool
```

## 环境要求

```bash
# 激活虚拟环境
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate

# 依赖包已安装：
# - torch, torchvision
# - opencv-python
# - segment-anything
# - matplotlib
# - ipywidgets (用于Jupyter交互)
# - tqdm
```

## 注意事项

1. **点击点坐标**: 模板文件中的坐标是归一化的`[0,1]`范围
2. **掩码保存**: 所有掩码都包含affordance点的可视化标记
3. **输出目录**: 所有输出文件自动保存到`output/`子目录
4. **GPU支持**: 工具自动检测并使用CUDA加速（如果可用）
5. **错误处理**: 工具包含完善的错误处理和状态提示

## 故障排除

### 环境问题
- 确保激活了正确的虚拟环境
- 检查SAM模型路径是否正确

### 模型加载问题
- 确保SAM模型路径正确
- 检查CUDA环境是否正确配置

### 内存不足
- 使用CPU模式：`--device cpu`
- 减少批处理大小
- 关闭其他占用GPU的程序 