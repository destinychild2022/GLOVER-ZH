# Affordance数据提取工具

这个工具用于从机械臂操作数据中提取物体的affordance信息，包括机械臂末端执行器的6DOF位姿和对应的RGB图像。

## 功能特性

- 从H5文件中提取机械臂末端执行器的位置和姿态信息
- 加载对应的RGB和深度图像
- 支持多个相机视角（头部、左手、右手）
- 自动处理数据目录结构
- 生成标准化的affordance数据集
- 提供数据可视化工具
- **支持GLOVER训练格式转换**

## 数据结构

输入数据按以下结构组织：
```
/mnt/data-oss/rap-prod-bak/AgibotWorld-Challenge/AgiBotWorldChallenge-2025/iros_agibot_sim/
├── 2810130/                    # 任务ID
│   ├── 3335440/               # Episode ID
│   │   ├── A2D0015AB00061/   # Trial ID
│   │   │   ├── 12052046/     # Timestep ID
│   │   │   │   ├── aligned_joints.h5  # 机械臂数据
│   │   │   │   ├── camera/           # 相机数据
│   │   │   │   │   ├── head_color.jpg
│   │   │   │   │   ├── hand_left_color.jpg
│   │   │   │   │   ├── hand_right_color.jpg
│   │   │   │   │   ├── head_depth.png
│   │   │   │   │   ├── hand_left_depth.png
│   │   │   │   │   ├── hand_right_depth.png
│   │   │   │   │   └── time_stamp.json
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 基础Affordance数据提取

```bash
# 使用默认参数
python3 extract.py \
    --data_root /mnt/data-oss/rap-prod-bak/AgibotWorld-Challenge/AgiBotWorldChallenge-2025/iros_agibot_sim \
    --output_dir ./extracted_affordance_data

# 限制处理的任务和episode数量
python3 extract.py \
    --data_root /mnt/data-oss/rap-prod-bak/AgibotWorld-Challenge/AgiBotWorldChallenge-2025/iros_agibot_sim \
    --output_dir ./extracted_affordance_data \
    --max_tasks 5 \
    --max_episodes_per_task 10
```

### 2. GLOVER格式数据提取

```bash
# 提取GLOVER兼容的affordance数据
python3 extract_for_glover.py \
    --data_root /mnt/data-oss/rap-prod-bak/AgibotWorld-Challenge/AgiBotWorldChallenge-2025/iros_agibot_sim \
    --output_dir ./glover_affordance_data \
    --max_tasks 5 \
    --max_episodes_per_task 10
```

### 3. 使用Shell脚本

```bash
# 基础提取
chmod +x run_extraction.sh
./run_extraction.sh

# GLOVER格式提取
chmod +x run_glover_extraction.sh
./run_glover_extraction.sh
```

### 4. 可视化提取的数据

```bash
# 查看基础数据统计信息
python3 visualize_affordance.py \
    --data_dir ./extracted_affordance_data \
    --mode stats

# 可视化GLOVER格式数据
python3 visualize_glover_data.py \
    --data_dir ./glover_affordance_data \
    --mode stats

# 可视化多个样本
python3 visualize_glover_data.py \
    --data_dir ./glover_affordance_data \
    --mode multiple \
    --num_samples 4
```

## 输出数据格式

### 基础格式

提取的数据按以下结构保存：

```
extracted_affordance_data/
├── images/                    # 图像文件
│   ├── 2810130_3335440_000000_rgb.jpg
│   ├── 2810130_3335440_000000_depth.png
│   └── ...
├── poses/                     # 位姿数据
│   ├── 2810130_poses.json
│   └── ...
├── metadata/                  # 元数据
│   ├── 2810130_metadata.json
│   └── ...
└── extraction_stats.json      # 总体统计信息
```

### GLOVER格式

```
glover_affordance_data/
├── images/                    # RGB图像
│   ├── 2810130_3335440_000000.jpg
│   └── ...
├── masks/                     # Affordance掩码
│   ├── 2810130_3335440_000000.png
│   └── ...
├── annotations/               # 标注数据
│   ├── 2810130_annotations.json
│   └── ...
├── metadata/                  # 元数据
│   ├── 2810130_metadata.json
│   └── ...
└── extraction_stats.json      # 总体统计信息
```

### 位姿数据格式 (poses.json)

```json
[
  {
    "episode_id": "3335440",
    "task_id": "2810130",
    "timestamp": 1234567890.123,
    "position": [x, y, z],
    "orientation": [w, x, y, z],  // 四元数格式
    "rgb_image": "2810130_3335440_000000_rgb.jpg",
    "depth_image": "2810130_3335440_000000_depth.png"
  }
]
```

### GLOVER标注格式 (annotations.json)

```json
[
  {
    "image_path": "2810130_3335440_000000.jpg",
    "mask_path": "2810130_3335440_000000.png",
    "question": "<image>\nWhere should I interact with the object to pick up it? Please output segmentation mask.",
    "answer": "You can interact with the highlighted area [SEG]."
    "affordance_point": [x, y],  // 2D像素坐标
    "position_3d": [x, y, z],    // 3D位置
    "orientation_3d": [w, x, y, z],  // 3D姿态
    "timestamp": 1234567890.123,
    "task_id": "2810130",
    "episode_id": "3335440"
  }
]
```

## 数据说明

### 机械臂数据
- **位置**: 3D坐标 (x, y, z)，单位为米
- **姿态**: 四元数 (w, x, y, z)，表示末端执行器的旋转
- **时间戳**: 数据采集的时间戳

### 图像数据
- **RGB图像**: 彩色图像，格式为JPG
- **深度图像**: 深度图，格式为PNG（可选）
- **相机视角**: 支持头部、左手、右手三个视角

### Affordance信息
每个样本包含：
1. 机械臂末端执行器的6DOF位姿
2. 对应的RGB图像
3. 可选的深度图像
4. 时间戳和任务信息

### GLOVER兼容性
GLOVER格式数据额外包含：
1. **Affordance掩码**: 二值化的affordance区域
2. **2D Affordance点**: 从3D位姿投影得到的图像坐标
3. **标准问答格式**: 符合GLOVER训练要求的问题和答案
4. **分割标记**: 包含[SEG]标记的答案文本

## 关键改进

### 3D到2D投影
- 将机械臂末端执行器的3D位置投影到2D图像坐标
- 使用相机内参进行透视投影
- 生成以投影点为中心的高斯掩码

### GLOVER格式转换
- 生成标准的问题格式："Where should I interact with the {object} to pick up it?"
- 生成标准的答案格式："You can interact with the highlighted area [SEG]."
- 创建二值化的affordance掩码
- 保存完整的3D和2D信息

## 注意事项

1. 确保有足够的磁盘空间存储提取的数据
2. 处理大量数据时建议使用`--max_tasks`和`--max_episodes_per_task`参数限制处理范围
3. 图像文件可能较大，注意存储空间
4. 深度图像为可选数据，某些样本可能没有深度信息
5. **相机内参需要根据实际相机参数调整**
6. **3D到2D投影是简化模型，实际应用中需要准确的相机外参**

## 故障排除

### 常见问题

1. **H5文件读取错误**: 确保h5py库版本兼容
2. **图像文件不存在**: 检查相机数据目录结构
3. **内存不足**: 减少`--max_episodes_per_task`参数值
4. **磁盘空间不足**: 检查输出目录的可用空间
5. **投影坐标异常**: 检查相机内参设置
6. **掩码生成失败**: 确保affordance点在图像范围内

### 调试模式

可以修改提取脚本中的日志输出来调试问题：

```python
# 在AffordanceExtractor类中添加调试信息
print(f"Processing task {task_id}, episode {episode_id}")
print(f"3D position: {position}")
print(f"2D projection: {affordance_point}")
```

## 扩展功能

### 添加新的相机视角
在`load_camera_data`方法中添加新的相机类型：

```python
rgb_files = {
    'head': camera_dir / "head_color.jpg",
    'hand_left': camera_dir / "hand_left_color.jpg", 
    'hand_right': camera_dir / "hand_right_color.jpg",
    'new_camera': camera_dir / "new_camera_color.jpg"  # 添加新相机
}
```

### 改进3D到2D投影
在`project_3d_to_2d`方法中使用更准确的相机模型：

```python
def project_3d_to_2d(self, point_3d: np.ndarray, camera_name: str) -> Tuple[int, int]:
    # 使用完整的相机模型，包括外参
    camera_extrinsics = self.get_camera_extrinsics(camera_name)
    # 实现完整的投影变换
    return projected_point
```

### 添加相机位姿信息
如果需要相机位姿信息，可以在`AffordanceData`中添加`camera_pose`字段，并在相应的数据加载函数中实现相机位姿的提取。 