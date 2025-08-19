# 非抓取帧提取脚本

这个脚本用于从机械臂操作数据中抽取左右夹爪都不闭合的时间帧，用于生成affordance标注数据集。

## 功能特点

- **智能帧选择**: 自动检测左右夹爪都不闭合的时间段
- **多样化采样**: 从每个episode中选择时间上均匀分布的帧
- **多相机支持**: 提取头部、左手、右手相机的图像
- **任务分类**: 按任务ID组织数据，便于后续标注
- **完整元数据**: 保存3D位置、方向、时间戳等完整信息

## 使用方法

### 1. 快速运行

使用提供的运行脚本：

```bash
cd /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/annotation
chmod +x run_extract_non_grasp.sh
./run_extract_non_grasp.sh
```

### 2. 自定义参数运行

```bash
python extract_non_grasp_frames.py \
    --data_root /mnt/data-oss/rap-prod-bak/AgibotWorld-Challenge/AgiBotWorldChallenge-2025/iros_agibot_sim \
    --output_dir /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/annotation/non_grasp_frames \
    --frames_per_episode 5 \
    --grasp_threshold 50.0 \
    --max_episodes_per_task 10
```

### 3. 参数说明

- `--data_root`: 数据集根目录路径
- `--output_dir`: 输出目录路径
- `--frames_per_episode`: 每个episode提取的帧数（默认5）
- `--grasp_threshold`: 夹爪闭合阈值（默认50.0）
- `--max_episodes_per_task`: 每个任务最大处理的episode数（默认None，处理所有）
- `--max_tasks`: 最大处理的任务数（默认None，处理所有）

## 输出结构

```
non_grasp_frames/
├── images/                          # 所有提取的图像
│   ├── 0_episode_001_frame_0123_head.jpg
│   ├── 0_episode_001_frame_0123_hand_left.jpg
│   ├── 0_episode_001_frame_0123_hand_right.jpg
│   └── ...
├── annotations/                     # 每个任务的标注文件
│   ├── 0_non_grasp_annotations.json
│   ├── 1_non_grasp_annotations.json
│   └── ...
├── task_datasets/                   # 按任务分类的数据集
│   ├── task_0/
│   │   ├── images/
│   │   ├── 0_annotations.json
│   │   └── dataset_info.json
│   ├── task_1/
│   │   ├── images/
│   │   ├── 1_annotations.json
│   │   └── dataset_info.json
│   └── ...
├── summary_dataset/                 # 汇总数据集
│   ├── images/
│   ├── all_annotations.json
│   └── summary_info.json
└── extraction_stats.json           # 提取统计信息
```

## 标注数据格式

每个标注包含以下信息：

```json
{
  "head_image": "0_episode_001_frame_0123_head.jpg",
  "hand_left_image": "0_episode_001_frame_0123_hand_left.jpg",
  "hand_right_image": "0_episode_001_frame_0123_hand_right.jpg",
  "position_3d": [0.123, 0.456, 0.789],
  "orientation_3d": [0.0, 0.0, 0.0, 1.0],
  "timestamp": 12.345,
  "frame_idx": 123,
  "left_effector_position": 15.2,
  "right_effector_position": 12.8,
  "camera_name": "head",
  "task_id": "0",
  "episode_id": "001"
}
```

## 使用建议

1. **调整阈值**: 根据实际数据调整`grasp_threshold`参数
2. **控制数量**: 通过`frames_per_episode`和`max_episodes_per_task`控制数据量
3. **分批处理**: 对于大数据集，可以分批处理不同任务
4. **质量检查**: 运行后检查提取的图像质量和数量

## 后续步骤

提取完成后，可以使用Web标注工具进行affordance标注：

```bash
cd /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/annotation
python web_annotation.py \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
    --image_dir /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/annotation/non_grasp_frames/summary_dataset/images \
    --host 0.0.0.0 \
    --port 5000
```

## 注意事项

- 确保有足够的磁盘空间存储提取的图像
- 处理大量数据时可能需要较长时间
- 建议先在小数据集上测试参数设置
- 检查提取的图像质量，确保没有损坏的文件 