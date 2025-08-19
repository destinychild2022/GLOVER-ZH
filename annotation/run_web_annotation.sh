#!/bin/bash

# Web可视化标注工具启动脚本
# 
# 快速修改图像目录：
# 1. 修改下面的 IMAGE_DIR 变量
# 2. 例如：IMAGE_DIR="/path/to/your/images/task_X"
# 3. 保存并重新运行脚本
# 
# 支持的目录格式：
# - task_0, task_1, task_2, ... (自动创建对应的输出目录)
# - 任意其他目录名 (使用默认输出目录)

# 设置模型路径
SAM_CHECKPOINT="/mnt/data-oss/rap-prod-bak/GLOVER/model/SAM-vit-h/sam_vit_h_4b8939.pth"

# LISA模型已废弃，不再需要设置路径
# LISA_MODEL_PATH="/mnt/data-oss/rap-prod-bak/GLOVER/model/LISA_Plus_7b"

# 设置图像目录
IMAGE_DIR="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/annotation/non_grasp_frames_json/images/task_4"

# 设置GPU编号（可以通过环境变量或参数设置）
GPU_ID=${GPU_ID:-2}  # 默认使用GPU 2
GPU_IDS="0,1"  # 使用GPU 2和3

# 检查模型文件是否存在
if [ ! -f "$SAM_CHECKPOINT" ]; then
    echo "Error: SAM checkpoint not found at $SAM_CHECKPOINT"
    exit 1
fi

# 检查图像目录是否存在
if [ ! -d "$IMAGE_DIR" ]; then
    echo "Error: Image directory not found at $IMAGE_DIR"
    exit 1
fi

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    # 当使用CUDA_VISIBLE_DEVICES时，设备索引会重新映射
    # 如果设置CUDA_VISIBLE_DEVICES=2,3，那么cuda:0对应物理GPU 2，cuda:1对应物理GPU 3
    DEVICE="cuda:0"  # 使用第一个可见的GPU
    echo "CUDA detected, using GPU 2,3 (映射为cuda:0,1)"
    
    # 检查指定的GPU是否存在
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits | grep -q "^2$"
    if [ $? -ne 0 ]; then
        echo "Warning: GPU 2 not found, available GPUs:"
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits
        echo "Using GPU 0,1 instead"
        GPU_IDS="0,1"
    fi
else
    DEVICE="cpu"
    echo "CUDA not detected, using CPU"
fi

# 显示参数
echo "=== Web可视化标注工具启动 ==="
echo "SAM Checkpoint: $SAM_CHECKPOINT"
echo "Image Directory: $IMAGE_DIR"
echo "Device: $DEVICE"
echo "GPU IDs: $GPU_IDS (使用GPU 2和3)"
echo ""

# 获取本机IP地址
HOST_IP=$(hostname -I | awk '{print $1}')
PORT=5000

echo "Web标注工具将在以下地址启动:"
echo "本地访问: http://localhost:$PORT"
echo "远程访问: http://$HOST_IP:$PORT"
echo ""
echo "请在浏览器中访问上述地址之一"
echo ""

# 启动Web标注工具
echo "Starting Web Annotation Tool..."
export CUDA_VISIBLE_DEVICES=$GPU_IDS
python web_annotation.py \
    --sam_checkpoint "$SAM_CHECKPOINT" \
    --device "$DEVICE" \
    --image_dir "$IMAGE_DIR" \
    --host "0.0.0.0" \
    --port "$PORT" 