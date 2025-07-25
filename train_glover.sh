#!/bin/bash

# Activate virtual environment
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate

# 设置CUDA内存优化环境变量
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 如果不是在 tmux 会话中，则自动新建一个叫 glover 的 tmux 会话并在里面运行本脚本
# if [ -z "$TMUX" ]; then
#     # 创建日志文件名，包含时间戳
#     log_file="training_log_$(date +%Y%m%d_%H%M%S).txt"
#     tmux new-session -d -s glover "bash $0 2>&1 | tee $log_file"
#     echo "已在 tmux 会话 glover 中启动训练。你可以用命令：tmux attach -t glover 查看进度。"
#     echo "训练日志已保存到: $log_file"
#     exit 0
# fi

# 清理环境变量中的 /root/.nvm 路径
export PATH=$(echo $PATH | tr ':' '\n' | grep -v '/root/.nvm' | paste -sd: -)
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v '/root/.nvm' | paste -sd: -)

# 设置环境变量以避免权限问题
export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 确保不会访问 /root/.nvm
unset NVM_DIR
unset NVM_BIN
unset NVM_PATH

export CUDA_VISIBLE_DEVICES=0,1

deepspeed --num_gpus=2 --master_port=20983 train_glover.py \
  --version="/mnt/data-oss/rap-prod-bak/GLOVER/model/LISA_Plus_7b" \
  --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
  --dataset_dir='/mnt/data-oss/rap-prod-bak/GLOVER/dataset/HOVA-500K' \
  --dataset="3doi"  \
  --log_base_dir="/mnt/data-oss/rap-prod-bak/GLOVER/runs" \
  --sample_rates="1" \
  --exp_name="glover" \
  --lr=0.0005 \
  --epochs=3 \
  --batch_size=32 \
  --steps_per_epoch=60 \
  --ce_loss_weight=0.01
  #||ego4d||epic100||handal" \
  #   --dataset="3doi"  \
  # --sample_rates="1,1,1,1" \