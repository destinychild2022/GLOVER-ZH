# Activate virtual environment
source /mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/.venv/bin/activate
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

python infer.py \
    --version="/mnt/data-oss/rap-prod-bak/GLOVER/runs/glover/trained_model" \
    --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
    --model_arch="glover" \
    --precision="bf16" \
    --local-rank=0 \
    --vis_save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/infer/vis_output" \
    --vis_argmax_save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/infer/vis_argmax_output" \
    --image_path="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/images/%s.png" \
    --objects="cup" \
    --actions="grab" \
    --prompt="Where should I interact with the %s to %s it? Please output segmentation mask." \
    --load_in_8bit \
    --image_size=512 \
    --model_max_length=256