CUDA_VISIBLE_DEVICES=0 python eval.py \
    --dataset_dir='/mnt/data-oss/rap-prod-bak/GLOVER/dataset/HOVA-500K' \
    --version="/mnt/data-oss/rap-prod-bak/GLOVER/output/runs/glover/ckpt_model/global_step392" \
    --vision-tower="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2" \
    --model_arch="glover++" \
    --use_text_emb_in_suffix_sam \
    --vis_save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/eval/vis_output" \
    --vis_argmax_save_path="/mnt/data-oss/rap-prod-bak/GLOVER/output/eval/vis_argmax_output"
