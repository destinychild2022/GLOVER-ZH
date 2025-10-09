#!/usr/bin/env python3
"""
GLOVER-Qwen 权重合并脚本
将转换后的权重重新组织成标准的 Hugging Face 模型格式
"""

import os
import sys
import argparse
import torch
import transformers

from transformers import AutoTokenizer, AutoModelForCausalLM

from model.GLOVER_qwen import GloverQwenForCausalLM
from utils.utils import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def parse_args():
    parser = argparse.ArgumentParser(description="GLOVER-Qwen 权重合并")
    parser.add_argument("--version", type=str, required=True, help="Qwen 预训练模型路径")
    parser.add_argument("--weight_dir", type=str, required=True, help="转换后的权重目录")
    parser.add_argument("--save_path", type=str, required=True, help="输出模型路径")
    parser.add_argument("--vision-tower", type=str, required=True, help="Vision tower 路径")
    parser.add_argument("--sam_vit_path", type=str, required=True, help="SAM 模型路径")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "bf16", "fp16"])
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=== GLOVER-Qwen 权重合并 ===")
    print(f"预训练模型: {args.version}")
    print(f"权重目录: {args.weight_dir}")
    print(f"输出路径: {args.save_path}")
    
    # 1. 加载 tokenizer
    print("正在加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 添加特殊 token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    
    # 2. 加载预训练模型
    print("正在加载预训练模型...")
    model_args = {
        "train_mask_decoder": True,
        "out_dim": 256,
        "ce_loss_weight": 0.0,
        "dice_loss_weight": 0.5,
        "bce_loss_weight": 2.0,
        "kl_loss_weight": 0.1,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": None,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": True,
    }
    
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    model = GloverQwenForCausalLM.from_pretrained(
        args.version,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        **model_args
    )
    
    # 3. 加载 SAM 权重
    print("正在加载 SAM 权重...")
    suffix_sam_weight = torch.load(args.sam_vit_path, map_location="cpu")
    model.get_model().visual_model1.load_state_dict(suffix_sam_weight, strict=True)
    
    # 4. 设置模型配置
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 5. 加载训练后的权重
    print("正在加载训练后的权重...")
    
    # 检查权重文件
    weight_index_file = os.path.join(args.weight_dir, "pytorch_model.bin.index.json")
    if not os.path.exists(weight_index_file):
        raise FileNotFoundError(f"权重索引文件不存在: {weight_index_file}")
    
    # 加载权重
    from transformers import load_sharded_checkpoint
    load_sharded_checkpoint(model, args.weight_dir, strict=True)
    
    print("权重加载完成！")
    
    # 6. 保存完整模型
    print(f"正在保存完整模型到: {args.save_path}")
    os.makedirs(args.save_path, exist_ok=True)
    
    # 保存模型权重
    model.save_pretrained(args.save_path, safe_serialization=False)
    
    # 保存 tokenizer
    tokenizer.save_pretrained(args.save_path)
    
    print("=== 模型保存完成 ===")
    print(f"完整模型保存在: {args.save_path}")
    
    # 列出生成的文件
    print("\n生成的文件:")
    for file in os.listdir(args.save_path):
        file_path = os.path.join(args.save_path, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file} ({size:,} bytes)")
        else:
            print(f"  {file}/ (目录)")

if __name__ == "__main__":
    main()
