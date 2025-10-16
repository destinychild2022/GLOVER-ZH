#!/usr/bin/env python3
"""
合并Qwen LoRA权重的脚本
"""

import argparse
import os
import sys
import torch
import transformers
from transformers import AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from model.GLOVER_qwen import GloverQwenForCausalLM

def parse_args():
    parser = argparse.ArgumentParser(description="Merge Qwen LoRA weights")
    parser.add_argument("--version", type=str, required=True, help="Base model path")
    parser.add_argument("--weight", type=str, required=True, help="Fine-tuned model weight path")
    parser.add_argument("--save_path", type=str, required=True, help="Save path for merged model")
    parser.add_argument("--model_arch", type=str, default="glover_qwen", help="Model architecture")
    parser.add_argument("--vision-tower", type=str, required=True, help="Vision tower path")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,lm_head,gate_proj,up_proj,down_proj", help="LoRA target modules")
    parser.add_argument("--model_max_length", type=int, default=512, help="Model max length")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True, help="Use mm start end tokens")
    parser.add_argument("--train_mask_decoder", action="store_true", default=True, help="Train mask decoder")
    parser.add_argument("--out_dim", type=int, default=256, help="Output dimension")
    parser.add_argument("--conv_type", type=str, default="llava_v1", choices=["llava_v1", "llava_llama_2"], help="Conversation type")
    parser.add_argument("--sam_vit_path", type=str, required=True, help="SAM model path")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 50)
    print("开始合并Qwen LoRA权重...")
    print("=" * 50)
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    # 1. 创建tokenizer
    print("1. 创建tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    
    # 2. 设置模型参数（与训练脚本保持一致）
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": 0.1,  # 与训练脚本保持一致
        "dice_loss_weight": 0.5,
        "bce_loss_weight": 2.0,
        "kl_loss_weight": 0.5,  # 与训练脚本保持一致
        "seg_token_idx": args.seg_token_idx,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "image_size": 1024,  # SAM的image_size
        "precision": args.precision,
        "vision_pretrained": None,  # 与训练脚本保持一致，稍后单独加载SAM权重
    }
    
    # 设置精度
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    # 3. 创建模型
    print("2. 创建GLOVER-Qwen模型...")
    model = GloverQwenForCausalLM.from_pretrained(
        args.version, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True,
        **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 设置seg_token_idx（与训练脚本保持一致）
    model.seg_token_idx = args.seg_token_idx
    print(f"✅ 设置seg_token_idx: {args.seg_token_idx}")
    
    # 3.1. 初始化processor（与训练脚本保持一致）
    print("2.1. 初始化processor...")
    try:
        model.processor = AutoProcessor.from_pretrained(args.version, trust_remote_code=True)
        print("✅ 成功初始化processor")
    except Exception as e:
        print(f"⚠️ 初始化processor失败: {e}")
        print("继续执行，但可能影响推理功能...")
    
    # 4. 加载SAM权重（与训练脚本保持一致）
    print("3. 加载SAM权重...")
    try:
        suffix_sam_weight = torch.load(args.sam_vit_path, map_location="cpu")
        # 检查SAM权重是否包含完整的模型结构
        if 'model' in suffix_sam_weight:
            # 如果权重文件包含'model'键，提取实际权重
            sam_state_dict = suffix_sam_weight['model']
        else:
            # 直接使用权重字典
            sam_state_dict = suffix_sam_weight
        
        # 加载到visual_model1（SAM模型）
        missing_keys, unexpected_keys = model.model.visual_model1.load_state_dict(sam_state_dict, strict=False)
        print("✅ 成功加载SAM权重")
        if missing_keys:
            print(f"SAM权重缺失的键: {len(missing_keys)} 个")
        if unexpected_keys:
            print(f"SAM权重意外的键: {len(unexpected_keys)} 个")
    except Exception as e:
        print(f"❌ 加载SAM权重失败: {e}")
        print("⚠️ 继续执行，但SAM功能可能不可用")
    
    # 5. 初始化GLOVER组件
    print("4. 初始化GLOVER组件...")
    try:
        # 初始化视觉模块
        model.get_model().initialize_vision_modules(model.get_model().config)
        # Qwen2.5-VL使用内置视觉编码器，不需要外部vision_tower
        vision_tower = model.get_model().get_vision_tower()
        if vision_tower is not None:
            vision_tower.to(dtype=torch_dtype)
            print("设置外部vision_tower精度")
        else:
            print("使用Qwen2.5-VL内置视觉编码器，跳过外部vision_tower设置")
        
        # 初始化GLOVER模块（包括SAM和text_hidden_fcs）
        model.get_model().initialize_glover_modules(model.get_model().config)
        print("✅ 成功初始化GLOVER组件")
    except Exception as e:
        print(f"⚠️ 初始化GLOVER组件时出现警告: {e}")
        print("继续执行...")
    
    # 6. 设置LoRA配置（与训练脚本保持一致）
    print("5. 设置LoRA配置...")
    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "visual_model1",  # 与训练脚本保持一致
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))
        
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 7. 调整token embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # 8. 加载训练后的权重（参考原始merge_weights.py的方式）
    print("6. 加载训练后的权重...")
    
    # 检查是否是分片文件目录
    if os.path.isdir(args.weight):
        print(f"检测到权重目录: {args.weight}")
        
        # 检查是否存在index.json文件（标准HF分片权重格式）
        index_file = os.path.join(args.weight, "pytorch_model.bin.index.json")
        if os.path.exists(index_file):
            print("✅ 找到index.json文件，使用标准方式加载分片权重")
            
            # 读取index.json文件
            import json
            with open(index_file, "r") as f:
                index_data = json.load(f)
            
            # 获取所有唯一的分片文件名（去重）
            unique_bin_files = list(set(index_data["weight_map"].values()))
            bin_files = [os.path.join(args.weight, f) for f in unique_bin_files]
            print(f"将加载 {len(bin_files)} 个唯一分片文件: {[os.path.basename(f) for f in bin_files]}")
            
            # 合并加载所有分片
            trained_state_dict = {}
            for bf in bin_files:
                print(f"加载分片: {os.path.basename(bf)}")
                sd_part = torch.load(bf, map_location="cpu")
                trained_state_dict.update(sd_part)
        else:
            raise ValueError(f"在目录 {args.weight} 中没有找到 pytorch_model.bin.index.json 文件！请确保这是一个有效的分片权重目录。")
    else:
        # 单个文件，直接加载（完全按照原始merge_weights.py的方式）
        print(f"加载单个权重文件: {args.weight}")
        trained_state_dict = torch.load(args.weight, map_location="cpu")
    
    # 加载权重到模型（使用strict=False，因为SAM权重是单独加载的）
    missing_keys, unexpected_keys = model.load_state_dict(trained_state_dict, strict=False)
    print("✅ 成功加载训练后的权重")
    if missing_keys:
        print(f"缺失的键（预期，因为SAM权重单独加载）: {len(missing_keys)} 个")
    if unexpected_keys:
        print(f"意外的键: {len(unexpected_keys)} 个")
    
    # 尝试加载LoRA权重
    if os.path.isdir(args.weight):
        lora_weights_path = os.path.join(args.weight, "lora_weights")
        if os.path.exists(lora_weights_path):
            print("6.1. 加载LoRA权重...")
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, lora_weights_path)
                print("✅ 成功加载LoRA权重")
            except Exception as e:
                print(f"⚠️ 加载LoRA权重失败: {e}")
                print("继续使用基础模型权重...")
    
    # 9. 合并LoRA权重
    print("7. 合并LoRA权重...")
    model = model.merge_and_unload()
    
    # 10. 保存模型（参考原始merge_weights.py的方式）
    print("8. 保存合并后的模型...")
    state_dict = {}
    for k, v in model.state_dict().items():
        if "vision_tower" not in k:
            state_dict[k] = v
    model.save_pretrained(args.save_path, state_dict=state_dict, safe_serialization=False)
    tokenizer.save_pretrained(args.save_path)
    
    print("=" * 50)
    print("权重合并完成！")
    print(f"合并后的模型保存在: {args.save_path}")
    print("=" * 50)

if __name__ == "__main__":
    main()

