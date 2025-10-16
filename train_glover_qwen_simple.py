#!/usr/bin/env python3
"""
简化的GLOVER-Qwen训练脚本
参考原始train_glover_plus.py的方式
"""

import os
import sys
import logging
from datetime import datetime

# 添加当前目录到Python路径，确保能找到segment_anything模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import argparse
import time
import json
import torch
# from analyze_labels_distribution import analyze_labels_distribution, analyze_batch_labels_distribution
import torch.nn as nn
import torch.distributed
import deepspeed
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model.GLOVER_qwen import GloverQwenForCausalLM
from peft import LoraConfig, get_peft_model
from utils.dataset import HybridDataset, collate_fn
from utils.utils import dict_to_cuda, AverageMeter, ProgressMeter
from model.llava import conversation as conversation_lib
from utils.utils import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# 添加swanlab支持
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    print("Warning: swanlab not available, falling back to tensorboard only")
    SWANLAB_AVAILABLE = False

# 设置环境变量避免bitsandbytes权限问题
os.environ['HOME'] = '/tmp'
os.environ['USERPROFILE'] = '/tmp'
os.environ['XDG_CONFIG_HOME'] = '/tmp'
os.environ['XDG_CACHE_HOME'] = '/tmp'
os.environ['NVM_DIR'] = '/tmp'
os.environ['NODE_PATH'] = '/tmp'
os.environ['BITSANDBYTES_FUNCTIONAL'] = '1'
os.environ['BITSANDBYTES_CUDA_SETUP'] = '0'
os.environ['BITSANDBYTES_NO_CUDA'] = '1'

def parse_args():
    parser = argparse.ArgumentParser(description="GLOVER-Qwen Simple Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--version", type=str, required=True, help="Qwen model path")
    parser.add_argument("--vision-tower", type=str, required=True, help="Vision tower path")
    parser.add_argument("--sam_vit_path", type=str, required=True, help="SAM model path")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset directory")
    parser.add_argument("--dataset", type=str, default="3doi", help="Dataset name")
    parser.add_argument("--sample_rates", type=str, default="1", help="Sample rates")
    parser.add_argument("--exp_name", type=str, default="glover_qwen_simple", help="Experiment name")
    parser.add_argument("--log_base_dir", type=str, default="./output", help="Log directory")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--steps_per_epoch", type=int, default=4, help="Steps per epoch")
    parser.add_argument("--grad_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp32", "bf16", "fp16"], help="Precision")
    parser.add_argument("--image_size", type=int, default=None, help="Image size (backward compatibility, defaults to Qwen image_size)")
    parser.add_argument("--qwen_image_size", type=int, default=None, help="Image size for Qwen processor (defaults to model config)")
    parser.add_argument("--sam_image_size", type=int, default=1024, help="Image size for SAM model")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True, help="Use multimodal start/end tokens")
    parser.add_argument("--conv_type", type=str, default="llava_v1", help="Conversation type")
    parser.add_argument("--debug_fast", action="store_true", help="仅跑1个batch加速调试")
    parser.add_argument("--debug_mode", action="store_true", help="调试模式：减少数据集大小，优化加载时间")
    parser.add_argument("--debug_samples", type=int, default=10, help="调试模式下的样本数量")
    parser.add_argument("--vis_save_path", type=str, default="./vis_output", help="Visualization save path")
    parser.add_argument("--model_arch", type=str, default="glover_qwen", help="Model architecture")
    parser.add_argument("--ce_loss_weight", type=float, default=0.01, help="CE loss weight")
    parser.add_argument("--dice_loss_weight", type=float, default=0.5, help="Dice loss weight")
    parser.add_argument("--bce_loss_weight", type=float, default=2.0, help="BCE loss weight")
    parser.add_argument("--use_text_emb_in_suffix_sam", action="store_true", help="Use text embedding in suffix SAM")
    parser.add_argument("--kl_loss_weight", type=float, default=0.0, help="KL loss weight")
    parser.add_argument("--train_firs_mask_decoder", action="store_true", help="Train first mask decoder")
    parser.add_argument("--resume_from_tokenizer", type=str, default=None, help="Resume from saved tokenizer path")
    parser.add_argument("--resume_from_model", type=str, default=None, help="Resume from saved model path")
    # 移除宽松模式参数，现在默认严格模式
    parser.add_argument("--use_diff_lr", action="store_true", help="Use different learning rates")
    parser.add_argument("--lr_ratio", type=float, default=0.1, help="Learning rate ratio")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,lm_head,gate_proj,up_proj,down_proj", help="LoRA target modules")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for saving tokenizer and model states")
    return parser.parse_args()

def setup_logging(args):
    """设置日志记录"""
    import torch.distributed as dist
    
    # 获取当前进程的rank
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # 创建日志目录
    log_dir = os.path.join(args.vis_save_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名（包含时间戳和rank）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}_rank{local_rank}.log")
    
    # 配置日志 - 每个进程使用独立的日志文件
    logging.basicConfig(
        level=logging.INFO,
        format=f'[Rank {local_rank}] %(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) if local_rank == 0 else logging.NullHandler()  # 只有rank 0输出到控制台
        ]
    )
    
    logger = logging.getLogger(__name__)
    if local_rank == 0:  # 只有主进程记录日志文件路径
        logger.info(f"日志文件保存到: {log_file}")
    return logger

def main():
    args = parse_args()
    
    # 设置日志记录
    logger = setup_logging(args)
    
    logger.info("=== 简化的GLOVER-Qwen训练 ===")
    logger.info(f"模型路径: {args.version}")
    logger.info(f"数据集: {args.dataset}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"学习率: {args.lr}")
    
    # 1. 加载tokenizer
    logger.info("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    # 确保pad_token不为None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    
    # ===== 读取 Qwen 视觉塔配置 =====
    def get_qwen_vision_config(model_path):
        """读取Qwen2.5-VL模型的视觉塔配置"""
        from transformers import AutoConfig
        
        try:
            # 尝试读取模型配置
            cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            
            # 检查是否有vision_config字段（Qwen2.5-VL标准格式）
            if hasattr(cfg, "vision_config") and cfg.vision_config is not None:
                patch_size = getattr(cfg.vision_config, 'patch_size', None)
                
                # Qwen2.5-VL使用动态分辨率，尝试从config中读取image_size
                if patch_size is not None:
                    # 尝试从vision_config中读取image_size
                    if hasattr(cfg.vision_config, 'image_size'):
                        image_size = cfg.vision_config.image_size
                    elif hasattr(cfg.vision_config, 'size'):
                        # 有些模型使用size字段
                        size_config = cfg.vision_config.size
                        if isinstance(size_config, dict):
                            image_size = size_config.get('shortest_edge', 448)
                        else:
                            image_size = size_config
                    else:
                        # 如果都没有找到，使用默认值并给出警告
                        image_size = 448
                        print(f"⚠️ 警告：无法从模型config中读取image_size，使用默认值 {image_size}")
                        print(f"   建议：检查模型config是否包含vision_config.image_size或vision_config.size")
                    
                    return image_size, patch_size
            
            # 检查是否直接在config中有这些字段（兼容格式）
            if hasattr(cfg, "image_size") and hasattr(cfg, "patch_size"):
                image_size = getattr(cfg, 'image_size', None)
                patch_size = getattr(cfg, 'patch_size', None)
                if image_size is not None and patch_size is not None:
                    return image_size, patch_size
            
            # 如果都没有找到，抛出错误
            raise ValueError(f"模型 {model_path} 不包含视觉塔配置")
            
        except Exception as e:
            raise ValueError(f"读取模型配置失败: {e}")
    
    try:
        qwen_image_size, qwen_patch_size = get_qwen_vision_config(args.version)
        print(f"✅ Qwen视觉塔配置: image_size={qwen_image_size}, patch_size={qwen_patch_size}")
    except Exception as e:
        print(f"❌ {e}")
        print("请确认加载的是Qwen2.5-VL模型，包含vision_config字段")
        sys.exit(1)

    # SAM 的 image_size 仍然固定 1024
    sam_image_size = 1024

    # 确保参数同步
    args.qwen_image_size = qwen_image_size
    args.sam_image_size = sam_image_size
    args.patch_size = qwen_patch_size
    print(f"[INFO] Qwen image_size={args.qwen_image_size}, SAM image_size={args.sam_image_size}, patch_size={args.patch_size}")
    
    # 1.3 设置参数
    # 对于Qwen processor模式，使用Qwen的image_size
    if not hasattr(args, 'qwen_image_size') or args.qwen_image_size is None:
        args.qwen_image_size = qwen_image_size
        print(f"使用Qwen默认的 image_size: {args.qwen_image_size}")
    else:
        print(f"使用用户指定的Qwen image_size: {args.qwen_image_size}")
    
    # 对于SAM，使用SAM的image_size
    if not hasattr(args, 'sam_image_size') or args.sam_image_size is None:
        args.sam_image_size = sam_image_size
        print(f"使用SAM默认的 image_size: {args.sam_image_size}")
    else:
        print(f"使用用户指定的SAM image_size: {args.sam_image_size}")
    
    # 为了向后兼容，保留原来的image_size参数
    if not hasattr(args, 'image_size') or args.image_size is None:
        args.image_size = args.qwen_image_size  # 默认使用Qwen的image_size
        print(f"向后兼容：image_size = {args.image_size}")
    
    # 检查是否需要从保存的tokenizer恢复
    if args.resume_from_tokenizer:
        try:
            # 加载保存的tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.resume_from_tokenizer,
                trust_remote_code=True
            )
            
            # 加载token信息
            token_info_path = os.path.join(os.path.dirname(args.resume_from_tokenizer), "token_info.json")
            if os.path.exists(token_info_path):
                with open(token_info_path, 'r') as f:
                    token_info = json.load(f)
                seg_token_idx = token_info["seg_token_idx"]
                pass  # 从保存状态恢复
            else:
                # 如果没有token_info.json，重新获取
                seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
                
        except Exception as e:
            print(f"[ERROR] 恢复tokenizer失败: {e}")
            print(f"[ERROR] 无法从 {args.resume_from_tokenizer} 恢复tokenizer，这可能导致训练状态不一致。")
            print(f"[ERROR] 建议检查tokenizer路径是否正确，或者重新开始训练。")
            raise RuntimeError(f"Tokenizer恢复失败: {e}。请检查路径 {args.resume_from_tokenizer} 是否正确，或重新开始训练。")
    
    # 如果没有从保存状态恢复，则正常添加tokens
    if not args.resume_from_tokenizer:
        # 记录添加token前的词汇表大小
        original_vocab_size = len(tokenizer)
        # 按固定顺序添加tokens，确保ID稳定
        tokens_to_add = []
        
        # 1. 先添加[SEG] token
        tokens_to_add.append("[SEG]")
        
        # 2. 如果需要，添加multimodal start/end tokens
        if args.use_mm_start_end:
            tokens_to_add.extend([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        
        # 3. 一次性添加所有tokens
        num_added_tokens = tokenizer.add_tokens(tokens_to_add)
        
        # 4. 获取[SEG] token的最终ID
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        
        # 5. 验证其他tokens的ID
        if args.use_mm_start_end:
            im_start_id = tokenizer(DEFAULT_IM_START_TOKEN, add_special_tokens=False).input_ids[0]
            im_end_id = tokenizer(DEFAULT_IM_END_TOKEN, add_special_tokens=False).input_ids[0]
    else:
        # 从恢复状态设置变量
        original_vocab_size = len(tokenizer)  # 已经包含新tokens
        num_added_tokens = 0  # 不需要再添加
        tokens_to_add = []  # 已经添加过了
    
    # 2. 初始化SwanLab（如果可用）
    swanlab_run = None
    if SWANLAB_AVAILABLE:
        try:
            # 设置SwanLab API key
            os.environ['SWANLAB_API_KEY'] = 'BBd5HKuM6sIhTwyWmgZ6Z'
            
            swanlab_run = swanlab.init(
                experiment_name=args.exp_name,
                config={
                    "version": args.version,
                    "vision_tower": args.vision_tower,
                    "dataset": args.dataset,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "steps_per_epoch": args.steps_per_epoch,
                    "ce_loss_weight": args.ce_loss_weight,
                    "dice_loss_weight": args.dice_loss_weight,
                    "bce_loss_weight": args.bce_loss_weight,
                    "kl_loss_weight": args.kl_loss_weight,
                    "precision": args.precision,
                    "model_arch": args.model_arch,
                }
            )
            print(f"SwanLab experiment initialized successfully")
        except Exception as e:
            print(f"Warning: Failed to initialize SwanLab: {e}")
            swanlab_run = None
    else:
        print("SwanLab not available, skipping initialization")

    # 3. 加载模型
    logger.info("正在加载千问模型...")
    model_args = {
        "train_mask_decoder": True,
        "out_dim": 256,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "kl_loss_weight": args.kl_loss_weight,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": None,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "image_size": args.sam_image_size,  # 传递SAM的image_size给模型
        "precision": args.precision,  # 传递精度信息给模型
    }
    
    print(f"模型参数: {model_args}")
    
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # 模型加载选项
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        **model_args
    }
    
    try:
        print("正在加载预训练模型...")
        model = GloverQwenForCausalLM.from_pretrained(
            args.version, 
            **load_kwargs
        )
        print("✅ 成功加载预训练模型")
    except Exception as e:
        print(f"❌ 加载预训练模型失败: {e}")
        raise e


    # 3. 使用之前读取的Qwen视觉塔配置
    print("🔍 使用已读取的Qwen视觉塔配置...")
    print(f"✅ Qwen视觉塔配置: image_size={qwen_image_size}, patch_size={qwen_patch_size}")
    
    # 4. 加载SAM权重
    print("正在加载SAM权重...")
    suffix_sam_weight = torch.load(args.sam_vit_path, map_location="cpu")
    model.model.visual_model1.load_state_dict(suffix_sam_weight, strict=True)
    print("✅ 成功加载SAM权重")
    
    # 验证模型结构
    print("🔍 验证模型结构...")
    print(f"模型类型: {type(model)}")
    print(f"是否有model_forward方法: {hasattr(model, 'model_forward')}")
    print(f"是否有get_model方法: {hasattr(model, 'get_model')}")
    
    # 不在此处将整模型搬到GPU，交由DeepSpeed接管设备放置，避免OOM

    # 4. 设置模型配置
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # 5. 初始化视觉组件
    print("正在初始化视觉组件...")
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()

    # 对于Qwen2.5-VL，vision_tower可能为None，因为使用内置视觉编码器
    if vision_tower is not None:
        for p in vision_tower.parameters():
            p.requires_grad = False
        print("冻结外部视觉编码器参数")
    else:
        print("使用Qwen2.5-VL内置视觉编码器，无需冻结外部参数")
    
    # mm_projector对视觉-语言对齐很重要，应该训练
    if model.get_model().mm_projector is not None:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        print("启用多模态投影器训练")
    else:
        print("使用Qwen2.5-VL内置投影器，无需额外训练")

    # 6. 设置对话模板
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # 7. 配置真正的LoRA微调
    print("正在配置LoRA微调...")
    
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
                                "visual_model1",
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
        
        # 验证LoRA目标模块
        logger.info(f"LoRA目标模块: {args.lora_target_modules.split(',')}")
        logger.info(f"找到的模块数量: {len(lora_target_modules)}")
        
        # 检查是否有模块未找到
        requested_modules = set(args.lora_target_modules.split(","))
        found_modules = set()
        for module_name in lora_target_modules:
            for req_module in requested_modules:
                if req_module in module_name:
                    found_modules.add(req_module)
                    break
        
        missing_modules = requested_modules - found_modules
        if missing_modules:
            logger.warning(f"以下模块未找到: {missing_modules}")
        else:
            logger.info("所有请求的LoRA模块都已找到")
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        print(f"成功应用LoRA，目标模块: {lora_target_modules}")

    # 8. 设置其他可训练参数
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in [
                    "lora_A",
                    "lora_B",
                    "lm_head",
                    "embed_tokens",
                    "text_hidden_fcs",
                ]
            ]
        ):
            # 修复：当ce_loss_weight > 0时，启用这些参数的梯度
            p.requires_grad = (args.ce_loss_weight > 0.0)
            if args.ce_loss_weight > 0.0:
                print(f"✅ 启用参数训练: {n}, requires_grad={p.requires_grad}")
            else:
                print(f"❌ 禁用参数训练: {n}, requires_grad={p.requires_grad}")

    for n, p in model.named_parameters():
        if "visual_model1." in n:
            if "prompt_encoder" in n:
                p.requires_grad = False
            elif "image_encoder" in n:
                p.requires_grad = False
            elif "mask_decoder" in n:
                p.requires_grad = True
        if "visual_model." in n:
            p.requires_grad = False

    # 9. 调整token embeddings并保存
    # 检查是否需要resize
    if len(tokenizer) > original_vocab_size:
        # 保存resize前的embedding状态
        original_embedding_size = model.get_input_embeddings().weight.shape[0]
        
        # 执行resize
        model.resize_token_embeddings(len(tokenizer))
        
        # 验证resize结果
        new_embedding_size = model.get_input_embeddings().weight.shape[0]
        
        # 保存tokenizer和模型状态，方便resume
        if args.output_dir:
            tokenizer_save_path = os.path.join(args.output_dir, "tokenizer_with_new_tokens")
            model_save_path = os.path.join(args.output_dir, "model_with_resized_embeddings")
            
            tokenizer.save_pretrained(tokenizer_save_path)
            model.save_pretrained(model_save_path, safe_serialization=False)
            
            # 保存token信息
            token_info = {
                "original_vocab_size": original_vocab_size,
                "new_vocab_size": len(tokenizer),
                "seg_token_idx": seg_token_idx,
                "tokens_added": tokens_to_add,
                "num_added_tokens": num_added_tokens
            }
            if args.use_mm_start_end:
                token_info.update({
                    "im_start_token": DEFAULT_IM_START_TOKEN,
                    "im_end_token": DEFAULT_IM_END_TOKEN,
                    "im_start_id": im_start_id,
                    "im_end_id": im_end_id
                })
            
            token_info_path = os.path.join(args.output_dir, "token_info.json")
            with open(token_info_path, 'w') as f:
                json.dump(token_info, f, indent=2)
    
    # 10. 准备数据集
    print("正在准备数据集...")
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    
    # 计算每个epoch的样本数
    samples_per_epoch = args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size
    print(f"📊 每个epoch使用 {samples_per_epoch} 个样本")
    train_dataset = HybridDataset(
        base_image_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=args.version,  # 在Qwen模式下使用Qwen模型路径，让数据集检测到Qwen
        samples_per_epoch=samples_per_epoch,
        precision=args.precision,
        image_size=args.sam_image_size,  # 使用SAM的image_size
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
    )
    print(f"训练数据集大小: {len(train_dataset)}")

    # 11. DeepSpeed配置
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (0.9, 0.95),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.steps_per_epoch * args.epochs // 10,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

    # 12. 初始化DeepSpeed
    print("正在初始化DeepSpeed...")
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
            use_qwen_mode=True,  # 启用Qwen2.5-VL模式
            processor=model.processor,  # 传递processor
            image_size=qwen_image_size,   # !! 使用从模型config读取的Qwen image_size
            patch_size=qwen_patch_size    # !! 使用从模型config读取的patch_size
        ),
        config=ds_config,
    )

    # 13. 开始训练
    logger.info("开始训练...")
    train_iter = iter(train_loader)
    
    # 设置保存目录
    save_dir = os.path.join(args.vis_save_path, "ckpt_model")
    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    
    total_epochs = 1 if args.debug_fast else args.epochs
    for epoch in range(total_epochs):
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            None,  # writer
            train_iter,
            args,
            tokenizer,
            swanlab_run,
        )
        
        # 保存检查点
        if not args.debug_fast:  # 只在非调试模式下保存
            if args.local_rank == 0:
                print(f"保存第 {epoch+1} 轮检查点到: {save_dir}")
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            
            # 生成 zero_to_fp32.py 脚本
            if args.local_rank == 0:
                zero_to_fp32_script = os.path.join(args.vis_save_path, "zero_to_fp32.py")
                with open(zero_to_fp32_script, "w") as f:
                    f.write('''#!/usr/bin/env python3
"""
DeepSpeed ZeRO 检查点转换为 FP32 权重脚本
使用方法: python zero_to_fp32.py <checkpoint_dir> <output_dir>
"""

import os
import sys
import argparse
import torch
import deepspeed

def convert_checkpoint():
    parser = argparse.ArgumentParser(description="Convert DeepSpeed ZeRO checkpoint to FP32")
    parser.add_argument("checkpoint_dir", help="DeepSpeed checkpoint directory")
    parser.add_argument("output_dir", help="Output directory for FP32 weights")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_dir):
        print(f"错误: 检查点目录不存在: {args.checkpoint_dir}")
        sys.exit(1)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"正在转换检查点: {args.checkpoint_dir} -> {args.output_dir}")
    
    # 使用 DeepSpeed 的转换工具
    try:
        deepspeed.checkpoint.zero_to_fp32_weights(
            args.checkpoint_dir, 
            args.output_dir
        )
        print(f"转换完成！FP32 权重保存在: {args.output_dir}")
    except Exception as e:
        print(f"转换失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    convert_checkpoint()
''')
                os.chmod(zero_to_fp32_script, 0o755)  # 设置可执行权限
                print(f"已生成 zero_to_fp32.py 脚本: {zero_to_fp32_script}")
                print(f"使用方法: python {zero_to_fp32_script} {save_dir} <output_dir>")
                
                # 保存训练元数据
                meta_log_path = os.path.join(args.vis_save_path, f"meta_log_epoch_{epoch+1}.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "args": vars(args),
                    "timestamp": time.time()
                }, meta_log_path)
                print(f"已保存训练元数据: {meta_log_path}")
                
                # 保存 LoRA 权重（如果使用 LoRA）
                if hasattr(model, 'peft_config') and model.peft_config:
                    lora_save_dir = os.path.join(args.vis_save_path, "lora_weights")
                    os.makedirs(lora_save_dir, exist_ok=True)
                    model.save_pretrained(lora_save_dir)
                    print(f"已保存 LoRA 权重到: {lora_save_dir}")
                
                # 保存 tokenizer
                tokenizer_save_dir = os.path.join(args.vis_save_path, "tokenizer")
                os.makedirs(tokenizer_save_dir, exist_ok=True)
                tokenizer.save_pretrained(tokenizer_save_dir)
                print(f"已保存 tokenizer 到: {tokenizer_save_dir}")
                
                # 保存训练配置
                config_save_path = os.path.join(args.vis_save_path, "training_config.json")
                with open(config_save_path, 'w') as f:
                    json.dump(vars(args), f, indent=2)
                print(f"已保存训练配置到: {config_save_path}")

def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
    tokenizer,
    swanlab_run=None,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    kl_losses = AverageMeter("KlLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, losses, ce_losses, mask_losses, kl_losses],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    import time
    end = time.time()
    
    total_steps = 1 if args.debug_fast else args.steps_per_epoch
    for global_step in range(total_steps):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            # 精度转换
            if "pixel_values" in input_dict:
                # Qwen processor模式：处理pixel_values
                if args.precision == "fp16":
                    input_dict["pixel_values"] = input_dict["pixel_values"].half()
                elif args.precision == "bf16":
                    input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
                else:
                    input_dict["pixel_values"] = input_dict["pixel_values"].float()
            else:
                # 传统模式：处理images和images_clip
                if args.precision == "fp16":
                    input_dict["images"] = input_dict["images"].half()
                    input_dict["images_clip"] = input_dict["images_clip"].half()
                elif args.precision == "bf16":
                    input_dict["images"] = input_dict["images"].bfloat16()
                    input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                else:
                    input_dict["images"] = input_dict["images"].float()
                    input_dict["images_clip"] = input_dict["images_clip"].float()
            
            # 重命名attention_masks为attention_mask以匹配模型期望
            if "attention_masks" in input_dict:
                input_dict["attention_mask"] = input_dict.pop("attention_masks")
            
            # 调试：打印input_dict中的所有键
            print(f"🔍 input_dict 包含的键: {list(input_dict.keys())}")
            
            # 确保所有必需的参数都存在
            if "pixel_values" in input_dict:
                # Qwen processor模式
                required_params = ['pixel_values', 'image_grid_thw', 'input_ids', 'labels', 'attention_mask', 'offset', 'masks_list', 'resize_list']
            else:
                # 传统模式
                required_params = ['images', 'images_clip', 'input_ids', 'labels', 'attention_mask', 'offset', 'masks_list', 'resize_list']
            missing_params = []
            
            for param in required_params:
                if param not in input_dict:
                    missing_params.append(param)
            
            # 严格模式：缺失关键参数直接抛出异常，不进行任何补全
            if missing_params:
                raise ValueError(f"缺少必需参数: {missing_params}。请检查数据管道配置。\n"
                               f"这可能是由于数据加载器配置错误或模型模式不匹配导致的。\n"
                               f"请检查：\n"
                               f"1. 是否正确设置了use_qwen_mode参数\n"
                               f"2. 数据加载器是否正确配置了processor\n"
                               f"3. 模型是否支持当前的数据模式")
                
            # 使用标准的模型调用方式，确保PEFT正确处理
            # 在DeepSpeed环境中，使用module属性访问底层模型
            base_model = model.module if hasattr(model, 'module') else model
            
            # 添加tokenizer到input_dict中
            input_dict['tokenizer'] = tokenizer
            


            # 使用forward方法进行训练
            # 直接传递关键参数，避免PEFT包装器过滤
            # 根据数据模式选择正确的参数
            if "pixel_values" in input_dict:
                # Qwen processor模式
                output_dict = base_model.forward(
                    input_ids=input_dict['input_ids'],
                    attention_mask=input_dict['attention_mask'],
                    labels=input_dict['labels'],
                    pixel_values=input_dict['pixel_values'],
                    image_grid_thw=input_dict['image_grid_thw'],
                    offset=input_dict['offset'],
                    masks_list=input_dict['masks_list'],
                    resize_list=input_dict['resize_list'],
                    tokenizer=input_dict['tokenizer'],
                    use_text_emb_in_suffix_sam=args.use_text_emb_in_suffix_sam,
                    inference=False,  # 训练时必须是False，确保进行分割训练
                    images=input_dict.get('images', None),  # 传递原始图像数据
                )
            else:
                # 传统模式
                output_dict = base_model.forward(
                    input_ids=input_dict['input_ids'],
                    attention_mask=input_dict['attention_mask'],
                    labels=input_dict['labels'],
                    images=input_dict['images'],
                    images_clip=input_dict['images_clip'],
                    offset=input_dict['offset'],
                    masks_list=input_dict['masks_list'],
                    resize_list=input_dict['resize_list'],
                    tokenizer=input_dict['tokenizer'],
                    use_text_emb_in_suffix_sam=args.use_text_emb_in_suffix_sam,
                    inference=input_dict.get('inference', False),
                )

            # 调试：检查模型输出的键
            print(f"[DEBUG] 模型输出键: {list(output_dict.keys())}")
            
            # 检查是否有[SEG] token相关的调试信息
            if "pred_masks" in output_dict:
                pred_masks = output_dict["pred_masks"]
                if pred_masks and len(pred_masks) > 0:
                    print(f"[DEBUG] 生成了 {len(pred_masks)} 个掩码")
                else:
                    print(f"[WARNING] 没有生成掩码，可能没有[SEG] token")
            
            # 检查labels中是否包含[SEG] token
            if "labels" in input_dict:
                labels = input_dict["labels"]
                seg_token_id = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
                seg_token_count = (labels == seg_token_id).sum().item()
                if seg_token_count > 0:
                    print(f"[DEBUG] 批次中包含 {seg_token_count} 个[SEG] token (ID: {seg_token_id})")
                else:
                    print(f"[WARNING] 批次中没有[SEG] token，这可能导致模型无法学会生成[SEG] token")
            
            # 检查input_ids中是否包含[SEG] token
            if "input_ids" in input_dict:
                input_ids = input_dict["input_ids"]
                seg_token_id = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
                seg_token_count = (input_ids == seg_token_id).sum().item()
                if seg_token_count > 0:
                    print(f"[DEBUG] 输入中包含 {seg_token_count} 个[SEG] token (ID: {seg_token_id})")
                else:
                    print(f"[DEBUG] 输入中没有[SEG] token（这是正常的，因为[SEG] token应该在labels中）")
            
            loss = output_dict["loss"]
            
            # 安全地获取其他损失字段，如果不存在则使用默认值
            ce_loss = output_dict.get("ce_loss", loss)  # 如果没有ce_loss，使用总损失
            mask_loss = output_dict.get("mask_loss", torch.tensor(0.0, device=loss.device))
            kl_loss = output_dict.get("kl_loss", torch.tensor(0.0, device=loss.device))

            # 获取批次大小
            if "pixel_values" in input_dict:
                batch_size = input_dict["pixel_values"].size(0)
            else:
                batch_size = input_dict["images"].size(0)
            
            losses.update(loss.item(), batch_size)
            ce_losses.update(ce_loss.item(), batch_size)
            mask_losses.update(mask_loss.item(), batch_size)
            kl_losses.update(kl_loss.item(), batch_size)
            
            # SwanLab记录（如果可用）
            if swanlab_run is not None:
                try:
                    swanlab.log({
                        "train/loss": loss.item(),
                        "train/ce_loss": ce_loss.item(),
                        "train/mask_loss": mask_loss.item(),
                        "train/kl_loss": kl_loss.item(),
                        "train/learning_rate": scheduler.get_last_lr()[0] if scheduler else 0,
                        "train/epoch": epoch,
                        "train/step": global_step,
                    })
                except Exception as e:
                    print(f"Warning: Failed to log to SwanLab: {e}")
            
            # 记录到本地日志文件
            if global_step % 10 == 0:  # 每10步记录一次
                logging.info(f"Epoch {epoch}, Step {global_step}: "
                           f"Loss={loss.item():.4f}, "
                           f"CE={ce_loss.item():.4f}, "
                           f"Mask={mask_loss.item():.4f}, "
                           f"KL={kl_loss.item():.4f}")
            
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % 1 == 0:  # 每步都打印
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()
                losses.all_reduce()
                ce_losses.all_reduce()
                mask_losses.all_reduce()
                kl_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_losses.reset()
            kl_losses.reset()

    return train_iter

if __name__ == "__main__":
    main()


