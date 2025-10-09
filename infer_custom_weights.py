#!/usr/bin/env python3
"""
使用自定义训练的GLOVER++权重进行推理
支持加载转换后的PyTorch格式权重文件或合并后的模型目录
"""

# 设置环境变量解决bitsandbytes权限问题
import os
import tempfile

# 创建临时目录作为HOME
temp_home = tempfile.mkdtemp()
os.environ['HOME'] = temp_home
os.environ['USERPROFILE'] = temp_home  # Windows兼容
os.environ['XDG_CONFIG_HOME'] = os.path.join(temp_home, '.config')
os.environ['XDG_CACHE_HOME'] = os.path.join(temp_home, '.cache')

# 设置bitsandbytes相关环境变量
os.environ['BITSANDBYTES_FUNCTIONAL'] = '1'
os.environ['BITSANDBYTES_CUDA_SETUP'] = '0'
os.environ['BITSANDBYTES_NO_CUDA'] = '1'  # 禁用CUDA相关检查

# 设置其他可能需要的环境变量
os.environ['NVM_DIR'] = os.path.join(temp_home, '.nvm')
os.environ['NODE_PATH'] = os.path.join(temp_home, '.node')

# 创建必要的目录
os.makedirs(os.environ['XDG_CONFIG_HOME'], exist_ok=True)
os.makedirs(os.environ['XDG_CACHE_HOME'], exist_ok=True)
os.makedirs(os.environ['NVM_DIR'], exist_ok=True)

import argparse
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from utils.visualizer import draw_affordance, draw_affordance_center
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="GLOVER++ Custom Weights Inference")
    
    # 模型相关参数
    parser.add_argument("--base_model", default="/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus", 
                       help="基础模型路径")
    parser.add_argument("--custom_weights", default=None,
                       help="自定义训练权重路径 (.bin文件)")
    parser.add_argument("--merged_model", default="/mnt/data-oss/rap-prod-bak/GLOVER/output/finetune/agi_sim/merged_model",
                       help="合并后的模型目录路径")
    parser.add_argument("--use_merged_model", action="store_true", default=True,
                       help="使用合并后的模型而不是原始权重文件")
    parser.add_argument("--vision_tower", default="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2",
                       help="视觉编码器路径")
    
    # 推理参数
    parser.add_argument("--precision", default="bf16", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--max_new_tokens", default=32, type=int)
    
    # 输入输出参数
    parser.add_argument("--image_path", default="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/images/%s.png", 
                       help="输入图像路径模板，使用%s作为对象名称占位符")
    parser.add_argument("--objects", default="bread,ham", help="对象名称列表，用逗号分隔")
    parser.add_argument("--actions", default="grab,grab", help="动作列表，用逗号分隔")
    parser.add_argument("--prompt", default="Where should I interact with the %s to %s it? Please output segmentation mask.", 
                       help="推理提示词模板，使用%s作为对象和动作占位符")
    parser.add_argument("--output_dir", default="./inference_results", help="输出目录")
    
    # 其他参数
    parser.add_argument("--use_text_emb_in_suffix_sam", action="store_true", default=True)
    parser.add_argument("--device", default="cuda:0", help="推理设备")
    
    return parser.parse_args()

def preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
               pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
               img_size=1024) -> torch.Tensor:
    """图像预处理"""
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def load_custom_model(args):
    """加载自定义训练的模型"""
    logger.info("开始加载模型...")
    
    # 设置数据类型
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    
    if args.use_merged_model and os.path.exists(args.merged_model):
        # 使用合并后的模型
        logger.info(f"加载合并后的模型: {args.merged_model}")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.merged_model,
            cache_dir=None,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        
        # 加载合并后的模型
        from model.GLOVER_plus import GloverForCausalLM
        model = GloverForCausalLM.from_pretrained(
            args.merged_model,
            low_cpu_mem_usage=True,
            vision_tower=args.vision_tower,
            seg_token_idx=seg_token_idx,
            torch_dtype=torch_dtype
        )
        
        logger.info("成功加载合并后的模型")
        
    else:
        # 使用原始方式：基础模型+权重文件
        logger.info(f"加载基础模型: {args.base_model}")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            cache_dir=None,
            model_max_length=args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
        seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        
        # 加载基础模型
        from model.GLOVER_plus import GloverForCausalLM
        model = GloverForCausalLM.from_pretrained(
            args.base_model,
            low_cpu_mem_usage=True,
            vision_tower=args.vision_tower,
            seg_token_idx=seg_token_idx,
            torch_dtype=torch_dtype
        )
        
        # 加载自定义权重
        if args.custom_weights and os.path.exists(args.custom_weights):
            logger.info(f"加载自定义权重: {args.custom_weights}")
            custom_state_dict = torch.load(args.custom_weights, map_location='cpu')
            
            # 过滤掉不兼容的权重（如优化器状态等）
            model_state_dict = {}
            for key, value in custom_state_dict.items():
                if key in model.state_dict():
                    model_state_dict[key] = value
            
            # 加载权重
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            logger.info(f"缺失的键: {len(missing_keys)}")
            logger.info(f"意外的键: {len(unexpected_keys)}")
        else:
            logger.info("使用基础模型权重（未加载自定义权重）")
    
    # 设置模型配置
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 初始化视觉模块
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)
    
    # 移动到设备
    if args.precision == "bf16":
        model = model.bfloat16().to(args.device)
    elif args.precision == "fp16":
        model = model.half().to(args.device)
    else:
        model = model.float().to(args.device)
    
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.device)
    
    model.eval()
    logger.info("模型加载完成")
    
    return model, tokenizer, seg_token_idx

def inference_single_image(args, model, tokenizer, seg_token_idx, obj, action):
    """对单张图像进行推理"""
    # 构建图像路径和提示词
    if "%s" in args.image_path:
        # 如果图像路径包含占位符，则进行格式化
        image_path = args.image_path % obj
    else:
        # 否则直接使用提供的图像路径
        image_path = args.image_path
    
    prompt = args.prompt % (obj, action)
    
    logger.info(f"处理对象: {obj}, 动作: {action}")
    logger.info(f"图像路径: {image_path}")
    logger.info(f"提示词: {prompt}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查图像是否存在
    if not os.path.exists(image_path):
        logger.warning(f"图像文件不存在: {image_path}")
        return
    
    # 加载和预处理图像
    image_np = cv2.imread(image_path)
    if image_np is None:
        logger.warning(f"无法加载图像: {image_path}")
        return
    
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size = image_np.shape[:2]
    
    # CLIP图像处理
    clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)
    image_clip = clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(args.device)
    
    if args.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif args.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()
    
    # SAM图像处理
    transform = ResizeLongestSide(args.image_size)
    image = transform.apply_image(image_np)
    resize_size = image.shape[:2]
    
    image = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).to(args.device)
    if args.precision == "bf16":
        image = image.bfloat16()
    elif args.precision == "fp16":
        image = image.half()
    else:
        image = image.float()
    
    # 构建提示词
    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if args.use_text_emb_in_suffix_sam:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
    
    # 构建对话
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    conv.messages = []
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()
    
    # Tokenize
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(args.device)
    
    # 推理
    logger.info("开始推理...")
    with torch.no_grad():
        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            [resize_size],
            [original_size],
            max_new_tokens=args.max_new_tokens,
            tokenizer=tokenizer,
            use_text_emb_in_suffix_sam=args.use_text_emb_in_suffix_sam
        )
    
    # 处理输出
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
    text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", "").replace("  ", " ")
    logger.info(f"生成的文本: {text_output}")
    
    # 保存结果
    base_name = f"{obj}_{action}"
    
    # 保存文本结果
    text_output_path = os.path.join(args.output_dir, f"{base_name}_output.txt")
    with open(text_output_path, 'w', encoding='utf-8') as f:
        f.write(f"Object: {obj}\n")
        f.write(f"Action: {action}\n")
        f.write(f"Prompt: {args.prompt % (obj, action)}\n")
        f.write(f"Generated Text: {text_output}\n")
    
    # 处理分割掩码
    if pred_masks and len(pred_masks) > 0:
        for i, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue
            
            pred_mask = pred_mask[0].sigmoid()
            pred_mask = pred_mask.detach().cpu().numpy()
            
            # 保存原始图像
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_original.png"), image_bgr)
            
            # 保存affordance可视化
            aff_vis, _ = draw_affordance(image_bgr, pred_mask)
            aff_path = os.path.join(args.output_dir, f"{base_name}_affordance_{i}.png")
            cv2.imwrite(aff_path, aff_vis)
            logger.info(f"保存affordance结果: {aff_path}")
            
            # 保存中心点可视化
            aff_vis_center = draw_affordance_center(image_bgr, pred_mask)
            center_path = os.path.join(args.output_dir, f"{base_name}_center_{i}.png")
            cv2.imwrite(center_path, aff_vis_center)
            logger.info(f"保存中心点结果: {center_path}")
            
            # 保存掩码
            mask_path = os.path.join(args.output_dir, f"{base_name}_mask_{i}.npy")
            np.save(mask_path, pred_mask)
            logger.info(f"保存掩码: {mask_path}")
    
    logger.info(f"对象 {obj} 的推理完成")

def main():
    args = parse_args()
    
    # 检查文件存在性
    if args.use_merged_model:
        if not os.path.exists(args.merged_model):
            raise ValueError(f"合并后的模型不存在: {args.merged_model}")
    else:
        if not os.path.exists(args.base_model):
            raise ValueError(f"基础模型不存在: {args.base_model}")
        if not args.custom_weights:
            raise ValueError("当 --use_merged_model 为 False 时，必须提供 --custom_weights 参数")
        if not os.path.exists(args.custom_weights):
            raise ValueError(f"自定义权重不存在: {args.custom_weights}")
    
    # 加载模型
    model, tokenizer, seg_token_idx = load_custom_model(args)
    
    try:
        # 解析对象和动作列表
        objects = [obj.strip() for obj in args.objects.split(",")]
        actions = [action.strip() for action in args.actions.split(",")]
        
        if len(objects) != len(actions):
            raise ValueError("对象和动作列表长度不匹配")
        
        logger.info(f"将处理 {len(objects)} 个对象: {objects}")
        logger.info(f"对应的动作: {actions}")
        
        # 对每个对象进行推理
        for obj, action in zip(objects, actions):
            try:
                inference_single_image(args, model, tokenizer, seg_token_idx, obj, action)
            except Exception as e:
                logger.error(f"处理对象 {obj} 时出错: {e}")
                continue
        
        logger.info(f"所有推理完成，结果保存在: {args.output_dir}")
        
    finally:
        # 清理内存
        del model
        torch.cuda.empty_cache()
        logger.info("内存清理完成")

if __name__ == "__main__":
    main() 