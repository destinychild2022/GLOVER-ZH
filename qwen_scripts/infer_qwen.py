#!/usr/bin/env python3
"""
GLOVER-Qwen推理脚本
基于训练脚本和原始推理脚本修改
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from model.GLOVER_qwen import GloverQwenForCausalLM
from model.llava import conversation as conversation_lib
from utils.utils import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN
from utils.visualizer import draw_affordance, draw_affordance_center


def parse_args():
    parser = argparse.ArgumentParser(description="GLOVER-Qwen推理")
    parser.add_argument("--version", type=str, required=True, help="合并后的模型路径")
    parser.add_argument("--sam_vit_path", type=str, required=True, help="SAM模型路径")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str, help="可视化保存路径")
    parser.add_argument("--vis_argmax_save_path", default="./vis_argmax_output", type=str, help="argmax可视化保存路径")
    parser.add_argument("--image_path", default="", type=str, help="图像路径模板")
    parser.add_argument("--prompt", default="", type=str, help="提示词模板")
    parser.add_argument("--objects", default="", type=str, help="对象列表，逗号分隔")
    parser.add_argument("--actions", default="", type=str, help="动作列表，逗号分隔")
    
    parser.add_argument("--use_text_emb_in_suffix_sam", action="store_true", default=False, help="在SAM中使用文本嵌入")
    parser.add_argument("--precision", default="fp32", type=str, choices=["fp32", "bf16", "fp16"], help="推理精度")
    parser.add_argument("--image_size", default=1024, type=int, help="图像尺寸")
    parser.add_argument("--model_max_length", default=512, type=int, help="模型最大长度")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True, help="使用多模态开始/结束token")
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"], help="对话类型")
    parser.add_argument("--local_rank", default=0, type=int, help="本地rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="8位量化加载")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="4位量化加载")
    
    return parser.parse_args()


def preprocess_image_for_sam(image_np, img_size=1024):
    """为SAM预处理图像"""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    
    # 调整图像尺寸
    h, w = image_np.shape[:2]
    if h > w:
        new_h, new_w = img_size, int(w * img_size / h)
    else:
        new_h, new_w = int(h * img_size / w), img_size
    
    image_resized = cv2.resize(image_np, (new_w, new_h))
    
    # 转换为tensor并归一化
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).contiguous().float()
    image_tensor = (image_tensor - pixel_mean) / pixel_std
    
    # 填充到正方形
    h, w = image_tensor.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    image_tensor = F.pad(image_tensor, (0, padw, 0, padh))
    
    return image_tensor, (new_h, new_w)


def infer(args):
    print("=" * 60)
    print("GLOVER-Qwen推理开始")
    print(f"模型路径: {args.version}")
    print(f"SAM路径: {args.sam_vit_path}")
    print("=" * 60)

    # 创建保存目录
    os.makedirs(args.vis_save_path, exist_ok=True)
    os.makedirs(args.vis_argmax_save_path, exist_ok=True)
    
    # 解析对象和动作
    objs = args.objects.split(",") if args.objects else []
    actions = args.actions.split(",") if args.actions else []
    
    if len(objs) != len(actions):
        print(f"错误：对象数量({len(objs)})与动作数量({len(actions)})不匹配")
        return

    # 1. 加载tokenizer
    print("1. 加载tokenizer...")
    # 使用合并后模型中的tokenizer
    tokenizer_path = os.path.join(args.version, "tokenizer") if os.path.exists(os.path.join(args.version, "tokenizer")) else args.version
    
    print(f"使用tokenizer路径: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
    )
    print("✅ 成功加载tokenizer")
    
    # 确保pad_token不为None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 获取[SEG] token的ID（tokenizer应该已经包含了[SEG] token）
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    print(f"[SEG] token ID: {seg_token_idx}")
    
    # 验证[SEG] token是否正确添加
    if seg_token_idx is None or len(tokenizer("[SEG]", add_special_tokens=False).input_ids) == 0:
        print("[ERROR] [SEG] token not found in tokenizer!")
        print(f"[ERROR] tokenizer vocab size: {len(tokenizer)}")
        print(f"[ERROR] [SEG] token encoding result: {tokenizer('[SEG]', add_special_tokens=False)}")
        raise ValueError("SEG token not properly added to tokenizer")
    else:
        print(f"[INFO] [SEG] token successfully found with ID: {seg_token_idx}")

    # 2. 设置精度
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # 3. 加载模型
    print("2. 加载GLOVER-Qwen模型...")
    model_args = {
        "train_mask_decoder": True,
        "out_dim": 256,
        "ce_loss_weight": 0.1,  # 与训练脚本保持一致
        "dice_loss_weight": 0.5,
        "bce_loss_weight": 2.0,
        "kl_loss_weight": 0.5,
        "seg_token_idx": seg_token_idx,
        "vision_tower": "/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL",  # 使用原始Qwen模型路径
        "use_mm_start_end": args.use_mm_start_end,
        "image_size": args.image_size,
        "precision": args.precision,
        "vision_pretrained": None,  # 与训练脚本保持一致，稍后单独加载SAM权重
    }
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    load_kwargs = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "device_map": "auto" if device == "cuda" else None,  # 自动设备映射
        **model_args
    }
    
    # 添加量化配置
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model1"],
            ),
        })
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model1"],
                load_in_8bit=True,
            ),
        })

    # 直接加载合并后的模型
    print("正在加载合并后的模型...")
    model = GloverQwenForCausalLM.from_pretrained(args.version, **load_kwargs)
    print("✅ 成功加载合并后的模型")
    
    # 设置processor - 这是关键步骤！
    print("正在设置processor...")
    try:
        # 尝试从模型路径加载processor
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.version, trust_remote_code=True)
        model.processor = processor
        print("✅ 成功设置processor")
    except Exception as e:
        print(f"⚠️ 无法从模型路径加载processor: {e}")
        print("尝试从原始Qwen模型路径加载processor...")
        try:
            # 从原始Qwen模型路径加载processor
            original_qwen_path = "/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL"
            processor = AutoProcessor.from_pretrained(original_qwen_path, trust_remote_code=True)
            model.processor = processor
            print("✅ 成功从原始Qwen模型路径加载processor")
        except Exception as e2:
            print(f"❌ 无法加载processor: {e2}")
            print("将使用tokenizer作为fallback")
            model.processor = None
    
    # 加载SAM权重
    print("正在加载SAM权重...")
    # 直接加载到GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_weight = torch.load(args.sam_vit_path, map_location=device)
    model.get_model().visual_model1.load_state_dict(sam_weight, strict=True)
    print("✅ 成功加载SAM权重到GPU")

    # 4. 设置模型配置
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # 5. 初始化视觉组件
    print("3. 初始化视觉组件...")
    try:
        model.get_model().initialize_vision_modules(model.get_model().config)
        vision_tower = model.get_model().get_vision_tower()
        
        # 对于Qwen2.5-VL，vision_tower可能为None
        if vision_tower is not None:
            vision_tower.to(dtype=torch_dtype)
            print("设置外部vision_tower精度")
        else:
            print("使用Qwen2.5-VL内置视觉编码器")
        
        model.get_model().initialize_glover_modules(model.get_model().config)
        print("✅ 视觉组件初始化完成")
    except Exception as e:
        print(f"⚠️ 视觉组件初始化失败: {e}")
        print("继续使用默认配置")

    # 6. 设置设备
    if torch.cuda.is_available():
        # 如果模型还没有在GPU上，移动到GPU
        if next(model.parameters()).device.type != 'cuda':
            model = model.cuda()
        
        # 确保SAM模型的所有组件都在GPU上
        if hasattr(model.get_model(), 'visual_model1'):
            sam_model = model.get_model().visual_model1
            # 将SAM模型移动到GPU
            sam_model = sam_model.cuda()
            # 确保所有子模块都在GPU上
            for name, module in sam_model.named_modules():
                if hasattr(module, 'to'):
                    # 检查模块是否有参数，如果有参数则检查设备
                    if hasattr(module, 'parameters') and any(p.device.type != 'cuda' for p in module.parameters()):
                        module.to('cuda')
                    elif not hasattr(module, 'parameters'):
                        # 对于没有参数的模块，直接移动到GPU
                        module.to('cuda')
            print("✅ SAM模型已完全移动到GPU")
        
        # 确保所有模型参数都在GPU上
        for name, param in model.named_parameters():
            if param.device.type != 'cuda':
                param.data = param.data.cuda()
        
        print("✅ 模型已完全移动到GPU")
    else:
        print("使用CPU推理")

    # 7. 设置对话模板
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # 8. 开始推理
    print("4. 开始推理...")
    model.eval()
    
    prompt_template = args.prompt
    image_path_template = args.image_path
    
    for i, (obj, action) in enumerate(zip(objs, actions)):
        print(f"\n处理 {i+1}/{len(objs)}: {obj} - {action}")
        
        # 构建提示词
        try:
            prompt = prompt_template % (obj, action)
        except TypeError as e:
            print(f"⚠️ 提示词模板格式化错误: {e}")
            print(f"提示词模板: {prompt_template}")
            print(f"对象: {obj}, 动作: {action}")
            # 使用简单的字符串替换作为备选方案
            prompt = prompt_template.replace("%s", obj, 1).replace("%s", action, 1)
            print(f"使用备选方案后的提示词: {prompt}")
        
        # 尝试不同的文件扩展名
        image_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            filename = obj + ext
            test_path = image_path_template % filename
            if os.path.exists(test_path):
                image_path = test_path
                break
        
        if image_path is None:
            print(f"文件不存在: {obj} (尝试了 .jpg, .png, .jpeg)")
            continue
        
        # 创建保存路径
        save_path_dir = os.path.join(args.vis_save_path, obj)
        os.makedirs(save_path_dir, exist_ok=True)
        save_path = os.path.join(save_path_dir, f"{obj}_{action}.png")
        
        argmax_save_path_dir = os.path.join(args.vis_argmax_save_path, obj)
        os.makedirs(argmax_save_path_dir, exist_ok=True)
        argmax_save_path = os.path.join(argmax_save_path_dir, f"{obj}_{action}.png")

        # 构建完整的提示词
        print(f"原始提示词: {prompt}")
        
        # 在提示词前添加<image>token，让processor自动处理
        prompt_with_image = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        print(f"添加<image>token后: {prompt_with_image}")
        
        # 设置对话
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        conv.append_message(conv.roles[0], prompt_with_image)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        print(f"最终提示词: {prompt}")

        # 加载和预处理图像
        print(f"加载图像: {image_path}")
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size = image_np.shape[:2]

        # 使用Qwen的processor处理图像
        processor = model.processor
        if processor is None:
            print("❌ 模型没有processor，从原始模型路径加载...")
            # 从原始Qwen模型路径加载processor
            from transformers import AutoProcessor
            try:
                # 使用原始Qwen模型路径
                original_qwen_path = "/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL"
                processor = AutoProcessor.from_pretrained(original_qwen_path, trust_remote_code=True)
                print("✅ 成功从原始模型路径加载processor")
            except Exception as e:
                print(f"❌ 无法加载processor: {e}")
                continue
        
        # 重要：使用训练时保存的tokenizer（包含[SEG] token）
        print("使用训练时保存的tokenizer...")
        processor.tokenizer = tokenizer
        print("✅ 已设置processor使用训练时保存的tokenizer")
        
        # 将numpy图像转换为PIL图像
        from PIL import Image
        pil_image = Image.fromarray(image_np)
        
        # 使用与训练时相同的方式：使用apply_chat_template
        try:
            # 构建与训练时相同的消息格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 检查是否有processor
            if hasattr(model, 'processor') and model.processor is not None:
                processor = model.processor
                print("✅ 使用模型内置processor")
            else:
                print("❌ 模型没有processor，从原始模型路径加载...")
                from transformers import AutoProcessor
                original_qwen_path = "/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL"
                processor = AutoProcessor.from_pretrained(original_qwen_path, trust_remote_code=True)
                print("✅ 成功从原始模型路径加载processor")
            
            # 使用训练时保存的tokenizer
            if hasattr(processor, 'tokenizer'):
                processor.tokenizer = tokenizer
                print("✅ 已设置processor使用训练时保存的tokenizer")
            
            # 使用processor的apply_chat_template，与训练时保持一致
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            print(f"Chat template处理后的文本: {text[:100]}...")
            
            # 使用processor处理
            inputs = processor(
                text=text,
                images=pil_image,
                return_tensors="pt",
                padding=True
            )
            print(f"✅ 成功处理图像，输入键: {list(inputs.keys())}")
            
            # 检查image_grid_thw是否存在且不为None
            if 'image_grid_thw' in inputs:
                print(f"image_grid_thw: {inputs['image_grid_thw']}")
                if inputs['image_grid_thw'] is None:
                    print("⚠️ image_grid_thw为None，尝试从pixel_values推断...")
                    if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                        # 从pixel_values推断grid_thw
                        pixel_values = inputs['pixel_values']
                        if len(pixel_values.shape) == 4:  # [batch, channels, height, width]
                            batch_size = pixel_values.shape[0]
                            # 为每个batch创建grid_thw
                            grid_thw_list = []
                            for i in range(batch_size):
                                # 假设图像被分割成网格，这里使用默认值
                                # 实际应该根据图像的实际分割情况来设置
                                grid_thw_list.append([1, 16, 16])  # [t, h, w]
                            inputs['image_grid_thw'] = grid_thw_list
                            print(f"✅ 设置image_grid_thw为: {inputs['image_grid_thw']}")
                        else:
                            print("❌ pixel_values形状不正确，无法推断grid_thw")
                    else:
                        print("❌ pixel_values也为None，无法推断grid_thw")
            else:
                print("❌ 没有找到image_grid_thw键")
        except Exception as e:
            print(f"❌ 图像处理失败: {e}")
            continue
        
        # 移动到GPU
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # 精度转换
        if args.precision == "fp16":
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].half()
        elif args.precision == "bf16":
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor) and inputs[key].dtype == torch.float32:
                    inputs[key] = inputs[key].bfloat16()

        # 推理
        print("执行推理...")
        with torch.no_grad():
            try:
                # 准备推理参数
                # 对于推理，我们需要提供images参数（原始图像数据用于SAM）
                # 将PIL图像转换为numpy数组，然后转换为tensor
                image_np = np.array(pil_image)
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
                
                # 应用精度转换（与inputs保持一致）
                if args.precision == "fp16":
                    image_tensor = image_tensor.half()
                elif args.precision == "bf16":
                    image_tensor = image_tensor.bfloat16()
                
                # 使用evaluate方法进行推理（文本生成+分割）
                print("使用evaluate方法进行推理...")
                
                with torch.no_grad():
                    # 使用模型的evaluate方法进行完整的推理（文本生成+分割）
                    outputs = model.evaluate(
                        image_clip=inputs['pixel_values'],
                        image=image_tensor,  # 原始图像数据用于SAM
                        input_ids=inputs['input_ids'],
                        resize_list=[original_size],
                        original_size_list=[original_size],
                        max_new_tokens=100,
                        tokenizer=tokenizer,
                        use_text_emb_in_suffix_sam=args.use_text_emb_in_suffix_sam,
                        image_grid_thw=inputs.get('image_grid_thw', None),
                    )
                    
                    # 处理推理结果
                    if outputs is not None:
                        print("✅ 推理成功完成")
                        
                        # 检查是否有生成的文本
                        if hasattr(outputs, 'generated_text') and outputs.generated_text:
                            generated_text = outputs.generated_text
                            print(f"生成的文本: {generated_text}")
                            
                            # 检查是否包含[SEG] token
                            if "[SEG]" in generated_text:
                                print("✅ 找到[SEG] token")
                            else:
                                print("❌ 生成的文本中没有找到[SEG] token")
                        else:
                            print("❌ 没有生成文本")
                    else:
                        print("❌ 推理失败，返回结果为空")
                
                print(f"✅ 推理完成，输出类型: {type(outputs)}")
                
                # 检查输出
                if outputs is None:
                    print("❌ 没有生成[SEG] token，跳过这个样本")
                    continue
                
                # 获取预测的掩码
                pred_masks = None
                if hasattr(outputs, 'pred_masks') and outputs.pred_masks:
                    pred_masks = outputs.pred_masks
                elif isinstance(outputs, dict) and 'pred_masks' in outputs:
                    pred_masks = outputs['pred_masks']
                
                if pred_masks and len(pred_masks) > 0:
                    print(f"生成了 {len(pred_masks)} 个掩码")
                else:
                    print("未生成掩码")
                    continue
                
                if pred_masks and len(pred_masks) > 0:
                    print(f"生成了 {len(pred_masks)} 个掩码")
                    
                    for j, pred_mask in enumerate(pred_masks):
                        if pred_mask is None or pred_mask.shape[0] == 0:
                            print(f"掩码 {j} 为空，跳过")
                            continue
                        
                        # 处理掩码
                        if len(pred_mask.shape) == 3:
                            pred_mask = pred_mask[0]  # 取第一个掩码
                        
                        pred_mask = pred_mask.sigmoid()
                        pred_mask = pred_mask.detach().cpu().numpy()
                        
                        # 确保掩码是2D的
                        if len(pred_mask.shape) == 3:
                            pred_mask = pred_mask[0]
                        
                        # 调整掩码尺寸到原始图像尺寸
                        pred_mask_resized = cv2.resize(pred_mask, (original_size[1], original_size[0]))
                        
                        # 可视化
                        image_vis = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        try:
                            aff_vis, _ = draw_affordance(image_vis, pred_mask_resized)
                            cv2.imwrite(save_path, aff_vis)
                            
                            aff_vis_center = draw_affordance_center(image_vis, pred_mask_resized)
                            cv2.imwrite(argmax_save_path, aff_vis_center)
                            
                            print(f"保存可视化结果: {save_path}")
                            print(f"保存中心点结果: {argmax_save_path}")
                        except Exception as vis_e:
                            print(f"可视化失败: {vis_e}")
                else:
                    print("未生成掩码")
                    print(f"输出键: {list(outputs.keys()) if isinstance(outputs, dict) else '不是字典'}")
                    
            except Exception as e:
                print(f"推理过程中出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\n" + "=" * 60)
    print("推理完成！")
    print(f"结果保存在: {args.vis_save_path}")
    print("=" * 60)

    # 清理内存
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    infer(args)
