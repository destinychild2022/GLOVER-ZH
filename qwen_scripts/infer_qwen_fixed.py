import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

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
import pdb


def parse_args(args):
    parser = argparse.ArgumentParser(description="GLOVER-Qwen chat")
    parser.add_argument("--version", default="/path/to/GLOVER-Qwen model")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--vis_argmax_save_path", default="./vis_argmax_output", type=str
    )
    parser.add_argument("--image_path", default="", type=str)
    parser.add_argument("--prompt", default="", type=str)
    parser.add_argument("--objects", default="", type=str)
    parser.add_argument("--actions", default="", type=str)

    parser.add_argument(
        "--use_text_emb_in_suffix_sam", action="store_true", default=False
    )
    parser.add_argument(
        "--model_arch",
        default="glover_qwen",
        type=str,
        choices=["glover_qwen"],
    )

    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower",
        default="/path/to/clip-vit-large-patch14",
        type=str,
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_llama_2",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def infer(args):
    print("--------------------------------------------------------")
    print("generating with GLOVER-Qwen model: %s..." % args.version)

    # 设置离线模式，避免网络连接问题
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'

    # 确保输出目录存在
    os.makedirs(args.vis_save_path, exist_ok=True)
    os.makedirs(args.vis_argmax_save_path, exist_ok=True)
    objs = args.objects.split(",")
    actions = args.actions.split(",")

    # Create model - 使用训练好的 tokenizer
    print("从训练好的模型加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "/mnt/data-oss/rap-prod-bak/GLOVER/output/qwen2/tokenizer",  # 使用最新的训练好的tokenizer
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,  # 重要：Qwen tokenizer 需要这个
    )
    tokenizer.pad_token = tokenizer.unk_token
    
    # 检查 [SEG] token 是否存在
    if "[SEG]" in tokenizer.get_vocab():
        print("✅ [SEG] token 已存在于训练好的 tokenizer 中")
    else:
        print("❌ [SEG] token 不存在，这不应该发生")
    
    # 检查图像相关的 token
    if "<im_start>" in tokenizer.get_vocab():
        print("✅ 图像相关 token 已存在")
    else:
        print("❌ 图像相关 token 不存在")
    
    # 确保这些变量在全局作用域中可用
    from utils.utils import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    
    # 使用与GLOVER_plus.py相同的设置方式，但使用Qwen中SEG token的正确ID
    args.seg_token_idx = 78759  # Qwen中SEG的token ID
    print(f"SEG token ID: {args.seg_token_idx}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    # 加载基础模型 + LoRA权重 + GLOVER组件
    print("加载基础模型 + LoRA权重 + GLOVER组件...")
    from model.GLOVER_qwen import GloverQwenForCausalLM
    from peft import PeftModel
    
    # 首先加载基础模型
    print("加载基础Qwen2.5-VL模型...")
    base_model = GloverQwenForCausalLM.from_pretrained(
        "/mnt/data-oss/rap-prod-bak/GLOVER/model/Qwen-7B-VL",  # 基础模型路径
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        vision_tower=args.vision_tower,  # 传递本地CLIP路径
        **kwargs
    )
    
    # 初始化GLOVER特有的组件
    print("初始化GLOVER组件...")
    base_model.get_model().initialize_vision_modules(base_model.get_model().config)
    base_model.get_model().initialize_glover_modules(base_model.get_model().config)
    
    # 然后加载LoRA权重
    print("加载LoRA权重...")
    lora_path = "/mnt/data-oss/rap-prod-bak/GLOVER/output/qwen2/lora_weights"
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 加载训练后的GLOVER组件权重
    print("加载训练后的GLOVER组件权重...")
    checkpoint_path = "/mnt/data-oss/rap-prod-bak/GLOVER/output/qwen2/ckpt_model/global_step1960/mp_rank_00_model_states.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state_dict = checkpoint['module']
        
        # 提取GLOVER组件的权重
        glover_weights = {}
        for key, value in model_state_dict.items():
            if 'text_hidden_fcs' in key:
                # 移除base_model前缀，适配当前模型结构
                new_key = key.replace('base_model.model.model.', '')
                glover_weights[new_key] = value
                print(f"加载GLOVER权重: {new_key}")
        
        # 加载GLOVER权重到模型
        if glover_weights:
            model.get_model().load_state_dict(glover_weights, strict=False)
            print(f"成功加载 {len(glover_weights)} 个GLOVER组件权重")
        else:
            print("警告：未找到GLOVER组件权重")
    else:
        print("警告：训练检查点文件不存在，使用初始化的GLOVER组件")
    
    print("模型加载完成！")

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    if vision_tower is not None:
        vision_tower.to(dtype=torch_dtype)
    else:
        print("使用Qwen2.5-VL内置视觉编码器，无需单独处理vision_tower")

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
        # 确保SAM模型也使用bf16精度
        if hasattr(model, 'get_model') and hasattr(model.get_model(), 'visual_model1'):
            model.get_model().visual_model1 = model.get_model().visual_model1.bfloat16()
            print("SAM模型已转换为bf16精度")
    elif args.precision == "fp16":
        model = model.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    if vision_tower is not None:
        vision_tower.to(device=args.local_rank)
    else:
        print("使用Qwen2.5-VL内置视觉编码器，无需单独处理vision_tower设备")

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()

    prompt_template = args.prompt
    image_path_template = args.image_path
    save_path_template = args.vis_save_path
    argmax_save_path_template = args.vis_argmax_save_path
    for _, (obj, action) in enumerate(zip(objs, actions)):
        prompt = prompt_template % (obj, action)
        image_path = image_path_template % obj  # 格式化图像路径，替换 %s 为对象名
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue
        save_path_dir = os.path.join(save_path_template, obj)
        # 确保父目录存在
        os.makedirs(save_path_template, exist_ok=True)
        try:
            os.makedirs(save_path_dir, exist_ok=True)
        except FileExistsError:
            if os.path.isfile(save_path_dir):
                os.remove(save_path_dir)
                os.makedirs(save_path_dir, exist_ok=True)
        save_path = os.path.join(save_path_dir, args.version.split("/")[-1] + ".png")

        argmax_save_path_dir = os.path.join(argmax_save_path_template, obj)
        # 确保父目录存在
        os.makedirs(argmax_save_path_template, exist_ok=True)
        try:
            os.makedirs(argmax_save_path_dir, exist_ok=True)
        except FileExistsError:
            if os.path.isfile(argmax_save_path_dir):
                os.remove(argmax_save_path_dir)
                os.makedirs(argmax_save_path_dir, exist_ok=True)
        argmax_save_path = os.path.join(
            argmax_save_path_dir, args.version.split("/")[-1] + ".png"
        )

        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )

        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        print(f"原始input_ids shape: {input_ids.shape}")
        print(f"原始input_ids: {input_ids}")
        input_ids = input_ids.unsqueeze(0).cuda()
        print(f"处理后的input_ids shape: {input_ids.shape}")

        # 使用 evaluate 方法进行推理
        print("使用 evaluate 方法进行推理")
        
        with torch.no_grad():
            output_ids, pred_masks = model.evaluate(
                image_clip=image_clip,  # CLIP处理后的图像特征
                image=image,            # SAM处理用的原始图像
                input_ids=input_ids,
                resize_list=resize_list,
                original_size_list=original_size_list,
                max_new_tokens=512,    # 使用与原始GLOVER相同的生成长度
                tokenizer=tokenizer,
                use_text_emb_in_suffix_sam=args.use_text_emb_in_suffix_sam,
            )
        # 确保output_ids和IMAGE_TOKEN_INDEX在同一设备上
        if output_ids.device != torch.device('cuda'):
            output_ids = output_ids.cuda()
        
        # 过滤图像token
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
        print(f"过滤后的output_ids shape: {output_ids.shape}")

        # 解码文本输出
        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output for %s: %s" % (obj, text_output))
        
        # 检查生成的token中是否包含[SEG] token ID - 检查所有可能的[SEG] token ID
        possible_seg_ids = [151665, 151666, 151667, 151655]
        found_seg_tokens = []
        
        for seg_id in possible_seg_ids:
            if seg_id in output_ids:
                found_seg_tokens.append(seg_id)
        
        if found_seg_tokens:
            print(f"✅ 模型成功生成了 [SEG] token，ID: {found_seg_tokens}")
        else:
            print("❌ 在生成的token中没有找到[SEG] token")
            print(f"生成的token IDs: {output_ids.tolist()}")
            print("这可能是模型训练不充分或生成参数需要调整")
        
        # 额外检查文本输出中是否包含[SEG]字符串（仅供参考）
        if "[SEG]" not in text_output:
            print(f"注意：文本输出中没有显示'[SEG]'字符串，但token ID检查: {'成功' if found_seg_tokens else '失败'}")
        else:
            print("✅ 文本输出中也显示了[SEG]字符串")

        print(f"总共生成了 {len(pred_masks)} 个掩码")
        
        # 处理掩码 - 现在应该只有一个掩码
        if len(pred_masks) > 0:
            pred_mask = pred_masks[0]
            if pred_mask.shape[0] == 0:
                print("掩码为空，跳过")
            else:
                print(f"处理掩码，原始形状: {pred_mask.shape}")
                
                # 应用 sigmoid 并转换为 numpy
                pred_mask = pred_mask.sigmoid()
                pred_mask = pred_mask.detach().cpu().numpy()
                
                print(f"图像尺寸: {image_np.shape}")
                print(f"原始掩码尺寸: {pred_mask.shape}")
                
                # 确保掩码是2D的 - 关键修复！
                if len(pred_mask.shape) == 4:
                    # (batch, channel, height, width) -> (height, width)
                    pred_mask = pred_mask[0, 0]  # 取第一个batch和第一个通道
                elif len(pred_mask.shape) == 3:
                    # (batch, height, width) -> (height, width)
                    pred_mask = pred_mask[0]  # 取第一个batch
                elif len(pred_mask.shape) == 2:
                    # 已经是2D，保持不变
                    pass
                else:
                    print(f"警告：未知的掩码形状 {pred_mask.shape}")
                
                print(f"处理后掩码尺寸: {pred_mask.shape}")
                
                # 分析掩码的数值分布来选择合适的阈值
                print(f"掩码数值统计: min={pred_mask.min():.3f}, max={pred_mask.max():.3f}, mean={pred_mask.mean():.3f}")
                
                # 尝试不同的阈值，找到最合适的
                best_threshold = 0.5
                best_ratio = 0.0
                
                for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    test_mask = (pred_mask > threshold).astype(np.uint8)
                    foreground_ratio = np.sum(test_mask > 0) / test_mask.size
                    print(f"阈值 {threshold}: 前景比例 {foreground_ratio:.3f}")
                    
                    # 选择前景比例在10%-30%之间的阈值
                    if 0.1 <= foreground_ratio <= 0.3:
                        best_threshold = threshold
                        best_ratio = foreground_ratio
                        break
                
                print(f"选择阈值: {best_threshold}, 前景比例: {best_ratio:.3f}")
                
                # 生成二值掩码
                binary_mask = (pred_mask > best_threshold).astype(np.uint8) * 255
                
                # 形态学操作来平滑掩码
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)  # 填充小洞
                binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)   # 去除噪声
                
                print(f"二值掩码统计: 前景像素 {np.sum(binary_mask > 0)}, 背景像素 {np.sum(binary_mask == 0)}")
                
                # 调整掩码尺寸以匹配图像
                if pred_mask.shape != image_np.shape[:2]:
                    if image_np.shape[1] > 0 and image_np.shape[0] > 0:
                        try:
                            pred_mask = cv2.resize(pred_mask, (image_np.shape[1], image_np.shape[0]))
                            binary_mask = cv2.resize(binary_mask, (image_np.shape[1], image_np.shape[0]))
                            print(f"调整后掩码尺寸: {pred_mask.shape}")
                        except Exception as e:
                            print(f"调整掩码尺寸失败: {e}")
                            print(f"保持原始掩码尺寸: {pred_mask.shape}")
                    else:
                        print(f"图像尺寸无效，跳过调整: {image_np.shape}")
                
                # 确保二值掩码是2D的
                if len(binary_mask.shape) == 3:
                    binary_mask = binary_mask[:, :, 0]  # 取第一个通道
                elif len(binary_mask.shape) == 4:
                    binary_mask = binary_mask[0, :, :, 0]  # 取第一个batch和第一个通道
                
                # 保存二值掩码
                binary_save_path = save_path.replace('.png', '_binary_mask.png')
                cv2.imwrite(binary_save_path, binary_mask)
                print(f"保存二值掩码到: {binary_save_path}")
                
                # 可视化 - 使用已经处理好的2D掩码
                image_np_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                pred_mask_vis = pred_mask.astype(np.float32)
                
                try:
                    aff_vis, _ = draw_affordance(image_np_bgr, pred_mask_vis)
                    cv2.imwrite(save_path, aff_vis)
                    print(f"保存可视化结果到: {save_path}")
                    
                    aff_vis_center = draw_affordance_center(image_np_bgr, pred_mask_vis)
                    cv2.imwrite(argmax_save_path, aff_vis_center)
                    print(f"保存中心点结果到: {argmax_save_path}")
                except Exception as e:
                    print(f"可视化失败: {e}")
                    print(f"pred_mask_vis形状: {pred_mask_vis.shape}")
                    print(f"image_np_bgr形状: {image_np_bgr.shape}")
        else:
            print("没有生成任何掩码")

    del model
    if 'image_clip' in locals():
        del image_clip
    if 'image' in locals():
        del image
    if 'input_ids' in locals():
        del input_ids
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    infer(args)
