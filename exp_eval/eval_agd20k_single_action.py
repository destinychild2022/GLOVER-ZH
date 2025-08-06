#!/usr/bin/env python3
import argparse
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from tqdm import tqdm
import glob
from PIL import Image, ImageDraw, ImageFont

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from utils.visualizer import draw_affordance_center


def cal_kl(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> float:
    """计算Kullback-Leibler散度"""
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    kld = np.sum(map2 * np.log(map2 / (map1 + eps) + eps))
    return kld


def cal_sim(pred: np.ndarray, gt: np.ndarray, eps=1e-12, is_part=False) -> float:
    """计算相似度"""
    map1 = pred / (pred.sum() + eps)
    if is_part:
        map2 = gt
    else:
        map2 = gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)


def image_binary(image, threshold):
    """将图像二值化"""
    output = np.zeros(image.size).reshape(image.shape)
    for xx in range(image.shape[0]):
        for yy in range(image.shape[1]):
            if image[xx][yy] > threshold:
                output[xx][yy] = 1
    return output


def cal_nss(pred: np.ndarray, gt: np.ndarray) -> float:
    """计算归一化扫描路径显著性"""
    pred = pred / 255.0
    gt = gt / 255.0
    std = np.std(pred)
    u = np.mean(pred)

    smap = (pred - u) / std
    fixation_map = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-12)
    fixation_map = image_binary(fixation_map, 0.1)

    nss = smap * fixation_map
    nss = np.sum(nss) / np.sum(fixation_map + 1e-12)
    return nss


def parse_args(args):
    parser = argparse.ArgumentParser(description="GLOVER++ AGD20K Simple Padding Evaluation")
    parser.add_argument("--version", default="/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus", type=str)
    parser.add_argument("--vision-tower", default="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2", type=str)
    parser.add_argument("--model_arch", default="glover++", type=str)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_text_emb_in_suffix_sam", action="store_true", default=True)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--conv_type", default="llava_v1", type=str)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--dataset_dir", default="/mnt/data-oss/rap-prod-bak/GLOVER/dataset/AGD20K/AGD20K/AGD20K/Unseen/testset", type=str)
    parser.add_argument("--output_dir", default="./agd20k_simple_padding_results", type=str)
    parser.add_argument("--test_action", default="carry", type=str, help="要测试的动作名称")
    parser.add_argument("--max_samples", default=3, type=int, help="最大测试样本数")
    return parser.parse_args(args)


def preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
               pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
               img_size=1024) -> torch.Tensor:
    """图像预处理 - 使用padding而不是resize"""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad to square without resizing
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def load_single_action_data(dataset_dir, test_action, max_samples=3):
    """加载单个动作的数据"""
    img_paths = []
    gt_paths = []
    actions = []
    nouns = []
    
    egocentric_dir = os.path.join(dataset_dir, "egocentric")
    gt_dir = os.path.join(dataset_dir, "GT")
    
    action_path = os.path.join(egocentric_dir, test_action)
    gt_action_path = os.path.join(gt_dir, test_action)
    
    if not os.path.exists(action_path):
        print(f"错误: 动作目录不存在 {action_path}")
        return img_paths, gt_paths, actions, nouns
    
    if not os.path.exists(gt_action_path):
        print(f"错误: GT动作目录不存在 {gt_action_path}")
        return img_paths, gt_paths, actions, nouns
    
    print(f"正在加载动作 '{test_action}' 的数据...")
    
    # 遍历该动作下的所有物体
    for object_dir in os.listdir(action_path):
        object_path = os.path.join(action_path, object_dir)
        gt_object_path = os.path.join(gt_action_path, object_dir)
        
        if not os.path.isdir(object_path) or not os.path.isdir(gt_object_path):
            continue
            
        print(f"  处理物体: {object_dir}")
        
        # 获取该物体下的所有图片
        image_files = glob.glob(os.path.join(object_path, "*.jpg"))
        
        for i, img_file in enumerate(image_files):
            if len(img_paths) >= max_samples:
                break
                
            # 构造对应的GT mask文件路径
            base_name = os.path.splitext(os.path.basename(img_file))[0]
            gt_file = os.path.join(gt_object_path, f"{base_name}.png")
            
            if os.path.exists(gt_file):
                img_paths.append(img_file)
                gt_paths.append(gt_file)
                actions.append(test_action)
                nouns.append(object_dir)
                print(f"    添加样本 {len(img_paths)}: {os.path.basename(img_file)}")
            
            if len(img_paths) >= max_samples:
                break
        
        if len(img_paths) >= max_samples:
            break
    
    print(f"加载了 {len(img_paths)} 个样本")
    return img_paths, gt_paths, actions, nouns


def pad_image_to_square(image, target_size=1024):
    """将图像padding到正方形，使用白色填充"""
    h, w = image.shape[:2]
    
    # 如果图像已经大于目标尺寸，需要先resize
    if h > target_size or w > target_size:
        # 保持长宽比，将最长边resize到target_size
        scale = target_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        from PIL import Image
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        image = np.array(pil_image)
        h, w = new_h, new_w
    
    # 计算padding
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    
    # 计算padding位置（居中）
    pad_h_top = pad_h // 2
    pad_w_left = pad_w // 2
    
    # 使用白色填充图像
    if len(image.shape) == 3:
        padded_image = np.full((target_size, target_size, image.shape[2]), 255, dtype=image.dtype)
    else:
        padded_image = np.full((target_size, target_size), 255, dtype=image.dtype)
    
    # 将原图像放在中心
    padded_image[pad_h_top:pad_h_top+h, pad_w_left:pad_w_left+w] = image
    
    return padded_image


def pad_mask_to_square(mask, target_size=1024):
    """将mask padding到正方形，使用黑色填充"""
    h, w = mask.shape[:2]
    
    # 如果mask已经大于目标尺寸，需要先resize
    if h > target_size or w > target_size:
        # 保持长宽比，将最长边resize到target_size
        scale = target_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        from PIL import Image
        pil_mask = Image.fromarray(mask)
        pil_mask = pil_mask.resize((new_w, new_h), Image.NEAREST)
        mask = np.array(pil_mask)
        h, w = new_h, new_w
    
    # 计算padding
    pad_h = max(0, target_size - h)
    pad_w = max(0, target_size - w)
    
    # 计算padding位置（居中）
    pad_h_top = pad_h // 2
    pad_w_left = pad_w // 2
    
    # 使用黑色填充mask
    if len(mask.shape) == 3:
        padded_mask = np.zeros((target_size, target_size, mask.shape[2]), dtype=mask.dtype)
    else:
        padded_mask = np.zeros((target_size, target_size), dtype=mask.dtype)
    
    # 将原mask放在中心
    padded_mask[pad_h_top:pad_h_top+h, pad_w_left:pad_w_left+w] = mask
    
    return padded_mask


def save_mask_comparison(pred_mask, gt_mask, original_image, save_path, sample_idx):
    """保存预测mask和GT mask的对比图"""
    try:
        # 确保所有图像都是1024x1024
        assert pred_mask.shape == (1024, 1024), f"预测mask尺寸错误: {pred_mask.shape}"
        assert gt_mask.shape == (1024, 1024), f"GT mask尺寸错误: {gt_mask.shape}"
        
        # 将mask转换为0-255范围用于显示
        pred_mask_display = (pred_mask * 255).astype(np.uint8)
        gt_mask_display = (gt_mask * 255).astype(np.uint8)
        
        # 创建对比图
        # 第一行：原图、预测mask、GT mask
        # 第二行：预测mask和GT mask的叠加对比
        
        # 创建画布 (1024*3, 1024*2)
        canvas_width = 1024 * 3
        canvas_height = 1024 * 2
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        
        # 第一行：原图
        if len(original_image.shape) == 3:
            original_pil = Image.fromarray(original_image)
        else:
            original_pil = Image.fromarray(original_image, mode='L').convert('RGB')
        original_pil = original_pil.resize((1024, 1024))
        canvas.paste(original_pil, (0, 0))
        
        # 第一行：预测mask（深红色显示）
        pred_mask_colored = np.zeros((1024, 1024, 3), dtype=np.uint8)
        # 使用更深的红色，并且只在mask值大于阈值时显示
        threshold = 0.1  # 设置阈值，只显示明显的预测区域
        pred_mask_binary = (pred_mask > threshold).astype(np.uint8) * 255
        pred_mask_colored[:, :, 0] = pred_mask_binary  # 红色通道
        pred_mask_pil = Image.fromarray(pred_mask_colored)
        canvas.paste(pred_mask_pil, (1024, 0))
        
        # 第一行：GT mask（深绿色显示）
        gt_mask_colored = np.zeros((1024, 1024, 3), dtype=np.uint8)
        # 使用更深的绿色
        gt_mask_binary = (gt_mask > 0.1).astype(np.uint8) * 255
        gt_mask_colored[:, :, 1] = gt_mask_binary  # 绿色通道
        gt_mask_pil = Image.fromarray(gt_mask_colored)
        canvas.paste(gt_mask_pil, (2048, 0))
        
        # 第二行：叠加对比
        overlay = np.zeros((1024, 1024, 3), dtype=np.uint8)
        overlay[:, :, 0] = pred_mask_binary  # 预测mask（红色）
        overlay[:, :, 1] = gt_mask_binary    # GT mask（绿色）
        # 重叠区域会显示为黄色
        overlay_pil = Image.fromarray(overlay)
        canvas.paste(overlay_pil, (0, 1024))
        
        # 添加文字标签
        draw = ImageDraw.Draw(canvas)
        try:
            # 尝试使用默认字体
            font = ImageFont.load_default()
        except:
            font = None
        
        # 添加标签
        labels = [
            "Original Image", "Predicted Mask (Red)", "GT Mask (Green)",
            "Overlay (Red=Pred, Green=GT, Yellow=Overlap)"
        ]
        positions = [(10, 10), (1034, 10), (2058, 10), (10, 1034)]
        
        for label, pos in zip(labels, positions):
            if font:
                draw.text(pos, label, fill=(255, 255, 255), font=font)
            else:
                draw.text(pos, label, fill=(255, 255, 255))
        
        # 保存图片
        canvas.save(save_path)
        print(f"对比图已保存: {save_path}")
        
    except Exception as e:
        print(f"保存对比图失败: {e}")


def create_visualization_dir(output_dir):
    """创建可视化输出目录"""
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir


def main(args):
    args = parse_args(args)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建可视化目录
    vis_dir = create_visualization_dir(args.output_dir)
    
    print(f"开始测试动作: {args.test_action}")
    print(f"最大样本数: {args.max_samples}")
    print("注意: 使用简化版本，直接使用原始图像尺寸进行测试")
    print(f"可视化结果将保存到: {vis_dir}")
    
    # 创建模型
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        })
    elif args.load_in_8bit:
        kwargs.update({
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        })

    if args.model_arch == "glover++":
        from model.GLOVER_plus import GloverForCausalLM
    elif args.model_arch == "glover":
        from model.GLOVER import GloverForCausalLM

    model = GloverForCausalLM.from_pretrained(
        args.version,
        low_cpu_mem_usage=True,
        vision_tower=args.vision_tower,
        seg_token_idx=args.seg_token_idx,
        **kwargs,
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().to(f"cuda:{args.local_rank}")
    elif args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed
        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().to(f"cuda:{args.local_rank}")
    elif args.precision == "fp32":
        model = model.float().to(f"cuda:{args.local_rank}")

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)

    model.eval()
    print("模型加载完成")

    # 加载数据
    img_paths, gt_paths, actions, nouns = load_single_action_data(
        args.dataset_dir, args.test_action, args.max_samples
    )
    
    if len(img_paths) == 0:
        print("未找到数据，请检查数据集路径和动作名称")
        return

    # 评估指标
    KLs = []
    SIM = []
    NSS = []
    
    print(f"开始评估 {len(img_paths)} 个样本...")
    
    for i in tqdm(range(len(img_paths)), desc=f"评估 {args.test_action}"):
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        object_name = nouns[i]
        action_name = actions[i]
        
        prompt = f"<image>\nWhere should I interact with the {object_name} to {action_name} it?"
        prompt = prompt + " Please output segmentation mask."
        
        if args.use_mm_start_end:
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        # 加载图像
        image_path = img_paths[i]
        try:
            # 尝试使用PIL加载图像（避免cv2依赖问题）
            from PIL import Image
            import numpy as np
            
            # 加载图像
            pil_image = Image.open(image_path)
            image_np = np.array(pil_image)
            
            # 如果是RGBA，转换为RGB
            if len(image_np.shape) == 3 and image_np.shape[2] == 4:
                image_np = image_np[:, :, :3]
            
            if image_np is None:
                print(f"跳过图像 {image_path} - 无法加载")
                continue
                
        except Exception as e:
            print(f"跳过图像 {image_path} - 加载失败: {e}")
            continue
            
        # 将原图padding到1024×1024（白色填充）
        padded_image = pad_image_to_square(image_np, 1024)
        original_size_list = [image_np.shape[:2]]

        # 加载GT mask并padding到1024×1024（黑色填充）
        gt_path = gt_paths[i]
        try:
            gt_pil = Image.open(gt_path)
            gt_mask = np.array(gt_pil)
            
            # 如果是RGB，转换为灰度
            if len(gt_mask.shape) == 3:
                gt_mask = gt_mask[:, :, 0]  # 取第一个通道
            
            # 归一化到0-1
            gt_mask = gt_mask.astype(np.float32) / 255.0
            
            # 将GT mask padding到1024×1024（黑色填充）
            gt_mask = pad_mask_to_square(gt_mask, 1024)
            
        except Exception as e:
            print(f"处理GT mask失败 {gt_path}: {e}")
            gt_mask = np.zeros((1024, 1024), dtype=np.float32)

        # 图像预处理 - 使用padding后的图像
        image_clip = clip_image_processor.preprocess(padded_image, return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(f"cuda:{args.local_rank}")
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        # 使用padding后的图像进行SAM处理
        image = transform.apply_image(padded_image)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .to(f"cuda:{args.local_rank}")
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(f"cuda:{args.local_rank}")

        # 模型推理
        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
            **({"use_text_emb_in_suffix_sam": args.use_text_emb_in_suffix_sam} if args.model_arch == "glover++" else {}),
        )

        # 处理预测结果
        for j, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask[0].sigmoid()
            pred_mask = pred_mask.detach().cpu().numpy()
            
            # 将预测mask padding到1024×1024（黑色填充）
            pred_mask = pad_mask_to_square(pred_mask, 1024)
            
            # 现在pred_mask和gt_mask都是1024×1024，可以正确比对
            kld = cal_kl(pred_mask, gt_mask)
            sim = cal_sim(pred_mask, gt_mask)
            nss = cal_nss(pred_mask, gt_mask)
            
            print(f"样本 {i+1}: KLD={kld:.4f}, SIM={sim:.4f}, NSS={nss:.4f}")

            KLs.append(kld)
            SIM.append(sim)
            NSS.append(nss)

            # 保存可视化结果
            save_path = os.path.join(vis_dir, f"sample_{i+1}_pred_gt_comparison.png")
            save_mask_comparison(pred_mask, gt_mask, padded_image, save_path, i+1)

    # 计算最终结果
    if len(KLs) > 0:
        mKLD = np.mean(KLs)
        mSIM = np.mean(SIM)
        mNSS = np.mean(NSS)

        print("\n" + "="*50)
        print(f"动作 '{args.test_action}' 的评估结果（简化版本）")
        print("="*50)
        print(f"mKLD: {mKLD:.4f}")
        print(f"mSIM: {mSIM:.4f}")
        print(f"mNSS: {mNSS:.4f}")
        print(f"样本数: {len(KLs)}")
        print("注意: 这是简化版本的测试结果")

        # 保存结果
        results = {
            "action": args.test_action,
            "mKLD": float(mKLD),
            "mSIM": float(mSIM),
            "mNSS": float(mNSS),
            "samples": len(KLs),
            "note": "简化版本测试结果"
        }

        output_file = os.path.join(args.output_dir, f"{args.test_action}_eval_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n结果已保存到: {output_file}")
    else:
        print("没有成功处理的样本")


if __name__ == "__main__":
    main(sys.argv[1:]) 