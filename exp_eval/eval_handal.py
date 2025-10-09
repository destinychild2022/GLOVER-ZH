import argparse
import os
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont
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
from utils.visualizer import draw_affordance_center
import pdb
import json
from tqdm import tqdm
import pdb
import glob
import random


def save_mask_comparison(pred_mask, gt_mask, original_image, save_path, sample_idx, object_name):
    """保存预测mask和GT mask的对比图"""
    try:
        # 确保所有图像都是相同尺寸
        h, w = pred_mask.shape[:2]
        
        # 将mask转换为0-255范围用于显示
        pred_mask_display = (pred_mask * 255).astype(np.uint8)
        gt_mask_display = (gt_mask * 255).astype(np.uint8)
        
        # 创建对比图
        # 第一行：原图、预测mask、GT mask
        # 第二行：预测mask和GT mask的叠加对比
        
        # 创建画布 (w*3, h*2)
        canvas_width = w * 3
        canvas_height = h * 2
        canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))
        
        # 第一行：原图
        if len(original_image.shape) == 3:
            original_pil = Image.fromarray(original_image)
        else:
            original_pil = Image.fromarray(original_image, mode='L').convert('RGB')
        original_pil = original_pil.resize((w, h))
        canvas.paste(original_pil, (0, 0))
        
        # 第一行：预测mask（深红色显示）
        pred_mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
        # 显示连续概率值，不进行二值化，并增强红色显示
        pred_mask_normalized = (pred_mask * 255).astype(np.uint8)
        # 增强红色显示：使用更亮的红色
        pred_mask_colored[:, :, 0] = np.clip(pred_mask_normalized * 2, 0, 255)  # 红色通道增强
        pred_mask_pil = Image.fromarray(pred_mask_colored)
        canvas.paste(pred_mask_pil, (w, 0))
        
        # 第一行：GT mask（深绿色显示）
        gt_mask_colored = np.zeros((h, w, 3), dtype=np.uint8)
        # 显示连续概率值，并增强绿色显示
        gt_mask_normalized = (gt_mask * 255).astype(np.uint8)
        # 增强绿色显示：使用更亮的绿色
        gt_mask_colored[:, :, 1] = np.clip(gt_mask_normalized * 2, 0, 255)  # 绿色通道增强
        gt_mask_pil = Image.fromarray(gt_mask_colored)
        canvas.paste(gt_mask_pil, (w*2, 0))
        
        # 第二行：叠加对比
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        # 增强叠加显示
        overlay[:, :, 0] = np.clip(pred_mask_normalized * 2, 0, 255)  # 预测mask（红色）
        overlay[:, :, 1] = np.clip(gt_mask_normalized * 2, 0, 255)    # GT mask（绿色）
        # 重叠区域会显示为黄色
        overlay_pil = Image.fromarray(overlay)
        canvas.paste(overlay_pil, (0, h))
        
        # 添加文字标签
        draw = ImageDraw.Draw(canvas)
        try:
            # 尝试使用默认字体
            font = ImageFont.load_default()
        except:
            font = None
        
        # 添加标签
        labels = [
            f"Original Image ({object_name})", 
            "Predicted Mask (Red)", 
            "GT Mask (Green)",
            "Overlay (Red=Pred, Green=GT, Yellow=Overlap)"
        ]
        positions = [(10, 10), (w+10, 10), (w*2+10, 10), (10, h+10)]
        
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


# 理想值为 0
def cal_kl(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    kld = np.sum(map2 * np.log(map2 / (map1 + eps) + eps))
    return kld


def cal_sim(pred: np.ndarray, gt: np.ndarray, eps=1e-12, is_part=False) -> np.ndarray:
    map1 = pred / (pred.sum() + eps)
    if is_part:
        map2 = gt
    else:
        map2 = gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)

    return np.sum(intersection)


def image_binary(image, threshold):
    output = np.zeros(image.size).reshape(image.shape)
    for xx in range(image.shape[0]):
        for yy in range(image.shape[1]):
            if image[xx][yy] > threshold:
                output[xx][yy] = 1
    return output


def cal_nss(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
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
    parser = argparse.ArgumentParser(description="GLOVER++ HANDAL Evaluation")
    parser.add_argument("--version", default="/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument(
        "--vision-tower",
        default="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2",
        type=str,
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--use_text_emb_in_suffix_sam",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--dataset_dir",
        default="/mnt/data-oss/rap-prod-bak/GLOVER/dataset/HOVA-500K/HANDAL",
        type=str,
    )
    parser.add_argument(
        "--model_arch",
        default="glover++",
        type=str,
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    parser.add_argument("--max_samples", default=5, type=int, help="最大测试样本数")
    parser.add_argument("--seed", default=42, type=int, help="随机种子")
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


def load_handal_data(dataset_dir, max_samples=5, seed=42):
    """加载HANDAL数据集，随机选择max_samples个样本"""
    img_paths = []
    gt_paths = []
    actions = []
    nouns = []
    
    # 设置随机种子
    random.seed(seed)
    
    print("正在加载HANDAL数据集...")
    
    # HANDAL数据集结构：
    # /dataset_dir/images/handal_dataset_*/test/*/rgb/*.jpg
    # /dataset_dir/images/handal_dataset_*/test/*/mask_parts/*.png
    
    # 获取所有物体类型目录
    object_types = ["handal_dataset_spatulas", "handal_dataset_strainers", "handal_dataset_utensils", "handal_dataset_whisks"]
    
    all_image_files = []
    all_gt_files = []
    all_object_names = []
    
    for object_type in object_types:
        # 从目录名提取物体名称
        object_name = object_type.replace("handal_dataset_", "")
        object_dir = os.path.join(dataset_dir, "images", object_type, "test")
        if not os.path.exists(object_dir):
            print(f"跳过不存在的目录: {object_dir}")
            continue
            
        # 遍历每个场景目录
        for scene_dir in os.listdir(object_dir):
            scene_path = os.path.join(object_dir, scene_dir)
            if not os.path.isdir(scene_path):
                continue
                
            rgb_dir = os.path.join(scene_path, "rgb")
            mask_parts_dir = os.path.join(scene_path, "mask_parts")  # 使用mask_parts而不是mask
            
            if not os.path.exists(rgb_dir) or not os.path.exists(mask_parts_dir):
                continue
                
            # 获取所有图像文件
            image_files = glob.glob(os.path.join(rgb_dir, "*.jpg"))
            
            for img_file in image_files:
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                # GT文件名格式：{base_name}_000000_handle.png
                gt_file = os.path.join(mask_parts_dir, f"{base_name}_000000_handle.png")
                
                if os.path.exists(gt_file):
                    all_image_files.append(img_file)
                    all_gt_files.append(gt_file)
                    all_object_names.append(object_name)
    
    print(f"找到 {len(all_image_files)} 个有效样本")
    
    # 随机选择max_samples个样本
    if len(all_image_files) > max_samples:
        selected_indices = random.sample(range(len(all_image_files)), max_samples)
        selected_image_files = [all_image_files[i] for i in selected_indices]
        selected_gt_files = [all_gt_files[i] for i in selected_indices]
        selected_object_names = [all_object_names[i] for i in selected_indices]
    else:
        selected_image_files = all_image_files
        selected_gt_files = all_gt_files
        selected_object_names = all_object_names
        print(f"图像数量少于{max_samples}，使用所有图像")
    
    print(f"随机选择了 {len(selected_image_files)} 个样本")
    
    for i, (img_file, gt_file, object_name) in enumerate(zip(selected_image_files, selected_gt_files, selected_object_names)):
        img_paths.append(img_file)
        gt_paths.append(gt_file)
        actions.append("pick up")  # HANDAL数据集主要是pick up动作
        nouns.append(object_name)  # 使用真实的物体名称
        print(f"  添加样本 {i+1}: {os.path.basename(img_file)} -> {os.path.basename(gt_file)} (物体: {object_name})")
    
    print(f"成功加载了 {len(img_paths)} 个样本")
    return img_paths, gt_paths, actions, nouns


def main(args):
    args = parse_args(args)
    
    # 创建输出目录
    output_dir = "exp_eval"  # 使用exp_eval目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建可视化目录
    vis_dir = create_visualization_dir(output_dir)
    
    print(f"开始HANDAL数据集评估...")
    print(f"最大样本数: {args.max_samples}")
    print(f"随机种子: {args.seed}")
    print(f"可视化结果将保存到: {vis_dir}")
    
    # Create model
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
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
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

    img_paths, gt_paths, actions, nouns = load_handal_data(args.dataset_dir, args.max_samples, args.seed)

    KLs = []
    SIM = []
    NSS = []

    for i in tqdm(range(len(img_paths))):
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        object = nouns[i]
        action = actions[i]
        prompt = f"<image>\nWhere should I interact with the {object} to {action} it?"
        prompt = prompt + " Please output segmentation mask."
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_path = img_paths[i]
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        gt_path = gt_paths[i]
        gt_mask = Image.open(gt_path)
        gt_mask = np.array(gt_mask) / 255.0

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .to(f"cuda:{args.local_rank}")
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

        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
            **(
                {"use_text_emb_in_suffix_sam": args.use_text_emb_in_suffix_sam}
                if args.model_arch == "glover++"
                else {}
            ),
        )

        for j, pred_mask in enumerate(pred_masks):
            if pred_mask.shape[0] == 0:
                continue

            pred_mask = pred_mask[0].sigmoid()
            pred_mask = pred_mask.detach().cpu().numpy()

            kld, sim, nss = (
                cal_kl(pred_mask, gt_mask),
                cal_sim(pred_mask, gt_mask),
                cal_nss(pred_mask, gt_mask),
            )

            KLs.append(kld)
            SIM.append(sim)
            NSS.append(nss)

            # 保存可视化结果
            if args.max_samples <= 10:
                # 样本数较少时，保存所有样本的对比图
                save_path = os.path.join(vis_dir, f"sample_{i+1}_{object}.png")
                save_mask_comparison(pred_mask, gt_mask, image_np, save_path, i+1, object)
            else:
                # 样本数较多时，只保存前10张和每1000张保存一张
                if i < 10 or i % 100 == 0:
                    save_path = os.path.join(vis_dir, f"sample_{i+1}_{object}.png")
                    save_mask_comparison(pred_mask, gt_mask, image_np, save_path, i+1, object)

        if i % 500 == 0 and i != 0:
            print("mKLD: ", sum(KLs) / len(KLs))
            print("mSIM: ", sum(SIM) / len(SIM))
            print("mNSS: ", sum(NSS) / len(NSS))
    
    mKLD = sum(KLs) / len(KLs)
    mSIM = sum(SIM) / len(SIM)
    mNSS = sum(NSS) / len(NSS)

    print("mKLD: ", mKLD)
    print("mSIM: ", mSIM)
    print("mNSS: ", mNSS)
    
    # 保存结果到文件
    results_file = os.path.join(output_dir, "handal_eval_results.txt")
    with open(results_file, "a") as f:
        f.write("model name: %s\n" % args.version.split("/")[-1])
        f.write("mKLD: %s\n" % str(mKLD))
        f.write("mSIM: %s\n" % str(mSIM))
        f.write("mNSS: %s\n" % str(mNSS))
        f.write("\n")
    
    print(f"结果已保存到: {results_file}")
    print(f"可视化结果已保存到: {vis_dir}")


if __name__ == "__main__":
    main(sys.argv[1:]) 