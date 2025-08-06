import argparse
import os
import sys
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from tqdm import tqdm
import glob

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
    parser = argparse.ArgumentParser(description="GLOVER++ AGD20K Evaluation")
    parser.add_argument("--version", default="/mnt/data-oss/rap-prod-bak/GLOVER/model/GLOVER_plus", type=str)
    parser.add_argument("--vision-tower", default="/mnt/data-oss/rap-prod-bak/GLOVER/model/clip-vit-large-patch14-2", type=str)
    parser.add_argument("--model_arch", default="glover++", type=str)
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=64, type=int)
    parser.add_argument("--use_text_emb_in_suffix_sam", action="store_true", default=True)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--conv_type", default="llava_v1", type=str)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--dataset_dir", default="/mnt/data-oss/rap-prod-bak/GLOVER/dataset/AGD20K/AGD20K/AGD20K/Unseen/testset", type=str)
    parser.add_argument("--output_dir", default="./agd20k_results", type=str)
    parser.add_argument("--max_samples_per_object", default=3, type=int, help="每种物体最大测试样本数")
    parser.add_argument("--use_json_points", action="store_true", default=False, help="是否使用json文件中的点坐标生成mask")
    parser.add_argument("--pred_threshold", default=0.01, type=float, help="预测mask的阈值")
    parser.add_argument("--gt_threshold", default=0.1, type=float, help="GT mask的阈值")
    return parser.parse_args(args)


def preprocess(x, pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
               pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
               img_size=1024) -> torch.Tensor:
    """图像预处理"""
    x = (x - pixel_mean) / pixel_std
    return x


def pad_image_to_square(image, target_size=1024):
    """将图像padding到正方形，使用白色填充"""
    h, w = image.shape[:2]
    
    # 如果图像已经大于目标尺寸，需要先resize
    if h > target_size or w > target_size:
        # 保持长宽比，将最长边resize到target_size
        scale = target_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
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


def create_mask_from_points(points, image_shape, radius=10):
    """从点坐标创建mask"""
    mask = np.zeros(image_shape[:2], dtype=np.float32)
    
    if not points or len(points) == 0:
        return mask
    
    h, w = image_shape[:2]
    
    for point in points:
        if len(point) >= 2:
            x, y = int(point[0]), int(point[1])
            # 确保坐标在图像范围内
            x = max(0, min(x, w-1))
            y = max(0, min(y, h-1))
            
            # 在点周围画圆
            cv2.circle(mask, (x, y), radius, 1.0, -1)
    
    return mask


def load_gt_from_json(json_path, image_shape, radius=20):
    """从json文件加载GT点坐标并生成mask"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 提取点坐标 - 处理LabelMe格式
        points = []
        if 'shapes' in data:
            shapes = data['shapes']
            for shape in shapes:
                if shape.get('shape_type') == 'point':
                    # 对于点标注，取第一个点
                    if 'points' in shape and len(shape['points']) > 0:
                        point = shape['points'][0]
                        points.append(point)
        
        # 从点坐标生成mask，使用更大的半径
        mask = create_mask_from_points(points, image_shape, radius)
        return mask
        
    except Exception as e:
        print(f"Error loading GT from json {json_path}: {e}")
        return np.zeros(image_shape[:2], dtype=np.float32)


def save_mask_comparison(pred_mask, gt_mask, original_image, save_path, sample_idx, pred_threshold=0.05, gt_threshold=0.1):
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
        # 显示连续概率值，不进行二值化，并增强红色显示
        pred_mask_normalized = (pred_mask * 255).astype(np.uint8)
        # 增强红色显示：使用更亮的红色
        pred_mask_colored[:, :, 0] = np.clip(pred_mask_normalized * 2, 0, 255)  # 红色通道增强
        pred_mask_pil = Image.fromarray(pred_mask_colored)
        canvas.paste(pred_mask_pil, (1024, 0))
        
        # 第一行：GT mask（深绿色显示）
        gt_mask_colored = np.zeros((1024, 1024, 3), dtype=np.uint8)
        # 显示连续概率值，并增强绿色显示
        gt_mask_normalized = (gt_mask * 255).astype(np.uint8)
        # 增强绿色显示：使用更亮的绿色
        gt_mask_colored[:, :, 1] = np.clip(gt_mask_normalized * 2, 0, 255)  # 绿色通道增强
        gt_mask_pil = Image.fromarray(gt_mask_colored)
        canvas.paste(gt_mask_pil, (2048, 0))
        
        # 第二行：叠加对比
        overlay = np.zeros((1024, 1024, 3), dtype=np.uint8)
        # 增强叠加显示
        overlay[:, :, 0] = np.clip(pred_mask_normalized * 2, 0, 255)  # 预测mask（红色）
        overlay[:, :, 1] = np.clip(gt_mask_normalized * 2, 0, 255)    # GT mask（绿色）
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


def load_agd20k_data(dataset_dir, max_samples_per_object=3, use_json_points=False):
    """加载AGD20K数据集，每种物体最多取max_samples_per_object个样本"""
    img_paths = []
    gt_paths = []
    actions = []
    nouns = []
    
    # 遍历egocentric目录下的所有动作文件夹
    egocentric_dir = os.path.join(dataset_dir, "egocentric")
    
    if use_json_points:
        # 使用json文件中的点坐标，json文件就在egocentric目录中
        gt_dir = egocentric_dir
    else:
        # 使用mask图片
        gt_dir = os.path.join(dataset_dir, "GT")
    
    if not os.path.exists(egocentric_dir):
        print(f"Error: {egocentric_dir} does not exist")
        return img_paths, gt_paths, actions, nouns
    
    if not os.path.exists(gt_dir):
        print(f"Error: {gt_dir} does not exist")
        return img_paths, gt_paths, actions, nouns
    
    print("正在加载AGD20K数据集...")
    print(f"使用GT数据格式: {'JSON点坐标' if use_json_points else 'Mask图片'}")
    
    for action_dir in os.listdir(egocentric_dir):
        action_path = os.path.join(egocentric_dir, action_dir)
        gt_action_path = os.path.join(gt_dir, action_dir)
        
        if not os.path.isdir(action_path) or not os.path.isdir(gt_action_path):
            continue
            
        print(f"  处理动作: {action_dir}")
        
        # 遍历每个动作下的物体文件夹
        for object_dir in os.listdir(action_path):
            object_path = os.path.join(action_path, object_dir)
            gt_object_path = os.path.join(gt_action_path, object_dir)
            
            if not os.path.isdir(object_path) or not os.path.isdir(gt_object_path):
                continue
                
            print(f"    处理物体: {object_dir}")
            
            # 获取该物体文件夹下的所有图片
            image_files = glob.glob(os.path.join(object_path, "*.jpg"))
            
            sample_count = 0
            for img_file in image_files:
                if sample_count >= max_samples_per_object:
                    break
                    
                # 构造对应的GT文件路径
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                if use_json_points:
                    gt_file = os.path.join(gt_object_path, f"{base_name}.json")
                else:
                    gt_file = os.path.join(gt_object_path, f"{base_name}.png")
                
                if os.path.exists(gt_file):
                    img_paths.append(img_file)
                    gt_paths.append(gt_file)
                    actions.append(action_dir)
                    nouns.append(object_dir)
                    sample_count += 1
                    print(f"      添加样本 {sample_count}: {os.path.basename(img_file)}")
                
                if sample_count >= max_samples_per_object:
                    break
    
    print(f"加载了 {len(img_paths)} 个样本")
    return img_paths, gt_paths, actions, nouns


def load_gt_mask(gt_path, image_shape, use_json_points=False):
    """加载GT mask图片或从json生成mask"""
    try:
        if use_json_points:
            # 从json文件加载点坐标并生成mask
            return load_gt_from_json(gt_path, image_shape)
        else:
            # 加载mask图片
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                print(f"Error loading GT mask: {gt_path}")
                return np.zeros(image_shape[:2], dtype=np.float32)
            
            # 归一化到0-1
            gt_mask = gt_mask.astype(np.float32) / 255.0
            
            # 如果mask尺寸与图像不匹配，进行resize
            if gt_mask.shape != image_shape[:2]:
                gt_mask = cv2.resize(gt_mask, (image_shape[1], image_shape[0]))
            
            return gt_mask
    except Exception as e:
        print(f"Error loading GT mask {gt_path}: {e}")
        return np.zeros(image_shape[:2], dtype=np.float32)


def main(args):
    args = parse_args(args)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建可视化目录
    vis_dir = create_visualization_dir(args.output_dir)
    
    print(f"开始AGD20K数据集评估...")
    print(f"每种物体最大样本数: {args.max_samples_per_object}")
    print(f"使用JSON点坐标: {args.use_json_points}")
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

    # 加载AGD20K数据
    img_paths, gt_paths, actions, nouns = load_agd20k_data(
        args.dataset_dir, args.max_samples_per_object, args.use_json_points
    )
    
    if len(img_paths) == 0:
        print("No data found. Please check the dataset path.")
        return

    # 评估指标
    KLs = []
    SIM = []
    NSS = []
    
    # 按动作类型分组统计
    action_metrics = {}
    # 按动作-物体分组统计
    action_object_metrics = {}
    
    print(f"开始评估 {len(img_paths)} 个样本...")
    
    for i in tqdm(range(len(img_paths)), desc="Evaluating"):
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        object_name = nouns[i]
        action_name = actions[i]
        
        # 构建提示词
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
            # 使用原始图像尺寸加载GT
            gt_mask = load_gt_mask(gt_path, image_np.shape, args.use_json_points)
            
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
            # 使用连续概率值进行计算，而不是二值化
            kld = cal_kl(pred_mask, gt_mask)
            sim = cal_sim(pred_mask, gt_mask)
            nss = cal_nss(pred_mask, gt_mask)
            
            # 添加调试信息
            if i < 5:  # 只对前5个样本显示调试信息
                print(f"样本 {i+1} 预测mask统计:")
                print(f"  最大值: {np.max(pred_mask):.4f}")
                print(f"  最小值: {np.min(pred_mask):.4f}")
                print(f"  平均值: {np.mean(pred_mask):.4f}")
                print(f"  标准差: {np.std(pred_mask):.4f}")
                print(f"  大于0.1的像素数: {np.sum(pred_mask > 0.1)}")
                print(f"  大于0.05的像素数: {np.sum(pred_mask > 0.05)}")
                print(f"  大于0.01的像素数: {np.sum(pred_mask > 0.01)}")

            KLs.append(kld)
            SIM.append(sim)
            NSS.append(nss)
            
            # 按动作类型分组
            if action_name not in action_metrics:
                action_metrics[action_name] = {"KLs": [], "SIM": [], "NSS": []}
            action_metrics[action_name]["KLs"].append(kld)
            action_metrics[action_name]["SIM"].append(sim)
            action_metrics[action_name]["NSS"].append(nss)
            
            # 按动作-物体分组
            action_object_key = f"{action_name}_{object_name}"
            if action_object_key not in action_object_metrics:
                action_object_metrics[action_object_key] = {"KLs": [], "SIM": [], "NSS": []}
            action_object_metrics[action_object_key]["KLs"].append(kld)
            action_object_metrics[action_object_key]["SIM"].append(sim)
            action_object_metrics[action_object_key]["NSS"].append(nss)

            # 保存可视化结果（每种物体保存前3张图片）
            object_samples = len(action_object_metrics[action_object_key]["KLs"])
            if object_samples <= 3:
                save_path = os.path.join(vis_dir, f"{action_name}_{object_name}_sample_{object_samples}.png")
                save_mask_comparison(pred_mask, gt_mask, padded_image, save_path, i+1, 
                                   args.pred_threshold, args.gt_threshold)

        # 每100个样本打印一次进度
        if (i + 1) % 100 == 0:
            print(f"\nProcessed {i + 1}/{len(img_paths)} samples")
            print(f"Current mKLD: {np.mean(KLs):.4f}")
            print(f"Current mSIM: {np.mean(SIM):.4f}")
            print(f"Current mNSS: {np.mean(NSS):.4f}")

    # 计算最终结果
    mKLD = np.mean(KLs)
    mSIM = np.mean(SIM)
    mNSS = np.mean(NSS)

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"mKLD: {mKLD:.4f}")
    print(f"mSIM: {mSIM:.4f}")
    print(f"mNSS: {mNSS:.4f}")
    print(f"Total samples: {len(KLs)}")
    print(f"GT数据格式: {'JSON点坐标' if args.use_json_points else 'Mask图片'}")

    # 按动作类型输出结果
    print("\n" + "="*50)
    print("RESULTS BY ACTION")
    print("="*50)
    for action, metrics in action_metrics.items():
        if len(metrics["KLs"]) > 0:
            print(f"{action}:")
            print(f"  mKLD: {np.mean(metrics['KLs']):.4f}")
            print(f"  mSIM: {np.mean(metrics['SIM']):.4f}")
            print(f"  mNSS: {np.mean(metrics['NSS']):.4f}")
            print(f"  Samples: {len(metrics['KLs'])}")

    # 按动作-物体输出结果
    print("\n" + "="*50)
    print("RESULTS BY ACTION-OBJECT")
    print("="*50)
    for action_object, metrics in action_object_metrics.items():
        if len(metrics["KLs"]) > 0:
            print(f"{action_object}:")
            print(f"  mKLD: {np.mean(metrics['KLs']):.4f}")
            print(f"  mSIM: {np.mean(metrics['SIM']):.4f}")
            print(f"  mNSS: {np.mean(metrics['NSS']):.4f}")
            print(f"  Samples: {len(metrics['KLs'])}")

    # 保存结果
    results = {
        "overall": {
            "mKLD": float(mKLD),
            "mSIM": float(mSIM),
            "mNSS": float(mNSS),
            "total_samples": len(KLs),
            "gt_format": "JSON点坐标" if args.use_json_points else "Mask图片"
        },
        "by_action": {},
        "by_action_object": {}
    }
    
    for action, metrics in action_metrics.items():
        if len(metrics["KLs"]) > 0:
            results["by_action"][action] = {
                "mKLD": float(np.mean(metrics["KLs"])),
                "mSIM": float(np.mean(metrics["SIM"])),
                "mNSS": float(np.mean(metrics["NSS"])),
                "samples": len(metrics["KLs"])
            }
    
    for action_object, metrics in action_object_metrics.items():
        if len(metrics["KLs"]) > 0:
            results["by_action_object"][action_object] = {
                "mKLD": float(np.mean(metrics["KLs"])),
                "mSIM": float(np.mean(metrics["SIM"])),
                "mNSS": float(np.mean(metrics["NSS"])),
                "samples": len(metrics["KLs"])
            }

    # 保存到文件
    suffix = "_json" if args.use_json_points else "_mask"
    output_file = os.path.join(args.output_dir, f"agd20k_eval_results{suffix}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main(sys.argv[1:]) 