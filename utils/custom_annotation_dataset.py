import os
import json
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from model.llava.mm_utils import tokenizer_image_token
from .utils import DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN
from model.segment_anything.utils.transforms import ResizeLongestSide


class CustomAnnotationDataset(Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        annotation_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        data_name="custom_annotation",
    ):
        self.samples_per_epoch = samples_per_epoch
        self.base_image_dir = base_image_dir
        self.annotation_dir = annotation_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.data_name = data_name

        # 加载所有task的标注数据
        self.all_annotations = []
        self.all_image_paths = []
        self.all_mask_paths = []
        
        # 支持的task列表
        task_list = ["task_0", "task_1", "task_2", "task_3", "task_4"]
        
        for task in task_list:
            task_annotation_file = os.path.join(annotation_dir, task, "annotations.json")
            task_image_dir = os.path.join(base_image_dir, task)
            
            if os.path.exists(task_annotation_file) and os.path.exists(task_image_dir):
                print(f"Loading annotations from {task}")
                with open(task_annotation_file, 'r', encoding='utf-8') as f:
                    task_annotations = json.load(f)
                
                for annotation in task_annotations:
                    # 构建图像路径
                    image_path = os.path.join(task_image_dir, annotation['img_name'])
                    # 构建mask路径
                    mask_path = os.path.join(annotation_dir, annotation['gt_path'])
                    
                    # 检查文件是否存在
                    if os.path.exists(image_path) and os.path.exists(mask_path):
                        self.all_annotations.append(annotation)
                        self.all_image_paths.append(image_path)
                        self.all_mask_paths.append(mask_path)
                    else:
                        print(f"Warning: Missing files for {annotation['img_name']}")
                        print(f"  Image: {image_path} (exists: {os.path.exists(image_path)})")
                        print(f"  Mask: {mask_path} (exists: {os.path.exists(mask_path)})")
        
        print(f"Loaded {len(self.all_annotations)} valid annotations from {len(task_list)} tasks")
        
        if len(self.all_annotations) == 0:
            raise ValueError("No valid annotations found!")

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_size - h
        padw = self.image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        # 随机选择一个标注
        idx = random.randint(0, len(self.all_annotations) - 1)
        annotation = self.all_annotations[idx]
        image_path = self.all_image_paths[idx]
        mask_path = self.all_mask_paths[idx]

        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取mask
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        # 处理mask：提取affordance区域和affordance点
        if len(mask.shape) == 3:
            # RGB格式 - 处理OpenCV BGR格式保存的红色affordance点
            # OpenCV中红色是(0,0,255)，对应RGB中的蓝色通道
            red_channel = mask[:, :, 0]  # RGB中的红色通道
            green_channel = mask[:, :, 1]  # RGB中的绿色通道
            blue_channel = mask[:, :, 2]  # RGB中的蓝色通道
            
            # Affordance区域：黑色物体区域（所有通道都接近0）
            affordance_region = (red_channel < 50) & (green_channel < 50) & (blue_channel < 50)
            
            # Prioritize using affordance_points coordinates from JSON
            if 'affordance_points' in annotation and annotation['affordance_points']:
                # Use affordance point coordinates saved in JSON
                affordance_point = annotation['affordance_points'][0]  # Take the first affordance point
                # print(f"Using JSON affordance point coordinates: ({affordance_point[0]}, {affordance_point[1]})")
            else:
                # 如果没有JSON坐标，尝试从mask中提取
                # Affordance点：OpenCV BGR格式的红色点，在RGB中是蓝色通道高值
                # OpenCV红色(0,0,255) -> RGB蓝色通道高值
                red_points = (blue_channel > 200) & (red_channel < 100) & (green_channel < 100)
                
                # 找到affordance点的坐标
                affordance_point_coords = np.where(red_points)
                if len(affordance_point_coords[0]) > 0:
                    # 取第一个红色点作为affordance点
                    affordance_y, affordance_x = affordance_point_coords[0][0], affordance_point_coords[1][0]
                    affordance_point = [affordance_x, affordance_y]
                else:
                    # 如果没有找到红色点，使用affordance区域的中心
                    region_coords = np.where(affordance_region)
                    if len(region_coords[0]) > 0:
                        center_y = np.mean(region_coords[0])
                        center_x = np.mean(region_coords[1])
                        affordance_point = [int(center_x), int(center_y)]
                    else:
                        affordance_point = [image.shape[1]//2, image.shape[0]//2]  # 图像中心
        else:
            # 灰度格式
            affordance_region = mask < 50  # 黑色区域作为affordance区域
            # 对于灰度图，假设中心点为affordance点
            affordance_point = [image.shape[1]//2, image.shape[0]//2]
        
        # 转换为二值mask：affordance区域为1，背景为0
        binary_mask = np.where(affordance_region, 1.0, 0.0).astype(np.float32)

        # 预处理图像
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]
        image = self.transform.apply_image(image)
        resize = image.shape[:2]

        # 构建对话 - 使用GLOVER++原始格式，包含affordance点坐标
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        
        # 构建问题：使用GLOVER++原始格式
        object_category = annotation.get('object', 'object')
        action_category = annotation.get('action', 'grasp')
        
        # 构建问题文本：完全匹配GLOVER++原始格式
        question = f"<image>\nWhere should I interact with the {object_category} to {action_category} it? Please output segmentation mask."
        
        # 构建回答：包含affordance点坐标
        # 使用之前提取的affordance点坐标
        x, y = affordance_point
        answer = f"You can interact with the highlighted area [SEG] at point ({x}, {y})."
        
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())

        # 预处理图像
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        
        # 根据precision设置数据类型
        if self.precision == "fp16":
            image = image.half()
            image_clip = image_clip.half()
        elif self.precision == "bf16":
            image = image.bfloat16()
            image_clip = image_clip.bfloat16()
        else:
            image = image.float()
            image_clip = image_clip.float()

        # 处理mask - 使用float32格式，与GLOVER++一致
        masks = torch.from_numpy(binary_mask)
        masks = masks.unsqueeze(0)  # 添加batch维度

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            resize,
            question,
        )


def init_custom_annotation(base_image_dir, annotation_dir):
    """初始化自定义标注数据集"""
    # 这个函数是为了兼容现有的数据集接口
    return [], [], [], [] 