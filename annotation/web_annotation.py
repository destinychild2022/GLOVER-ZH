#!/usr/bin/env python3
"""
基于Web的可视化标注工具
使用Flask创建Web界面，支持点击添加affordance点，使用SAM进行分割
使用基于文件名的智能默认类别生成物体和动作类别
"""

import os
import json
import cv2
import numpy as np
import torch
import argparse
import base64
import glob
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# SAM imports
from segment_anything import SamPredictor, sam_model_registry

# LISA imports (已废弃，移除相关导入)
# try:
#     from transformers import LlavaForConditionalGeneration, LlavaProcessor, AutoConfig
#     LISA_AVAILABLE = True
# except ImportError:
#     LISA_AVAILABLE = False
#     print("Warning: LISA model not available, using default categories")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 增加到100MB
app.config['MAX_CONTENT_LENGTH'] = None  # 或者完全禁用限制

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('output/masks', exist_ok=True)

class WebAnnotator:
    def __init__(self, sam_checkpoint: str, device: str = "cuda", image_dir: str = None, lisa_model_path: str = None):
        """
        初始化Web标注器
        
        Args:
            sam_checkpoint: SAM模型路径
            device: 设备类型
            image_dir: 图像目录路径
            lisa_model_path: LISA模型路径（已废弃，保留参数以兼容旧代码）
        """
        self.device = device
        self.image_dir = image_dir
        
        # 设置输出目录，与IMAGE_DIR的最后一级目录名相同
        if image_dir:
            image_dir_path = Path(image_dir)
            task_name = image_dir_path.name  # 获取最后一级目录名
            self.output_dir = f"output/{task_name}"
            print(f"输出目录: {self.output_dir}")
        else:
            self.output_dir = "output/default"
            print(f"输出目录: {self.output_dir}")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "masks"), exist_ok=True)
        
        # 初始化SAM模型
        print("Loading SAM model...")
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        
        # LISA模型已废弃，不再初始化
        self.lisa_model = None
        self.lisa_processor = None
        if lisa_model_path:
            print("注意: LISA模型参数已提供但将被忽略，LISA功能已禁用")
        
        # 预定义的物体和动作类别 - 针对日常生活场景优化
        self.object_categories = [
            # 机械零件
            "adjusting screw", "set screw", "ring", "pin", "rod", "plug", "cone spring", "ball", 
            "spring seat", "core", "spring", "seat", "cone sleeve", "glyd ring", "snap ring", 
            "screw seat", "nut", "cone core", "step seal", "damping core",

        ]
        
        # 手部相关的动作类别 - 针对日常生活场景优化
        self.action_categories = [
            # 基本抓取动作
            "grasp", "hold", "lift", "move", "place", "put", "take", "pick", "carry", "pass",
            # 开关动作
            "open", "close", "turn_on", "turn_off", "switch", "press", "push", "pull",
            # 旋转动作
            "turn", "rotate", "twist", "screw", "unscrew",
            # 使用动作
            "use", "operate", "control", "adjust", "set", "activate", "deactivate",
            # 清洁动作
            "clean", "wash", "wipe", "sweep", "dust", "scrub", "rinse",
            # 烹饪动作
            "cook", "heat", "cool", "mix", "stir", "pour", "fill", "empty", "cut", "slice", "chop",
            # 触摸动作
            "touch", "tap", "click", "drag", "drop", "throw", "catch", "squeeze", "pinch", "grip",
            # 其他动作
            "fold", "unfold", "tear", "rip", "bend", "rub", "massage", "pat", "slap"
        ]
        
        # 标注数据存储
        self.annotations = []
        
        # 获取图像列表
        self.image_list = self.get_image_list()
        
        # 当前图像索引
        self.current_image_index = 0
        
        # 图像缓存，提高加载速度
        self.image_cache = {}
        self.max_cache_size = 10  # 最多缓存10张图片
        
        # 记忆功能：记录常用的物体和动作类别
        self.memory_file = self.get_memory_filename()
        self.load_memory()  # 同时加载标注数据
        
        # 设置当前图像索引为第一个未标注的图像
        self.set_current_to_first_unannotated()
        
        # 预加载下一张图片
        self.preload_next_image()
    
    def get_memory_filename(self) -> str:
        """根据图像目录获取记忆文件名"""
        try:
            # 从图像目录路径中提取任务名称
            image_dir_path = Path(self.image_dir)
            
            # 查找包含task_的目录名
            for part in image_dir_path.parts:
                if part.startswith('task_'):
                    task_name = part
                    break
            else:
                # 如果没有找到task_开头的目录，使用图像目录的最后一部分
                task_name = image_dir_path.name
            
            memory_filename = f"annotation_memory_{task_name}.json"
            print(f"记忆文件: {memory_filename}")
            return memory_filename
            
        except Exception as e:
            print(f"获取记忆文件名失败: {e}")
            return "annotation_memory_default.json"
    
    def load_memory(self):
        """加载记忆数据"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                    self.object_memory = memory_data.get('objects', {})
                    self.action_memory = memory_data.get('actions', {})
                    self.annotations = memory_data.get('annotations', [])  # 同时加载标注数据
                    print(f"加载记忆数据: {len(self.object_memory)} 个物体类别, {len(self.action_memory)} 个动作类别, {len(self.annotations)} 条标注")
            else:
                self.object_memory = {}
                self.action_memory = {}
                self.annotations = []
                print("创建新的记忆数据")
        except Exception as e:
            print(f"加载记忆数据失败: {e}")
            self.object_memory = {}
            self.action_memory = {}
            self.annotations = []
    
    def save_memory(self):
        """保存记忆数据"""
        try:
            memory_data = {
                'objects': self.object_memory,
                'actions': self.action_memory,
                'annotations': self.annotations  # 同时保存标注数据
            }
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)
            print(f"保存记忆数据: {len(self.object_memory)} 个物体类别, {len(self.action_memory)} 个动作类别, {len(self.annotations)} 条标注")
        except Exception as e:
            print(f"保存记忆数据失败: {e}")
    
    def update_memory(self, object_category: str, action_category: str):
        """更新记忆数据"""
        # 更新物体类别记忆
        if object_category in self.object_memory:
            self.object_memory[object_category] += 1
        else:
            self.object_memory[object_category] = 1
        
        # 更新动作类别记忆
        if action_category in self.action_memory:
            self.action_memory[action_category] += 1
        else:
            self.action_memory[action_category] = 1
        
        # 保存记忆数据
        self.save_memory()
    
    def get_suggestions(self) -> Tuple[List[str], List[str]]:
        """获取建议的物体和动作类别"""
        # 按使用频率排序
        sorted_objects = sorted(self.object_memory.items(), key=lambda x: x[1], reverse=True)
        sorted_actions = sorted(self.action_memory.items(), key=lambda x: x[1], reverse=True)
        
        # 取前20个最常用的（增加数量）
        object_suggestions = [obj for obj, _ in sorted_objects[:20]]
        action_suggestions = [act for act, _ in sorted_actions[:20]]
        
        return object_suggestions, action_suggestions
    
    def get_image_list(self):
        """获取图像目录下的所有图像文件，按数字大小排序"""
        if not self.image_dir or not os.path.exists(self.image_dir):
            return []
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file in os.listdir(self.image_dir):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(file)
        
        # 自然排序：按照文件名中的数字大小排序
        def natural_sort_key(filename):
            """提取文件名中的数字用于排序"""
            # 提取文件名中的所有数字
            parts = re.split(r'(\d+)', filename)
            # 将数字部分转换为整数，非数字部分保持原样
            return [int(part) if part.isdigit() else part.lower() for part in parts]
        
        return sorted(image_files, key=natural_sort_key)
    
    def predict_categories(self, image_path: str, mask: np.ndarray) -> Tuple[str, str]:
        """
        返回最常用的物体和动作类别
        只返回真正使用频率高的类别，不返回默认的object和grasp
        """
        # 获取频率最高的类别
        object_suggestions, action_suggestions = self.get_suggestions()
        
        # 只返回真正有使用记录的类别
        if object_suggestions and len(object_suggestions) > 0:
            default_object = object_suggestions[0]
        else:
            # 如果没有使用记录，返回空字符串，让前端使用预定义类别
            default_object = ""
            
        if action_suggestions and len(action_suggestions) > 0:
            default_action = action_suggestions[0]
        else:
            # 如果没有使用记录，返回空字符串，让前端使用预定义类别
            default_action = ""
        
        return default_object, default_action
    
    def segment_object(self, image: np.ndarray, points: List[Tuple[int, int]], 
                      labels: List[int]) -> Tuple[np.ndarray, List[float]]:
        """
        使用SAM模型分割物体
        """
        # 设置图像
        self.predictor.set_image(image)
        
        # 转换为numpy数组
        points_array = np.array(points)
        labels_array = np.array(labels)
        
        # 预测掩码 - 使用更快的设置
        masks, scores, logits = self.predictor.predict(
            point_coords=points_array,
            point_labels=labels_array,
            multimask_output=False,  # 只输出一个mask以提高速度
            return_logits=False  # 不返回logits以提高速度
        )
        
        # 获取mask
        mask = masks[0] if len(masks) > 0 else np.zeros(image.shape[:2], dtype=bool)
        
        # 计算边界框
        y_indices, x_indices = np.where(mask)
        if len(y_indices) > 0 and len(x_indices) > 0:
            x_min, x_max = x_indices.min(), x_indices.max()
            y_min, y_max = y_indices.min(), y_indices.max()
            
            # 归一化坐标
            height, width = image.shape[:2]
            bbox = [
                x_min / width,  # x
                y_min / height,  # y
                (x_max - x_min) / width,  # width
                (y_max - y_min) / height  # height
            ]
        else:
            bbox = [0, 0, 0, 0]
            
        return mask, bbox
    
    def save_annotation(self, image_path: str, bbox: List[float], 
                       affordance: List[float], object_category: str, 
                       action_category: str, mask: np.ndarray = None, 
                       points: List[Tuple[int, int]] = None,
                       affordance_points: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        保存标注数据
        """
        # 读取图像获取尺寸 - 修复图像路径问题
        print(f"保存标注图像路径: {image_path}")
        
        # 检查图像路径是否存在，如果不存在则尝试重新构造路径
        if not os.path.exists(image_path):
            print(f"图像路径不存在: {image_path}")
            # 尝试从文件名重新构造路径
            filename = os.path.basename(image_path)
            if filename in self.image_list:
                image_path = os.path.join(self.image_dir, filename)
                print(f"重新构造图像路径: {image_path}")
            else:
                raise ValueError(f"Image file not found: {image_path}")
        
        # 尝试从缓存获取图像
        image = None
        filename = os.path.basename(image_path)
        
        # 尝试从缓存获取
        if filename in self.image_cache:
            image = self.image_cache[filename]
            print(f"从缓存获取图像: {filename}")
        
        # 如果缓存中没有，直接读取
        if image is None:
            image = cv2.imread(image_path)
            print(f"直接读取图像: {image_path}")
        
        # 检查图像是否成功加载
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        height, width = image.shape[:2]
        print(f"图像尺寸: {width}x{height}")
        
        # 生成掩码文件名：原名_物体类别_mask.png
        base_name = Path(image_path).stem
        # 清理物体类别名称，移除特殊字符，避免文件名问题
        safe_object_category = object_category.replace('/', '_').replace('\\', '_').replace(' ', '_')
        mask_filename = f"{base_name}_{safe_object_category}_mask.png"
        
        # 保存掩码到output目录
        mask_path = os.path.join(self.output_dir, "masks", mask_filename)
        
        # 如果传入了mask数据，保存mask文件
        if mask is not None and len(mask) > 0:
            # 创建带affordance点的mask图像（白色背景，黑色物体，红色点）
            mask_with_points = np.ones((height, width, 3), dtype=np.uint8) * 255  # 白色背景
            
            # 将mask数据调整到正确的尺寸
            if mask.shape != (height, width):
                # 如果mask尺寸不匹配，需要调整
                print(f"调整mask尺寸: {mask.shape} -> ({height}, {width})")
                
                # 调整mask尺寸到实际图像尺寸
                from PIL import Image
                
                # 将mask转换为PIL图像（mask已经是numpy数组）
                mask_array = (mask.astype(np.uint8) * 255)  # 转换为0-255范围
                mask_pil = Image.fromarray(mask_array)
                
                # 调整尺寸到实际图像尺寸
                mask_resized = mask_pil.resize((width, height), Image.NEAREST)
                mask_resized_array = np.array(mask_resized) > 127  # 转换回布尔值
                
                # 使用调整后的mask
                mask_with_points[mask_resized_array] = [0, 0, 0]  # 黑色物体
                print(f"mask尺寸调整完成: {mask_resized_array.shape}")
            else:
                mask_with_points[mask] = [0, 0, 0]  # 黑色物体
            
            # 如果有affordance点，在mask上绘制红色圆点
            if affordance_points:
                for point in affordance_points:
                    x, y = int(point[0]), int(point[1])
                    # 确保点在图像范围内
                    if 0 <= x < width and 0 <= y < height:
                        cv2.circle(mask_with_points, (x, y), 5, (0, 0, 255), -1)  # 红色圆点
            
            cv2.imwrite(mask_path, mask_with_points)
            print(f"保存mask文件: {mask_path}")
        else:
            # 如果没有mask数据，检查是否已经有mask文件
            if not os.path.exists(mask_path):
                print(f"警告: 没有mask数据且mask文件不存在: {mask_path}")
                # 创建一个基本的mask文件（白色背景）
                basic_mask = np.ones((height, width, 3), dtype=np.uint8) * 255
                cv2.imwrite(mask_path, basic_mask)
                print(f"创建基本mask文件: {mask_path}")
            else:
                print(f"使用已存在的mask文件: {mask_path}")
        
        # 创建标注数据
        annotation = {
            "img_name": os.path.basename(image_path),
            "bbox": bbox,
            "affordance": affordance,
            "height": height,
            "width": width,
            "object": object_category,
            "action": action_category,
            "gt_path": os.path.relpath(mask_path, "output"),  # 使用相对于output目录的路径
            "points": points if points else [],  # 保存分割点
            "affordance_points": affordance_points if affordance_points else []  # 保存affordance点
        }
        
        return annotation
    
    def get_next_image_index(self):
        """获取下一张图像的索引"""
        if self.current_image_index < len(self.image_list) - 1:
            return self.current_image_index + 1
        return 0  # 循环到第一张

    def preload_next_image(self):
        """预加载下一张图片，提高加载速度"""
        if self.current_image_index < len(self.image_list) - 1:
            next_index = self.current_image_index + 1
            next_image_name = self.image_list[next_index]
            if next_image_name not in self.image_cache:
                image_path = os.path.join(self.image_dir, next_image_name)
                if os.path.exists(image_path):
                    try:
                        image = cv2.imread(image_path)
                        self.image_cache[next_image_name] = image
                        print(f"预加载图片: {next_image_name}")
                    except Exception as e:
                        print(f"预加载图片失败: {next_image_name}, 错误: {e}")
                else:
                    print(f"预加载图片不存在: {next_image_name}")
    
    def get_cached_image(self, image_name: str) -> np.ndarray:
        """获取缓存的图像，如果没有则加载并缓存"""
        if image_name in self.image_cache:
            return self.image_cache[image_name]
        
        image_path = os.path.join(self.image_dir, image_name)
        if os.path.exists(image_path):
            try:
                image = cv2.imread(image_path)
                # 管理缓存大小
                if len(self.image_cache) >= self.max_cache_size:
                    # 删除最旧的缓存
                    oldest_key = next(iter(self.image_cache))
                    del self.image_cache[oldest_key]
                
                self.image_cache[image_name] = image
                return image
            except Exception as e:
                print(f"加载图片失败: {image_name}, 错误: {e}")
                return None
        else:
            print(f"图片不存在: {image_name}")
            return None
    
    def update_current_image_index(self, new_index: int):
        """更新当前图像索引并预加载下一张图片"""
        self.current_image_index = new_index
        # 异步预加载下一张图片
        import threading
        threading.Thread(target=self.preload_next_image, daemon=True).start()

    def load_saved_annotations(self):
        """加载已保存的标注数据"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    memory_data = json.load(f)
                    self.annotations = memory_data.get('annotations', [])
                    print(f"加载已保存的标注数据: {len(self.annotations)} 条标注")
            else:
                self.annotations = []
                print("没有已保存的标注数据")
        except Exception as e:
            print(f"加载已保存的标注数据失败: {e}")
            self.annotations = []

    def set_current_to_first_unannotated(self):
        """设置当前图像索引为第一个未标注的图像"""
        # 获取已标注的图像列表
        annotated_images = [ann['img_name'] for ann in self.annotations]
        
        # 检查已存在的mask文件，标记已标注的图片（支持新格式：原名_物体类别_mask.png）
        mask_dir = os.path.join(self.output_dir, "masks")
        if os.path.exists(mask_dir):
            for image_name in self.image_list:
                base_name = Path(image_name).stem
                # 查找所有可能的mask文件（支持新格式和旧格式）
                # 查找所有以base_name开头的mask文件
                mask_pattern = os.path.join(mask_dir, f"{base_name}_*_mask.png")
                old_mask_pattern = os.path.join(mask_dir, f"{base_name}_mask.png")
                matching_masks = glob.glob(mask_pattern) + glob.glob(old_mask_pattern)
                if matching_masks and image_name not in annotated_images:
                    # 如果存在mask文件但不在标注列表中，也标记为已标注
                    annotated_images.append(image_name)
        
        # 找到第一个未标注的图像
        for i, image_name in enumerate(self.image_list):
            if image_name not in annotated_images:
                self.current_image_index = i
                print(f"设置当前图像索引为第一个未标注的图像: {image_name} (索引: {i})")
                return
        
        # 如果所有图像都已标注，设置为第一张图像
        self.current_image_index = 0
        print("所有图像都已标注，设置为第一张图像")

# 全局标注器实例
annotator = None

@app.errorhandler(500)
def internal_error(error):
    """处理500错误"""
    print(f"500错误: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found_error(error):
    """处理404错误"""
    print(f"404错误: {error}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    """处理所有异常"""
    print(f"未处理的异常: {e}")
    import traceback
    traceback.print_exc()
    return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """主页"""
    # 获取已标注的图像列表
    annotated_images = [ann['img_name'] for ann in annotator.annotations]
    
    # 检查已存在的mask文件，标记已标注的图片
    mask_dir = os.path.join(annotator.output_dir, "masks")
    if os.path.exists(mask_dir):
        for image_name in annotator.image_list:
            base_name = Path(image_name).stem
            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(mask_dir, mask_filename)
            if os.path.exists(mask_path) and image_name not in annotated_images:
                # 如果存在mask文件但不在标注列表中，也标记为已标注
                annotated_images.append(image_name)
    
    return render_template('index.html', 
                         object_categories=annotator.object_categories,
                         action_categories=annotator.action_categories,
                         image_list=annotator.image_list,
                         current_index=annotator.current_image_index,
                         annotated_images=annotated_images)

@app.route('/load_image/<filename>')
def load_image(filename):
    """加载指定图像"""
    if filename not in annotator.image_list:
        return jsonify({'error': 'Image not found'}), 404
    
    image_path = os.path.join(annotator.image_dir, filename)
    
    try:
        # 使用缓存获取图像
        image = annotator.get_cached_image(filename)
        if image is None:
            return jsonify({'error': 'Failed to load image'}), 500
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 编码为base64
        _, buffer = cv2.imencode('.jpg', image_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])  # 降低质量以提高速度
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 更新当前图像索引（如果不同）
        current_index = annotator.image_list.index(filename)
        if current_index != annotator.current_image_index:
            annotator.update_current_image_index(current_index)
        
        # 查找已保存的标注信息（同一张图片可能有多个不同物体类别的标注）
        # 返回该图片的所有标注，前端显示最后一个（最新的）
        saved_annotations = [ann for ann in annotator.annotations if ann['img_name'] == filename]
        saved_annotation = saved_annotations[-1] if saved_annotations else None
        
        # 如果没有找到标注信息，检查是否有mask文件
        if not saved_annotation:
            base_name = Path(filename).stem
            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(annotator.output_dir, "masks", mask_filename)
            if os.path.exists(mask_path):
                # 创建基本的标注信息
                # 获取频率最高的物体和动作类别作为默认值
                object_suggestions, action_suggestions = annotator.get_suggestions()
                # 只使用真正有使用记录的类别
                default_object = object_suggestions[0] if object_suggestions else "object"
                default_action = action_suggestions[0] if action_suggestions else "grasp"
                
                saved_annotation = {
                    "img_name": filename,
                    "object": default_object,  # 使用频率最高的物体类别
                    "action": default_action,   # 使用频率最高的动作类别
                    "gt_path": os.path.relpath(mask_path, "output"),  # 使用相对路径
                    "points": [],
                    "affordance_points": []
                }
        
        # 如果有已保存的标注信息，确保包含所有必要字段
        if saved_annotation:
            # 确保物体和动作类别字段存在
            if 'object' not in saved_annotation or not saved_annotation['object']:
                saved_annotation['object'] = "object"
            if 'action' not in saved_annotation or not saved_annotation['action']:
                saved_annotation['action'] = "grasp"
            # 确保其他字段存在
            if 'points' not in saved_annotation:
                saved_annotation['points'] = []
            if 'affordance_points' not in saved_annotation:
                saved_annotation['affordance_points'] = []
            if 'gt_path' not in saved_annotation:
                base_name = Path(filename).stem
                mask_filename = f"{base_name}_mask.png"
                mask_full_path = os.path.join(annotator.output_dir, "masks", mask_filename)
                saved_annotation['gt_path'] = os.path.relpath(mask_full_path, "output")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': image_path,
            'image_data': f'data:image/jpeg;base64,{img_base64}',
            'width': image.shape[1],
            'height': image.shape[0],
            'current_index': annotator.current_image_index,
            'total_count': len(annotator.image_list),
            'saved_annotation': saved_annotation,  # 显示最后一个标注（最新的）
            'all_annotations': saved_annotations,  # 该图片的所有标注（用于显示标注数量）
            'annotation_count': len(saved_annotations)  # 该图片的标注数量
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load_first_image')
def load_first_image():
    """自动加载第一张图像"""
    if not annotator.image_list:
        return jsonify({'error': 'No images found'}), 404
    
    first_image = annotator.image_list[0]
    return load_image(first_image)

@app.route('/load_first_unannotated_image')
def load_first_unannotated_image():
    """自动加载第一张未标注的图像"""
    if not annotator.image_list:
        return jsonify({'error': 'No images found'}), 404
    
    # 获取已标注的图像列表
    annotated_images = [ann['img_name'] for ann in annotator.annotations]
    
    # 检查已存在的mask文件，标记已标注的图片
    mask_dir = os.path.join(annotator.output_dir, "masks")
    if os.path.exists(mask_dir):
        for image_name in annotator.image_list:
            base_name = Path(image_name).stem
            mask_filename = f"{base_name}_mask.png"
            mask_path = os.path.join(mask_dir, mask_filename)
            if os.path.exists(mask_path) and image_name not in annotated_images:
                # 如果存在mask文件但不在标注列表中，也标记为已标注
                annotated_images.append(image_name)
    
    # 找到第一个未标注的图像
    for i, image_name in enumerate(annotator.image_list):
        if image_name not in annotated_images:
            print(f"找到第一个未标注的图像: {image_name} (索引: {i})")
            # 设置当前图像索引
            annotator.current_image_index = i
            return load_image(image_name)
    
    # 如果所有图像都已标注，返回第一张图像
    print("所有图像都已标注，返回第一张图像")
    annotator.current_image_index = 0
    first_image = annotator.image_list[0]
    return load_image(first_image)

@app.route('/get_image_list')
def get_image_list():
    """获取图像列表"""
    return jsonify({
        'images': annotator.image_list,
        'count': len(annotator.image_list),
        'current_index': annotator.current_image_index
    })

@app.route('/segment', methods=['POST'])
def segment():
    """分割物体"""
    data = request.json
    image_path = data.get('image_path')
    points = data.get('points', [])
    labels = data.get('labels', [])
    
    if not image_path or not points:
        return jsonify({'error': 'Missing image path or points'}), 400
    
    try:
        # 1. 读取图像 - 修复图像路径问题
        print(f"分割图像路径: {image_path}")
        
        # 检查图像路径是否存在，如果不存在则尝试重新构造路径
        if not os.path.exists(image_path):
            print(f"图像路径不存在: {image_path}")
            # 尝试从文件名重新构造路径
            filename = os.path.basename(image_path)
            if filename in annotator.image_list:
                image_path = os.path.join(annotator.image_dir, filename)
                print(f"重新构造图像路径: {image_path}")
            else:
                return jsonify({'error': f'Image file not found: {image_path}'}), 404
        
        # 使用缓存获取图像，如果缓存中没有则直接读取
        image = None
        filename = os.path.basename(image_path)
        
        # 尝试从缓存获取
        if filename in annotator.image_cache:
            image = annotator.image_cache[filename]
            print(f"从缓存获取图像: {filename}")
        
        # 如果缓存中没有，直接读取
        if image is None:
            image = cv2.imread(image_path)
            print(f"直接读取图像: {image_path}")
        
        # 检查图像是否成功加载
        if image is None:
            return jsonify({'error': f'Failed to load image: {image_path}'}), 500
        
        print(f"图像形状: {image.shape}")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 使用SAM分割物体
        mask, bbox = annotator.segment_object(image_rgb, points, labels)
        
        # 3. 预测物体和动作类别
        print(f"开始调用类别预测，图像路径: {image_path}")
        object_category, action_category = annotator.predict_categories(image_path, mask)
        print(f"类别预测完成，结果: 物体={object_category}, 动作={action_category}")
        
        # 4. 生成mask对比图（白色背景，黑色物体，红色affordance点）
        mask_image = np.ones_like(image_rgb) * 255  # 白色背景
        mask_image[mask] = [0, 0, 0]  # 黑色物体
        
        # 在mask上只添加affordance点（红色点），不添加绿色和蓝色点
        # 注意：这里我们只显示affordance点，因为绿色和蓝色点是辅助分割用的，不应该显示在最终mask上
        # 如果需要显示affordance点，需要从points中筛选出对应的affordance点
        # 由于当前实现中affordance点单独存储，这里暂时不在mask上显示点
        # 如果需要显示，可以在前端传递affordance_points参数
        
        # 编码掩码为base64
        _, buffer = cv2.imencode('.png', mask_image)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 获取当前图像索引
        current_filename = os.path.basename(image_path)
        current_index = annotator.image_list.index(current_filename) if current_filename in annotator.image_list else 0
        
        return jsonify({
            'success': True,
            'mask_data': f'data:image/png;base64,{mask_base64}',
            'bbox': bbox,
            'mask': mask.tolist(),  # 返回完整的mask数据
            'predicted_object': object_category,
            'predicted_action': action_category,
            'points': points,  # 返回点击的点用于保存
            'current_index': current_index,
            'total_count': len(annotator.image_list)
        })
        
    except Exception as e:
        print(f"分割错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    """保存标注"""
    try:
        data = request.json
        if not data:
            print("Error: No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400
            
        image_path = data.get('image_path')
        bbox = data.get('bbox')
        affordance = data.get('affordance')
        object_category = data.get('object_category')
        action_category = data.get('action_category')
        mask = data.get('mask')
        points = data.get('points', [])  # 分割点
        affordance_points = data.get('affordance_points', [])  # affordance点
        
        print(f"保存标注 - 图像路径: {image_path}")
        print(f"保存标注 - 物体类别: {object_category}")
        print(f"保存标注 - 动作类别: {action_category}")
        print(f"保存标注 - mask类型: {type(mask)}")
        print(f"保存标注 - 分割点数量: {len(points)}")
        print(f"保存标注 - affordance点数量: {len(affordance_points)}")
        
        if not image_path:
            return jsonify({'error': 'Missing image_path'}), 400
        if not bbox:
            return jsonify({'error': 'Missing bbox'}), 400
        if not mask:
            return jsonify({'error': 'Missing mask'}), 400
        if not object_category or not object_category.strip():
            return jsonify({'error': 'Missing or empty object_category'}), 400
        if not action_category or not action_category.strip():
            return jsonify({'error': 'Missing or empty action_category'}), 400
            
        # 转换mask为numpy数组
        try:
            if isinstance(mask, list):
                # 处理实际的mask数据（二维数组）
                print("处理实际的mask数据")
                mask = np.array(mask, dtype=bool)
                print(f"保存标注 - mask形状: {mask.shape}")
            elif isinstance(mask, dict) and 'hasMask' in mask:
                # 处理简化的mask信息 - 只保存mask文件，不保存完整数据
                print("处理简化的mask信息")
                # 这里我们不需要重建完整的mask，因为mask已经保存为文件
                # 只需要创建一个空的mask用于后续处理
                mask = None  # 设为None，让save_annotation方法处理
            else:
                # 处理普通的mask数据
                mask = np.array(mask)
            print(f"保存标注 - mask处理完成")
        except Exception as e:
            print(f"保存标注 - mask转换失败: {e}")
            return jsonify({'error': f'Invalid mask data: {e}'}), 400
        
        # 保存标注
        try:
            annotation = annotator.save_annotation(
                image_path, bbox, affordance, object_category, 
                action_category, mask, points, affordance_points
            )
            print(f"保存标注 - 标注创建成功")
        except Exception as e:
            print(f"保存标注 - 创建标注失败: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Failed to create annotation: {e}'}), 500
        
        # 添加到标注列表
        try:
            # 检查是否已存在该图片和相同物体类别的标注，如果存在则覆盖
            # 同一张图片可以标注多个不同物体，每个物体类别作为独立条目存储
            image_filename = os.path.basename(image_path)
            existing_annotation_index = None
            for i, existing_ann in enumerate(annotator.annotations):
                # 只有当图片名和物体类别都相同时，才认为是同一个标注，需要覆盖
                if existing_ann['img_name'] == image_filename and existing_ann.get('object') == object_category:
                    existing_annotation_index = i
                    break
            
            if existing_annotation_index is not None:
                # 覆盖已存在的标注（同一图片、同一物体类别）
                annotator.annotations[existing_annotation_index] = annotation
                print(f"保存标注 - 覆盖已存在的标注: {image_filename} (物体: {object_category})")
            else:
                # 添加新的标注（新图片或新物体类别）
                annotator.annotations.append(annotation)
                print(f"保存标注 - 添加新标注: {image_filename} (物体: {object_category})")
            
            # 更新记忆数据
            annotator.update_memory(object_category, action_category)
            print(f"更新记忆数据: {object_category} -> {action_category}")
            
            # 保存记忆数据（包括标注数据）
            annotator.save_memory()
            print(f"保存记忆数据成功")
            
            # 自动保存到annotations.json文件
            try:
                output_path = os.path.join(annotator.output_dir, "annotations.json")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(annotator.annotations, f, indent=4, ensure_ascii=False)
                
                print(f"自动保存到annotations.json: {len(annotator.annotations)}条标注")
            except Exception as e:
                print(f"自动保存到annotations.json失败: {e}")
            
        except Exception as e:
            print(f"保存标注 - 添加到列表失败: {e}")
            return jsonify({'error': f'Failed to add annotation to list: {e}'}), 500
        
        # 不再自动跳转到下一张图片，保持当前图片继续标注
        print(f"保存标注成功: {os.path.basename(image_path)}")
        print(f"当前索引: {annotator.current_image_index}")
        print(f"总图片数: {len(annotator.image_list)}")
        print(f"继续在当前图片上标注，可以添加更多标注")
        
        return jsonify({
            'success': True,
            'message': f'标注已保存并自动更新到JSON文件！总计: {len(annotator.annotations)}。可以继续在当前图片上添加更多标注。',
            'current_index': annotator.current_image_index,
            'total_count': len(annotator.image_list),
            'refresh_suggestions': True  # 告诉前端需要刷新建议
        })
        
    except Exception as e:
        print(f"保存标注失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/save_dataset', methods=['POST'])
def save_dataset():
    """保存完整数据集"""
    try:
        output_path = os.path.join(annotator.output_dir, "annotations.json")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 打印调试信息
        print(f"保存数据集 - 标注数量: {len(annotator.annotations)}")
        print(f"保存数据集 - 标注内容: {annotator.annotations}")
        
        # 保存标注数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotator.annotations, f, indent=4, ensure_ascii=False)
        
        print(f"数据集已保存到: {output_path}")
        
        return jsonify({
            'success': True,
            'message': f'数据集已保存到: {output_path} (共{len(annotator.annotations)}条标注)',
            'count': len(annotator.annotations)
        })
        
    except Exception as e:
        print(f"保存数据集失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/delete_annotation', methods=['POST'])
def delete_annotation():
    """删除当前图像的标注和原图文件"""
    try:
        data = request.json
        image_path = data.get('image_path')
        
        if not image_path:
            return jsonify({'error': 'Missing image_path'}), 400
        
        # 获取图像文件名
        image_filename = os.path.basename(image_path)
        
        # 从标注列表中删除，并获取要删除的mask文件路径
        mask_files_to_delete = []
        remaining_annotations = []
        for ann in annotator.annotations:
            if ann['img_name'] == image_filename:
                # 如果标注中有gt_path，记录要删除的mask文件
                if 'gt_path' in ann and ann['gt_path']:
                    mask_full_path = os.path.join("output", ann['gt_path'])
                    if os.path.exists(mask_full_path):
                        mask_files_to_delete.append(mask_full_path)
            else:
                remaining_annotations.append(ann)
        annotator.annotations = remaining_annotations
        
        # 删除所有相关的mask文件（支持新格式和旧格式）
        base_name = Path(image_filename).stem
        mask_dir = os.path.join(annotator.output_dir, "masks")
        # 查找所有以base_name开头的mask文件
        mask_pattern = os.path.join(mask_dir, f"{base_name}_*_mask.png")
        old_mask_pattern = os.path.join(mask_dir, f"{base_name}_mask.png")
        all_matching_masks = glob.glob(mask_pattern) + glob.glob(old_mask_pattern)
        
        # 删除所有找到的mask文件
        for mask_path in all_matching_masks:
            if os.path.exists(mask_path):
                os.remove(mask_path)
                print(f"删除mask文件: {mask_path}")
        
        # 也删除从annotation中获取的mask文件路径
        for mask_path in mask_files_to_delete:
            if os.path.exists(mask_path) and mask_path not in all_matching_masks:
                os.remove(mask_path)
                print(f"删除mask文件: {mask_path}")
        
        # 删除原图文件
        original_image_path = os.path.join(annotator.image_dir, image_filename)
        if os.path.exists(original_image_path):
            os.remove(original_image_path)
            print(f"删除原图文件: {original_image_path}")
            
            # 从图像列表中移除
            if image_filename in annotator.image_list:
                annotator.image_list.remove(image_filename)
                print(f"从图像列表中移除: {image_filename}")
        else:
            print(f"原图文件不存在: {original_image_path}")
        
        # 从图像缓存中移除
        if image_filename in annotator.image_cache:
            del annotator.image_cache[image_filename]
            print(f"从图像缓存中移除: {image_filename}")
        
        # 保存记忆数据（包括更新后的标注数据）
        annotator.save_memory()
        print(f"保存记忆数据成功")
        
        # 自动更新annotations.json文件
        try:
            output_path = os.path.join(annotator.output_dir, "annotations.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(annotator.annotations, f, indent=4, ensure_ascii=False)
            
            print(f"自动更新annotations.json: {len(annotator.annotations)}条标注")
        except Exception as e:
            print(f"自动更新annotations.json失败: {e}")
        
        # 删除后不自动切换到下一张图片，保持当前索引
        # 如果当前索引超出范围，则设置为0
        if annotator.current_image_index >= len(annotator.image_list):
            annotator.current_image_index = 0
        
        print(f"删除图片成功: {image_filename}")
        print(f"当前索引: {annotator.current_image_index}")
        print(f"剩余图片数: {len(annotator.image_list)}")
        
        return jsonify({
            'success': True,
            'message': f'图片已删除！总计: {len(annotator.annotations)}',
            'current_index': annotator.current_image_index,
            'total_count': len(annotator.image_list)
        })
        
    except Exception as e:
        print(f"删除图片失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/delete_mask', methods=['POST'])
def delete_mask():
    """根据图片名称和物体名称删除对应的mask和标注记录"""
    try:
        data = request.json
        image_name = data.get('image_name')
        object_category = data.get('object_category')
        
        if not image_name or not image_name.strip():
            return jsonify({'error': 'Missing image_name'}), 400
        if not object_category or not object_category.strip():
            return jsonify({'error': 'Missing or empty object_category'}), 400
        
        image_name = image_name.strip()
        object_category = object_category.strip()
        
        # 验证图片名称是否有效
        if image_name not in annotator.image_list:
            return jsonify({'error': f'Image not found: {image_name}. Available images: {annotator.image_list[:10]}...'}), 400
        
        # 使用提供的图片文件名
        image_filename = image_name
        base_name = Path(image_filename).stem
        
        # 清理物体类别名称，用于匹配文件名
        safe_object_category = object_category.replace('/', '_').replace('\\', '_').replace(' ', '_')
        mask_filename = f"{base_name}_{safe_object_category}_mask.png"
        mask_path = os.path.join(annotator.output_dir, "masks", mask_filename)
        
        deleted_mask = False
        deleted_annotations = 0
        
        # 删除mask文件
        if os.path.exists(mask_path):
            os.remove(mask_path)
            deleted_mask = True
            print(f"删除mask文件: {mask_path}")
        else:
            print(f"mask文件不存在: {mask_path}")
        
        # 从标注列表中删除匹配的记录
        remaining_annotations = []
        for ann in annotator.annotations:
            if ann['img_name'] == image_filename and ann.get('object') == object_category:
                deleted_annotations += 1
                print(f"从标注列表中删除: {image_filename} - {object_category}")
            else:
                remaining_annotations.append(ann)
        
        annotator.annotations = remaining_annotations
        
        # 更新memory文件中的标注数据
        annotator.save_memory()
        
        # 自动更新annotations.json文件
        try:
            output_path = os.path.join(annotator.output_dir, "annotations.json")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(annotator.annotations, f, indent=4, ensure_ascii=False)
            
            print(f"自动更新annotations.json: {len(annotator.annotations)}条标注")
        except Exception as e:
            print(f"自动更新annotations.json失败: {e}")
        
        message_parts = []
        if deleted_mask:
            message_parts.append(f"已删除mask文件")
        if deleted_annotations > 0:
            message_parts.append(f"已删除{deleted_annotations}条标注记录")
        
        if not deleted_mask and deleted_annotations == 0:
            return jsonify({
                'success': False,
                'message': f'未找到对应的mask文件或标注记录: {image_filename} - {object_category}',
                'current_index': annotator.current_image_index,
                'total_count': len(annotator.image_list),
                'remaining_annotations': len(annotator.annotations)
            })
        
        return jsonify({
            'success': True,
            'message': f'删除成功！{", ".join(message_parts)}。剩余标注: {len(annotator.annotations)}条',
            'current_index': annotator.current_image_index,
            'total_count': len(annotator.image_list),
            'remaining_annotations': len(annotator.annotations)
        })
        
    except Exception as e:
        print(f"删除mask失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/get_annotations', methods=['GET'])
def get_annotations():
    """获取所有标注"""
    return jsonify({
        'annotations': annotator.annotations,
        'count': len(annotator.annotations)
    })

@app.route('/get_suggestions', methods=['GET'])
def get_suggestions():
    """获取建议的物体和动作类别"""
    try:
        object_suggestions, action_suggestions = annotator.get_suggestions()
        return jsonify({
            'success': True,
            'object_suggestions': object_suggestions,
            'action_suggestions': action_suggestions,
            'object_memory': annotator.object_memory,
            'action_memory': annotator.action_memory
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/output/masks/<filename>')
def get_mask_file(filename):
    """获取mask文件"""
    mask_path = os.path.join(annotator.output_dir, "masks", filename)
    if os.path.exists(mask_path):
        return send_from_directory(os.path.join(annotator.output_dir, "masks"), filename)
    else:
        return jsonify({'error': 'Mask file not found'}), 404

@app.route('/output/<task_name>/masks/<filename>')
def get_task_mask_file(task_name, filename):
    """动态获取任意任务目录下的mask文件"""
    mask_path = os.path.join("output", task_name, "masks", filename)
    if os.path.exists(mask_path):
        return send_from_directory(os.path.join("output", task_name, "masks"), filename)
    else:
        return jsonify({'error': 'Mask file not found'}), 404

@app.route('/output/task_0/masks/<filename>')
def get_task0_mask_file(filename):
    """获取task_0目录下的mask文件（保持向后兼容）"""
    mask_path = os.path.join(annotator.output_dir, "masks", filename)
    if os.path.exists(mask_path):
        return send_from_directory(os.path.join(annotator.output_dir, "masks"), filename)
    else:
        return jsonify({'error': 'Mask file not found'}), 404

@app.route('/get_output_info', methods=['GET'])
def get_output_info():
    """获取输出目录信息"""
    return jsonify({
        'output_dir': annotator.output_dir,
        'masks_dir': os.path.join(annotator.output_dir, "masks")
    })

def create_html_template():
    """创建HTML模板"""
    template_dir = 'templates'
    os.makedirs(template_dir, exist_ok=True)
    
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <title>可视化标注工具</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { display: flex; gap: 20px; }
        .control-panel { width: 300px; }
        .image-panel { flex: 1; }
        .image-display { display: flex; gap: 10px; }
        .canvas-container { position: relative; border: 1px solid #ccc; }
        canvas { border: 1px solid #ddd; }
        .button { margin: 5px; padding: 8px 16px; background: #007bff; color: white; border: none; cursor: pointer; }
        .button:hover { background: #0056b3; }
        .button.success { background: #28a745; }
        .button.danger { background: #dc3545; }
        .button.danger:hover { background: #c82333; }
        .input-group { margin: 10px 0; }
        label { display: block; margin-bottom: 5px; }
        select, input { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }
        input:focus { outline: none; border-color: #007bff; box-shadow: 0 0 0 2px rgba(0,123,255,0.25); }
        .status { margin: 10px 0; padding: 10px; background: #f8f9fa; }
        .image-list { max-height: 200px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; }
        .image-item { padding: 5px; cursor: pointer; border-bottom: 1px solid #eee; }
        .image-item:hover { background: #f0f0f0; }
        .image-item.active { background: #007bff; color: white; }
        .image-item.annotated { background: #e8f5e8; border-left: 4px solid #28a745; }
        .image-item.annotated .annotation-status { color: #28a745; font-weight: bold; }
        .progress { margin: 10px 0; }
        .progress-bar { width: 100%; height: 20px; background: #f0f0f0; border-radius: 10px; overflow: hidden; }
        .progress-fill { height: 100%; background: #007bff; transition: width 0.3s; }
        .predicted-category { background: #e7f3ff; padding: 5px; margin: 5px 0; border-radius: 3px; }
        .suggestions { background: #fff3cd; padding: 5px; margin: 5px 0; border-radius: 3px; border-left: 3px solid #ffc107; }
        .suggestions small { color: #856404; }
        /* Toast通知样式 */
        .toast {
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: #28a745;
            color: white;
            padding: 12px 24px;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            font-size: 16px;
            font-weight: 500;
            opacity: 0;
            transition: opacity 0.3s ease-in-out, transform 0.3s ease-in-out;
            pointer-events: none;
        }
        .toast.show {
            opacity: 1;
            transform: translateX(-50%) translateY(0);
        }
        .toast.hide {
            opacity: 0;
            transform: translateX(-50%) translateY(-20px);
        }
    </style>
</head>
<body>
    <h1>可视化标注工具</h1>
    
    <div class="container">
        <div class="control-panel">
            <h3>控制面板</h3>
            
            <div class="progress">
                <div>进度: <span id="progressText">{{ current_index + 1 }}/{{ image_list|length }}</span></div>
                <div class="progress-bar">
                    <div class="progress-fill" id="progressBar" style="width: {{ ((current_index + 1) / image_list|length * 100) if image_list else 0 }}%"></div>
                </div>
            </div>
            
            <div class="input-group">
                <label>选择图像:</label>
                <div class="image-list" id="imageList">
                    {% for image in image_list %}
                    <div class="image-item {% if loop.index0 == current_index %}active{% endif %} {% if image in annotated_images %}annotated{% endif %}" onclick="loadImage('{{ image }}')">
                        {{ image }}
                        <span class="annotation-status" id="status-{{ image }}" style="display: none; color: green; font-size: 12px;">✓ 已标注</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div class="input-group">
                <label>点击模式: <small style="color: #666;">(按Q键切换)</small></label>
                <input type="radio" name="mode" value="affordance" checked> 交互点(红) - 主要分割点
                <input type="radio" name="mode" value="foreground"> 前景点(绿) - 辅助分割
                <input type="radio" name="mode" value="background"> 背景点(蓝) - 排除区域
            </div>
            
            <div class="input-group">
                <label>物体类别:</label>
                <div class="predicted-category" id="predictedObject" style="display: none;">
                    预测: <span id="predictedObjectText"></span>
                </div>
                <div class="suggestions" id="objectSuggestions" style="display: none;">
                    <small>常用: <span id="objectSuggestionsText"></span></small>
                </div>
                <input type="text" id="objectCategory" list="objectCategories" placeholder="输入或选择物体类别，按TAB键智能补全(优先常用类别)" value="object">
                <datalist id="objectCategories">
                    {% for category in object_categories %}
                    <option value="{{ category }}">
                    {% endfor %}
                </datalist>
            </div>
            
            <div class="input-group">
                <label>动作类别:</label>
                <div class="predicted-category" id="predictedAction" style="display: none;">
                    预测: <span id="predictedActionText"></span>
                </div>
                <div class="suggestions" id="actionSuggestions" style="display: none;">
                    <small>常用: <span id="actionSuggestionsText"></span></small>
                </div>
                <input type="text" id="actionCategory" list="actionCategories" placeholder="输入或选择动作类别，按TAB键智能补全(优先常用类别)" value="grasp">
                <datalist id="actionCategories">
                    {% for category in action_categories %}
                    <option value="{{ category }}">
                    {% endfor %}
                </datalist>
            </div>
            
            <div class="input-group">
                <button class="button" onclick="segmentObject()">分割物体</button>
                <button class="button success" onclick="saveAnnotation()">保存标注(自动更新JSON)</button>
                <button class="button danger" onclick="deleteAnnotation()">删除原图</button>
                <button class="button" onclick="clearPoints()">清除点</button>
                <button class="button" onclick="skipCurrentImage()">跳过当前图片</button>
                <button class="button" onclick="saveDataset()">保存数据集</button>
            </div>
            
            <div class="input-group">
                <label>删除Mask (按图片名称和物体名称):</label>
                <div style="display: flex; gap: 5px; flex-direction: column;">
                    <input type="text" id="deleteMaskImageName" list="imageNameList" placeholder="图片名称(支持自动补全)" style="width: 100%;">
                    <datalist id="imageNameList">
                        {% for image in image_list %}
                        <option value="{{ image }}">
                        {% endfor %}
                    </datalist>
                    <input type="text" id="deleteMaskObjectCategory" list="objectCategories" placeholder="物体名称(支持自动补全)" style="width: 100%;">
                </div>
                <button class="button danger" onclick="deleteMask()" style="margin-top: 5px; width: 100%;">删除Mask</button>
            </div>
            
            <div class="status" id="status">就绪</div>
        </div>
        
        <div class="image-panel">
            <h3>图像显示</h3>
            <div class="image-display">
                <div class="canvas-container">
                    <h4>原图</h4>
                    <canvas id="imageCanvas" width="800" height="600"></canvas>
                </div>
                <div class="canvas-container">
                    <h4>分割掩码</h4>
                    <canvas id="maskCanvas" width="800" height="600"></canvas>
                </div>
            </div>
        </div>
    </div>

    <!-- Toast通知容器 -->
    <div id="toast" class="toast"></div>

    <script>
        let currentImage = null;
        let currentImagePath = null;
        let points = [];  // 分割点
        let labels = [];  // 分割点标签
        let affordancePoints = [];  // affordance点
        let currentMask = null;
        let currentBbox = null;
        let canvas = document.getElementById('imageCanvas');
        let maskCanvas = document.getElementById('maskCanvas');
        let ctx = canvas.getContext('2d');
        let maskCtx = maskCanvas.getContext('2d');
        let imageScale = 1.0;
        let imageOffsetX = 0;
        let imageOffsetY = 0;
        let outputInfo = null;  // 存储输出目录信息
        
        // Toast通知函数
        function showToast(message, duration = 2000) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.classList.remove('hide');
            toast.classList.add('show');
            
            // 延迟后自动隐藏
            setTimeout(() => {
                toast.classList.remove('show');
                toast.classList.add('hide');
                // 动画结束后移除类
                setTimeout(() => {
                    toast.classList.remove('hide');
                }, 300);
            }, duration);
        }
        
        // 获取输出目录信息
        function getOutputInfo() {
            fetch('/get_output_info')
            .then(response => response.json())
            .then(data => {
                outputInfo = data;
                console.log('输出目录信息:', outputInfo);
            })
            .catch(error => {
                console.error('获取输出目录信息失败:', error);
                // 使用默认路径
                outputInfo = {
                    output_dir: 'output/default',
                    masks_dir: 'output/default/masks'
                };
            });
        }
        
        // 获取mask文件路径
        function getMaskPath(filename) {
            if (outputInfo) {
                const baseName = filename.replace(/\.[^/.]+$/, "");
                // 使用outputInfo中的masks_dir，但需要转换为URL路径
                const masksDir = outputInfo.masks_dir.replace('output/', 'output/');
                return `/${masksDir}/${baseName}_mask.png`;
            } else {
                // 使用默认路径
                const baseName = filename.replace(/\.[^/.]+$/, "");
                return `/output/masks/${baseName}_mask.png`;
            }
        }
        
        // 从gt_path获取mask文件路径
        function getMaskPathFromGtPath(gtPath) {
            // 检查是否是任务目录格式：task_X/masks/... 或其他目录/masks/...
            const taskMatch = gtPath.match(/^(task_\d+)\/masks\//);
            if (taskMatch) {
                // 新的路径格式：task_X/masks/...
                return `/output/${gtPath}`;
            } else if (gtPath.includes('/masks/')) {
                // 格式：目录名/masks/文件名 (如 robot_arm_02/masks/1_object_mask.png)
                // 直接拼接output前缀
                return `/output/${gtPath}`;
            } else if (gtPath.startsWith('output/')) {
                // 完整路径格式：output/task_X/masks/...
                return `/${gtPath}`;
            } else {
                // 旧路径格式：masks/... 或直接文件名
                return `/output/masks/${gtPath}`;
            }
        }
        
        function updateAnnotationStatuses() {
            // 获取所有标注信息
            fetch('/get_annotations')
            .then(response => response.json())
            .then(data => {
                const annotations = data.annotations || [];
                const annotatedImages = new Set(annotations.map(a => a.img_name));
                
                // 计算每张图片的标注数量
                const annotationCounts = {};
                annotations.forEach(ann => {
                    const imgName = ann.img_name;
                    annotationCounts[imgName] = (annotationCounts[imgName] || 0) + 1;
                });
                
                // 更新所有图像的状态
                document.querySelectorAll('.image-item').forEach(item => {
                    const imageName = item.textContent.split('✓')[0].trim(); // 移除状态文本
                    const statusElement = item.querySelector('.annotation-status');
                    
                    // 只检查已标注的图像是否有对应的mask文件
                    if (annotatedImages.has(imageName)) {
                        // 获取该图片的标注数量
                        const count = annotationCounts[imageName] || 0;
                        
                        // 找到对应的标注信息（用于获取gt_path）
                        const annotation = annotations.find(a => a.img_name === imageName);
                        let maskPath;
                        
                        if (annotation && annotation.gt_path) {
                            // 使用标注中的gt_path
                            maskPath = getMaskPathFromGtPath(annotation.gt_path);
                        } else {
                            // 使用默认路径
                            maskPath = getMaskPath(imageName);
                        }
                        
                        // 检查mask文件是否存在
                        fetch(maskPath, { method: 'HEAD' })
                        .then(response => {
                            const hasMaskFile = response.ok;
                            
                            if (hasMaskFile) {
                                if (statusElement) {
                                    // 显示标注数量：已标注xx张
                                    statusElement.textContent = count > 1 ? `✓ 已标注${count}张` : '✓ 已标注';
                                    statusElement.style.display = 'inline-block';
                                    statusElement.style.color = 'green';
                                }
                                item.classList.add('annotated'); // 添加已标注样式
                            } else {
                                // 有标注但没有mask文件，显示警告
                                if (statusElement) {
                                    statusElement.textContent = count > 1 ? `⚠ 标注不完整(${count}张)` : '⚠ 标注不完整';
                                    statusElement.style.display = 'inline-block';
                                    statusElement.style.color = 'orange';
                                }
                                item.classList.add('annotated');
                            }
                        })
                        .catch(() => {
                            // 检查失败，假设已标注
                            if (statusElement) {
                                // 显示标注数量：已标注xx张
                                statusElement.textContent = count > 1 ? `✓ 已标注${count}张` : '✓ 已标注';
                                statusElement.style.display = 'inline-block';
                                statusElement.style.color = 'green';
                            }
                            item.classList.add('annotated');
                        });
                    } else {
                        // 未标注的图像
                        if (statusElement) {
                            statusElement.style.display = 'none';
                        }
                        item.classList.remove('annotated'); // 移除已标注样式
                    }
                });
            })
            .catch(error => {
                console.error('获取标注状态失败:', error);
            });
        }
        
        // 页面加载时更新标注状态
        window.onload = function() {
            getOutputInfo();  // 获取输出目录信息
            loadFirstUnannotatedImage();
            // 延迟更新标注状态，确保DOM已加载
            setTimeout(updateAnnotationStatuses, 100);
            // 加载建议
            loadSuggestions();
            // 设置TAB补全
            setupTabCompletion();
        };
        
        function updateStatus(message) {
            document.getElementById('status').textContent = message;
        }
        
        function updateProgress(current, total) {
            const progressText = document.getElementById('progressText');
            const progressBar = document.getElementById('progressBar');
            progressText.textContent = `${current}/${total}`;
            progressBar.style.width = `${(current / total) * 100}%`;
        }
        
        function loadFirstImage() {
            fetch('/load_first_image')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadImageData(data);
                } else {
                    updateStatus('加载第一张图片失败: ' + data.error);
                }
            })
            .catch(error => {
                updateStatus('加载第一张图片错误: ' + error);
            });
        }
        
        function loadFirstUnannotatedImage() {
            fetch('/load_first_unannotated_image')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadImageData(data);
                } else {
                    updateStatus('加载第一个未标注图片失败: ' + data.error);
                }
            })
            .catch(error => {
                updateStatus('加载第一个未标注图片错误: ' + error);
            });
        }
        
        function loadImageData(data) {
            currentImagePath = data.filepath;
            currentImage = new Image();
            currentImage.onload = function() {
                // 计算缩放比例以适应画布
                const canvasWidth = 800;
                const canvasHeight = 600;
                const scaleX = canvasWidth / this.width;
                const scaleY = canvasHeight / this.height;
                imageScale = Math.min(scaleX, scaleY);
                
                // 计算居中偏移
                const scaledWidth = this.width * imageScale;
                const scaledHeight = this.height * imageScale;
                imageOffsetX = (canvasWidth - scaledWidth) / 2;
                imageOffsetY = (canvasHeight - scaledHeight) / 2;
                
                // 设置画布尺寸
                canvas.width = canvasWidth;
                canvas.height = canvasHeight;
                maskCanvas.width = canvasWidth;
                maskCanvas.height = canvasHeight;
                
                // 绘制图像
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(this, imageOffsetX, imageOffsetY, scaledWidth, scaledHeight);
                
                // 清除mask canvas
                maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                
                // 重置状态
                points = [];
                labels = [];
                affordancePoints = [];
                currentMask = null;
                currentBbox = null;
                
                // 自动切换回红色点标注模式（交互点）
                document.querySelector('input[name="mode"][value="affordance"]').checked = true;
                
                // 隐藏预测类别
                document.getElementById('predictedObject').style.display = 'none';
                document.getElementById('predictedAction').style.display = 'none';
                
                // 处理已保存的标注信息
                if (data.saved_annotation) {
                    console.log('找到已保存的标注:', data.saved_annotation);
                    // 设置物体类别
                    document.getElementById('objectCategory').value = data.saved_annotation.object || 'object';
                    // 设置动作类别
                    document.getElementById('actionCategory').value = data.saved_annotation.action || 'grasp';
                    
                    // 恢复保存的点信息
                    if (data.saved_annotation.points && data.saved_annotation.points.length > 0) {
                        points = data.saved_annotation.points;
                        labels = points.map(() => 1); // 假设所有点都是前景点
                        console.log('恢复保存的分割点:', points);
                    }
                    
                    if (data.saved_annotation.affordance_points && data.saved_annotation.affordance_points.length > 0) {
                        affordancePoints = data.saved_annotation.affordance_points;
                        console.log('恢复保存的affordance点:', affordancePoints);
                    }
                    
                    // 显示已保存的标注信息
                    const annotationCount = data.annotation_count || 0;
                    const statusText = annotationCount > 1 
                        ? `已加载图像: ${data.filename} (当前显示: ${data.saved_annotation.object} - ${data.saved_annotation.action}, 共${annotationCount}个标注)`
                        : `已加载图像: ${data.filename} (已标注: ${data.saved_annotation.object} - ${data.saved_annotation.action})`;
                    updateStatus(statusText);
                    // 显示已标注状态（更新标注数量）
                    const statusElement = document.getElementById(`status-${data.filename}`);
                    if (statusElement) {
                        const annotationCount = data.annotation_count || 0;
                        if (annotationCount > 1) {
                            statusElement.textContent = `✓ 已标注${annotationCount}张`;
                        } else {
                            statusElement.textContent = '✓ 已标注';
                        }
                        statusElement.style.display = 'inline-block';
                        statusElement.style.color = 'green';
                    }
                    // 添加已标注样式
                    const imageItem = document.querySelector(`[onclick*="${data.filename}"]`);
                    if (imageItem) {
                        imageItem.classList.add('annotated');
                    }
                    
                    // 重新绘制保存的点
                    setTimeout(() => {
                        // 重新绘制图像
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(currentImage, imageOffsetX, imageOffsetY, 
                                    currentImage.width * imageScale, 
                                    currentImage.height * imageScale);
                        
                        // 绘制保存的点
                        points.forEach((point, index) => {
                            const color = labels[index] === 1 ? 'green' : 'blue';
                            drawPoint(canvas, point[0], point[1], color);
                        });
                        
                        affordancePoints.forEach(point => {
                            drawPoint(canvas, point[0], point[1], 'red');
                        });
                        
                        // 如果有mask文件，自动显示mask
                        if (data.saved_annotation.gt_path) {
                            const maskImg = new Image();
                            maskImg.onload = function() {
                                maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                                maskCtx.drawImage(this, imageOffsetX, imageOffsetY, 
                                                currentImage.width * imageScale, 
                                                currentImage.height * imageScale);
                                
                                // 在mask上绘制affordance点
                                affordancePoints.forEach(point => {
                                    drawPoint(maskCanvas, point[0], point[1], 'red');
                                });
                            };
                            // 使用gt_path中的相对路径，转换为完整的URL路径
                            const gtPath = data.saved_annotation.gt_path;
                            const maskUrl = getMaskPathFromGtPath(gtPath);
                            maskImg.src = maskUrl;
                        }
                    }, 100);
                } else {
                    // 检查是否有mask文件
                    const baseName = data.filename.replace(/\.[^/.]+$/, "");
                    const maskPath = getMaskPath(data.filename);
                    
                    // 基于文件名预测默认类别
                    const predictedCategories = predictCategoriesFromFilename(data.filename);
                    
                    fetch(maskPath, { method: 'HEAD' })
                    .then(response => {
                        if (response.ok) {
                            // 有mask文件但没有标注信息，显示为已标注
                            updateStatus(`已加载图像: ${data.filename} (已有mask文件)`);
                            // 通过updateAnnotationStatuses更新标注数量（如果有的话）
                            updateAnnotationStatuses();
                            const statusElement = document.getElementById(`status-${data.filename}`);
                            if (statusElement && !statusElement.textContent) {
                                // 如果updateAnnotationStatuses没有更新，则显示默认状态
                                statusElement.textContent = '✓ 已标注';
                                statusElement.style.display = 'inline-block';
                                statusElement.style.color = 'green';
                            }
                            const imageItem = document.querySelector(`[onclick*="${data.filename}"]`);
                            if (imageItem) {
                                imageItem.classList.add('annotated');
                            }
                        } else {
                            // 没有mask文件，优先使用频率最高的类别，如果没有则使用预测的默认值
                            let defaultObject = predictedCategories.object;
                            let defaultAction = predictedCategories.action;
                            
                            // 尝试从建议中获取频率最高的类别（只使用真正有使用记录的）
                            if (window.objectSuggestions && window.objectSuggestions.length > 0) {
                                defaultObject = window.objectSuggestions[0];
                            }
                            if (window.actionSuggestions && window.actionSuggestions.length > 0) {
                                defaultAction = window.actionSuggestions[0];
                            }
                            
                            document.getElementById('objectCategory').value = defaultObject;
                            document.getElementById('actionCategory').value = defaultAction;
                            updateStatus(`已加载图像: ${data.filename} (默认: ${defaultObject} - ${defaultAction})`);
                            const statusElement = document.getElementById(`status-${data.filename}`);
                            if (statusElement) {
                                statusElement.style.display = 'none';
                            }
                            const imageItem = document.querySelector(`[onclick*="${data.filename}"]`);
                            if (imageItem) {
                                imageItem.classList.remove('annotated');
                            }
                        }
                    })
                    .catch(() => {
                        // 检查失败，优先使用频率最高的类别，如果没有则使用预测的默认值
                        let defaultObject = predictedCategories.object;
                        let defaultAction = predictedCategories.action;
                        
                        // 尝试从建议中获取频率最高的类别（只使用真正有使用记录的）
                        if (window.objectSuggestions && window.objectSuggestions.length > 0) {
                            defaultObject = window.objectSuggestions[0];
                        }
                        if (window.actionSuggestions && window.actionSuggestions.length > 0) {
                            defaultAction = window.actionSuggestions[0];
                        }
                        
                        document.getElementById('objectCategory').value = defaultObject;
                        document.getElementById('actionCategory').value = defaultAction;
                        updateStatus(`已加载图像: ${data.filename} (默认: ${defaultObject} - ${defaultAction})`);
                        const statusElement = document.getElementById(`status-${data.filename}`);
                        if (statusElement) {
                            statusElement.style.display = 'none';
                        }
                        const imageItem = document.querySelector(`[onclick*="${data.filename}"]`);
                        if (imageItem) {
                            imageItem.classList.remove('annotated');
                        }
                    });
                }
                
                // 检查是否有已保存的mask
                const baseName = data.filename.replace(/\.[^/.]+$/, "");
                const maskPath = getMaskPath(data.filename);
                
                // 尝试加载已保存的mask
                const maskImg = new Image();
                maskImg.onload = function() {
                    maskCtx.drawImage(this, imageOffsetX, imageOffsetY, 
                                    currentImage.width * imageScale, 
                                    currentImage.height * imageScale);
                    if (data.saved_annotation) {
                        updateStatus(`已加载图像和mask: ${data.filename} (已标注: ${data.saved_annotation.object} - ${data.saved_annotation.action})`);
                    } else {
                        updateStatus(`已加载图像和mask: ${data.filename}`);
                    }
                };
                maskImg.onerror = function() {
                    if (data.saved_annotation) {
                        updateStatus(`已加载图像: ${data.filename} (已标注: ${data.saved_annotation.object} - ${data.saved_annotation.action})`);
                    } else {
                        updateStatus(`已加载图像: ${data.filename}`);
                    }
                };
                // 如果有已保存的标注信息，优先使用gt_path
                if (data.saved_annotation && data.saved_annotation.gt_path) {
                    const gtPath = data.saved_annotation.gt_path;
                    maskImg.src = getMaskPathFromGtPath(gtPath);
                } else {
                    maskImg.src = maskPath;
                }
                
                // 更新选中状态
                document.querySelectorAll('.image-item').forEach(item => {
                    item.classList.remove('active');
                });
                const currentItem = document.querySelector(`[onclick*="${data.filename}"]`);
                if (currentItem) {
                    currentItem.classList.add('active');
                }
                
                updateProgress(data.current_index + 1, data.total_count);
            };
            currentImage.src = data.image_data;
        }
        
        function loadImage(filename) {
            // 更新选中状态
            document.querySelectorAll('.image-item').forEach(item => {
                item.classList.remove('active');
            });
            event.target.classList.add('active');
            
            fetch(`/load_image/${filename}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    loadImageData(data);
                } else {
                    updateStatus('加载失败: ' + data.error);
                }
            })
            .catch(error => {
                updateStatus('加载错误: ' + error);
            });
        }
        
        function getMousePos(canvas, evt) {
            const rect = canvas.getBoundingClientRect();
            const x = evt.clientX - rect.left;
            const y = evt.clientY - rect.top;
            
            // 转换到图像坐标系
            const imageX = (x - imageOffsetX) / imageScale;
            const imageY = (y - imageOffsetY) / imageScale;
            
            return { x: imageX, y: imageY };
        }
        
        function drawPoint(canvas, x, y, color) {
            const ctx = canvas.getContext('2d');
            const canvasX = x * imageScale + imageOffsetX;
            const canvasY = y * imageScale + imageOffsetY;
            
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(canvasX, canvasY, 5, 0, 2 * Math.PI);
            ctx.fill();
        }
        
        canvas.addEventListener('click', function(evt) {
            if (!currentImage) {
                updateStatus('请先选择图像');
                return;
            }
            
            const pos = getMousePos(canvas, evt);
            const mode = document.querySelector('input[name="mode"]:checked').value;
            
            // 检查点击是否在图像范围内
            if (pos.x < 0 || pos.y < 0 || pos.x > currentImage.width || pos.y > currentImage.height) {
                return;
            }
            
            // 添加点
            if (mode === 'foreground') {
                points.push([pos.x, pos.y]);
                labels.push(1); // 前景点
            } else if (mode === 'background') {
                points.push([pos.x, pos.y]);
                labels.push(0); // 背景点
            } else if (mode === 'affordance') {
                // affordance点只记录为affordance点，不用于分割
                affordancePoints.push([pos.x, pos.y]); // 只记录为affordance点
            }
            
            // 绘制点
            const color = mode === 'foreground' ? 'green' : (mode === 'background' ? 'blue' : 'red');
            drawPoint(canvas, pos.x, pos.y, color);
            
            updateStatus(`添加${mode === 'foreground' ? '前景' : (mode === 'background' ? '背景' : '交互')}点: (${Math.round(pos.x)}, ${Math.round(pos.y)}) - ${mode === 'affordance' ? '主要分割点' : (mode === 'foreground' ? '辅助分割点' : '排除区域')}`);
        });
        
        function segmentObject() {
            if (!currentImagePath || (points.length === 0 && affordancePoints.length === 0)) {
                updateStatus('请先选择图像并添加分割点（前景点/交互点或背景点）');
                return;
            }
            
            // 如果没有分割点，使用affordance点作为前景点
            let segmentPoints = points;
            let segmentLabels = labels;
            if (points.length === 0 && affordancePoints.length > 0) {
                segmentPoints = affordancePoints;
                segmentLabels = affordancePoints.map(() => 1); // 所有affordance点作为前景点
            }
            
            updateStatus('正在分割物体...');
            
            fetch('/segment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_path: currentImagePath,
                    points: segmentPoints,
                    labels: segmentLabels
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // 保存完整的mask数据
                    currentMask = data.mask;
                    currentBbox = data.bbox;
                    
                    // 显示掩码
                    const maskImg = new Image();
                    maskImg.onload = function() {
                        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
                        maskCtx.drawImage(this, imageOffsetX, imageOffsetY, 
                                        currentImage.width * imageScale, 
                                        currentImage.height * imageScale);
                        
                        // 只在mask上绘制红色affordance点，不绘制绿色和蓝色点
                        // 绿色和蓝色点是辅助分割用的，不应该显示在最终mask上
                        affordancePoints.forEach(point => {
                            drawPoint(maskCanvas, point[0], point[1], 'red');
                        });
                    };
                    maskImg.src = data.mask_data;
                    
                    // 更新物体类别 - 只在用户没有输入内容时才使用预测的类别
                    if (data.predicted_object && data.predicted_object !== 'unknown') {
                        document.getElementById('predictedObjectText').textContent = data.predicted_object;
                        document.getElementById('predictedObject').style.display = 'block';
                        console.log('预测的物体类别:', data.predicted_object, '但保留用户输入');
                    } else {
                        console.log('物体类别预测失败或为空:', data.predicted_object);
                    }
                    
                    // 更新动作类别 - 只在用户没有输入内容时才使用预测的类别
                    if (data.predicted_action && data.predicted_action !== 'unknown') {
                        document.getElementById('predictedActionText').textContent = data.predicted_action;
                        document.getElementById('predictedAction').style.display = 'block';
                        console.log('预测的动作类别:', data.predicted_action, '但保留用户输入');
                    } else {
                        console.log('动作类别预测失败或为空:', data.predicted_action);
                    }
                    
                    updateStatus('分割完成！');
                } else {
                    updateStatus('分割失败: ' + data.error);
                }
            })
            .catch(error => {
                updateStatus('分割错误: ' + error);
            });
        }
        
        function saveAnnotation() {
            if (!currentImagePath || !currentMask) {
                updateStatus('请先完成分割');
                return;
            }
            
            const affordance = affordancePoints.length > 0 ? [affordancePoints[0][0] / currentImage.width, affordancePoints[0][1] / currentImage.height] : [0.5, 0.5];
            
            updateStatus('正在保存标注...');
            
            // 直接使用现有的mask数据
            saveAnnotationWithMask(currentMask);
        }
        
        function saveAnnotationWithMask(maskData) {
            const affordance = affordancePoints.length > 0 ? [affordancePoints[0][0] / currentImage.width, affordancePoints[0][1] / currentImage.height] : [0.5, 0.5];
            
            fetch('/save_annotation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_path: currentImagePath,
                    bbox: currentBbox,
                    affordance: affordance,
                    object_category: document.getElementById('objectCategory').value.trim(),
                    action_category: document.getElementById('actionCategory').value.trim(),
                    mask: maskData,  // 使用原始mask数据
                    points: points,
                    affordance_points: affordancePoints
                })
            })
            .then(response => {
                // 检查响应类型
                const contentType = response.headers.get('content-type');
                if (!contentType || !contentType.includes('application/json')) {
                    throw new Error(`Expected JSON response but got ${contentType}`);
                }
                return response.json();
            })
            .then(data => {
                if (data.success) {
                    updateStatus(data.message);
                    console.log('保存成功，继续在当前图片上标注');
                    
                    // 显示顶部提示"已保存"
                    showToast('已保存', 2000);
                    
                    // 如果需要刷新建议，重新加载
                    if (data.refresh_suggestions) {
                        console.log('刷新建议数据...');
                        loadSuggestions();
                    }
                    
                    // 更新标注状态显示（会显示标注数量）
                    updateAnnotationStatuses();
                    
                    // 不再自动切换，保持当前图片继续标注
                    // 清除当前的点，准备添加新的标注
                    points = [];
                    labels = [];
                    affordancePoints = [];
                    currentMask = null;
                    currentBbox = null;
                    
                    // 重新绘制图像（清除点）
                    if (currentImage) {
                        ctx.clearRect(0, 0, canvas.width, canvas.height);
                        ctx.drawImage(currentImage, imageOffsetX, imageOffsetY, 
                                    currentImage.width * imageScale, 
                                    currentImage.height * imageScale);
                    }
                    
                    // 保持mask显示，但清除点标记
                    console.log('已保存标注，可以继续添加新的标注点');
                } else {
                    updateStatus('保存失败: ' + data.error);
                }
            })
            .catch(error => {
                updateStatus('保存错误: ' + error);
                console.error('保存错误:', error);
            });
        }
        
        // RLE压缩mask数据
        function compressMask(mask) {
            const height = mask.length;
            const width = mask[0].length;
            const compressed = [];
            
            for (let i = 0; i < height; i++) {
                let count = 0;
                let current = mask[i][0];
                
                for (let j = 0; j < width; j++) {
                    if (mask[i][j] === current) {
                        count++;
                    } else {
                        compressed.push(count);
                        current = mask[i][j];
                        count = 1;
                    }
                }
                compressed.push(count); // 添加最后一组
            }
            
            return {
                shape: [height, width],
                data: compressed,
                firstValue: mask[0][0] // 记录第一个值
            };
        }
        
        function clearPoints() {
            points = [];
            labels = [];
            affordancePoints = [];
            currentMask = null;
            currentBbox = null;
            
            // 重新绘制图像
            if (currentImage) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(currentImage, imageOffsetX, imageOffsetY, 
                            currentImage.width * imageScale, 
                            currentImage.height * imageScale);
                
                // 清除mask canvas
                maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
            }
            
            // 隐藏预测类别
            document.getElementById('predictedObject').style.display = 'none';
            document.getElementById('predictedAction').style.display = 'none';
            
            updateStatus('已清除所有点');
        }

        function deleteAnnotation() {
            if (!currentImagePath) {
                updateStatus('没有当前图片可删除');
                return;
            }
            
            if (!confirm('确定要删除当前图片吗？这将删除图片文件、标注信息和mask文件。此操作不可恢复！')) {
                return;
            }
            
            updateStatus('正在删除图片...');
            
            fetch('/delete_annotation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image_path: currentImagePath
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateStatus(data.message);
                    console.log('删除成功，当前索引:', data.current_index);
                    
                    // 从图像列表中移除被删除的图片
                    const deletedImageName = currentImagePath.split('/').pop();
                    const imageItems = document.querySelectorAll('.image-item');
                    imageItems.forEach(item => {
                        if (item.textContent.includes(deletedImageName)) {
                            item.remove();
                        }
                    });
                    
                    // 重新加载当前索引位置的图片
                    if (data.total_count > 0) {
                        updateStatus('重新加载当前索引位置的图片...');
                        
                        // 获取当前索引位置的图片
                        fetch('/get_image_list')
                        .then(response => response.json())
                        .then(imageListData => {
                            if (imageListData.images && imageListData.images.length > 0) {
                                const currentIndex = Math.min(data.current_index, imageListData.images.length - 1);
                                const currentImage = imageListData.images[currentIndex];
                                
                                console.log('加载当前索引位置的图片:', currentImage);
                                
                                fetch(`/load_image/${currentImage}`)
                                .then(response => response.json())
                                .then(loadData => {
                                    if (loadData.success) {
                                        loadImageData(loadData);
                                        // 更新选中状态
                                        document.querySelectorAll('.image-item').forEach(item => {
                                            item.classList.remove('active');
                                        });
                                        const currentItem = document.querySelector(`[onclick*="${currentImage}"]`);
                                        if (currentItem) {
                                            currentItem.classList.add('active');
                                        }
                                    } else {
                                        updateStatus('加载当前图片失败: ' + loadData.error);
                                    }
                                })
                                .catch(error => {
                                    updateStatus('加载当前图片错误: ' + error);
                                });
                            } else {
                                updateStatus('没有更多图片，标注完成！');
                            }
                        })
                        .catch(error => {
                            updateStatus('获取图片列表失败: ' + error);
                        });
                    } else {
                        updateStatus('没有更多图片，标注完成！');
                    }
                } else {
                    updateStatus('删除失败: ' + data.error);
                }
            })
            .catch(error => {
                updateStatus('删除错误: ' + error);
                console.error('删除错误:', error);
            });
        }
        
        function deleteMask() {
            const imageNameInput = document.getElementById('deleteMaskImageName');
            const objectCategoryInput = document.getElementById('deleteMaskObjectCategory');
            
            const imageName = imageNameInput.value.trim();
            const objectCategory = objectCategoryInput.value.trim();
            
            if (!imageName) {
                updateStatus('请输入图片名称');
                return;
            }
            
            if (!objectCategory) {
                updateStatus('请输入物体名称');
                return;
            }
            
            // 验证图片名称是否存在
            fetch('/get_image_list')
            .then(response => response.json())
            .then(imageListData => {
                const images = imageListData.images || [];
                
                if (!images.includes(imageName)) {
                    updateStatus(`图片名称不存在: ${imageName}`);
                    return;
                }
                
                if (!confirm(`确定要删除图片 "${imageName}" 中物体类别为 "${objectCategory}" 的mask和标注记录吗？此操作不可恢复！`)) {
                    return;
                }
                
                updateStatus('正在删除mask和标注记录...');
                
                fetch('/delete_mask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        image_name: imageName,
                        object_category: objectCategory
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateStatus(data.message);
                        showToast(data.message, 3000);
                        
                        // 清空输入框
                        imageNameInput.value = '';
                        objectCategoryInput.value = '';
                        
                        // 更新标注状态显示
                        updateAnnotationStatuses();
                        
                        // 如果删除的是当前图片的标注，刷新当前图片显示
                        if (currentImagePath) {
                            const currentFilename = currentImagePath.split('/').pop();
                            if (currentFilename === imageName) {
                                // 重新加载当前图片以刷新显示
                                fetch(`/load_image/${imageName}`)
                                .then(response => response.json())
                                .then(loadData => {
                                    if (loadData.success) {
                                        loadImageData(loadData);
                                    }
                                })
                                .catch(error => {
                                    console.error('重新加载图片失败:', error);
                                });
                            }
                        }
                    } else {
                        updateStatus(data.message || '删除失败: ' + (data.error || '未知错误'));
                        showToast(data.message || '删除失败', 3000);
                    }
                })
                .catch(error => {
                    updateStatus('删除错误: ' + error);
                    showToast('删除错误: ' + error, 3000);
                    console.error('删除错误:', error);
                });
            })
            .catch(error => {
                updateStatus('获取图片列表失败: ' + error);
                console.error('获取图片列表失败:', error);
            });
        }

        function skipCurrentImage() {
            if (!currentImagePath) {
                updateStatus('没有当前图片可跳过');
                return;
            }
            
            updateStatus('跳过当前图片...');
            
            // 获取下一张图片
            const currentFilename = currentImagePath.split('/').pop();
            
            // 获取图像列表
            fetch('/get_image_list')
            .then(response => response.json())
            .then(data => {
                const currentIndex = data.images.indexOf(currentFilename);
                if (currentIndex !== -1) {
                    const nextIndex = (currentIndex + 1) % data.images.length;
                    const nextImage = data.images[nextIndex];
                    
                    console.log('跳过到下一张图片:', nextImage);
                    fetch(`/load_image/${nextImage}`)
                    .then(response => response.json())
                    .then(loadData => {
                        if (loadData.success) {
                            loadImageData(loadData);
                            // 更新选中状态
                            document.querySelectorAll('.image-item').forEach(item => {
                                item.classList.remove('active');
                            });
                            const nextItem = document.querySelector(`[onclick*="${nextImage}"]`);
                            if (nextItem) {
                                nextItem.classList.add('active');
                            }
                        } else {
                            updateStatus('加载下一张图片失败: ' + loadData.error);
                        }
                    })
                    .catch(error => {
                        updateStatus('加载下一张图片错误: ' + error);
                    });
                } else {
                    updateStatus('无法找到当前图片的索引');
                }
            })
            .catch(error => {
                updateStatus('获取图像列表失败: ' + error);
            });
        }
        
        function saveDataset() {
            fetch('/save_dataset', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    updateStatus(data.message);
                } else {
                    updateStatus('保存数据集失败: ' + data.error);
                }
            })
            .catch(error => {
                updateStatus('保存数据集错误: ' + error);
            });
        }
        
        function loadSuggestions() {
            fetch('/get_suggestions')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // 保存建议数据到全局变量
                    window.objectSuggestions = data.object_suggestions || [];
                    window.actionSuggestions = data.action_suggestions || [];
                    
                    // 显示物体类别建议
                    if (data.object_suggestions && data.object_suggestions.length > 0) {
                        const objectSuggestionsText = data.object_suggestions.slice(0, 10).join(', ');
                        document.getElementById('objectSuggestionsText').textContent = objectSuggestionsText;
                        document.getElementById('objectSuggestions').style.display = 'block';
                    }
                    
                    // 显示动作类别建议
                    if (data.action_suggestions && data.action_suggestions.length > 0) {
                        const actionSuggestionsText = data.action_suggestions.slice(0, 10).join(', ');
                        document.getElementById('actionSuggestionsText').textContent = actionSuggestionsText;
                        document.getElementById('actionSuggestions').style.display = 'block';
                    }
                    
                    console.log('加载建议成功:', data);
                } else {
                    console.error('加载建议失败:', data.error);
                }
            })
            .catch(error => {
                console.error('加载建议错误:', error);
            });
        }
        
        // TAB键自动补全功能
        
        function setupTabCompletion() {
            const objectInput = document.getElementById('objectCategory');
            const actionInput = document.getElementById('actionCategory');
            const deleteMaskObjectInput = document.getElementById('deleteMaskObjectCategory');
            
            // 使用全局建议数据，如果没有则获取
            if (!window.objectSuggestions || !window.actionSuggestions) {
                fetch('/get_suggestions')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        window.objectSuggestions = data.object_suggestions || [];
                        window.actionSuggestions = data.action_suggestions || [];
                    }
                })
                .catch(error => {
                    console.error('获取建议失败:', error);
                    window.objectSuggestions = [];
                    window.actionSuggestions = [];
                });
            }
            
            // 智能自动补全函数
            function findBestMatch(input, suggestions, allCategories) {
                const inputLower = input.toLowerCase();
                
                // 首先在常用建议中查找以输入开头的（优先常用类别）
                const suggestionStartsWith = suggestions.filter(sug => 
                    sug.toLowerCase().startsWith(inputLower)
                );
                
                if (suggestionStartsWith.length > 0) {
                    return suggestionStartsWith[0];
                }
                
                // 然后在预定义类别中查找以输入开头的
                const startsWithMatches = allCategories.filter(cat => 
                    cat.toLowerCase().startsWith(inputLower)
                );
                
                if (startsWithMatches.length > 0) {
                    return startsWithMatches[0];
                }
                
                // 在常用建议中查找包含输入的
                const suggestionContains = suggestions.filter(sug => 
                    sug.toLowerCase().includes(inputLower)
                );
                
                if (suggestionContains.length > 0) {
                    return suggestionContains[0];
                }
                
                // 最后在预定义类别中查找包含输入的
                const containsMatches = allCategories.filter(cat => 
                    cat.toLowerCase().includes(inputLower)
                );
                
                if (containsMatches.length > 0) {
                    return containsMatches[0];
                }
                
                // 如果都没有找到，返回第一个常用建议
                return suggestions.length > 0 ? suggestions[0] : '';
            }
            
            // 物体类别TAB补全
            objectInput.addEventListener('keydown', function(e) {
                if (e.key === 'Tab') {
                    e.preventDefault();
                    const currentValue = this.value.trim();
                    const allCategories = {{ object_categories | tojson }};
                    const bestMatch = findBestMatch(currentValue, window.objectSuggestions || [], allCategories);
                    if (bestMatch) {
                        this.value = bestMatch;
                        this.focus();
                    }
                }
            });
            
            // 动作类别TAB补全
            actionInput.addEventListener('keydown', function(e) {
                if (e.key === 'Tab') {
                    e.preventDefault();
                    const currentValue = this.value.trim();
                    const allCategories = {{ action_categories | tojson }};
                    const bestMatch = findBestMatch(currentValue, window.actionSuggestions || [], allCategories);
                    if (bestMatch) {
                        this.value = bestMatch;
                        this.focus();
                    }
                }
            });
            
            // 删除Mask的物体类别TAB补全
            if (deleteMaskObjectInput) {
                deleteMaskObjectInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Tab') {
                        e.preventDefault();
                        const currentValue = this.value.trim();
                        const allCategories = {{ object_categories | tojson }};
                        const bestMatch = findBestMatch(currentValue, window.objectSuggestions || [], allCategories);
                        if (bestMatch) {
                            this.value = bestMatch;
                            this.focus();
                        }
                    }
                });
            }
            
            // 删除Mask的图片名称TAB补全
            const deleteMaskImageInput = document.getElementById('deleteMaskImageName');
            if (deleteMaskImageInput) {
                deleteMaskImageInput.addEventListener('keydown', function(e) {
                    if (e.key === 'Tab') {
                        e.preventDefault();
                        const currentValue = this.value.trim();
                        const imageList = {{ image_list | tojson }};
                        
                        // 查找匹配的图片名称
                        const inputLower = currentValue.toLowerCase();
                        const matches = imageList.filter(img => 
                            img.toLowerCase().startsWith(inputLower)
                        );
                        
                        if (matches.length > 0) {
                            this.value = matches[0];
                            this.focus();
                        }
                    }
                });
            }
            
            // Q键快捷键切换点击模式
            document.addEventListener('keydown', function(e) {
                if (e.key === 'q' || e.key === 'Q') {
                    e.preventDefault();
                    const currentMode = document.querySelector('input[name="mode"]:checked').value;
                    let nextMode;
                    
                    switch (currentMode) {
                        case 'affordance':
                            nextMode = 'foreground';
                            break;
                        case 'foreground':
                            nextMode = 'background';
                            break;
                        case 'background':
                            nextMode = 'affordance';
                            break;
                        default:
                            nextMode = 'affordance';
                    }
                    
                    document.querySelector(`input[name="mode"][value="${nextMode}"]`).checked = true;
                    
                    // 显示切换提示
                    const modeNames = {
                        'affordance': '交互点(红)',
                        'foreground': '前景点(绿)', 
                        'background': '背景点(蓝)'
                    };
                    updateStatus(`点击模式已切换为: ${modeNames[nextMode]}`);
                }
            });
        }
        
        function predictCategoriesFromFilename(filename) {
            // 返回最常用的物体和动作类别
            // 只返回真正有使用记录的类别
            let defaultObject = 'object';  // 默认值
            let defaultAction = 'grasp';   // 默认值
            
            if (window.objectSuggestions && window.objectSuggestions.length > 0) {
                defaultObject = window.objectSuggestions[0];
            }
            if (window.actionSuggestions && window.actionSuggestions.length > 0) {
                defaultAction = window.actionSuggestions[0];
            }
            
            return { object: defaultObject, action: defaultAction };
        }
    </script>
</body>
</html>
    '''
    
    with open(os.path.join(template_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description="Web可视化标注工具")
    parser.add_argument("--sam_checkpoint", type=str, required=True,
                       help="Path to SAM model checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--image_dir", type=str, default="/mnt/data-cpfs/rap_mani/harrison.zhou/harrison_workspace/GLOVER/images",
                       help="Directory containing images to annotate")
    parser.add_argument("--lisa_model_path", type=str, default=None,
                       help="Path to LISA model (已废弃，不再使用)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000,
                       help="Port to bind to")
    
    args = parser.parse_args()
    
    # 创建HTML模板
    create_html_template()
    
    # 初始化全局标注器
    global annotator
    annotator = WebAnnotator(args.sam_checkpoint, args.device, args.image_dir, args.lisa_model_path)
    
    print(f"Web标注工具启动在 http://{args.host}:{args.port}")
    print(f"图像目录: {args.image_dir}")
    print(f"找到 {len(annotator.image_list)} 张图像")
    print("使用最常用的物体和动作类别作为默认值")
    print("请在浏览器中访问上述地址")
    
    # 启动Flask应用
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main() 
