#!/usr/bin/env python3
"""
交互式标注工具
基于matplotlib，可以在Jupyter notebook中运行
"""

import os
import json
import cv2
import numpy as np
import torch
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, RadioButtons, TextBox
import ipywidgets as widgets
from IPython.display import display, clear_output

# SAM imports
from segment_anything import SamPredictor, sam_model_registry

class InteractiveAnnotator:
    def __init__(self, sam_checkpoint: str, device: str = "cuda"):
        """
        初始化交互式标注器
        
        Args:
            sam_checkpoint: SAM模型路径
            device: 设备类型
        """
        self.device = device
        
        # 初始化SAM模型
        print("Loading SAM model...")
        self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        
        # 预定义的物体和动作类别
        self.object_categories = [
            "cup", "bowl", "plate", "fork", "spoon", "knife", "pan", "pot", 
            "blender", "microwave", "refrigerator", "dishwasher", "toaster",
            "phone", "laptop", "book", "pen", "paper", "chair", "table",
            "door", "window", "light", "switch", "remote", "keys", "wallet",
            "clothes", "shoes", "bag", "bottle", "can", "box", "container"
        ]
        
        self.action_categories = [
            "grasp", "lift", "move", "place", "put", "take", "pick", "hold",
            "open", "close", "push", "pull", "turn", "rotate", "press",
            "pour", "fill", "empty", "clean", "wash", "wipe", "sweep",
            "cut", "slice", "chop", "mix", "stir", "cook", "heat", "cool"
        ]
        
        # 标注数据存储
        self.annotations = []
        self.current_image_path = None
        self.current_image = None
        self.current_points = []
        self.current_labels = []
        self.current_mask = None
        self.current_bbox = None
        
        # 交互状态
        self.click_mode = "foreground"  # "foreground" or "background"
        
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
        
        # 预测掩码
        masks, scores, logits = self.predictor.predict(
            point_coords=points_array,
            point_labels=labels_array,
            multimask_output=True
        )
        
        # 选择最佳掩码
        best_mask_idx = np.argmax(scores)
        mask = masks[best_mask_idx]
        
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
                       action_category: str, mask: np.ndarray, points: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        保存标注数据
        """
        # 读取图像获取尺寸
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        
        # 生成掩码文件名
        base_name = Path(image_path).stem
        mask_filename = f"{base_name}_mask.png"
        
        # 保存掩码到output目录
        mask_path = os.path.join("output", "masks", mask_filename)
        os.makedirs(os.path.join("output", "masks"), exist_ok=True)
        
        # 在mask上绘制affordance点
        mask_with_points = mask.copy().astype(np.uint8) * 255
        
        # 如果有affordance点，在mask上绘制红色圆点
        if points:
            for point in points:
                x, y = int(point[0]), int(point[1])
                # 确保点在图像范围内
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(mask_with_points, (x, y), 5, (255, 0, 0), -1)  # 红色圆点
        
        cv2.imwrite(mask_path, mask_with_points)
        
        # 创建标注数据
        annotation = {
            "img_name": os.path.basename(image_path),
            "bbox": bbox,
            "affordance": affordance,
            "height": height,
            "width": width,
            "object": object_category,
            "action": action_category,
            "gt_path": mask_path
        }
        
        return annotation

class InteractiveAnnotationGUI:
    def __init__(self, annotator: InteractiveAnnotator):
        """
        初始化交互式标注GUI界面
        
        Args:
            annotator: 交互式标注器实例
        """
        self.annotator = annotator
        self.fig = None
        self.ax = None
        self.canvas = None
        
        # 创建控件
        self.setup_widgets()
        
    def setup_widgets(self):
        """设置控件"""
        # 文件选择控件
        self.file_button = widgets.Button(description="Load Image")
        self.file_button.on_click(self.load_image)
        
        # 点击模式选择
        self.mode_radio = widgets.RadioButtons(
            options=['foreground', 'background'],
            description='Click Mode:',
            value='foreground'
        )
        self.mode_radio.observe(self.on_mode_change, names='value')
        
        # 物体类别选择
        self.object_dropdown = widgets.Dropdown(
            options=self.annotator.object_categories,
            description='Object:',
            value=self.annotator.object_categories[0]
        )
        
        # 动作类别选择
        self.action_dropdown = widgets.Dropdown(
            options=self.annotator.action_categories,
            description='Action:',
            value=self.annotator.action_categories[0]
        )
        
        # 操作按钮
        self.segment_button = widgets.Button(description="Segment")
        self.segment_button.on_click(self.segment_object)
        
        self.save_button = widgets.Button(description="Save Annotation")
        self.save_button.on_click(self.save_annotation)
        
        self.clear_button = widgets.Button(description="Clear Points")
        self.clear_button.on_click(self.clear_points)
        
        self.save_dataset_button = widgets.Button(description="Save Dataset")
        self.save_dataset_button.on_click(self.save_dataset)
        
        # 状态显示
        self.status_text = widgets.HTML(value="<b>Status:</b> No image loaded")
        
        # 布局
        self.controls = widgets.VBox([
            self.file_button,
            self.mode_radio,
            self.object_dropdown,
            self.action_dropdown,
            self.segment_button,
            self.save_button,
            self.clear_button,
            self.save_dataset_button,
            self.status_text
        ])
        
    def on_mode_change(self, change):
        """处理模式改变"""
        self.annotator.click_mode = change['new']
        
    def load_image(self, b):
        """加载图像"""
        from google.colab import files
        
        try:
            uploaded = files.upload()
            if uploaded:
                filename = list(uploaded.keys())[0]
                self.annotator.current_image_path = filename
                self.annotator.current_image = cv2.imread(filename)
                self.annotator.current_image = cv2.cvtColor(self.annotator.current_image, cv2.COLOR_BGR2RGB)
                
                # 重置状态
                self.annotator.current_points = []
                self.annotator.current_labels = []
                self.annotator.current_mask = None
                self.annotator.current_bbox = None
                
                # 显示图像
                self.display_image()
                self.status_text.value = f"<b>Status:</b> Loaded {filename}"
                
        except Exception as e:
            self.status_text.value = f"<b>Error:</b> {str(e)}"
    
    def display_image(self):
        """显示图像"""
        if self.annotator.current_image is None:
            return
            
        # 清除之前的图像
        if self.fig:
            plt.close(self.fig)
        
        # 创建新的图像
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # 显示图像
        self.ax.imshow(self.annotator.current_image)
        
        # 绘制点击的点
        if self.annotator.current_points:
            for i, (x, y) in enumerate(self.annotator.current_points):
                if self.annotator.current_labels[i] == 1:  # 前景点
                    self.ax.plot(x, y, 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)
                else:  # 背景点
                    self.ax.plot(x, y, 'bo', markersize=10, markeredgecolor='white', markeredgewidth=2)
        
        # 绘制分割掩码
        if self.annotator.current_mask is not None:
            mask_overlay = np.zeros_like(self.annotator.current_image)
            mask_overlay[self.annotator.current_mask] = [255, 0, 0]  # 红色
            self.ax.imshow(mask_overlay, alpha=0.3)
            
            # 绘制边界框
            if self.annotator.current_bbox:
                height, width = self.annotator.current_image.shape[:2]
                x, y, w, h = [self.annotator.current_bbox[0] * width, 
                             self.annotator.current_bbox[1] * height,
                             self.annotator.current_bbox[2] * width, 
                             self.annotator.current_bbox[3] * height]
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
                self.ax.add_patch(rect)
        
        self.ax.set_title("Click to add points (foreground/background mode)")
        self.ax.axis('off')
        
        # 绑定点击事件
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        plt.tight_layout()
        plt.show()
    
    def on_click(self, event):
        """处理点击事件"""
        if (self.annotator.current_image is None or 
            event.inaxes != self.ax or 
            event.button != 1):  # 只处理左键点击
            return
            
        x, y = int(event.xdata), int(event.ydata)
        
        # 根据模式添加点
        if self.annotator.click_mode == "foreground":
            self.annotator.current_points.append((x, y))
            self.annotator.current_labels.append(1)
        else:
            self.annotator.current_points.append((x, y))
            self.annotator.current_labels.append(0)
        
        # 重新显示图像
        self.display_image()
    
    def segment_object(self, b):
        """分割物体"""
        if self.annotator.current_image is None or not self.annotator.current_points:
            self.status_text.value = "<b>Error:</b> Please load an image and add points first!"
            return
            
        try:
            # 分割物体
            self.annotator.current_mask, self.annotator.current_bbox = self.annotator.segment_object(
                self.annotator.current_image, self.annotator.current_points, self.annotator.current_labels
            )
            
            self.display_image()
            self.status_text.value = "<b>Status:</b> Segmentation completed!"
            
        except Exception as e:
            self.status_text.value = f"<b>Error:</b> Segmentation failed: {str(e)}"
    
    def save_annotation(self, b):
        """保存标注"""
        if (self.annotator.current_image_path is None or 
            self.annotator.current_mask is None):
            self.status_text.value = "<b>Error:</b> Please complete segmentation first!"
            return
            
        try:
            # 计算交互点
            if self.annotator.current_points:
                affordance_point = self.annotator.current_points[0]
                height, width = self.annotator.current_image.shape[:2]
                affordance = [affordance_point[0] / width, affordance_point[1] / height]
            else:
                affordance = [0.5, 0.5]
            
            # 保存标注
            annotation = self.annotator.save_annotation(
                self.annotator.current_image_path,
                self.annotator.current_bbox,
                affordance,
                self.object_dropdown.value,
                self.action_dropdown.value,
                self.annotator.current_mask,
                self.annotator.current_points
            )
            
            # 添加到标注列表
            self.annotator.annotations.append(annotation)
            
            self.status_text.value = f"<b>Status:</b> Annotation saved! Total: {len(self.annotator.annotations)}"
            
        except Exception as e:
            self.status_text.value = f"<b>Error:</b> Save failed: {str(e)}"
    
    def clear_points(self, b):
        """清除所有点"""
        self.annotator.current_points = []
        self.annotator.current_labels = []
        self.annotator.current_mask = None
        self.annotator.current_bbox = None
        self.display_image()
        self.status_text.value = "<b>Status:</b> Points cleared"
    
    def save_dataset(self, b):
        """保存完整数据集"""
        if not self.annotator.annotations:
            self.status_text.value = "<b>Error:</b> No annotations to save!"
            return
            
        try:
            # 确保output目录存在
            os.makedirs("output", exist_ok=True)
            
            output_path = "output/annotations.json"
            with open(output_path, 'w') as f:
                json.dump(self.annotator.annotations, f, indent=4)
            
            self.status_text.value = f"<b>Status:</b> Dataset saved to {output_path}"
            
        except Exception as e:
            self.status_text.value = f"<b>Error:</b> Save dataset failed: {str(e)}"
    
    def display_controls(self):
        """显示控件"""
        display(self.controls)

def create_interactive_annotator(sam_checkpoint: str, device: str = "cuda"):
    """
    创建交互式标注器
    
    Args:
        sam_checkpoint: SAM模型路径
        device: 设备类型
        
    Returns:
        InteractiveAnnotationGUI: 交互式标注GUI
    """
    annotator = InteractiveAnnotator(sam_checkpoint, device)
    gui = InteractiveAnnotationGUI(annotator)
    return gui

# 使用示例
if __name__ == "__main__":
    # 在Jupyter notebook中使用
    # gui = create_interactive_annotator("/path/to/sam_checkpoint.pth")
    # gui.display_controls()
    pass 