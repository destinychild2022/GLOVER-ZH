#!/usr/bin/env python3
"""
GLOVER数据可视化脚本
用于验证affordance数据转换是否正确
"""

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import List, Dict
import random


def load_annotations(annotations_file: Path) -> List[Dict]:
    """加载标注数据"""
    with open(annotations_file, 'r') as f:
        return json.load(f)


def visualize_glover_sample(annotations: List[Dict], images_dir: Path, masks_dir: Path, sample_idx: int = 0):
    """可视化单个GLOVER样本"""
    if sample_idx >= len(annotations):
        print(f"样本索引 {sample_idx} 超出范围，最大索引为 {len(annotations) - 1}")
        return
    
    sample = annotations[sample_idx]
    
    # 加载RGB图像
    rgb_image_path = images_dir / sample['image_path']
    if not rgb_image_path.exists():
        print(f"RGB图像文件不存在: {rgb_image_path}")
        return
    
    rgb_image = cv2.imread(str(rgb_image_path))
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    
    # 加载掩码
    mask_path = masks_dir / sample['mask_path']
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        print(f"掩码文件不存在: {mask_path}")
        mask = None
    
    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 显示RGB图像
    axes[0].imshow(rgb_image)
    axes[0].set_title(f"RGB Image\nTask: {sample['task_id']}, Episode: {sample['episode_id']}")
    axes[0].axis('off')
    
    # 显示掩码
    if mask is not None:
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Affordance Mask")
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, "Mask not found", ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("Affordance Mask")
        axes[1].axis('off')
    
    # 显示叠加效果
    axes[2].imshow(rgb_image)
    if mask is not None:
        # 创建彩色掩码
        colored_mask = np.zeros_like(rgb_image)
        colored_mask[mask > 0] = [255, 0, 0]  # 红色
        axes[2].imshow(colored_mask, alpha=0.5)
    
    # 标记affordance点
    affordance_point = sample['affordance_point']
    axes[2].plot(affordance_point[0], affordance_point[1], 'go', markersize=10, label='Affordance Point')
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()


def visualize_multiple_samples(annotations: List[Dict], images_dir: Path, masks_dir: Path, num_samples: int = 4):
    """可视化多个GLOVER样本"""
    if len(annotations) == 0:
        print("没有可用的标注数据")
        return
    
    # 随机选择样本
    sample_indices = random.sample(range(len(annotations)), min(num_samples, len(annotations)))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices):
        sample = annotations[idx]
        
        # 加载RGB图像
        rgb_image_path = images_dir / sample['image_path']
        if rgb_image_path.exists():
            rgb_image = cv2.imread(str(rgb_image_path))
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            
            # 加载掩码
            mask_path = masks_dir / sample['mask_path']
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                # 创建叠加图像
                overlay = rgb_image.copy()
                colored_mask = np.zeros_like(rgb_image)
                colored_mask[mask > 0] = [255, 0, 0]
                overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
                
                axes[i].imshow(overlay)
                
                # 标记affordance点
                affordance_point = sample['affordance_point']
                axes[i].plot(affordance_point[0], affordance_point[1], 'go', markersize=8)
                
                axes[i].set_title(f"Task: {sample['task_id']}, Episode: {sample['episode_id']}\n"
                                 f"Point: ({affordance_point[0]}, {affordance_point[1]})")
            else:
                axes[i].imshow(rgb_image)
                axes[i].set_title(f"Task: {sample['task_id']}, Episode: {sample['episode_id']}\n(No mask)")
            
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, f"Image not found\n{sample['image_path']}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f"Task: {sample['task_id']}, Episode: {sample['episode_id']}")
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def show_statistics(annotations: List[Dict]):
    """显示数据统计信息"""
    if not annotations:
        print("没有可用的标注数据")
        return
    
    # 统计任务和episode
    tasks = set()
    episodes = set()
    positions_3d = []
    affordance_points = []
    
    for sample in annotations:
        tasks.add(sample['task_id'])
        episodes.add(sample['episode_id'])
        positions_3d.append(sample['position_3d'])
        affordance_points.append(sample['affordance_point'])
    
    positions_3d = np.array(positions_3d)
    affordance_points = np.array(affordance_points)
    
    print("=== GLOVER Affordance数据统计 ===")
    print(f"总样本数: {len(annotations)}")
    print(f"任务数: {len(tasks)}")
    print(f"Episode数: {len(episodes)}")
    print(f"任务列表: {sorted(list(tasks))}")
    print(f"Episode列表: {sorted(list(episodes))}")
    
    print("\n=== 3D位置统计 ===")
    print(f"X范围: [{positions_3d[:, 0].min():.3f}, {positions_3d[:, 0].max():.3f}]")
    print(f"Y范围: [{positions_3d[:, 1].min():.3f}, {positions_3d[:, 1].max():.3f}]")
    print(f"Z范围: [{positions_3d[:, 2].min():.3f}, {positions_3d[:, 2].max():.3f}]")
    print(f"位置均值: [{positions_3d[:, 0].mean():.3f}, {positions_3d[:, 1].mean():.3f}, {positions_3d[:, 2].mean():.3f}]")
    
    print("\n=== 2D Affordance点统计 ===")
    print(f"X范围: [{affordance_points[:, 0].min():.0f}, {affordance_points[:, 0].max():.0f}]")
    print(f"Y范围: [{affordance_points[:, 1].min():.0f}, {affordance_points[:, 1].max():.0f}]")
    print(f"点均值: [{affordance_points[:, 0].mean():.1f}, {affordance_points[:, 1].mean():.1f}]")
    
    # 检查GLOVER兼容性
    print("\n=== GLOVER兼容性检查 ===")
    for sample in annotations[:5]:  # 检查前5个样本
        has_image = 'image_path' in sample
        has_mask = 'mask_path' in sample
        has_question = 'question' in sample and '[SEG]' in sample['question']
        has_answer = 'answer' in sample and '[SEG]' in sample['answer']
        has_point = 'affordance_point' in sample
        
        print(f"样本 {sample.get('image_path', 'N/A')}:")
        print(f"  - 图像: {'✓' if has_image else '✗'}")
        print(f"  - 掩码: {'✓' if has_mask else '✗'}")
        print(f"  - 问题格式: {'✓' if has_question else '✗'}")
        print(f"  - 答案格式: {'✓' if has_answer else '✗'}")
        print(f"  - Affordance点: {'✓' if has_point else '✗'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize GLOVER affordance data")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing extracted GLOVER affordance data")
    parser.add_argument("--task_id", type=str, default=None,
                       help="Specific task ID to visualize")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Sample index to visualize (for single sample mode)")
    parser.add_argument("--num_samples", type=int, default=4,
                       help="Number of samples to visualize (for multiple samples mode)")
    parser.add_argument("--mode", choices=["single", "multiple", "stats"], default="multiple",
                       help="Visualization mode")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    annotations_dir = data_dir / "annotations"
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    
    if not data_dir.exists():
        print(f"数据目录不存在: {data_dir}")
        return
    
    if not annotations_dir.exists():
        print(f"标注目录不存在: {annotations_dir}")
        return
    
    # 查找标注文件
    if args.task_id:
        annotations_file = annotations_dir / f"{args.task_id}_annotations.json"
        if not annotations_file.exists():
            print(f"任务 {args.task_id} 的标注文件不存在: {annotations_file}")
            return
        annotations_files = [annotations_file]
    else:
        annotations_files = list(annotations_dir.glob("*_annotations.json"))
        if not annotations_files:
            print(f"在 {annotations_dir} 中没有找到标注文件")
            return
    
    # 加载所有标注数据
    all_annotations = []
    for annotations_file in annotations_files:
        annotations = load_annotations(annotations_file)
        all_annotations.extend(annotations)
    
    if not all_annotations:
        print("没有可用的标注数据")
        return
    
    print(f"加载了 {len(all_annotations)} 个样本")
    
    if args.mode == "single":
        visualize_glover_sample(all_annotations, images_dir, masks_dir, args.sample_idx)
    elif args.mode == "multiple":
        visualize_multiple_samples(all_annotations, images_dir, masks_dir, args.num_samples)
    elif args.mode == "stats":
        show_statistics(all_annotations)


if __name__ == "__main__":
    main() 