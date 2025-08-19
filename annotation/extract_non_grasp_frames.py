#!/usr/bin/env python3
"""
从机械臂操作数据中抽取非抓取时刻的帧
根据JSON文件获取task信息，按task_name英文命名保存
"""

import os
import json
import h5py
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
from dataclasses import dataclass
import random
import shutil


@dataclass
class NonGraspFrameData:
    """非抓取时刻帧数据结构"""
    episode_id: str
    task_id: str
    task_name: str
    timestamp: float
    frame_idx: int
    end_effector_position: np.ndarray  # (3,) xyz position
    end_effector_orientation: np.ndarray  # (4,) quaternion
    head_camera_image: np.ndarray  # 头部相机图像
    hand_left_camera_image: np.ndarray  # 左手相机图像
    hand_right_camera_image: np.ndarray  # 右手相机图像
    left_effector_position: float  # 左手夹爪位置
    right_effector_position: float  # 右手夹爪位置
    camera_name: str  # 主要使用的相机名称


class NonGraspFrameExtractor:
    """非抓取时刻帧提取器"""
    
    def __init__(self, data_root: str, output_dir: str):
        """
        Args:
            data_root: 数据根目录路径
            output_dir: 输出目录路径
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建输出子目录
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        
        # 夹爪闭合阈值
        self.grasp_threshold = 10  # 夹爪位置小于此值认为是张开状态
        
        # 每个episode抽取的帧数
        self.frames_per_episode = 20
        
        # 每个任务选择的episode数量
        self.episodes_per_task = 6
        
        # 随机种子
        random.seed(42)
        
        # 加载task信息
        self.task_info = self.load_task_info()
    
    def load_task_info(self) -> Dict[str, Dict]:
        """加载task信息从JSON文件"""
        task_info = {}
        
        # 查找所有task JSON文件
        json_files = list(self.data_root.glob("task_*.json"))
        json_files.sort()
        
        print(f"找到 {len(json_files)} 个task JSON文件")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # 提取task信息
                task_id = json_file.stem.split('_')[1]  # 从文件名提取task_id
                
                # 从JSON数据中提取task_name和episode信息
                if isinstance(data, list) and len(data) > 0:
                    # 假设JSON是episode列表格式
                    task_name = f"task_{task_id}"
                    episodes = []
                    
                    for episode_data in data:
                        if isinstance(episode_data, dict):
                            episode_id = episode_data.get('episode_id')
                            task_id_from_json = episode_data.get('task_id')
                            job_id = episode_data.get('job_id')
                            sn_code = episode_data.get('sn_code')
                            
                            if episode_id and task_id_from_json and job_id and sn_code:
                                episodes.append({
                                    'episode_id': episode_id,
                                    'task_id': task_id_from_json,
                                    'job_id': job_id,
                                    'sn_code': sn_code
                                })
                    
                    task_info[task_id] = {
                        'task_name': task_name,
                        'episodes': episodes,
                        'json_file': json_file.name
                    }
                    
                    print(f"  Task {task_id}: {task_name}, {len(episodes)} episodes")
                
            except Exception as e:
                print(f"加载 {json_file} 失败: {e}")
        
        return task_info
    
    def load_h5_data(self, trial_data_root_dir: Path) -> Optional[Dict]:
        """加载H5文件中的机械臂数据"""
        h5_file = trial_data_root_dir / "aligned_joints.h5"
        if not h5_file.exists():
            print(f"  - 在 {trial_data_root_dir} 中未找到H5文件")
            return None
        
        print(f"  - 找到H5文件: {h5_file}")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                data = {}
                
                # 检查是否有state数据
                if 'state' not in f:
                    print(f"  - 错误: 未找到state数据")
                    return None
                
                # 加载机器人基准位置
                if 'robot' in f['state'] and 'position' in f['state']['robot']:
                    robot_position = f['state']['robot']['position'][:]  # (T, 3)
                    print(f"  - 加载机器人基准位置: {robot_position.shape}")
                else:
                    print(f"  - 警告: 未找到机器人基准位置，使用零向量")
                    robot_position = np.zeros((f['timestamp'].shape[0], 3))
                
                # 加载末端执行器位置（相对于机器人基准位置）
                if 'end' in f['state'] and 'position' in f['state']['end']:
                    end_position_raw = f['state']['end']['position'][:]  # (T, 2, 3)
                    # 减去机器人基准位置
                    end_position_relative = end_position_raw - robot_position[:, np.newaxis, :]
                    data['end_position'] = end_position_relative
                    print(f"  - 加载末端执行器位置（相对坐标）: {data['end_position'].shape}")
                else:
                    print(f"  - 错误: 未找到末端执行器位置数据")
                    return None
                
                # 加载末端执行器方向
                if 'end' in f['state'] and 'orientation' in f['state']['end']:
                    data['end_orientation'] = f['state']['end']['orientation'][:]
                    print(f"  - 加载末端执行器方向: {data['end_orientation'].shape}")
                else:
                    print(f"  - 警告: 未找到末端执行器方向数据")
                    data['end_orientation'] = None
                
                # 加载夹爪状态数据
                if 'left_effector' in f['state'] and 'position' in f['state']['left_effector']:
                    data['left_effector_position'] = f['state']['left_effector']['position'][:]
                    print(f"  - 加载左手夹爪位置数据: {data['left_effector_position'].shape}")
                else:
                    print(f"  - 警告: 未找到左手夹爪位置数据")
                    return None
                
                if 'right_effector' in f['state'] and 'position' in f['state']['right_effector']:
                    data['right_effector_position'] = f['state']['right_effector']['position'][:]
                    print(f"  - 加载右手夹爪位置数据: {data['right_effector_position'].shape}")
                else:
                    print(f"  - 警告: 未找到右手夹爪位置数据")
                    return None
                
                # 加载时间戳
                data['timestamp'] = f['timestamp'][:]
                print(f"  - 加载时间戳: {data['timestamp'].shape}")
                
                return data
                
        except Exception as e:
            print(f"  - 加载H5文件失败: {e}")
            return None
    
    def find_non_grasp_frames(self, h5_data: Dict) -> List[int]:
        """找到左右夹爪都张开的帧"""
        left_effector_pos = h5_data['left_effector_position']
        right_effector_pos = h5_data['right_effector_position']
        timestamps = h5_data['timestamp']
        
        # 确保数据是一维数组
        if left_effector_pos.ndim > 1:
            left_effector_pos = left_effector_pos.squeeze()
        if right_effector_pos.ndim > 1:
            right_effector_pos = right_effector_pos.squeeze()
        
        print(f"  - 夹爪位置数据范围:")
        print(f"    左手夹爪: {left_effector_pos.min():.1f} - {left_effector_pos.max():.1f}")
        print(f"    右手夹爪: {right_effector_pos.min():.1f} - {right_effector_pos.max():.1f}")
        
        # 找到左右夹爪都张开的帧
        non_grasp_frames = []
        for i in range(len(left_effector_pos)):
            left_pos = left_effector_pos[i]
            right_pos = right_effector_pos[i]
            
            # 如果两个夹爪位置都小于阈值，认为是张开状态
            if left_pos < self.grasp_threshold and right_pos < self.grasp_threshold:
                non_grasp_frames.append(i)
        
        print(f"  - 找到 {len(non_grasp_frames)} 个夹爪张开帧")
        return non_grasp_frames
    
    def select_equidistant_frames(self, non_grasp_frames: List[int], h5_data: Dict) -> List[int]:
        """从非抓取帧中等距选择帧"""
        if len(non_grasp_frames) == 0:
            return []
        
        timestamps = h5_data['timestamp']
        selected_frames = []
        
        # 如果非抓取帧数量少于等于目标数量，全部选择
        if len(non_grasp_frames) <= self.frames_per_episode:
            selected_frames = non_grasp_frames
        else:
            # 等距选择帧
            step = len(non_grasp_frames) // self.frames_per_episode
            for i in range(self.frames_per_episode):
                idx = i * step
                if idx < len(non_grasp_frames):
                    selected_frames.append(non_grasp_frames[idx])
        
        print(f"  - 等距选择了 {len(selected_frames)} 个帧")
        return selected_frames
    
    def load_camera_image(self, camera_image_dir: Path, camera_name: str) -> Optional[np.ndarray]:
        """加载单个相机的图像"""
        rgb_file = camera_image_dir / f"{camera_name}_color.jpg"
        if rgb_file.exists():
            img = cv2.imread(str(rgb_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
        return None
    
    def process_episode(self, episode_dir: Path, task_id: str, task_name: str, episode_id: str) -> List[NonGraspFrameData]:
        """处理单个episode的非抓取帧数据"""
        frame_data = []
        
        print(f"  - 处理episode: {episode_dir.name}")
        
        # 直接使用episode目录，不需要查找trial目录
        trial_data_root_dir = episode_dir
        
        # 加载机械臂数据
        h5_data = self.load_h5_data(trial_data_root_dir)
        if h5_data is None:
            return frame_data
        
        # 找到非抓取帧
        non_grasp_frames = self.find_non_grasp_frames(h5_data)
        if not non_grasp_frames:
            print(f"  - 未找到夹爪张开帧")
            return frame_data
        
        # 等距选择帧
        selected_frames = self.select_equidistant_frames(non_grasp_frames, h5_data)
        
        # 处理每个选中的帧
        for frame_idx in selected_frames:
            timestamp = h5_data['timestamp'][frame_idx]
            left_effector_pos = h5_data['left_effector_position'][frame_idx]
            right_effector_pos = h5_data['right_effector_position'][frame_idx]
            
            # 加载相机图像
            camera_dir = trial_data_root_dir / "camera" / str(frame_idx)
            head_image = self.load_camera_image(camera_dir, 'head')
            hand_left_image = self.load_camera_image(camera_dir, 'hand_left')
            hand_right_image = self.load_camera_image(camera_dir, 'hand_right')
            
            if head_image is None:
                print(f"    - 跳过帧 {frame_idx}: 缺少头部相机图像")
                continue
            
            # 选择主要使用的相机（优先使用头部相机）
            camera_name = 'head'
            
            # 获取末端执行器位置和方向（使用左手位置作为主要位置）
            position = h5_data['end_position'][frame_idx][0]  # 左手位置
            orientation = h5_data['end_orientation'][frame_idx][0] if h5_data['end_orientation'] is not None else np.array([0, 0, 0, 1])
            
            frame_moment = NonGraspFrameData(
                episode_id=episode_id,
                task_id=task_id,
                task_name=task_name,
                timestamp=timestamp,
                frame_idx=frame_idx,
                end_effector_position=position,
                end_effector_orientation=orientation,
                head_camera_image=head_image,
                hand_left_camera_image=hand_left_image,
                hand_right_camera_image=hand_right_image,
                left_effector_position=left_effector_pos,
                right_effector_position=right_effector_pos,
                camera_name=camera_name
            )
            frame_data.append(frame_moment)
            print(f"    - 成功提取帧 {frame_idx} 的数据 (时间: {timestamp:.2f}s)")
        
        print(f"  - 总共提取了 {len(frame_data)} 个夹爪张开帧")
        return frame_data
    
    def save_frame_data(self, frame_data: List[NonGraspFrameData], task_id: str, task_name: str):
        """保存非抓取帧数据"""
        if not frame_data:
            return
        
        # 创建任务子文件夹
        task_images_dir = self.output_dir / "images" / task_name
        task_images_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存图像
        for i, data in enumerate(frame_data):
            # 保存头部相机图像
            head_filename = f"{task_id}_{data.episode_id}_frame_{data.frame_idx:04d}_head.jpg"
            head_path = task_images_dir / head_filename
            cv2.imwrite(str(head_path), cv2.cvtColor(data.head_camera_image, cv2.COLOR_RGB2BGR))
            
            # 保存左手相机图像（如果存在）
            if data.hand_left_camera_image is not None:
                hand_left_filename = f"{task_id}_{data.episode_id}_frame_{data.frame_idx:04d}_hand_left.jpg"
                hand_left_path = task_images_dir / hand_left_filename
                cv2.imwrite(str(hand_left_path), cv2.cvtColor(data.hand_left_camera_image, cv2.COLOR_RGB2BGR))
            
            # 保存右手相机图像（如果存在）
            if data.hand_right_camera_image is not None:
                hand_right_filename = f"{task_id}_{data.episode_id}_frame_{data.frame_idx:04d}_hand_right.jpg"
                hand_right_path = task_images_dir / hand_right_filename
                cv2.imwrite(str(hand_right_path), cv2.cvtColor(data.hand_right_camera_image, cv2.COLOR_RGB2BGR))
        
        # 保存标注数据
        annotations = []
        for i, data in enumerate(frame_data):
            annotation = {
                'head_image': f"{task_name}/{task_id}_{data.episode_id}_frame_{data.frame_idx:04d}_head.jpg",
                'hand_left_image': f"{task_name}/{task_id}_{data.episode_id}_frame_{data.frame_idx:04d}_hand_left.jpg" if data.hand_left_camera_image is not None else None,
                'hand_right_image': f"{task_name}/{task_id}_{data.episode_id}_frame_{data.frame_idx:04d}_hand_right.jpg" if data.hand_right_camera_image is not None else None,
                'position_3d': data.end_effector_position.tolist(),
                'orientation_3d': data.end_effector_orientation.tolist(),
                'timestamp': float(data.timestamp),
                'frame_idx': data.frame_idx,
                'left_effector_position': float(data.left_effector_position),
                'right_effector_position': float(data.right_effector_position),
                'camera_name': data.camera_name,
                'task_id': data.task_id,
                'task_name': data.task_name,
                'episode_id': data.episode_id
            }
            annotations.append(annotation)
        
        annotations_filename = f"{task_name}_annotations.json"
        annotations_path = self.output_dir / "annotations" / annotations_filename
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"  - 保存了 {len(frame_data)} 个夹爪张开帧的数据到 {task_name} 文件夹")
    
    def extract_all(self, max_tasks: Optional[int] = 10):
        """提取所有任务的非抓取帧数据"""
        
        # 获取前max_tasks个task
        task_ids = list(self.task_info.keys())[:max_tasks]
        
        total_samples = 0
        task_stats = {}
        
        for task_id in tqdm(task_ids, desc="Processing tasks"):
            task_info = self.task_info[task_id]
            task_name = task_info['task_name']
            available_episodes = task_info['episodes']
            
            print(f"\nProcessing task {task_id} ({task_name})")
            print(f"  - 可用episodes: {len(available_episodes)}")
            
            # 选择指定数量的episode
            if len(available_episodes) > self.episodes_per_task:
                selected_episodes = random.sample(available_episodes, self.episodes_per_task)
            else:
                selected_episodes = available_episodes
            
            print(f"  - 选择 {len(selected_episodes)} 个episode")
            
            task_frame_data = []
            
            for episode_info in tqdm(selected_episodes, desc=f"Task {task_id} episodes", leave=False):
                # 构建目录路径: task_id/job_id/sn_code/episode_id
                episode_dir = self.data_root / str(episode_info['task_id']) / str(episode_info['job_id']) / episode_info['sn_code'] / str(episode_info['episode_id'])
                
                if not episode_dir.exists():
                    print(f"  - Episode目录不存在: {episode_dir}")
                    continue
                
                episode_frame_data = self.process_episode(
                    episode_dir, 
                    str(task_id), 
                    task_name, 
                    str(episode_info['episode_id'])
                )
                task_frame_data.extend(episode_frame_data)
            
            if task_frame_data:
                self.save_frame_data(task_frame_data, str(task_id), task_name)
                total_samples += len(task_frame_data)
                task_stats[task_id] = {
                    'task_name': task_name,
                    'samples': len(task_frame_data),
                    'episodes': len(selected_episodes)
                }
                print(f"Task {task_id} ({task_name}): {len(task_frame_data)} non-grasp frames from {len(selected_episodes)} episodes")
        
        print(f"\nTotal non-grasp frames extracted: {total_samples}")
        
        # 保存总体统计信息
        stats = {
            'total_non_grasp_frames': total_samples,
            'tasks_processed': len(task_ids),
            'task_statistics': task_stats,
            'output_directory': str(self.output_dir),
            'frames_per_episode': self.frames_per_episode,
            'episodes_per_task': self.episodes_per_task,
            'grasp_threshold': self.grasp_threshold
        }
        
        stats_path = self.output_dir / "extraction_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"数据集生成完成！")
        print(f"输出目录: {self.output_dir}")
        print(f"图像目录: {self.output_dir}/images/ (按task_name分成子文件夹)")
        print(f"标注目录: {self.output_dir}/annotations")


def main():
    parser = argparse.ArgumentParser(description="Extract non-grasp frames from robot manipulation dataset")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for extracted non-grasp frame data")
    parser.add_argument("--max_tasks", type=int, default=10,
                       help="Maximum number of tasks to process (default: 10)")
    parser.add_argument("--frames_per_episode", type=int, default=20,
                       help="Number of frames to extract per episode (default: 20)")
    parser.add_argument("--episodes_per_task", type=int, default=6,
                       help="Number of episodes per task (default: 6)")
    parser.add_argument("--grasp_threshold", type=float, default=10.0,
                       help="Threshold for determining if gripper is open (default: 10.0)")
    
    args = parser.parse_args()
    
    # 创建提取器
    extractor = NonGraspFrameExtractor(args.data_root, args.output_dir)
    
    # 更新参数
    extractor.frames_per_episode = args.frames_per_episode
    extractor.episodes_per_task = args.episodes_per_task
    extractor.grasp_threshold = args.grasp_threshold
    
    # 开始提取
    extractor.extract_all(args.max_tasks)


if __name__ == "__main__":
    main() 