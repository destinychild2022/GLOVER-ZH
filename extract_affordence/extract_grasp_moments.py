#!/usr/bin/env python3
"""
抓取时刻数据提取和可视化脚本
从机械臂操作数据中提取抓取时刻，生成affordance mask，并创建可视化
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
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle


@dataclass
class GraspMomentData:
    """抓取时刻数据结构"""
    episode_id: str
    task_id: str
    grasp_timestamp: float
    pre_grasp_timestamp: float
    stable_close_timestamp: float
    end_effector_position: np.ndarray  # (3,) xyz position
    end_effector_orientation: np.ndarray  # (4,) quaternion
    pre_grasp_image: np.ndarray  # 抓取前的图像
    stable_close_image: np.ndarray  # 稳定闭合后的图像
    head_camera_image: np.ndarray  # 头部相机图像
    affordance_mask: np.ndarray  # (H, W) binary mask
    affordance_point: Tuple[int, int]  # (x, y) pixel coordinates
    camera_name: str  # 使用的相机名称
    head_affordance_point: Tuple[int, int]  # 头部相机中的affordance点


class GraspMomentExtractor:
    """抓取时刻数据提取器"""
    
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
        (self.output_dir / "masks").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        
        # 初始化相机参数存储
        self.camera_intrinsics = {}
        self.camera_extrinsics = {}
        
        # 加载相机参数
        self.load_camera_parameters()
    
    def load_camera_parameters(self):
        """加载相机参数"""
        print("开始加载相机参数...")
        
        # 使用已知的相机参数路径
        param_dir = Path("/mnt/data-oss/rap-prod-bak/AgibotWorld-Challenge/AgiBotWorldChallenge-2025/iros_agibot_sim/2810130/3335440/A2D0015AB00061/12052046/parameters/camera")
        
        if not param_dir.exists():
            print(f"相机参数目录不存在: {param_dir}")
            print("使用默认相机参数")
            self.setup_default_camera_parameters()
            return
        
        print(f"找到相机参数目录: {param_dir}")
        
        # 加载内参
        intrinsic_files = {
            'head': param_dir / 'head_intrinsic_params.json',
            'hand_left': param_dir / 'hand_left_intrinsic_params.json',
            'hand_right': param_dir / 'hand_right_intrinsic_params.json'
        }
        
        for camera_name, intrinsic_file in intrinsic_files.items():
            if intrinsic_file.exists():
                try:
                    with open(intrinsic_file, 'r') as f:
                        intrinsic_data = json.load(f)
                    
                    fx = intrinsic_data['fx']
                    fy = intrinsic_data['fy']
                    cx = intrinsic_data['ppx']
                    cy = intrinsic_data['ppy']
                    
                    intrinsic_matrix = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ])
                    
                    self.camera_intrinsics[camera_name] = {
                        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                        'matrix': intrinsic_matrix
                    }
                    
                    print(f"加载 {camera_name} 内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                    
                except Exception as e:
                    print(f"加载 {camera_name} 内参失败: {e}")
        
        # 加载外参
        extrinsic_files = {
            'head': param_dir / 'head_extrinsic_params_aligned.json',
            'hand_left': param_dir / 'hand_left_extrinsic_params_aligned.json',
            'hand_right': param_dir / 'hand_right_extrinsic_params_aligned.json'
        }
        
        for camera_name, extrinsic_file in extrinsic_files.items():
            if extrinsic_file.exists():
                try:
                    with open(extrinsic_file, 'r') as f:
                        extrinsic_data = json.load(f)
                    
                    if isinstance(extrinsic_data, list) and len(extrinsic_data) > 0:
                        extrinsic = extrinsic_data[0]['extrinsic']
                    else:
                        extrinsic = extrinsic_data['extrinsic']
                    
                    rotation_matrix = np.array(extrinsic['rotation_matrix'])
                    translation_vector = np.array(extrinsic['translation_vector'])
                    
                    self.camera_extrinsics[camera_name] = {
                        'rotation': rotation_matrix,
                        'translation': translation_vector
                    }
                    
                    print(f"加载 {camera_name} 外参: 旋转矩阵形状={rotation_matrix.shape}, 平移向量={translation_vector}")
                    
                except Exception as e:
                    print(f"加载 {camera_name} 外参失败: {e}")
        
        print(f"成功加载 {len(self.camera_intrinsics)} 个相机的内参")
        print(f"成功加载 {len(self.camera_extrinsics)} 个相机的外参")
    
    def setup_default_camera_parameters(self):
        """设置默认相机参数"""
        default_intrinsics = {
            'fx': 525.0, 'fy': 525.0, 'cx': 320.0, 'cy': 240.0,
            'matrix': np.array([[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1]])
        }
        
        default_extrinsics = {
            'rotation': np.eye(3),
            'translation': np.array([0, 0, 0])
        }
        
        for camera_name in ['head', 'hand_left', 'hand_right']:
            self.camera_intrinsics[camera_name] = default_intrinsics.copy()
            self.camera_extrinsics[camera_name] = default_extrinsics.copy()
    
    def world_to_camera(self, point_world: np.ndarray, camera_name: str) -> np.ndarray:
        """将世界坐标转换为相机坐标"""
        if camera_name not in self.camera_extrinsics:
            print(f"警告: 未找到相机 {camera_name} 的外参，使用默认变换")
            return point_world
        
        extrinsics = self.camera_extrinsics[camera_name]
        R = extrinsics['rotation']
        t = extrinsics['translation']
        
        point_camera = R @ point_world + t
        return point_camera
    
    def camera_to_image(self, point_camera: np.ndarray, camera_name: str) -> Tuple[int, int]:
        """将相机坐标转换为图像坐标"""
        if camera_name not in self.camera_intrinsics:
            print(f"警告: 未找到相机 {camera_name} 的内参，使用默认投影")
            return (320, 240)
        
        intrinsics = self.camera_intrinsics[camera_name]
        K = intrinsics['matrix']
        
        point_homogeneous = K @ point_camera
        x = int(point_homogeneous[0] / point_homogeneous[2])
        y = int(point_homogeneous[1] / point_homogeneous[2])
        
        return (x, y)
    
    def project_3d_to_2d(self, point_3d: np.ndarray, camera_name: str) -> Tuple[int, int]:
        """将3D点投影到2D图像坐标"""
        try:
            point_camera = self.world_to_camera(point_3d, camera_name)
            
            if point_camera[2] <= 0:
                print(f"警告: 点 {point_3d} 在相机 {camera_name} 后方")
                intrinsics = self.camera_intrinsics[camera_name]
                return (int(intrinsics['cx']), int(intrinsics['cy']))
            
            point_image = self.camera_to_image(point_camera, camera_name)
            
            # 根据相机类型设置图像尺寸
            if camera_name == 'head':
                max_width, max_height = 1280, 720
            else:  # hand_left, hand_right
                max_width, max_height = 848, 480
            
            x, y = point_image
            
            # 检查是否在视野范围内
            in_fov = (0 <= x < max_width) and (0 <= y < max_height)
            
            if not in_fov:
                print(f"警告: 点 {point_3d} 超出 {camera_name} 相机视野范围")
                # 返回图像中心作为fallback
                return (max_width // 2, max_height // 2)
            
            return (x, y)
            
        except Exception as e:
            print(f"投影失败: {e}")
            intrinsics = self.camera_intrinsics[camera_name]
            return (int(intrinsics['cx']), int(intrinsics['cy']))
    
    def select_best_camera(self, point_3d: np.ndarray) -> str:
        """选择最佳相机视角"""
        best_camera = None
        best_score = float('inf')
        
        for camera_name in ['head', 'hand_left', 'hand_right']:
            try:
                point_camera = self.world_to_camera(point_3d, camera_name)
                
                # 检查点是否在相机前方
                if point_camera[2] <= 0:
                    continue
                
                # 投影到图像坐标
                point_image = self.camera_to_image(point_camera, camera_name)
                x, y = point_image
                
                # 设置图像尺寸
                if camera_name == 'head':
                    max_width, max_height = 1280, 720
                else:
                    max_width, max_height = 848, 480
                
                # 计算视野内得分
                if 0 <= x < max_width and 0 <= y < max_height:
                    # 计算距离图像中心的距离（越小越好）
                    center_x, center_y = max_width // 2, max_height // 2
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    
                    # 考虑相机Z距离（越近越好）
                    z_distance = point_camera[2]
                    
                    # 综合得分（距离中心越近，Z距离越近，得分越低）
                    score = distance + z_distance * 0.1
                    
                    if score < best_score:
                        best_score = score
                        best_camera = camera_name
                
            except Exception as e:
                print(f"选择相机时出错 {camera_name}: {e}")
                continue
        
        return best_camera or 'head'  # 默认使用head相机
    
    def create_affordance_mask(self, image_shape: Tuple[int, int], 
                              affordance_point: Tuple[int, int], 
                              radius: int = 30) -> np.ndarray:
        """创建affordance掩码"""
        H, W = image_shape
        mask = np.zeros((H, W), dtype=np.uint8)
        
        x, y = affordance_point
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        
        y_coords, x_coords = np.ogrid[:H, :W]
        distance = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
        
        sigma = radius / 3.0
        gaussian_mask = np.exp(-(distance**2) / (2 * sigma**2))
        
        threshold = 0.1
        mask = (gaussian_mask > threshold).astype(np.uint8) * 255
        
        return mask
    
    def detect_grasp_moments(self, h5_data: Dict) -> List[Dict]:
        """检测夹爪闭合时刻（position从接近0到接近1）"""
        if 'end_position' not in h5_data:
            return []
        
        positions = h5_data['end_position']  # (T, 2, 3)
        
        # 分析左右手的位置变化
        left_hand_positions = positions[:, 0, :]  # (T, 3)
        right_hand_positions = positions[:, 1, :]  # (T, 3)
        
        # 计算夹爪开合状态（通过末端执行器之间的距离来判断）
        gripper_distances = np.linalg.norm(left_hand_positions - right_hand_positions, axis=1)
        
        # 计算夹爪状态的变化
        gripper_changes = np.diff(gripper_distances)
        
        # 检测夹爪闭合时刻（距离突然减小）
        close_threshold = np.percentile(gripper_changes, 15)  # 取15%分位数作为闭合阈值
        
        grasp_moments = []
        
        # 检测夹爪闭合时刻
        for i, change in enumerate(gripper_changes):
            if change < close_threshold:  # 距离突然减小，表示夹爪闭合
                grasp_moments.append(i + 1)  # +1因为diff后索引偏移
        
        # 去重并排序
        grasp_moments = sorted(list(set(grasp_moments)))
        
        # 过滤太近的时刻（至少间隔20帧）
        if grasp_moments:
            filtered_moments = [grasp_moments[0]]
            for i in range(1, len(grasp_moments)):
                if grasp_moments[i] - grasp_moments[i-1] > 20:  # 至少间隔20帧
                    filtered_moments.append(grasp_moments[i])
            grasp_moments = filtered_moments
        
        # 为每个抓取时刻确定详细信息
        grasp_details = []
        for grasp_idx in grasp_moments:
            if grasp_idx >= len(positions):
                continue
            
            # 分析是左臂还是右臂在抓取
            # 通过比较左右手的位置变化来判断
            left_velocity = np.linalg.norm(np.diff(left_hand_positions[max(0, grasp_idx-5):grasp_idx+1], axis=0), axis=1)
            right_velocity = np.linalg.norm(np.diff(right_hand_positions[max(0, grasp_idx-5):grasp_idx+1], axis=0), axis=1)
            
            # 计算平均速度
            left_avg_velocity = np.mean(left_velocity) if len(left_velocity) > 0 else 0
            right_avg_velocity = np.mean(right_velocity) if len(right_velocity) > 0 else 0
            
            # 判断哪个手臂在抓取（速度更大的那个）
            if left_avg_velocity > right_avg_velocity:
                grasping_arm = 'left'
                camera_name = 'hand_left'
                position = left_hand_positions[grasp_idx]
                orientation = h5_data['end_orientation'][grasp_idx][0]  # 左手
            else:
                grasping_arm = 'right'
                camera_name = 'hand_right'
                position = right_hand_positions[grasp_idx]
                orientation = h5_data['end_orientation'][grasp_idx][1]  # 右手
            
            # 寻找抓取前的时刻（距离较大的时刻）
            pre_grasp_idx = grasp_idx
            for i in range(grasp_idx-1, max(0, grasp_idx-20), -1):
                if gripper_distances[i] > gripper_distances[grasp_idx] * 1.2:  # 距离明显更大
                    pre_grasp_idx = i
                    break
            
            # 寻找稳定闭合后的时刻（延后几帧，确保夹爪稳定闭合）
            stable_close_idx = grasp_idx
            for i in range(grasp_idx+1, min(len(gripper_distances), grasp_idx+15)):
                # 检查后续几帧的距离是否稳定在较小值
                if (gripper_distances[i] < gripper_distances[grasp_idx] * 0.9 and 
                    abs(gripper_distances[i] - gripper_distances[i-1]) < 0.01):  # 距离稳定且较小
                    stable_close_idx = i
                    break
            
            grasp_details.append({
                'grasp_idx': grasp_idx,
                'pre_grasp_idx': pre_grasp_idx,
                'stable_close_idx': stable_close_idx,
                'grasping_arm': grasping_arm,
                'camera_name': camera_name,
                'position': position,
                'orientation': orientation,
                'grasp_timestamp': h5_data['timestamp'][grasp_idx],
                'pre_grasp_timestamp': h5_data['timestamp'][pre_grasp_idx],
                'stable_close_timestamp': h5_data['timestamp'][stable_close_idx]
            })
        
        print(f"  - 检测到 {len(grasp_details)} 个夹爪闭合时刻")
        for i, detail in enumerate(grasp_details):
            print(f"    - 时刻 {i+1}: {detail['grasping_arm']}臂抓取, 相机: {detail['camera_name']}, "
                  f"抓取前: {detail['pre_grasp_idx']}, 抓取时: {detail['grasp_idx']}, 稳定闭合: {detail['stable_close_idx']}")
        
        return grasp_details
    
    def load_h5_data(self, trial_data_root_dir: Path) -> Optional[Dict]:
        """加载H5文件中的机械臂数据"""
        h5_file = trial_data_root_dir / "aligned_joints.h5"
        if not h5_file.exists():
            print(f"  - 在 {trial_data_root_dir} 中未找到H5文件")
            return None
        
        print(f"  - 找到H5文件: {h5_file}")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                data = {
                    'end_position': f['action']['end']['position'][:],
                    'end_orientation': f['action']['end']['orientation'][:],
                    'timestamp': f['timestamp'][:]
                }
                
                if 'joint' in f['action']:
                    data['joint_positions'] = f['action']['joint']['position'][:]
                    data['joint_velocities'] = f['action']['joint']['velocity'][:]
                
                print(f"  - 成功加载H5数据: end_position={data['end_position'].shape}")
                return data
                
        except Exception as e:
            print(f"  - 加载H5文件失败: {e}")
            return None
    
    def load_camera_image(self, camera_image_dir: Path, camera_name: str) -> Optional[np.ndarray]:
        """加载单个相机的图像"""
        rgb_file = camera_image_dir / f"{camera_name}_color.jpg"
        if rgb_file.exists():
            img = cv2.imread(str(rgb_file))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
        return None
    
    def process_episode(self, episode_dir: Path, task_id: str, episode_id: str) -> List[GraspMomentData]:
        """处理单个episode的抓取时刻数据"""
        grasp_data = []
        
        # 查找trial目录
        trial_dirs = []
        for item in episode_dir.iterdir():
            if item.is_dir() and item.name.startswith('A2D'):
                trial_dirs.append(item)
        
        if not trial_dirs:
            print(f"  - 未找到trial目录")
            return grasp_data
        
        trial_dir = trial_dirs[0]
        print(f"  - 处理trial: {trial_dir.name}")
        
        # 查找trial_data_root目录
        trial_data_root_dirs = []
        for item in trial_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                trial_data_root_dirs.append(item)
        
        if not trial_data_root_dirs:
            print(f"  - 未找到trial数据根目录")
            return grasp_data
        
        trial_data_root_dir = trial_data_root_dirs[0]
        print(f"  - 找到trial数据根目录: {trial_data_root_dir.name}")
        
        # 加载机械臂数据
        h5_data = self.load_h5_data(trial_data_root_dir)
        if h5_data is None:
            return grasp_data
        
        # 检测抓取时刻
        grasp_details = self.detect_grasp_moments(h5_data)
        
        if not grasp_details:
            print(f"  - 未检测到抓取时刻")
            return grasp_data
        
        # 处理每个抓取时刻
        for detail in grasp_details:
            grasp_idx = detail['grasp_idx']
            pre_grasp_idx = detail['pre_grasp_idx']
            stable_close_idx = detail['stable_close_idx']
            grasping_arm = detail['grasping_arm']
            camera_name = detail['camera_name']
            position = detail['position']
            orientation = detail['orientation']
            grasp_timestamp = detail['grasp_timestamp']
            pre_grasp_timestamp = detail['pre_grasp_timestamp']
            stable_close_timestamp = detail['stable_close_timestamp']
            
            # 根据抓取的手臂选择对应的相机视角
            if grasping_arm == 'left':
                best_camera = 'hand_left'
            elif grasping_arm == 'right':
                best_camera = 'hand_right'
            else:
                best_camera = self.select_best_camera(position)
            
            # 加载抓取前和抓取后的图像
            pre_grasp_image = self.load_camera_image(trial_data_root_dir / "camera" / str(pre_grasp_idx), best_camera)
            stable_close_image = self.load_camera_image(trial_data_root_dir / "camera" / str(stable_close_idx), best_camera)
            
            # 加载头部相机图像
            head_camera_image = self.load_camera_image(trial_data_root_dir / "camera" / str(stable_close_idx), 'head')
            
            if pre_grasp_image is None or stable_close_image is None:
                print(f"    - 跳过抓取时刻 {grasp_idx}: 缺少图像")
                continue
            
            # 将3D位姿投影到2D图像坐标
            affordance_point = self.project_3d_to_2d(position, best_camera)
            head_affordance_point = self.project_3d_to_2d(position, 'head')
            
            # 检查投影是否有效
            if best_camera == 'head':
                max_width, max_height = 1280, 720
            else:
                max_width, max_height = 848, 480
            
            x, y = affordance_point
            if 0 <= x < max_width and 0 <= y < max_height:
                # 创建affordance掩码
                affordance_mask = self.create_affordance_mask(
                    pre_grasp_image.shape[:2], affordance_point
                )
                
                grasp_moment = GraspMomentData(
                    episode_id=episode_id,
                    task_id=task_id,
                    grasp_timestamp=grasp_timestamp,
                    pre_grasp_timestamp=pre_grasp_timestamp,
                    stable_close_timestamp=stable_close_timestamp,
                    end_effector_position=position,
                    end_effector_orientation=orientation,
                    pre_grasp_image=pre_grasp_image,
                    stable_close_image=stable_close_image,
                    head_camera_image=head_camera_image,
                    affordance_mask=affordance_mask,
                    affordance_point=affordance_point,
                    camera_name=best_camera,
                    head_affordance_point=head_affordance_point
                )
                grasp_data.append(grasp_moment)
                print(f"    - 成功提取抓取时刻 {grasp_idx} 的数据 ({grasping_arm}臂, {best_camera}相机)")
            else:
                print(f"    - 跳过抓取时刻 {grasp_idx}: 投影点 ({x}, {y}) 超出图像范围")
        
        print(f"  - 总共提取了 {len(grasp_data)} 个抓取时刻")
        return grasp_data
    
    def create_visualization(self, grasp_data: List[GraspMomentData], task_id: str):
        """创建可视化"""
        if not grasp_data:
            return
        
        # 为每个episode创建可视化
        episodes = {}
        for data in grasp_data:
            if data.episode_id not in episodes:
                episodes[data.episode_id] = []
            episodes[data.episode_id].append(data)
        
        for episode_id, episode_data in episodes.items():
            # 创建大图 - 3行：抓取前、抓取后、头部相机
            num_moments = len(episode_data)
            fig, axes = plt.subplots(3, num_moments, figsize=(5*num_moments, 15))
            
            if num_moments == 1:
                axes = axes.reshape(3, 1)
            
            for i, data in enumerate(episode_data):
                # 抓取前的图像 + affordance mask
                ax1 = axes[0, i]
                ax1.imshow(data.pre_grasp_image)
                
                # 添加affordance mask
                mask_overlay = np.zeros_like(data.pre_grasp_image)
                mask_overlay[data.affordance_mask > 0] = [255, 0, 0]  # 红色
                ax1.imshow(mask_overlay, alpha=0.3)
                
                # 添加affordance点
                x, y = data.affordance_point
                circle = Circle((x, y), 10, color='red', fill=False, linewidth=2)
                ax1.add_patch(circle)
                
                ax1.set_title(f'Pre-grasp ({data.camera_name})\nAffordance Point: ({x}, {y})')
                ax1.axis('off')
                
                # 稳定闭合后的图像
                ax2 = axes[1, i]
                ax2.imshow(data.stable_close_image)
                ax2.set_title(f'Stable Close ({data.camera_name})\nTimestamp: {data.stable_close_timestamp:.2f}')
                ax2.axis('off')
                
                # 头部相机图像 + 6DOF位姿标注
                ax3 = axes[2, i]
                if data.head_camera_image is not None:
                    ax3.imshow(data.head_camera_image)
                    
                    # 添加6DOF位姿标注
                    head_x, head_y = data.head_affordance_point
                    circle = Circle((head_x, head_y), 15, color='red', fill=False, linewidth=3)
                    ax3.add_patch(circle)
                    
                    # 添加位姿信息文本
                    pos = data.end_effector_position
                    orient = data.end_effector_orientation
                    pose_text = f'Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})\nOrient: ({orient[0]:.2f}, {orient[1]:.2f}, {orient[2]:.2f}, {orient[3]:.2f})'
                    ax3.text(10, 30, pose_text, fontsize=8, color='white', 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7))
                    
                    ax3.set_title(f'Head Camera (6DOF Pose)\nPoint: ({head_x}, {head_y})')
                else:
                    ax3.text(0.5, 0.5, 'No Head Camera Image', ha='center', va='center', transform=ax3.transAxes)
                    ax3.set_title('Head Camera (No Image)')
                ax3.axis('off')
            
            # 保存可视化
            vis_filename = f"{task_id}_{episode_id}_gripper_close_with_head_visualization.png"
            vis_path = self.output_dir / "visualizations" / vis_filename
            plt.tight_layout()
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  - 保存可视化: {vis_filename}")
    
    def save_grasp_data(self, grasp_data: List[GraspMomentData], task_id: str):
        """保存抓取时刻数据"""
        if not grasp_data:
            return
        
        # 保存图像和掩码
        for i, data in enumerate(grasp_data):
            # 保存抓取前图像
            pre_grasp_filename = f"{task_id}_{data.episode_id}_pregrasp_{i:03d}.jpg"
            pre_grasp_path = self.output_dir / "images" / pre_grasp_filename
            cv2.imwrite(str(pre_grasp_path), cv2.cvtColor(data.pre_grasp_image, cv2.COLOR_RGB2BGR))
            
            # 保存稳定闭合后图像
            stable_close_filename = f"{task_id}_{data.episode_id}_stable_close_{i:03d}.jpg"
            stable_close_path = self.output_dir / "images" / stable_close_filename
            cv2.imwrite(str(stable_close_path), cv2.cvtColor(data.stable_close_image, cv2.COLOR_RGB2BGR))
            
            # 保存头部相机图像
            if data.head_camera_image is not None:
                head_camera_filename = f"{task_id}_{data.episode_id}_head_camera_{i:03d}.jpg"
                head_camera_path = self.output_dir / "images" / head_camera_filename
                cv2.imwrite(str(head_camera_path), cv2.cvtColor(data.head_camera_image, cv2.COLOR_RGB2BGR))
            
            # 保存affordance掩码
            mask_filename = f"{task_id}_{data.episode_id}_mask_{i:03d}.png"
            mask_path = self.output_dir / "masks" / mask_filename
            cv2.imwrite(str(mask_path), data.affordance_mask)
        
        # 保存标注数据
        annotations = []
        for i, data in enumerate(grasp_data):
            annotation = {
                'pregrasp_image': f"{task_id}_{data.episode_id}_pregrasp_{i:03d}.jpg",
                'stable_close_image': f"{task_id}_{data.episode_id}_stable_close_{i:03d}.jpg" if data.stable_close_image is not None else None,
                'head_camera_image': f"{task_id}_{data.episode_id}_head_camera_{i:03d}.jpg" if data.head_camera_image is not None else None,
                'mask_image': f"{task_id}_{data.episode_id}_mask_{i:03d}.png",
                'affordance_point': list(data.affordance_point),
                'head_affordance_point': list(data.head_affordance_point) if data.head_affordance_point is not None else None,
                'position_3d': data.end_effector_position.tolist(),
                'orientation_3d': data.end_effector_orientation.tolist(),
                'grasp_timestamp': float(data.grasp_timestamp),
                'pregrasp_timestamp': float(data.pre_grasp_timestamp),
                'stable_close_timestamp': float(data.stable_close_timestamp),
                'camera_name': data.camera_name,
                'task_id': data.task_id,
                'episode_id': data.episode_id
            }
            annotations.append(annotation)
        
        annotations_filename = f"{task_id}_grasp_annotations.json"
        annotations_path = self.output_dir / "annotations" / annotations_filename
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        # 创建可视化
        self.create_visualization(grasp_data, task_id)
    
    def extract_all(self, max_tasks: Optional[int] = None, max_episodes_per_task: Optional[int] = None):
        """提取所有任务的抓取时刻数据"""
        
        task_dirs = []
        for task_dir in self.data_root.iterdir():
            if task_dir.is_dir() and task_dir.name.isdigit():
                task_dirs.append(task_dir)
        
        task_dirs.sort()
        
        if max_tasks:
            task_dirs = task_dirs[:max_tasks]
        
        total_samples = 0
        
        for task_dir in tqdm(task_dirs, desc="Processing tasks"):
            task_id = task_dir.name
            print(f"\nProcessing task {task_id}")
            
            episode_dirs = []
            for episode_dir in task_dir.iterdir():
                if episode_dir.is_dir() and episode_dir.name.isdigit():
                    episode_dirs.append(episode_dir)
            
            episode_dirs.sort()
            
            if max_episodes_per_task:
                episode_dirs = episode_dirs[:max_episodes_per_task]
            
            task_grasp_data = []
            
            for episode_dir in tqdm(episode_dirs, desc=f"Task {task_id} episodes", leave=False):
                episode_id = episode_dir.name
                episode_grasp_data = self.process_episode(episode_dir, task_id, episode_id)
                task_grasp_data.extend(episode_grasp_data)
            
            if task_grasp_data:
                self.save_grasp_data(task_grasp_data, task_id)
                total_samples += len(task_grasp_data)
                print(f"Task {task_id}: {len(task_grasp_data)} grasp moments")
        
        print(f"\nTotal grasp moments extracted: {total_samples}")
        
        # 保存总体统计信息
        stats = {
            'total_grasp_moments': total_samples,
            'tasks_processed': len(task_dirs),
            'output_directory': str(self.output_dir)
        }
        
        stats_path = self.output_dir / "grasp_extraction_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract grasp moments from robot manipulation dataset")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for extracted grasp data")
    parser.add_argument("--max_tasks", type=int, default=None,
                       help="Maximum number of tasks to process")
    parser.add_argument("--max_episodes_per_task", type=int, default=None,
                       help="Maximum number of episodes per task to process")
    
    args = parser.parse_args()
    
    # 创建提取器
    extractor = GraspMomentExtractor(args.data_root, args.output_dir)
    
    # 开始提取
    extractor.extract_all(args.max_tasks, args.max_episodes_per_task)


if __name__ == "__main__":
    main() 