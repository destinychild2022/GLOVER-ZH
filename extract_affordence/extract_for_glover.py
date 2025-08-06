#!/usr/bin/env python3
"""
GLOVER Affordance数据提取脚本
从机械臂操作数据中提取物体的affordance信息，转换为GLOVER训练格式
包括将6DOF位姿转换为图像中的affordance位置点
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


@dataclass
class GLOVERAffordanceData:
    """GLOVER Affordance数据结构"""
    episode_id: str
    task_id: str
    timestamp: float
    end_effector_position: np.ndarray  # (3,) xyz position
    end_effector_orientation: np.ndarray  # (4,) quaternion
    rgb_image: np.ndarray  # (H, W, 3) RGB image
    affordance_mask: np.ndarray  # (H, W) binary mask
    affordance_point: Tuple[int, int]  # (x, y) pixel coordinates
    object_name: str  # 物体名称
    question: str  # 问题文本
    answer: str  # 答案文本


class GLOVERAffordanceExtractor:
    """GLOVER Affordance数据提取器"""
    
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
        (self.output_dir / "annotations").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
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
    
    def project_3d_to_2d_smart(self, point_3d: np.ndarray) -> Tuple[str, Tuple[int, int]]:
        """智能投影：选择最佳相机并投影"""
        best_camera = self.select_best_camera(point_3d)
        projected_point = self.project_3d_to_2d(point_3d, best_camera)
        return best_camera, projected_point
    
    def create_affordance_mask(self, image_shape: Tuple[int, int], 
                              affordance_point: Tuple[int, int], 
                              radius: int = 20) -> np.ndarray:
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
    
    def extract_6dof_pose(self, position: np.ndarray, orientation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """提取6DOF位姿"""
        if len(orientation) == 4:
            if np.linalg.norm(orientation[:3]) > np.linalg.norm(orientation[3:]):
                orientation = np.array([orientation[3], orientation[0], orientation[1], orientation[2]])
        
        orientation = orientation / np.linalg.norm(orientation)
        return position, orientation
    
    def generate_question_answer(self, object_name: str = "object") -> Tuple[str, str]:
        """生成GLOVER格式的问题和答案"""
        question = f"<image>\nWhere should I interact with the {object_name} to pick up it? Please output segmentation mask [SEG]."
        answer = "You can interact with the highlighted area [SEG]."
        return question, answer
    
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
    
    def detect_manipulation_moments(self, h5_data: Dict) -> List[int]:
        """检测机械臂操作物体的关键时刻"""
        if 'end_position' not in h5_data:
            return []
        
        positions = h5_data['end_position']  # (T, 2, 3)
        right_hand_positions = positions[:, 1, :]  # (T, 3)
        
        position_changes = np.linalg.norm(np.diff(right_hand_positions, axis=0), axis=1)
        velocities = position_changes / 0.1  # 假设时间间隔为0.1秒
        
        velocity_threshold = np.percentile(velocities, 75)
        
        manipulation_moments = []
        for i, velocity in enumerate(velocities):
            if velocity > velocity_threshold:
                manipulation_moments.append(i)
        
        if manipulation_moments:
            filtered_moments = [manipulation_moments[0]]
            for i in range(1, len(manipulation_moments)):
                if manipulation_moments[i] - manipulation_moments[i-1] > 5:
                    filtered_moments.append(manipulation_moments[i])
            manipulation_moments = filtered_moments
        
        print(f"  - 检测到 {len(manipulation_moments)} 个操作时刻: {manipulation_moments}")
        return manipulation_moments
    
    def load_camera_data(self, trial_dir: Path, target_timesteps: List[int] = None) -> Optional[Dict]:
        """加载相机数据"""
        # 查找时间步目录
        timestep_dirs = []
        for item in trial_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                timestep_dirs.append(item)
        
        timestep_dirs.sort(key=lambda x: int(x.name))
        
        if not timestep_dirs:
            print(f"  - 未找到时间步目录")
            return None
        
        print(f"  - 找到 {len(timestep_dirs)} 个时间步目录")
        
        # 限制处理的时间步数量
        if target_timesteps is not None:
            timestep_dirs = [timestep_dirs[i] for i in target_timesteps if i < len(timestep_dirs)]
        else:
            timestep_dirs = timestep_dirs[:10]  # 只处理前10个时间步
        
        camera_data = {}
        
        for timestep_dir in timestep_dirs:
            timestep_idx = int(timestep_dir.name)
            camera_dir = timestep_dir / "camera"
            
            if not camera_dir.exists():
                continue
            
            # 相机图像文件直接在camera目录下，不需要子目录
            rgb_files = {
                'head': camera_dir / "head_color.jpg",
                'hand_left': camera_dir / "hand_left_color.jpg", 
                'hand_right': camera_dir / "hand_right_color.jpg"
            }
            
            for camera_name, rgb_file in rgb_files.items():
                if rgb_file.exists():
                    img = cv2.imread(str(rgb_file))
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        camera_data[f'{camera_name}_rgb_{timestep_idx}'] = img
                        print(f"    - 加载 {camera_name} 图像: {rgb_file}")
        
        print(f"  - 加载了 {len(camera_data)} 张图像")
        return camera_data
    
    def load_camera_data_single_timestep(self, camera_image_dir: Path) -> Optional[Dict]:
        """加载单个时间步的相机数据"""
        camera_data = {}
        for camera_name in ['head', 'hand_left', 'hand_right']:
            rgb_file = camera_image_dir / f"{camera_name}_color.jpg"
            if rgb_file.exists():
                img = cv2.imread(str(rgb_file))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    camera_data[f'{camera_name}_rgb'] = img
                    print(f"    - 加载 {camera_name} 图像: {rgb_file}")
        return camera_data

    def process_episode(self, episode_dir: Path, task_id: str, episode_id: str) -> List[GLOVERAffordanceData]:
        """处理单个episode的数据"""
        affordance_data = []
        
        # 查找trial目录（每个episode只有一个trial）
        trial_dirs = []
        for item in episode_dir.iterdir():
            if item.is_dir() and item.name.startswith('A2D'):
                trial_dirs.append(item)
        
        if not trial_dirs:
            print(f"  - 未找到trial目录")
            return affordance_data
        
        trial_dir = trial_dirs[0]  # 每个episode只有一个trial
        print(f"  - 处理trial: {trial_dir.name}")
        
        # 查找trial_data_root目录（如12052046）
        trial_data_root_dirs = []
        for item in trial_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                trial_data_root_dirs.append(item)
        
        if not trial_data_root_dirs:
            print(f"  - 未找到trial数据根目录")
            return affordance_data
        
        trial_data_root_dir = trial_data_root_dirs[0]
        print(f"  - 找到trial数据根目录: {trial_data_root_dir.name}")
        
        # 加载机械臂数据
        h5_data = self.load_h5_data(trial_data_root_dir)
        if h5_data is None:
            return affordance_data
        
        # 检测操作时刻
        manipulation_moments = self.detect_manipulation_moments(h5_data)
        
        # 确定要处理的时间步
        if manipulation_moments:
            target_timesteps = manipulation_moments[:5]  # 只取前5个操作时刻
        else:
            target_timesteps = list(range(min(5, len(h5_data['end_position']))))
        
        print(f"  - 处理时间步: {target_timesteps}")
        
        # 处理每个时间步
        for timestep_idx in target_timesteps:
            if timestep_idx >= len(h5_data['end_position']):
                continue
            
            # 构建相机图像目录路径
            camera_image_dir = trial_data_root_dir / "camera" / str(timestep_idx)
            
            # 加载相机数据
            camera_data = self.load_camera_data_single_timestep(camera_image_dir)
            if camera_data is None:
                continue
            
            # 获取机械臂位姿数据
            position = h5_data['end_position'][timestep_idx][1]  # 右手末端执行器
            orientation = h5_data['end_orientation'][timestep_idx][1]
            timestamp = h5_data['timestamp'][timestep_idx]
            
            position, orientation = self.extract_6dof_pose(position, orientation)
            
            # 为每个相机视角创建affordance数据
            for camera_name in ['head', 'hand_left', 'hand_right']:
                rgb_key = f'{camera_name}_rgb'
                if rgb_key in camera_data:
                    rgb_image = camera_data[rgb_key]
                    
                    # 将3D位姿投影到2D图像坐标
                    affordance_point = self.project_3d_to_2d(position, camera_name)
                    
                    # 检查投影是否有效（在图像范围内）
                    if camera_name == 'head':
                        max_width, max_height = 1280, 720
                    else:
                        max_width, max_height = 848, 480
                    
                    x, y = affordance_point
                    if 0 <= x < max_width and 0 <= y < max_height:
                        # 创建affordance掩码
                        affordance_mask = self.create_affordance_mask(
                            rgb_image.shape[:2], affordance_point
                        )
                        
                        # 生成问题和答案
                        question, answer = self.generate_question_answer("object")
                        
                        affordance = GLOVERAffordanceData(
                            episode_id=episode_id,
                            task_id=task_id,
                            timestamp=timestamp,
                            end_effector_position=position,
                            end_effector_orientation=orientation,
                            rgb_image=rgb_image,
                            affordance_mask=affordance_mask,
                            affordance_point=affordance_point,
                            object_name="object",
                            question=question,
                            answer=answer
                        )
                        affordance_data.append(affordance)
                    else:
                        print(f"    - 跳过 {camera_name} 相机：投影点 ({x}, {y}) 超出图像范围")
                else:
                    print(f"    - 未找到 {camera_name} 图像在 {camera_image_dir}")
        
        print(f"  - 总共创建了 {len(affordance_data)} 个affordance样本")
        return affordance_data
    
    def save_glover_data(self, affordance_data: List[GLOVERAffordanceData], task_id: str):
        """保存GLOVER格式的数据"""
        if not affordance_data:
            return
        
        # 保存图像和掩码
        for i, data in enumerate(affordance_data):
            img_filename = f"{task_id}_{data.episode_id}_{i:06d}.jpg"
            img_path = self.output_dir / "images" / img_filename
            cv2.imwrite(str(img_path), cv2.cvtColor(data.rgb_image, cv2.COLOR_RGB2BGR))
            
            mask_filename = f"{task_id}_{data.episode_id}_{i:06d}.png"
            mask_path = self.output_dir / "masks" / mask_filename
            cv2.imwrite(str(mask_path), data.affordance_mask)
        
        # 保存标注数据
        annotations = []
        for i, data in enumerate(affordance_data):
            annotation = {
                'image_path': f"{task_id}_{data.episode_id}_{i:06d}.jpg",
                'mask_path': f"{task_id}_{data.episode_id}_{i:06d}.png",
                'question': data.question,
                'answer': data.answer,
                'affordance_point': list(data.affordance_point),  # 转换为list
                'position_3d': data.end_effector_position.tolist(),
                'orientation_3d': data.end_effector_orientation.tolist(),
                'timestamp': float(data.timestamp),  # 转换为float以支持JSON序列化
                'task_id': data.task_id,
                'episode_id': data.episode_id
            }
            annotations.append(annotation)
        
        annotations_filename = f"{task_id}_annotations.json"
        annotations_path = self.output_dir / "annotations" / annotations_filename
        with open(annotations_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        # 保存元数据
        metadata = {
            'task_id': task_id,
            'num_samples': len(affordance_data),
            'episode_ids': list(set([data.episode_id for data in affordance_data])),
            'data_format': {
                'image': 'RGB (H, W, 3)',
                'mask': 'Binary (H, W)',
                'affordance_point': '(x, y) pixel coordinates',
                'position_3d': 'xyz (3,)',
                'orientation_3d': 'quaternion (w, x, y, z) (4,)'
            },
            'glover_compatible': True
        }
        
        metadata_filename = f"{task_id}_metadata.json"
        metadata_path = self.output_dir / "metadata" / metadata_filename
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def extract_all(self, max_tasks: Optional[int] = None, max_episodes_per_task: Optional[int] = None, test_episode: Optional[str] = None):
        """提取所有任务的affordance数据"""
        
        if test_episode:
            # 测试单个episode
            print(f"测试单个episode: {test_episode}")
            task_id, episode_id = test_episode.split('/')
            
            task_dir = self.data_root / task_id
            episode_dir = task_dir / episode_id
            
            if not task_dir.exists():
                print(f"任务目录不存在: {task_dir}")
                return
            if not episode_dir.exists():
                print(f"Episode目录不存在: {episode_dir}")
                return
            
            print(f"处理任务 {task_id}, episode {episode_id}")
            
            episode_affordance_data = self.process_episode(episode_dir, task_id, episode_id)
            
            if episode_affordance_data:
                self.save_glover_data(episode_affordance_data, task_id)
                print(f"总共提取了 {len(episode_affordance_data)} 个样本")
            else:
                print("未提取到任何数据")
            
            return
        
        # 批量处理逻辑
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
            
            task_affordance_data = []
            
            for episode_dir in tqdm(episode_dirs, desc=f"Task {task_id} episodes", leave=False):
                episode_id = episode_dir.name
                episode_affordance_data = self.process_episode(episode_dir, task_id, episode_id)
                task_affordance_data.extend(episode_affordance_data)
            
            if task_affordance_data:
                self.save_glover_data(task_affordance_data, task_id)
                total_samples += len(task_affordance_data)
                print(f"Task {task_id}: {len(task_affordance_data)} samples")
        
        print(f"\nTotal samples extracted: {total_samples}")
        
        # 保存总体统计信息
        stats = {
            'total_samples': total_samples,
            'tasks_processed': len(task_dirs),
            'output_directory': str(self.output_dir),
            'glover_compatible': True
        }
        
        stats_path = self.output_dir / "extraction_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Extract GLOVER affordance data from robot manipulation dataset")
    parser.add_argument("--data_root", type=str, required=True,
                       help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for extracted affordance data")
    parser.add_argument("--max_tasks", type=int, default=None,
                       help="Maximum number of tasks to process")
    parser.add_argument("--max_episodes_per_task", type=int, default=None,
                       help="Maximum number of episodes per task to process")
    parser.add_argument("--test_episode", type=str, default=None,
                       help="Test a single episode in the format 'task_id/episode_id'")
    
    args = parser.parse_args()
    
    # 创建提取器
    extractor = GLOVERAffordanceExtractor(args.data_root, args.output_dir)
    
    # 开始提取
    extractor.extract_all(args.max_tasks, args.max_episodes_per_task, args.test_episode)


if __name__ == "__main__":
    main() 