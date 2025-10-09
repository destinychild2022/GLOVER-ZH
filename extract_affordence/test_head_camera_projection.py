#!/usr/bin/env python3
"""
头部相机投影测试脚本
测试3D点到2D图像的投影，验证外参矩阵变换的效果
"""

import os
import json
import numpy as np
import cv2
import h5py
from pathlib import Path
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class HeadCameraProjectionTester:
    """头部相机投影测试器"""
    
    def __init__(self, episode_path: str = None, frame_idx: int = None):
        # 初始化相机参数存储
        self.camera_intrinsics = {}
        self.camera_extrinsics = {}
        
        # 定义坐标系统变换矩阵（Y和Z轴反射变换）
        self.COORDSYS_TRANSFORM = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],  # Y轴反射
            [0, 0, -1, 0],  # Z轴反射
            [0, 0, 0, 1]
        ])
        
        # 加载相机参数
        self.load_camera_parameters(episode_path, frame_idx)
    
    def load_camera_parameters(self, episode_path: str = None, frame_idx: int = None):
        """加载相机参数"""
        print("开始加载相机参数...")
        
        # 如果提供了episode路径，使用该episode的相机参数
        if episode_path:
            param_dir = Path(episode_path) / "parameters" / "camera"
        else:
            # 使用已知的相机参数路径作为默认
            param_dir = Path("/mnt/data-oss/rap-prod-bak/AgibotWorld-Challenge/AgiBotWorldChallenge-2025/iros_agibot_sim/2810130/3335440/A2D0015AB00061/12052046/parameters/camera")
        
        if not param_dir.exists():
            print(f"相机参数目录不存在: {param_dir}")
            return
        
        print(f"找到相机参数目录: {param_dir}")
        
        # 只加载头部相机内参
        intrinsic_file = param_dir / 'head_intrinsic_params.json'
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
                
                self.camera_intrinsics['head'] = {
                    'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                    'matrix': intrinsic_matrix
                }
                
                print(f"加载 head 内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
                print(f"内参矩阵 K:\n{intrinsic_matrix}")
                
            except Exception as e:
                print(f"加载 head 内参失败: {e}")
        
        # 加载头部相机外参（按帧加载）
        extrinsic_file = param_dir / 'head_extrinsic_params_aligned.json'
        if extrinsic_file.exists():
            try:
                with open(extrinsic_file, 'r') as f:
                    extrinsic_data = json.load(f)
                
                print(f"外参文件包含 {len(extrinsic_data)} 帧数据")
                
                # 如果指定了帧索引，使用对应帧的外参
                if frame_idx is not None:
                    if 0 <= frame_idx < len(extrinsic_data):
                        extrinsic = extrinsic_data[frame_idx]['extrinsic']
                        print(f"使用第 {frame_idx} 帧的外参")
                    else:
                        print(f"警告: 帧索引 {frame_idx} 超出范围 [0, {len(extrinsic_data)-1}]，使用第0帧")
                        extrinsic = extrinsic_data[0]['extrinsic']
                else:
                    # 默认使用第0帧
                    extrinsic = extrinsic_data[0]['extrinsic']
                    print(f"使用第 0 帧的外参")
                
                rotation_matrix = np.array(extrinsic['rotation_matrix'])
                translation_vector = np.array(extrinsic['translation_vector'])
                
                print(f"\n原始外参 (第{frame_idx if frame_idx is not None else 0}帧):")
                print(f"旋转矩阵 R:\n{rotation_matrix}")
                print(f"平移向量 t: {translation_vector}")
                
                # 构建4x4相机外参矩阵
                camera_extrinsic = np.eye(4)
                camera_extrinsic[:3, :3] = rotation_matrix
                camera_extrinsic[:3, 3] = translation_vector
                
                print(f"\n原始4x4外参矩阵 T_camera_world:\n{camera_extrinsic}")
                
                # 应用坐标系统变换矩阵
                camera_extrinsic_transformed = camera_extrinsic @ self.COORDSYS_TRANSFORM
                
                print(f"\n坐标系统变换矩阵:\n{self.COORDSYS_TRANSFORM}")
                print(f"\n变换后的4x4外参矩阵 T_camera_world:\n{camera_extrinsic_transformed}")
                
                # 提取变换后的旋转矩阵和平移向量
                rotation_matrix_transformed = camera_extrinsic_transformed[:3, :3]
                translation_vector_transformed = camera_extrinsic_transformed[:3, 3]
                
                self.camera_extrinsics['head'] = {
                    'rotation': rotation_matrix_transformed,
                    'translation': translation_vector_transformed
                }
                
                print(f"\n变换后的外参:")
                print(f"旋转矩阵 R_transformed:\n{rotation_matrix_transformed}")
                print(f"平移向量 t_transformed: {translation_vector_transformed}")
                
            except Exception as e:
                print(f"加载 head 外参失败: {e}")
    
    def detect_grasp_moments_from_h5(self, h5_file_path: str) -> List[Dict]:
        """从H5文件中检测抓取时刻"""
        print(f"从H5文件检测抓取时刻: {h5_file_path}")
        
        grasp_moments = []
        
        try:
            with h5py.File(h5_file_path, 'r') as f:
                print(f"H5文件根目录包含: {list(f.keys())}")
                
                # 检查是否有state数据
                if 'state' not in f:
                    print("  - 错误: 未找到state数据")
                    return grasp_moments
                
                print(f"state目录包含: {list(f['state'].keys())}")
                
                # 检查是否有夹爪状态数据
                left_effector_available = 'left_effector' in f['state'] and 'position' in f['state']['left_effector']
                right_effector_available = 'right_effector' in f['state'] and 'position' in f['state']['right_effector']
                
                if not left_effector_available and not right_effector_available:
                    print("  - 缺少夹爪状态数据，无法检测抓取时刻")
                    print(f"  - left_effector可用: {left_effector_available}")
                    print(f"  - right_effector可用: {right_effector_available}")
                    return grasp_moments
                
                # 加载机器人基准位置
                if 'robot' in f['state'] and 'position' in f['state']['robot']:
                    robot_position = f['state']['robot']['position'][:]  # (T, 3)
                    print(f"  - 加载机器人基准位置: {robot_position.shape}")
                else:
                    print(f"  - 警告: 未找到机器人基准位置，使用零向量")
                    robot_position = np.zeros((f['timestamp'].shape[0], 3))
                
                # 加载末端执行器位置
                if 'end' in f['state'] and 'position' in f['state']['end']:
                    end_position_raw = f['state']['end']['position'][:]  # (T, 2, 3)
                    # 减去机器人基准位置
                    end_position_relative = end_position_raw - robot_position[:, np.newaxis, :]
                    positions = end_position_relative
                    print(f"  - 加载末端执行器位置（相对坐标）: {positions.shape}")
                    print(f"  - 位置范围: {positions.min():.3f} - {positions.max():.3f} 米")
                else:
                    print(f"  - 错误: 未找到末端执行器位置数据")
                    return grasp_moments
                
                timestamps = f['timestamp'][:]  # (T,)
                print(f"  - 加载时间戳: {timestamps.shape}")
                
                # 加载夹爪状态数据
                if left_effector_available:
                    left_effector_pos = f['state']['left_effector']['position'][:]  # (T,)
                    print(f"  - 夹爪位置数据范围 - 左手: {left_effector_pos.min():.1f} - {left_effector_pos.max():.1f}")
                    
                    # 检测左手夹爪闭合
                    left_grasp_moments = self._detect_effector_grasp(left_effector_pos, 'left', positions, timestamps, 0)
                    for moment in left_grasp_moments:
                        moment['grasping_arm'] = 'left'
                        moment['camera_name'] = 'hand_left'
                        grasp_moments.append(moment)
                
                if right_effector_available:
                    right_effector_pos = f['state']['right_effector']['position'][:]  # (T,)
                    print(f"  - 夹爪位置数据范围 - 右手: {right_effector_pos.min():.1f} - {right_effector_pos.max():.1f}")
                    
                    # 检测右手夹爪闭合
                    right_grasp_moments = self._detect_effector_grasp(right_effector_pos, 'right', positions, timestamps, 1)
                    for moment in right_grasp_moments:
                        moment['grasping_arm'] = 'right'
                        moment['camera_name'] = 'hand_right'
                        grasp_moments.append(moment)
                
                # 按时间戳排序
                grasp_moments.sort(key=lambda x: x['grasp_idx'])
                
                # 去重并过滤太近的时刻（至少间隔20帧）
                if grasp_moments:
                    filtered_moments = [grasp_moments[0]]
                    for i in range(1, len(grasp_moments)):
                        if grasp_moments[i]['grasp_idx'] - grasp_moments[i-1]['grasp_idx'] > 20:  # 至少间隔20帧
                            filtered_moments.append(grasp_moments[i])
                    grasp_moments = filtered_moments
                
                print(f"  - 检测到 {len(grasp_moments)} 个夹爪闭合时刻")
                for i, detail in enumerate(grasp_moments):
                    print(f"    - 时刻 {i+1}: {detail['grasping_arm']}臂抓取, 相机: {detail['camera_name']}, "
                          f"抓取前: {detail['pre_grasp_idx']}, 抓取时: {detail['grasp_idx']}, 稳定闭合: {detail['stable_close_idx']}")
                
        except Exception as e:
            print(f"  - 加载H5文件失败: {e}")
            import traceback
            traceback.print_exc()
        
        return grasp_moments
    
    def _detect_effector_grasp(self, effector_pos: np.ndarray, arm_name: str, positions: np.ndarray, timestamps: np.ndarray, arm_idx: int) -> List[Dict]:
        """检测单个夹爪的闭合时刻"""
        grasp_moments = []
        
        # 确保effector_pos是一维数组
        if effector_pos.ndim > 1:
            effector_pos = effector_pos.squeeze()
        
        # 计算夹爪位置的变化
        effector_changes = np.diff(effector_pos)
        
        # 检查是否有足够的数据
        if len(effector_changes) == 0:
            print(f"  - 警告: {arm_name}夹爪数据不足，无法检测抓取时刻")
            return grasp_moments
        
        # 检测夹爪闭合时刻（位置从接近0变化到100多）
        # 寻找位置突然增大的时刻
        close_threshold = np.percentile(effector_changes, 85)  # 取85%分位数作为闭合阈值
        
        for i, change in enumerate(effector_changes):
            if change > close_threshold and effector_pos[i+1] > 80:  # 位置突然增大且大于80
                grasp_idx = i + 1  # +1因为diff后索引偏移
                
                # 寻找抓取前的时刻（位置接近0的时刻）
                pre_grasp_idx = grasp_idx
                for j in range(grasp_idx-1, max(0, grasp_idx-30), -1):
                    if effector_pos[j] < 10:  # 位置接近0
                        pre_grasp_idx = j
                        break
                
                # 寻找稳定闭合后的时刻（延后几帧，确保夹爪稳定闭合）
                stable_close_idx = grasp_idx
                for j in range(grasp_idx+1, min(len(effector_pos), grasp_idx+20)):
                    # 检查后续几帧的位置是否稳定在较大值
                    if (effector_pos[j] > 80 and 
                        abs(effector_pos[j] - effector_pos[j-1]) < 5):  # 位置稳定且较大
                        stable_close_idx = j
                        break
                
                grasp_moments.append({
                    'grasp_idx': grasp_idx,
                    'pre_grasp_idx': pre_grasp_idx,
                    'stable_close_idx': stable_close_idx,
                    'grasp_timestamp': timestamps[grasp_idx],
                    'pre_grasp_timestamp': timestamps[pre_grasp_idx],
                    'stable_close_timestamp': timestamps[stable_close_idx],
                    'position': positions[grasp_idx][arm_idx],  # 对应手臂的位置
                    'effector_position_at_grasp': effector_pos[grasp_idx],
                    'effector_position_pre_grasp': effector_pos[pre_grasp_idx],
                    'effector_position_stable_close': effector_pos[stable_close_idx]
                })
        
        return grasp_moments
    
    def world_to_camera(self, point_world: np.ndarray, camera_name: str = 'head') -> np.ndarray:
        """将世界坐标转换为相机坐标"""
        if camera_name not in self.camera_extrinsics:
            print(f"警告: 未找到相机 {camera_name} 的外参")
            return point_world
        
        extrinsics = self.camera_extrinsics[camera_name]
        R = extrinsics['rotation']
        t = extrinsics['translation']
        
        print(f"\n=== {camera_name} 相机坐标转换调试信息 ===")
        print(f"[1] 相机内参矩阵 K:\n{self.camera_intrinsics[camera_name]['matrix']}")
        
        # 构建变换矩阵
        camera_extrinsic = np.eye(4)
        camera_extrinsic[:3, :3] = R
        camera_extrinsic[:3, 3] = t
        
        print(f"[2] 相机外参矩阵 T_camera_world:\n{camera_extrinsic}")
        print(f"[3] 要投影的 3D 点: {point_world}")
        
        # 将3D点转换为齐次坐标
        point_homogeneous = np.append(point_world, 1.0)
        print(f"[2.1] 世界点的齐次坐标: {point_homogeneous}")
        
        # 计算从世界到相机的变换矩阵（外参矩阵的逆）
        try:
            inv_camera_extrinsic = np.linalg.inv(camera_extrinsic)
            print(f"[2.2] 从世界到相机的变换矩阵 (inv(T_camera_world)):\n{inv_camera_extrinsic}")
        except np.linalg.LinAlgError:
            print("警告: 外参矩阵不可逆，使用伪逆")
            inv_camera_extrinsic = np.linalg.pinv(camera_extrinsic)
        
        # 应用变换
        point_camera_homogeneous = inv_camera_extrinsic @ point_homogeneous
        point_camera = point_camera_homogeneous[:3]
        print(f"[2.3] 3D点在相机坐标系下的坐标: {point_camera}")
        
        # 检查点是否在相机前方
        if point_camera[2] <= 0:
            print(f"❌ 错误: 点 {point_world} 在相机 {camera_name} 后方(深度Z = {point_camera[2]:.4f}), 无法投影。脚本终止。")
            return point_camera
        
        print(f"✅ 点 {point_world} 在相机 {camera_name} 前方(深度Z = {point_camera[2]:.4f})")
        return point_camera
    
    def camera_to_image(self, point_camera: np.ndarray, camera_name: str = 'head') -> Tuple[int, int]:
        """将相机坐标转换为图像坐标"""
        if camera_name not in self.camera_intrinsics:
            print(f"警告: 未找到相机 {camera_name} 的内参，使用默认投影")
            return (320, 240)
        
        intrinsics = self.camera_intrinsics[camera_name]
        K = intrinsics['matrix']
        
        print(f"\n=== {camera_name} 相机内参投影调试信息 ===")
        print(f"[3.1] 相机内参矩阵 K:\n{K}")
        print(f"[3.2] 相机坐标系下的点: {point_camera}")
        
        point_homogeneous = K @ point_camera
        print(f"[3.3] 内参投影后的齐次坐标: {point_homogeneous}")
        
        x = int(point_homogeneous[0] / point_homogeneous[2])
        y = int(point_homogeneous[1] / point_homogeneous[2])
        
        print(f"[3.4] 最终图像坐标: ({x}, {y})")
        
        return (x, y)
    
    def project_3d_to_2d(self, point_3d: np.ndarray, camera_name: str = 'head') -> Tuple[int, int]:
        """将3D点投影到2D图像坐标"""
        print(f"\n开始投影 3D 点 {point_3d} 到 {camera_name} 相机...")
        
        try:
            point_camera = self.world_to_camera(point_3d, camera_name)
            
            if point_camera[2] <= 0:
                print(f"❌ 错误: 点 {point_3d} 在相机 {camera_name} 后方，无法投影")
                intrinsics = self.camera_intrinsics[camera_name]
                return (int(intrinsics['cx']), int(intrinsics['cy']))
            
            point_image = self.camera_to_image(point_camera, camera_name)
            
            # 设置图像尺寸
            max_width, max_height = 1280, 720  # 头部相机图像尺寸
            
            x, y = point_image
            
            # 检查是否在视野范围内
            in_fov = (0 <= x < max_width) and (0 <= y < max_height)
            
            if not in_fov:
                print(f"⚠️ 警告: 点 {point_3d} 投影到 {camera_name} 相机坐标 ({x}, {y}) 超出视野范围 ({max_width}x{max_height})")
                # 返回图像中心作为fallback
                fallback_point = (max_width // 2, max_height // 2)
                print(f"使用fallback坐标: {fallback_point}")
                return fallback_point
            
            print(f"✅ {camera_name} 相机投影成功: 3D点 {point_3d} -> 2D坐标 ({x}, {y})")
            return (x, y)
            
        except Exception as e:
            print(f"❌ 投影失败: {e}")
            intrinsics = self.camera_intrinsics[camera_name]
            return (int(intrinsics['cx']), int(intrinsics['cy']))
    
    def project_3d_to_2d_with_frame(self, point_3d: np.ndarray, camera_name: str = 'head', frame_idx: int = None) -> Tuple[int, int]:
        """将3D点投影到2D图像坐标（使用指定帧的外参）"""
        print(f"\n开始投影 3D 点 {point_3d} 到 {camera_name} 相机 (帧 {frame_idx})...")
        
        # 如果指定了帧索引，重新加载对应帧的外参
        if frame_idx is not None:
            self.load_camera_parameters_for_frame(frame_idx)
        
        try:
            point_camera = self.world_to_camera(point_3d, camera_name)
            
            if point_camera[2] <= 0:
                print(f"❌ 错误: 点 {point_3d} 在相机 {camera_name} 后方，无法投影")
                intrinsics = self.camera_intrinsics[camera_name]
                return (int(intrinsics['cx']), int(intrinsics['cy']))
            
            point_image = self.camera_to_image(point_camera, camera_name)
            
            # 设置图像尺寸
            max_width, max_height = 1280, 720  # 头部相机图像尺寸
            
            x, y = point_image
            
            # 检查是否在视野范围内
            in_fov = (0 <= x < max_width) and (0 <= y < max_height)
            
            if not in_fov:
                print(f"⚠️ 警告: 点 {point_3d} 投影到 {camera_name} 相机坐标 ({x}, {y}) 超出视野范围 ({max_width}x{max_height})")
                # 返回图像中心作为fallback
                fallback_point = (max_width // 2, max_height // 2)
                print(f"使用fallback坐标: {fallback_point}")
                return fallback_point
            
            print(f"✅ {camera_name} 相机投影成功: 3D点 {point_3d} -> 2D坐标 ({x}, {y})")
            return (x, y)
            
        except Exception as e:
            print(f"❌ 投影失败: {e}")
            intrinsics = self.camera_intrinsics[camera_name]
            return (int(intrinsics['cx']), int(intrinsics['cy']))
    
    def load_camera_parameters_for_frame(self, frame_idx: int):
        """加载指定帧的相机参数"""
        print(f"重新加载第 {frame_idx} 帧的相机参数...")
        
        # 使用已知的相机参数路径
        param_dir = Path("/mnt/data-oss/rap-prod-bak/AgibotWorld-Challenge/AgiBotWorldChallenge-2025/iros_agibot_sim/2600003/2600003/3311058/A2D0015AB00061/12050785/parameters/camera")
        
        if not param_dir.exists():
            print(f"相机参数目录不存在: {param_dir}")
            return
        
        # 加载头部相机外参（按帧加载）
        extrinsic_file = param_dir / 'head_extrinsic_params_aligned.json'
        if extrinsic_file.exists():
            try:
                with open(extrinsic_file, 'r') as f:
                    extrinsic_data = json.load(f)
                
                print(f"外参文件包含 {len(extrinsic_data)} 帧数据")
                
                # 使用指定帧的外参
                if 0 <= frame_idx < len(extrinsic_data):
                    extrinsic = extrinsic_data[frame_idx]['extrinsic']
                    print(f"使用第 {frame_idx} 帧的外参")
                else:
                    print(f"警告: 帧索引 {frame_idx} 超出范围 [0, {len(extrinsic_data)-1}]，使用第0帧")
                    extrinsic = extrinsic_data[0]['extrinsic']
                
                rotation_matrix = np.array(extrinsic['rotation_matrix'])
                translation_vector = np.array(extrinsic['translation_vector'])
                
                print(f"原始外参 (第{frame_idx}帧):")
                print(f"旋转矩阵 R:\n{rotation_matrix}")
                print(f"平移向量 t: {translation_vector}")
                
                # 构建4x4相机外参矩阵
                camera_extrinsic = np.eye(4)
                camera_extrinsic[:3, :3] = rotation_matrix
                camera_extrinsic[:3, 3] = translation_vector
                
                print(f"原始4x4外参矩阵 T_camera_world:\n{camera_extrinsic}")
                
                # 应用坐标系统变换矩阵
                camera_extrinsic_transformed = camera_extrinsic @ self.COORDSYS_TRANSFORM
                
                print(f"坐标系统变换矩阵:\n{self.COORDSYS_TRANSFORM}")
                print(f"变换后的4x4外参矩阵 T_camera_world:\n{camera_extrinsic_transformed}")
                
                # 提取变换后的旋转矩阵和平移向量
                rotation_matrix_transformed = camera_extrinsic_transformed[:3, :3]
                translation_vector_transformed = camera_extrinsic_transformed[:3, 3]
                
                self.camera_extrinsics['head'] = {
                    'rotation': rotation_matrix_transformed,
                    'translation': translation_vector_transformed
                }
                
                print(f"变换后的外参:")
                print(f"旋转矩阵 R_transformed:\n{rotation_matrix_transformed}")
                print(f"平移向量 t_transformed: {translation_vector_transformed}")
                
            except Exception as e:
                print(f"加载 head 外参失败: {e}")
    
    def test_projection_with_image(self, image_path: str, test_points: list):
        """使用实际图像测试投影"""
        print(f"\n=== 使用图像测试投影 ===")
        print(f"图像路径: {image_path}")
        
        # 加载图像
        if not os.path.exists(image_path):
            print(f"❌ 图像文件不存在: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法加载图像: {image_path}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"图像尺寸: {image.shape}")
        
        # 测试每个3D点
        for i, point_3d in enumerate(test_points):
            print(f"\n--- 测试点 {i+1}: {point_3d} ---")
            
            # 投影到2D
            point_2d = self.project_3d_to_2d(point_3d)
            
            # 在图像上绘制点
            image_with_point = image.copy()
            x, y = point_2d
            
            # 绘制红色圆圈
            cv2.circle(image_with_point, (x, y), 15, (255, 0, 0), 3)
            
            # 添加坐标文本
            cv2.putText(image_with_point, f"({x}, {y})", (x+20, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 保存结果图像
            output_path = f"test_projection_result_{i+1}.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(image_with_point, cv2.COLOR_RGB2BGR))
            print(f"✅ 保存结果图像: {output_path}")
            
            # 显示图像
            plt.figure(figsize=(12, 8))
            plt.imshow(image_with_point)
            plt.title(f'3D点 {point_3d} -> 2D坐标 ({x}, {y})')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"test_projection_plot_{i+1}.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ 保存可视化图像: test_projection_plot_{i+1}.png")


def main():
    """主函数"""
    print("=== 头部相机投影测试脚本（多抓取时刻测试）===")
    
    # 使用正确的episode路径
    episode_path = "/mnt/data-oss/rap-prod-bak/AgibotWorld-Challenge/AgiBotWorldChallenge-2025/iros_agibot_sim/2600003/2600003/3311058/A2D0015AB00061/12050785"
    
    print(f"使用episode路径: {episode_path}")
    
    # 查找H5文件
    h5_file_path = f"{episode_path}/aligned_joints.h5"
    
    if not os.path.exists(h5_file_path):
        print(f"❌ H5文件不存在: {h5_file_path}")
        return
    
    # 创建测试器
    tester = HeadCameraProjectionTester(episode_path, 0)  # 初始使用第0帧参数
    
    # 从H5文件中检测抓取时刻
    grasp_moments = tester.detect_grasp_moments_from_h5(h5_file_path)
    
    if not grasp_moments:
        print("❌ 未检测到抓取时刻，使用默认测试点")
        # 使用默认测试点
        test_configs = [
            (0, np.array([0.5, 0.0, 0.8]), "第0帧末端位置"),
            (200, np.array([0.55, -0.05, 0.82]), "第200帧末端位置"),
            (400, np.array([0.58, -0.08, 0.84]), "第400帧末端位置"),
            (500, np.array([0.6, -0.1, 0.85]), "第500帧末端位置"),
            (600, np.array([0.62, -0.12, 0.87]), "第600帧末端位置"),
            (800, np.array([0.64, -0.15, 0.88]), "第800帧末端位置"),
            (1000, np.array([0.65, 0.05, 0.9]), "第1000帧末端位置"),
            (1607, np.array([0.68694925, -0.1927185, 0.81121844]), "第1607帧末端位置（抓取时刻）"),
        ]
    else:
        # 使用检测到的抓取时刻
        print(f"\n=== 使用检测到的 {len(grasp_moments)} 个抓取时刻进行测试 ===")
        
        # 选择前几个抓取时刻进行测试
        grasp_test_configs = []
        for i, moment in enumerate(grasp_moments[:6]):  # 最多测试6个抓取时刻
            frame_idx = moment['grasp_idx']
            position = moment['position']
            arm = moment['grasping_arm']
            description = f"第{frame_idx}帧{arm}臂抓取时刻"
            
            grasp_test_configs.append((frame_idx, position, description))
            print(f"  - 测试配置 {i+1}: {description}, 位置: {position}")
        
        # 添加指定的帧数测试点
        specified_frame_configs = [
            (0, np.array([0.5, 0.0, 0.8]), "第0帧指定测试点"),
            (200, np.array([0.55, -0.05, 0.82]), "第200帧指定测试点"),
            (400, np.array([0.58, -0.08, 0.84]), "第400帧指定测试点"),
            (600, np.array([0.62, -0.12, 0.87]), "第600帧指定测试点"),
            (800, np.array([0.64, -0.15, 0.88]), "第800帧指定测试点"),
        ]
        
        # 合并抓取时刻和指定帧数
        test_configs = grasp_test_configs + specified_frame_configs
        print(f"\n=== 总共测试 {len(test_configs)} 个配置（{len(grasp_test_configs)} 个抓取时刻 + {len(specified_frame_configs)} 个指定帧数）===")
    
    print(f"\n=== 测试不同抓取时刻的投影 ===")
    
    # 测试每个配置
    for frame_idx, point_3d, description in test_configs:
        print(f"\n--- {description}: 3D点 {point_3d} ---")
        
        # 使用对应帧的外参进行投影
        point_2d = tester.project_3d_to_2d_with_frame(point_3d, frame_idx=frame_idx)
        print(f"投影结果: {point_2d}")
    
    # 使用对应时刻的图像文件进行可视化
    print(f"\n=== 使用实际图像进行可视化测试 ===")
    
    # 为每个测试配置生成图像
    for i, (frame_idx, point_3d, description) in enumerate(test_configs):
        print(f"\n--- {description}: 3D点 {point_3d} 使用第{frame_idx}帧外参 ---")
        
        # 查找对应的图像文件
        image_path = f"{episode_path}/camera/{frame_idx}/head_color.jpg"
        
        if os.path.exists(image_path):
            print(f"找到图像: {image_path}")
            
            # 投影到2D
            point_2d = tester.project_3d_to_2d_with_frame(point_3d, frame_idx=frame_idx)
            
            # 在图像上绘制点
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_with_point = image.copy()
            x, y = point_2d
            
            # 绘制红色圆圈
            cv2.circle(image_with_point, (x, y), 15, (255, 0, 0), 3)
            
            # 添加坐标文本和帧信息
            text = f"Frame{frame_idx}: ({x}, {y})"
            cv2.putText(image_with_point, text, (x+20, y-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 添加3D位置信息
            pos_text = f"3D: ({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})"
            cv2.putText(image_with_point, pos_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 保存结果图像
            output_path = f"grasp_moment_{i+1:02d}_frame{frame_idx:04d}_result.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(image_with_point, cv2.COLOR_RGB2BGR))
            print(f"✅ 保存结果图像: {output_path}")
            
            # 显示图像
            plt.figure(figsize=(12, 8))
            plt.imshow(image_with_point)
            plt.title(f'{description}: 3D点 {point_3d} -> 2D坐标 ({x}, {y}) (帧{frame_idx})')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"grasp_moment_{i+1:02d}_frame{frame_idx:04d}_plot.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✅ 保存可视化图像: grasp_moment_{i+1:02d}_frame{frame_idx:04d}_plot.png")
        else:
            print(f"⚠️ 图像文件不存在: {image_path}")
            # 尝试查找相近帧的图像
            for offset in range(-5, 6):
                nearby_frame = frame_idx + offset
                nearby_image_path = f"{episode_path}/camera/{nearby_frame}/head_color.jpg"
                if os.path.exists(nearby_image_path):
                    print(f"使用相近帧图像: {nearby_image_path}")
                    break
            else:
                print(f"未找到相近帧的图像")


if __name__ == "__main__":
    main() 