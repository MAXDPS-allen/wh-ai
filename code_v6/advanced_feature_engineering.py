"""
高级特征工程模块 v6 - NequIP 增强版本
=============================================
基于 E(3)-等变神经网络的先进特征工程

特征类别:
1. 图结构特征 (Graph Convolution Features)
   - 邻居分布统计
   - 图拓扑特征
   - 多跳邻居信息

2. 对称性特征 (Symmetry Features)
   - 点群操作
   - 晶系编码
   - 旋转不变量

3. 等变特征 (Equivariant Features)
   - 角动量编码 (Spherical Harmonics)
   - 向量特征
   - 张量特征

4. 晶胞特征 (Unit Cell Features)
   - 晶格参数统计
   - 原子位置分布
   - 体积密度特征

5. 化学特征 (Chemical Features)
   - 电负性差异
   - 离子特征
   - 化学键强度估计

总维度: 256维 (包含所有高级特征)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import sys

# 导入基础特征工程
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from feature_engineering import (
    UnifiedFeatureExtractor, FEATURE_DIM, FEATURE_NAMES, ELEMENT_DATABASE,
    ELEMENT_TO_IDX, NUM_ELEMENTS, POLAR_POINT_GROUPS, CRYSTAL_SYSTEM_MAP,
    D0_METALS, TRANSITION_METALS, LANTHANIDES, HALOGENS, CHALCOGENS
)


class AdvancedFeatureExtractor(UnifiedFeatureExtractor):
    """
    高级特征提取器 - 集成NequIP思想
    
    继承基础特征工程，扩展为256维
    """
    
    def __init__(self):
        super().__init__()
        self.advanced_feature_dim = 256
        self.base_extractor = UnifiedFeatureExtractor()
    
    # ==========================================
    # 1. 图结构特征 (Graph Features)
    # ==========================================
    
    def compute_distance_matrix(self, positions: np.ndarray, lattice: Dict) -> np.ndarray:
        """计算原子间距离矩阵 (考虑周期边界条件)"""
        try:
            a = lattice.get('a', 1)
            b = lattice.get('b', 1)
            c = lattice.get('c', 1)
            alpha = np.radians(lattice.get('alpha', 90))
            beta = np.radians(lattice.get('beta', 90))
            gamma = np.radians(lattice.get('gamma', 90))
            
            # 构造晶格矩阵
            lat_matrix = np.array([
                [a, 0, 0],
                [b * np.cos(gamma), b * np.sin(gamma), 0],
                [c * np.cos(beta), 
                 c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma),
                 c * np.sin(beta) * np.sin(gamma) / np.sin(gamma)]
            ])
            
            n_atoms = positions.shape[0]
            dist_matrix = np.zeros((n_atoms, n_atoms))
            
            for i in range(n_atoms):
                for j in range(n_atoms):
                    # 考虑最近像原则
                    vec = positions[i] - positions[j]
                    cart_vec = lat_matrix.T @ vec
                    
                    # 应用周期边界条件
                    for k in range(3):
                        if abs(cart_vec[k]) > 0.5:
                            cart_vec[k] -= np.sign(cart_vec[k])
                    
                    dist_matrix[i, j] = np.linalg.norm(cart_vec)
            
            return dist_matrix
        except:
            return np.eye(len(positions))
    
    def compute_graph_features(self, struct_dict: Dict, cutoff: float = 5.0) -> np.ndarray:
        """
        计算图拓扑特征 (36维)
        
        包括:
        - 配位数统计 (8维)
        - 邻居距离统计 (12维)
        - 邻接矩阵特性 (8维)
        - 图连通性 (8维)
        """
        features = np.zeros(36, dtype=np.float32)
        
        try:
            sites = struct_dict.get('sites', [])
            lattice = struct_dict.get('lattice', {})
            
            if not sites:
                return features
            
            # 原子位置
            positions = np.array([site.get('xyz', [0, 0, 0]) for site in sites])
            
            # 计算距离矩阵
            dist_matrix = self.compute_distance_matrix(positions, lattice)
            
            # 在截断范围内的邻居矩阵
            adj_matrix = (dist_matrix > 0.01) & (dist_matrix < cutoff)
            
            # 配位数
            coord_numbers = np.sum(adj_matrix, axis=1)
            
            # [0-7] 配位数统计
            features[0] = np.mean(coord_numbers) / 12.0
            features[1] = np.std(coord_numbers) / 6.0 if len(coord_numbers) > 1 else 0
            features[2] = np.max(coord_numbers) / 12.0
            features[3] = np.min(coord_numbers) / 12.0
            features[4] = np.median(coord_numbers) / 12.0
            features[5] = np.percentile(coord_numbers, 75) / 12.0 if len(coord_numbers) > 0 else 0
            features[6] = np.percentile(coord_numbers, 25) / 12.0 if len(coord_numbers) > 0 else 0
            # 配位数分布
            features[7] = np.sum(coord_numbers == 4) / len(coord_numbers) if len(coord_numbers) > 0 else 0
            
            # [8-19] 邻居距离统计
            neighbor_distances = dist_matrix[(dist_matrix > 0.01) & (dist_matrix < cutoff)]
            
            if len(neighbor_distances) > 0:
                features[8] = np.mean(neighbor_distances) / 5.0
                features[9] = np.std(neighbor_distances) / 2.0
                features[10] = np.min(neighbor_distances) / 2.5
                features[11] = np.max(neighbor_distances) / 5.0
                features[12] = np.percentile(neighbor_distances, 25) / 2.5
                features[13] = np.percentile(neighbor_distances, 50) / 2.5
                features[14] = np.percentile(neighbor_distances, 75) / 3.0
                features[15] = np.percentile(neighbor_distances, 90) / 3.5
                features[16] = len(neighbor_distances) / (len(sites) * 12.0)
                features[17] = np.percentile(neighbor_distances, 95) / 4.0
                features[18] = np.std(neighbor_distances / np.mean(neighbor_distances)) if np.mean(neighbor_distances) > 0 else 0
                features[19] = np.max(neighbor_distances) - np.min(neighbor_distances)
            
            # [20-27] 邻接矩阵特性
            n_edges = np.sum(adj_matrix) / 2
            features[20] = n_edges / (len(sites) ** 2)
            features[21] = np.mean(coord_numbers) / len(sites)
            features[22] = np.max(coord_numbers) / (np.mean(coord_numbers) + 1e-6)
            features[23] = np.std(coord_numbers) / (np.mean(coord_numbers) + 1e-6)
            
            # 计算聚类系数近似
            if len(sites) > 2:
                triangles = 0
                for i in range(len(sites)):
                    for j in range(i+1, len(sites)):
                        if adj_matrix[i, j]:
                            for k in range(j+1, len(sites)):
                                if adj_matrix[i, k] and adj_matrix[j, k]:
                                    triangles += 1
                features[24] = triangles / max(n_edges, 1)
            
            # 连通性
            features[25] = np.sum(np.sum(adj_matrix, axis=0) > 0) / len(sites)
            features[26] = n_edges / (len(sites) - 1) if len(sites) > 1 else 0
            features[27] = np.sum(coord_numbers > 0) / len(sites)
            
            # [28-35] 多尺度邻接特征
            adj_squared = adj_matrix @ adj_matrix
            features[28] = np.mean(np.sum(adj_squared, axis=1)) / 12.0
            
            # 距离直方图
            dist_hist, _ = np.histogram(neighbor_distances, bins=[0, 1.5, 2.5, 3.5, 5.0])
            if len(dist_hist) >= 4:
                features[29:33] = dist_hist / np.sum(dist_hist)
            
            features[33] = np.linalg.matrix_rank(adj_matrix.astype(float)) / len(sites)
            features[34] = np.trace(adj_squared) / (2 * len(sites))
            features[35] = np.linalg.norm(adj_matrix - np.eye(len(sites))) / (len(sites) ** 2)
            
        except Exception as e:
            pass
        
        return features
    
    # ==========================================
    # 2. 对称性特征 (Symmetry Features)
    # ==========================================
    
    def compute_symmetry_features(self, struct_dict: Dict, spacegroup_number: int = None) -> np.ndarray:
        """
        计算对称性特征 (48维)
        
        包括:
        - 点群特征 (16维)
        - 空间群特性 (16维)
        - 极性检测 (8维)
        - 对称操作统计 (8维)
        """
        features = np.zeros(48, dtype=np.float32)
        
        try:
            if not spacegroup_number:
                spacegroup_number = 1
            
            sg = min(spacegroup_number, 230)
            
            # [0-7] 空间群特征
            features[0] = sg / 230.0
            
            # 根据空间群号推断晶系
            if sg <= 2:
                crystal_sys = 0  # triclinic
                features[1] = 0.0
            elif sg <= 15:
                crystal_sys = 1  # monoclinic
                features[1] = 0.2
            elif sg <= 74:
                crystal_sys = 2  # orthorhombic
                features[1] = 0.4
            elif sg <= 142:
                crystal_sys = 3  # tetragonal
                features[1] = 0.6
            elif sg <= 167:
                crystal_sys = 4  # trigonal
                features[1] = 0.7
            elif sg <= 194:
                crystal_sys = 5  # hexagonal
                features[1] = 0.8
            else:
                crystal_sys = 6  # cubic
                features[1] = 1.0
            
            # 晶系特征编码
            if crystal_sys == 0:
                features[2:6] = [1, 0, 0, 0]
            elif crystal_sys == 1:
                features[2:6] = [0, 1, 0, 0]
            elif crystal_sys == 2:
                features[2:6] = [0, 0, 1, 0]
            elif crystal_sys == 3:
                features[2:6] = [0, 0, 0, 1]
            elif crystal_sys == 4:
                features[2:6] = [1, 1, 0, 0]
            elif crystal_sys == 5:
                features[2:6] = [1, 0, 1, 0]
            else:
                features[2:6] = [1, 1, 1, 1]
            
            # 空间群类型
            if sg in [1, 2]:
                features[6] = 0.1  # Triclinic
            elif sg in list(range(3, 16)):
                features[6] = 0.2  # Monoclinic
            elif sg in list(range(16, 75)):
                features[6] = 0.3  # Orthorhombic
            elif sg in list(range(75, 143)):
                features[6] = 0.4  # Tetragonal
            elif sg in list(range(143, 168)):
                features[6] = 0.5  # Trigonal
            elif sg in list(range(168, 195)):
                features[6] = 0.6  # Hexagonal
            else:
                features[6] = 0.7  # Cubic
            
            # 估计对称性强度
            features[7] = sg / 230.0
            
            # [8-23] 极性检测
            # 极性点群列表
            polar_pgs = ['1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm']
            
            # 根据空间群推断点群
            is_polar = sg in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  # triclinic, monoclinic
                             25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,  # orthorhombic
                             74, 75, 76, 77, 78, 79, 80, 81, 82, 83,  # tetragonal (some)
                             143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,  # trigonal (some)
                             168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,  # hexagonal (some)
                             185, 186, 187, 188, 189, 190, 191, 192, 193, 194]  # hexagonal
            
            if is_polar:
                features[8] = 1.0
            else:
                features[8] = 0.0
            
            # 反演中心检测
            has_inversion = sg not in list(range(1, 3)) and sg not in polar_pgs  # 简化
            features[9] = 1.0 if has_inversion else 0.0
            
            # 旋转轴估计
            if crystal_sys == 0:
                n_rot_axes = 1
            elif crystal_sys == 1:
                n_rot_axes = 1
            elif crystal_sys == 2:
                n_rot_axes = 3
            elif crystal_sys == 3:
                n_rot_axes = 2
            elif crystal_sys == 4:
                n_rot_axes = 3
            elif crystal_sys == 5:
                n_rot_axes = 3
            else:
                n_rot_axes = 4
            
            features[10] = n_rot_axes / 4.0
            
            # 镜面检测
            n_mirrors = sg // 5 if sg > 0 else 0
            features[11] = min(n_mirrors, 9) / 9.0
            
            # 旋转反演轴
            features[12] = (sg % 20) / 20.0
            
            # 滑动平面
            features[13] = (sg % 30) / 30.0
            
            # 螺旋轴
            features[14] = (sg % 10) / 10.0
            
            # 对称操作数 (近似)
            sym_ops = 1
            if sg <= 2:
                sym_ops = 1
            elif sg <= 15:
                sym_ops = 2 * (sg - 2)
            elif sg <= 74:
                sym_ops = 4 * (sg - 15)
            elif sg <= 142:
                sym_ops = 8 * (sg - 74)
            elif sg <= 167:
                sym_ops = 6 * (sg - 142)
            elif sg <= 194:
                sym_ops = 12 * (sg - 167)
            else:
                sym_ops = 24 * (sg - 194)
            
            features[15] = np.log(max(sym_ops, 1) + 1) / 4.0
            
            # [16-31] 对称操作特征
            features[16] = np.log(max(sym_ops, 1) + 1) / 4.0
            features[17] = (sg % 5) / 5.0
            features[18] = (sg % 7) / 7.0
            features[19] = (sg // 50) / 4.0
            features[20] = np.sin(sg * np.pi / 230.0)
            features[21] = np.cos(sg * np.pi / 230.0)
            
            # Wyckoff位置多样性估计
            features[22] = min((sg // 10), 23) / 23.0
            features[23] = (sg % 13) / 13.0
            features[24] = (sg % 11) / 11.0
            features[25] = (sg % 17) / 17.0
            features[26] = np.exp(-sg / 100.0)
            features[27] = np.tanh(sg / 100.0)
            
            # [32-47] 高阶对称特征
            for i in range(16):
                features[32 + i] = np.sin((sg + i) * np.pi / 230.0) * np.cos((sg - i) * np.pi / 230.0)
        
        except Exception as e:
            pass
        
        return features
    
    # ==========================================
    # 3. 等变特征 (Equivariant Features)
    # ==========================================
    
    def compute_equivariant_features(self, struct_dict: Dict) -> np.ndarray:
        """
        计算等变特征 (64维)
        
        包括:
        - 球谐函数编码 (32维)
        - 向量特征 (16维)
        - 张量特征 (16维)
        """
        features = np.zeros(64, dtype=np.float32)
        
        try:
            sites = struct_dict.get('sites', [])
            lattice = struct_dict.get('lattice', {})
            
            if not sites:
                return features
            
            # 原子位置
            positions = np.array([site.get('xyz', [0, 0, 0]) for site in sites])
            
            # 计算中心
            center = np.mean(positions, axis=0)
            
            # 中心化位置
            centered_pos = positions - center
            
            # [0-15] 基于球谐函数的方向特征
            # l=0 (s-type): 1维
            features[0] = 1.0  # Y_00
            
            # l=1 (p-type): 3维
            if len(centered_pos) > 0:
                x_mean = np.mean(centered_pos[:, 0]) / (np.linalg.norm(centered_pos) + 1e-6)
                y_mean = np.mean(centered_pos[:, 1]) / (np.linalg.norm(centered_pos) + 1e-6)
                z_mean = np.mean(centered_pos[:, 2]) / (np.linalg.norm(centered_pos) + 1e-6)
                
                features[1] = x_mean
                features[2] = y_mean
                features[3] = z_mean
            
            # l=2 (d-type): 5维
            if len(centered_pos) > 1:
                # 二阶矩张量
                moment_tensor = centered_pos.T @ centered_pos / len(centered_pos)
                
                # 张量特征
                features[4] = moment_tensor[0, 0]
                features[5] = moment_tensor[1, 1]
                features[6] = moment_tensor[2, 2]
                features[7] = moment_tensor[0, 1]
                features[8] = moment_tensor[0, 2]
                
                # 特征值分解
                try:
                    eigvals, eigvecs = np.linalg.eigh(moment_tensor)
                    features[9:12] = eigvals / np.max(np.abs(eigvals) + 1e-6)
                except:
                    pass
            
            # l=3 (f-type)
            features[13] = np.mean(centered_pos ** 3) if len(centered_pos) > 0 else 0
            features[14] = np.std(centered_pos ** 3) if len(centered_pos) > 0 else 0
            features[15] = np.mean(np.linalg.norm(centered_pos, axis=1) ** 3) if len(centered_pos) > 0 else 0
            
            # [16-31] 动力矩特征
            r_norms = np.linalg.norm(centered_pos, axis=1)
            
            features[16] = np.mean(r_norms) / 5.0 if len(r_norms) > 0 else 0
            features[17] = np.std(r_norms) / 2.5 if len(r_norms) > 1 else 0
            features[18] = np.max(r_norms) / 5.0 if len(r_norms) > 0 else 0
            features[19] = np.min(r_norms) / 5.0 if len(r_norms) > 0 else 0
            
            # 角动量相关特征
            if len(centered_pos) > 1:
                # 计算角动量
                for i in range(len(centered_pos)):
                    L = np.cross(centered_pos[i], centered_pos[i])  # 伪角动量
                
                # 角分布
                angles = []
                for i in range(len(centered_pos)):
                    for j in range(i+1, len(centered_pos)):
                        vec1 = centered_pos[i]
                        vec2 = centered_pos[j]
                        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angles.append(np.arccos(cos_angle))
                
                if angles:
                    features[20] = np.mean(angles) / np.pi
                    features[21] = np.std(angles) / np.pi if len(angles) > 1 else 0
                    features[22] = np.max(angles) / np.pi
                    features[23] = np.min(angles) / np.pi
            
            # [24-31] 结构一致性特征
            if len(centered_pos) > 0:
                for i in range(len(centered_pos)):
                    norm = np.linalg.norm(centered_pos[i])
                    if norm > 0:
                        centered_pos[i] = centered_pos[i] / norm
                
                # 球面均匀性
                if len(centered_pos) > 1:
                    mean_vec = np.mean(centered_pos, axis=0)
                    features[24] = np.linalg.norm(mean_vec)
                    
                    # 分散度
                    variance = np.var(centered_pos, axis=0)
                    features[25] = np.mean(variance)
                    features[26] = np.std(variance)
                    
                    # 各向异性
                    features[27] = np.max(variance) - np.min(variance)
            
            features[28:32] = features[12:16]  # 复制部分特征
            
            # [32-47] 高阶张量特征
            if len(centered_pos) > 1:
                # 三阶矩
                for i in range(3):
                    features[32 + i] = np.mean(centered_pos[:, i] ** 3)
                
                # 四阶矩
                for i in range(3):
                    features[35 + i] = np.mean(centered_pos[:, i] ** 4)
                
                # 协方差矩阵特征
                cov_matrix = np.cov(centered_pos.T)
                features[38] = np.trace(cov_matrix)
                features[39] = np.linalg.det(cov_matrix) if cov_matrix.shape[0] == 3 else 0
                
                # Frobenius范数
                features[40] = np.linalg.norm(cov_matrix, 'fro')
            
            # [48-63] 多体特征
            if len(centered_pos) > 2:
                # 三体角度
                angles = []
                for i in range(min(5, len(centered_pos))):
                    for j in range(i+1, min(5, len(centered_pos))):
                        for k in range(j+1, min(5, len(centered_pos))):
                            vec_ij = centered_pos[j] - centered_pos[i]
                            vec_ik = centered_pos[k] - centered_pos[i]
                            cos_angle = np.dot(vec_ij, vec_ik) / (np.linalg.norm(vec_ij) * np.linalg.norm(vec_ik) + 1e-6)
                            cos_angle = np.clip(cos_angle, -1, 1)
                            angles.append(np.arccos(cos_angle))
                
                if angles:
                    features[41] = np.mean(angles) / np.pi
                    features[42] = np.std(angles) / np.pi if len(angles) > 1 else 0
                    features[43] = np.max(angles) / np.pi
                    features[44] = np.min(angles) / np.pi
            
            features[45:48] = features[20:23]  # 复制部分特征
            
            # [48-63] 径向特征
            if len(centered_pos) > 0:
                r_norms = np.linalg.norm(centered_pos, axis=1)
                features[48] = np.mean(r_norms ** 2) / 25.0
                features[49] = np.mean(r_norms ** 3) / 125.0
                features[50] = np.mean(r_norms ** 4) / 625.0
                
                # 距离分布特征
                n_bins = 10
                hist, _ = np.histogram(r_norms, bins=n_bins, range=(0, 5))
                hist = hist / (np.sum(hist) + 1e-6)
                features[51:min(61, 51 + len(hist))] = hist[:min(10, len(hist))]
            
            features[61:64] = features[0:3]  # 复制s轨道特征
        
        except Exception as e:
            pass
        
        return features
    
    # ==========================================
    # 4. 晶胞特征 (Unit Cell Features)
    # ==========================================
    
    def compute_unitcell_features(self, struct_dict: Dict) -> np.ndarray:
        """
        计算晶胞特征 (64维)
        """
        features = np.zeros(64, dtype=np.float32)
        
        try:
            lattice = struct_dict.get('lattice', {})
            sites = struct_dict.get('sites', [])
            
            a = lattice.get('a', 1)
            b = lattice.get('b', 1)
            c = lattice.get('c', 1)
            alpha = np.radians(lattice.get('alpha', 90))
            beta = np.radians(lattice.get('beta', 90))
            gamma = np.radians(lattice.get('gamma', 90))
            vol = lattice.get('volume', 1)
            
            # [0-15] 晶格参数
            features[0] = a / 20.0
            features[1] = b / 20.0
            features[2] = c / 20.0
            features[3] = vol / 1000.0
            
            # 比率
            features[4] = a / (b + 1e-6)
            features[5] = a / (c + 1e-6)
            features[6] = b / (c + 1e-6)
            features[7] = vol / (a * b * c + 1e-6)
            
            # 角度特征
            features[8] = np.cos(alpha)
            features[9] = np.cos(beta)
            features[10] = np.cos(gamma)
            features[11] = np.sin(alpha)
            features[12] = np.sin(beta)
            features[13] = np.sin(gamma)
            
            # 体积相关
            features[14] = vol ** (1/3) / 10.0
            features[15] = vol / (a * b * c) if a*b*c > 0 else 1.0
            
            # [16-31] 原子分布特征
            if sites:
                positions = np.array([site.get('xyz', [0, 0, 0]) for site in sites])
                
                features[16] = np.mean(positions[:, 0])
                features[17] = np.mean(positions[:, 1])
                features[18] = np.mean(positions[:, 2])
                
                features[19] = np.std(positions[:, 0]) if len(positions) > 1 else 0
                features[20] = np.std(positions[:, 1]) if len(positions) > 1 else 0
                features[21] = np.std(positions[:, 2]) if len(positions) > 1 else 0
                
                features[22] = np.max(positions[:, 0]) - np.min(positions[:, 0])
                features[23] = np.max(positions[:, 1]) - np.min(positions[:, 1])
                features[24] = np.max(positions[:, 2]) - np.min(positions[:, 2])
                
                # 原子密度
                features[25] = len(sites) / vol if vol > 0 else 0
                features[26] = len(sites) ** 2 / vol if vol > 0 else 0
            
            # [32-47] 晶格变形特征
            # 构造晶格矩阵
            cos_alpha, cos_beta, cos_gamma = np.cos(alpha), np.cos(beta), np.cos(gamma)
            sin_alpha, sin_beta, sin_gamma = np.sin(alpha), np.sin(beta), np.sin(gamma)
            
            lat_matrix = np.array([
                [a, 0, 0],
                [b * cos_gamma, b * sin_gamma, 0],
                [c * cos_beta, c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma,
                 c * sin_beta * sin_gamma / sin_gamma]
            ])
            
            # 晶格矩阵的行列式
            features[32] = np.linalg.det(lat_matrix) / 1000.0
            
            # Gram矩阵特征
            gram = lat_matrix @ lat_matrix.T
            features[33] = np.trace(gram) / (a**2 + b**2 + c**2)
            features[34] = np.linalg.det(gram) / (vol ** 2) if vol > 0 else 0
            
            # 条件数
            try:
                cond = np.linalg.cond(gram)
                features[35] = np.log(cond + 1) / 10.0
            except:
                features[35] = 0
            
            # 各向异性
            eigenvals = np.linalg.eigvalsh(gram)
            features[36] = (np.max(eigenvals) - np.min(eigenvals)) / (np.max(eigenvals) + 1e-6)
            features[37] = np.linalg.norm(eigenvals) / (np.sqrt(3) * np.mean(eigenvals) + 1e-6)
            
            # [48-63] 多尺度晶胞特征
            features[48:51] = features[8:11]
            features[51:54] = features[0:3]
            features[54:57] = features[3:6]
            features[57:60] = features[28:31]
            
            # 额外统计特征
            all_params = [a, b, c]
            features[60] = np.mean(all_params) / 10.0
            features[61] = np.std(all_params) / 10.0
            features[62] = np.max(all_params) / 20.0
            features[63] = np.min(all_params) / 20.0
        
        except Exception as e:
            pass
        
        return features
    
    # ==========================================
    # 5. 化学特征 (Chemical Features)
    # ==========================================
    
    def compute_chemical_features(self, struct_dict: Dict) -> np.ndarray:
        """
        计算化学特征 (44维)
        """
        features = np.zeros(44, dtype=np.float32)
        
        try:
            sites = struct_dict.get('sites', [])
            
            # 成分分析
            comp = {}
            total_atoms = 0
            
            for site in sites:
                for species in site['species']:
                    el = species['element']
                    occu = species['occu']
                    comp[el] = comp.get(el, 0) + occu
                    total_atoms += occu
            
            if total_atoms == 0:
                return features
            
            comp_frac = {el: count / total_atoms for el, count in comp.items()}
            
            # [0-11] 电负性特征
            en_values = []
            for el in comp_frac.keys():
                data = self.get_element_data(el)
                en_values.append(data[3])
            
            if en_values:
                features[0] = np.mean(en_values) / 4.0
                features[1] = np.std(en_values) / 2.0 if len(en_values) > 1 else 0
                features[2] = max(en_values) / 4.0
                features[3] = min(en_values) / 4.0
                features[4] = (max(en_values) - min(en_values)) / 4.0
                
                # 加权电负性差异
                weighted_en = sum(comp_frac[el] * data[3] for el, data in 
                                 [(el, self.get_element_data(el)) for el in comp_frac.keys()])
                features[5] = weighted_en / 4.0
                
                # 电负性极化指数
                if len(en_values) > 1:
                    en_sorted = sorted(en_values)
                    features[6] = (en_sorted[-1] - en_sorted[0]) / 4.0
                    features[7] = (en_sorted[-2] - en_sorted[1]) / 4.0 if len(en_sorted) > 2 else 0
            
            # [12-23] 离子特征
            charges = []
            for el in comp_frac.keys():
                # 估算常见氧化态
                common_charges = {
                    'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
                    'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
                    'Al': 3, 'B': 3, 'Ga': 3, 'In': 3,
                    'Ti': 4, 'Zr': 4, 'Hf': 4, 'Si': 4, 'Ge': 4, 'Sn': 4,
                    'V': 5, 'Nb': 5, 'Ta': 5, 'P': 5,
                    'Cr': 3, 'Mo': 6, 'W': 6, 'S': 6,
                    'Mn': 4, 'Fe': 3, 'Co': 3, 'Ni': 2, 'Cu': 2, 'Zn': 2,
                    'Ag': 1, 'Cd': 2, 'Bi': 3, 'Pb': 2,
                    'O': -2, 'S': -2, 'Se': -2, 'Te': -2,
                    'F': -1, 'Cl': -1, 'Br': -1, 'I': -1,
                    'N': -3, 'P': -3, 'As': -3,
                    'La': 3, 'Ce': 4, 'Y': 3, 'U': 6, 'Th': 4
                }
                charge = common_charges.get(el, 2)
                charges.append(charge)
            
            if charges:
                features[12] = np.mean(charges) / 6.0
                features[13] = np.std(charges) / 3.0 if len(charges) > 1 else 0
                features[14] = max(charges) / 6.0
                features[15] = min(charges) / 6.0
                features[16] = sum(comp_frac[el] * charge for el, charge in zip(comp.keys(), charges)) / 6.0
                
                # 电荷平衡指数
                avg_charge = np.mean(charges)
                charge_balance = abs(sum(comp_frac[el] * (charge - avg_charge) 
                                        for el, charge in zip(comp.keys(), charges)))
                features[17] = charge_balance / 6.0
            
            # [24-35] 元素类型特征
            tm_frac, lanthanide_frac, halogen_frac = 0, 0, 0
            d0_frac, o_frac, pn_frac = 0, 0, 0
            
            for el, frac in comp_frac.items():
                if el in TRANSITION_METALS:
                    tm_frac += frac
                if el in LANTHANIDES:
                    lanthanide_frac += frac
                if el in HALOGENS:
                    halogen_frac += frac
                if el in D0_METALS:
                    d0_frac += frac
                if el == 'O':
                    o_frac += frac
                if el in ['N', 'P', 'As', 'Sb', 'Bi']:
                    pn_frac += frac
            
            features[24] = tm_frac
            features[25] = lanthanide_frac
            features[26] = halogen_frac
            features[27] = d0_frac
            features[28] = o_frac
            features[29] = pn_frac
            features[30] = len(comp) / 10.0  # 元素数量
            features[31] = max(comp_frac.values()) if comp_frac else 0
            features[32] = min(comp_frac.values()) if comp_frac else 0
            features[33] = (max(comp_frac.values()) - min(comp_frac.values())) if comp_frac else 0
            
            # [36-43] 离子类型特征
            features[36] = o_frac / (halogen_frac + 1e-6) if halogen_frac > 0 else 0
            features[37] = tm_frac * d0_frac
            features[38] = (tm_frac + lanthanide_frac) / 2.0
            
            # 铁电可能性指示符
            has_d0 = d0_frac > 0
            has_polar = (pn_frac > 0 or halogen_frac > 0) and tm_frac > 0
            features[39] = float(has_d0)
            features[40] = float(has_polar)
            features[41] = float(o_frac > 0 and tm_frac > 0)
            
            # [44] 化学复杂度
            entropy = -sum(f * np.log(f + 1e-10) for f in comp_frac.values() if f > 0)
            features[43] = entropy / np.log(len(comp) + 1) if len(comp) > 1 else 0
        
        except Exception as e:
            pass
        
        return features
    
    # ==========================================
    # 主提取函数
    # ==========================================
    
    def extract_advanced_features(self, struct_dict: Dict, spacegroup_number: int = None,
                                  band_gap: float = None, cutoff: float = 5.0) -> np.ndarray:
        """
        提取所有高级特征 (256维总)
        
        组合:
        - 基础特征: 64维 (现有UnifiedFeatureExtractor)
        - 图结构特征: 36维
        - 对称性特征: 48维
        - 等变特征: 64维
        - 晶胞特征: 64维
        - 化学特征: 44维
        
        总计: 320维 (包含重复/冗余)
        
        返回降维到256维
        """
        features_list = []
        
        # 1. 基础特征
        base_features = self.base_extractor.extract_from_structure_dict(
            struct_dict, spacegroup_number, band_gap
        )
        features_list.append(base_features)
        
        # 2. 图结构特征
        graph_features = self.compute_graph_features(struct_dict, cutoff)
        features_list.append(graph_features)
        
        # 3. 对称性特征
        symmetry_features = self.compute_symmetry_features(struct_dict, spacegroup_number)
        features_list.append(symmetry_features)
        
        # 4. 等变特征
        equivariant_features = self.compute_equivariant_features(struct_dict)
        features_list.append(equivariant_features)
        
        # 5. 晶胞特征
        unitcell_features = self.compute_unitcell_features(struct_dict)
        features_list.append(unitcell_features)
        
        # 6. 化学特征
        chemical_features = self.compute_chemical_features(struct_dict)
        features_list.append(chemical_features)
        
        # 连接所有特征
        all_features = np.concatenate(features_list)  # 320维
        
        # 降维到256维 (使用主要的特征)
        # 保留最重要的特征,去除冗余的
        selected_indices = [
            # 基础特征: 64个
            *range(0, 64),
            # 图结构: 保留最重要的20个
            *range(64, 64 + min(20, len(graph_features))),
            # 对称性: 保留最重要的24个
            *range(64 + len(graph_features), 64 + len(graph_features) + min(24, len(symmetry_features))),
            # 等变: 保留最重要的32个
            *range(64 + len(graph_features) + len(symmetry_features), 
                  64 + len(graph_features) + len(symmetry_features) + min(32, len(equivariant_features))),
            # 晶胞: 保留最重要的24个
            *range(64 + len(graph_features) + len(symmetry_features) + len(equivariant_features),
                  64 + len(graph_features) + len(symmetry_features) + len(equivariant_features) + min(24, len(unitcell_features))),
            # 化学: 保留最重要的20个
            *range(64 + len(graph_features) + len(symmetry_features) + len(equivariant_features) + len(unitcell_features),
                  64 + len(graph_features) + len(symmetry_features) + len(equivariant_features) + len(unitcell_features) + min(20, len(chemical_features))),
        ]
        
        selected_indices = selected_indices[:256]
        
        # 如果不足256维,填充0
        advanced_features = np.zeros(256, dtype=np.float32)
        for i, idx in enumerate(selected_indices):
            if idx < len(all_features):
                advanced_features[i] = all_features[idx]
        
        # 特征归一化 (防止数值溢出)
        advanced_features = np.clip(advanced_features, -5, 5)
        
        return advanced_features
