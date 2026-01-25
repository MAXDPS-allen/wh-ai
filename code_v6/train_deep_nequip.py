#!/usr/bin/env python3
"""
NequIP增强型深度学习铁电分类器 v6 - 深度版本
使用深度神经网络自动学习高维特征表示

关键技术:
1. E(3)等变特征编码 (球谐函数 + 径向基函数)
2. 深度神经网络自动特征学习
3. 注意力机制捕捉原子间相互作用
4. Focal Loss 处理极端类别不平衡
5. 对比学习增强特征分离
6. 多尺度特征融合
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ============================================================================
# 第一部分: 高维特征工程 - 512维特征
# ============================================================================

class AdvancedFeatureExtractor:
    """
    高级NequIP风格特征提取器
    生成512维特征向量
    """
    
    # 元素属性字典
    ELEMENT_PROPERTIES = {
        'H': {'Z': 1, 'period': 1, 'group': 1, 'electronegativity': 2.20, 'atomic_radius': 0.53, 'ionization_energy': 13.598, 'electron_affinity': 0.754, 'valence': 1},
        'He': {'Z': 2, 'period': 1, 'group': 18, 'electronegativity': 0.0, 'atomic_radius': 0.31, 'ionization_energy': 24.587, 'electron_affinity': 0.0, 'valence': 0},
        'Li': {'Z': 3, 'period': 2, 'group': 1, 'electronegativity': 0.98, 'atomic_radius': 1.67, 'ionization_energy': 5.392, 'electron_affinity': 0.618, 'valence': 1},
        'Be': {'Z': 4, 'period': 2, 'group': 2, 'electronegativity': 1.57, 'atomic_radius': 1.12, 'ionization_energy': 9.323, 'electron_affinity': 0.0, 'valence': 2},
        'B': {'Z': 5, 'period': 2, 'group': 13, 'electronegativity': 2.04, 'atomic_radius': 0.87, 'ionization_energy': 8.298, 'electron_affinity': 0.277, 'valence': 3},
        'C': {'Z': 6, 'period': 2, 'group': 14, 'electronegativity': 2.55, 'atomic_radius': 0.67, 'ionization_energy': 11.260, 'electron_affinity': 1.263, 'valence': 4},
        'N': {'Z': 7, 'period': 2, 'group': 15, 'electronegativity': 3.04, 'atomic_radius': 0.56, 'ionization_energy': 14.534, 'electron_affinity': -0.07, 'valence': 5},
        'O': {'Z': 8, 'period': 2, 'group': 16, 'electronegativity': 3.44, 'atomic_radius': 0.48, 'ionization_energy': 13.618, 'electron_affinity': 1.461, 'valence': 6},
        'F': {'Z': 9, 'period': 2, 'group': 17, 'electronegativity': 3.98, 'atomic_radius': 0.42, 'ionization_energy': 17.423, 'electron_affinity': 3.401, 'valence': 7},
        'Na': {'Z': 11, 'period': 3, 'group': 1, 'electronegativity': 0.93, 'atomic_radius': 1.90, 'ionization_energy': 5.139, 'electron_affinity': 0.548, 'valence': 1},
        'Mg': {'Z': 12, 'period': 3, 'group': 2, 'electronegativity': 1.31, 'atomic_radius': 1.45, 'ionization_energy': 7.646, 'electron_affinity': 0.0, 'valence': 2},
        'Al': {'Z': 13, 'period': 3, 'group': 13, 'electronegativity': 1.61, 'atomic_radius': 1.18, 'ionization_energy': 5.986, 'electron_affinity': 0.441, 'valence': 3},
        'Si': {'Z': 14, 'period': 3, 'group': 14, 'electronegativity': 1.90, 'atomic_radius': 1.11, 'ionization_energy': 8.152, 'electron_affinity': 1.385, 'valence': 4},
        'P': {'Z': 15, 'period': 3, 'group': 15, 'electronegativity': 2.19, 'atomic_radius': 0.98, 'ionization_energy': 10.487, 'electron_affinity': 0.746, 'valence': 5},
        'S': {'Z': 16, 'period': 3, 'group': 16, 'electronegativity': 2.58, 'atomic_radius': 0.88, 'ionization_energy': 10.360, 'electron_affinity': 2.077, 'valence': 6},
        'Cl': {'Z': 17, 'period': 3, 'group': 17, 'electronegativity': 3.16, 'atomic_radius': 0.79, 'ionization_energy': 12.968, 'electron_affinity': 3.617, 'valence': 7},
        'K': {'Z': 19, 'period': 4, 'group': 1, 'electronegativity': 0.82, 'atomic_radius': 2.43, 'ionization_energy': 4.341, 'electron_affinity': 0.501, 'valence': 1},
        'Ca': {'Z': 20, 'period': 4, 'group': 2, 'electronegativity': 1.00, 'atomic_radius': 1.94, 'ionization_energy': 6.113, 'electron_affinity': 0.02, 'valence': 2},
        'Sc': {'Z': 21, 'period': 4, 'group': 3, 'electronegativity': 1.36, 'atomic_radius': 1.84, 'ionization_energy': 6.561, 'electron_affinity': 0.188, 'valence': 3},
        'Ti': {'Z': 22, 'period': 4, 'group': 4, 'electronegativity': 1.54, 'atomic_radius': 1.76, 'ionization_energy': 6.828, 'electron_affinity': 0.079, 'valence': 4},
        'V': {'Z': 23, 'period': 4, 'group': 5, 'electronegativity': 1.63, 'atomic_radius': 1.71, 'ionization_energy': 6.746, 'electron_affinity': 0.525, 'valence': 5},
        'Cr': {'Z': 24, 'period': 4, 'group': 6, 'electronegativity': 1.66, 'atomic_radius': 1.66, 'ionization_energy': 6.767, 'electron_affinity': 0.666, 'valence': 6},
        'Mn': {'Z': 25, 'period': 4, 'group': 7, 'electronegativity': 1.55, 'atomic_radius': 1.61, 'ionization_energy': 7.434, 'electron_affinity': 0.0, 'valence': 7},
        'Fe': {'Z': 26, 'period': 4, 'group': 8, 'electronegativity': 1.83, 'atomic_radius': 1.56, 'ionization_energy': 7.902, 'electron_affinity': 0.163, 'valence': 3},
        'Co': {'Z': 27, 'period': 4, 'group': 9, 'electronegativity': 1.88, 'atomic_radius': 1.52, 'ionization_energy': 7.881, 'electron_affinity': 0.661, 'valence': 3},
        'Ni': {'Z': 28, 'period': 4, 'group': 10, 'electronegativity': 1.91, 'atomic_radius': 1.49, 'ionization_energy': 7.640, 'electron_affinity': 1.156, 'valence': 2},
        'Cu': {'Z': 29, 'period': 4, 'group': 11, 'electronegativity': 1.90, 'atomic_radius': 1.45, 'ionization_energy': 7.726, 'electron_affinity': 1.228, 'valence': 2},
        'Zn': {'Z': 30, 'period': 4, 'group': 12, 'electronegativity': 1.65, 'atomic_radius': 1.42, 'ionization_energy': 9.394, 'electron_affinity': 0.0, 'valence': 2},
        'Ga': {'Z': 31, 'period': 4, 'group': 13, 'electronegativity': 1.81, 'atomic_radius': 1.36, 'ionization_energy': 5.999, 'electron_affinity': 0.30, 'valence': 3},
        'Ge': {'Z': 32, 'period': 4, 'group': 14, 'electronegativity': 2.01, 'atomic_radius': 1.25, 'ionization_energy': 7.900, 'electron_affinity': 1.233, 'valence': 4},
        'As': {'Z': 33, 'period': 4, 'group': 15, 'electronegativity': 2.18, 'atomic_radius': 1.14, 'ionization_energy': 9.815, 'electron_affinity': 0.81, 'valence': 5},
        'Se': {'Z': 34, 'period': 4, 'group': 16, 'electronegativity': 2.55, 'atomic_radius': 1.03, 'ionization_energy': 9.752, 'electron_affinity': 2.021, 'valence': 6},
        'Br': {'Z': 35, 'period': 4, 'group': 17, 'electronegativity': 2.96, 'atomic_radius': 0.94, 'ionization_energy': 11.814, 'electron_affinity': 3.365, 'valence': 7},
        'Rb': {'Z': 37, 'period': 5, 'group': 1, 'electronegativity': 0.82, 'atomic_radius': 2.65, 'ionization_energy': 4.177, 'electron_affinity': 0.486, 'valence': 1},
        'Sr': {'Z': 38, 'period': 5, 'group': 2, 'electronegativity': 0.95, 'atomic_radius': 2.19, 'ionization_energy': 5.695, 'electron_affinity': 0.05, 'valence': 2},
        'Y': {'Z': 39, 'period': 5, 'group': 3, 'electronegativity': 1.22, 'atomic_radius': 2.12, 'ionization_energy': 6.217, 'electron_affinity': 0.307, 'valence': 3},
        'Zr': {'Z': 40, 'period': 5, 'group': 4, 'electronegativity': 1.33, 'atomic_radius': 2.06, 'ionization_energy': 6.634, 'electron_affinity': 0.426, 'valence': 4},
        'Nb': {'Z': 41, 'period': 5, 'group': 5, 'electronegativity': 1.60, 'atomic_radius': 1.98, 'ionization_energy': 6.759, 'electron_affinity': 0.893, 'valence': 5},
        'Mo': {'Z': 42, 'period': 5, 'group': 6, 'electronegativity': 2.16, 'atomic_radius': 1.90, 'ionization_energy': 7.092, 'electron_affinity': 0.746, 'valence': 6},
        'Ru': {'Z': 44, 'period': 5, 'group': 8, 'electronegativity': 2.20, 'atomic_radius': 1.78, 'ionization_energy': 7.361, 'electron_affinity': 1.05, 'valence': 4},
        'Rh': {'Z': 45, 'period': 5, 'group': 9, 'electronegativity': 2.28, 'atomic_radius': 1.73, 'ionization_energy': 7.459, 'electron_affinity': 1.137, 'valence': 3},
        'Pd': {'Z': 46, 'period': 5, 'group': 10, 'electronegativity': 2.20, 'atomic_radius': 1.69, 'ionization_energy': 8.337, 'electron_affinity': 0.562, 'valence': 2},
        'Ag': {'Z': 47, 'period': 5, 'group': 11, 'electronegativity': 1.93, 'atomic_radius': 1.65, 'ionization_energy': 7.576, 'electron_affinity': 1.302, 'valence': 1},
        'Cd': {'Z': 48, 'period': 5, 'group': 12, 'electronegativity': 1.69, 'atomic_radius': 1.61, 'ionization_energy': 8.994, 'electron_affinity': 0.0, 'valence': 2},
        'In': {'Z': 49, 'period': 5, 'group': 13, 'electronegativity': 1.78, 'atomic_radius': 1.56, 'ionization_energy': 5.786, 'electron_affinity': 0.30, 'valence': 3},
        'Sn': {'Z': 50, 'period': 5, 'group': 14, 'electronegativity': 1.96, 'atomic_radius': 1.45, 'ionization_energy': 7.344, 'electron_affinity': 1.112, 'valence': 4},
        'Sb': {'Z': 51, 'period': 5, 'group': 15, 'electronegativity': 2.05, 'atomic_radius': 1.33, 'ionization_energy': 8.64, 'electron_affinity': 1.07, 'valence': 5},
        'Te': {'Z': 52, 'period': 5, 'group': 16, 'electronegativity': 2.10, 'atomic_radius': 1.23, 'ionization_energy': 9.010, 'electron_affinity': 1.971, 'valence': 6},
        'I': {'Z': 53, 'period': 5, 'group': 17, 'electronegativity': 2.66, 'atomic_radius': 1.15, 'ionization_energy': 10.451, 'electron_affinity': 3.059, 'valence': 7},
        'Cs': {'Z': 55, 'period': 6, 'group': 1, 'electronegativity': 0.79, 'atomic_radius': 2.98, 'ionization_energy': 3.894, 'electron_affinity': 0.472, 'valence': 1},
        'Ba': {'Z': 56, 'period': 6, 'group': 2, 'electronegativity': 0.89, 'atomic_radius': 2.53, 'ionization_energy': 5.212, 'electron_affinity': 0.14, 'valence': 2},
        'La': {'Z': 57, 'period': 6, 'group': 3, 'electronegativity': 1.10, 'atomic_radius': 2.50, 'ionization_energy': 5.577, 'electron_affinity': 0.47, 'valence': 3},
        'Ce': {'Z': 58, 'period': 6, 'group': 3, 'electronegativity': 1.12, 'atomic_radius': 2.48, 'ionization_energy': 5.539, 'electron_affinity': 0.50, 'valence': 4},
        'Pr': {'Z': 59, 'period': 6, 'group': 3, 'electronegativity': 1.13, 'atomic_radius': 2.47, 'ionization_energy': 5.473, 'electron_affinity': 0.50, 'valence': 4},
        'Nd': {'Z': 60, 'period': 6, 'group': 3, 'electronegativity': 1.14, 'atomic_radius': 2.45, 'ionization_energy': 5.525, 'electron_affinity': 0.50, 'valence': 3},
        'Sm': {'Z': 62, 'period': 6, 'group': 3, 'electronegativity': 1.17, 'atomic_radius': 2.42, 'ionization_energy': 5.644, 'electron_affinity': 0.50, 'valence': 3},
        'Eu': {'Z': 63, 'period': 6, 'group': 3, 'electronegativity': 1.20, 'atomic_radius': 2.40, 'ionization_energy': 5.670, 'electron_affinity': 0.50, 'valence': 3},
        'Gd': {'Z': 64, 'period': 6, 'group': 3, 'electronegativity': 1.20, 'atomic_radius': 2.38, 'ionization_energy': 6.150, 'electron_affinity': 0.50, 'valence': 3},
        'Tb': {'Z': 65, 'period': 6, 'group': 3, 'electronegativity': 1.20, 'atomic_radius': 2.37, 'ionization_energy': 5.864, 'electron_affinity': 0.50, 'valence': 4},
        'Dy': {'Z': 66, 'period': 6, 'group': 3, 'electronegativity': 1.22, 'atomic_radius': 2.35, 'ionization_energy': 5.939, 'electron_affinity': 0.50, 'valence': 3},
        'Ho': {'Z': 67, 'period': 6, 'group': 3, 'electronegativity': 1.23, 'atomic_radius': 2.33, 'ionization_energy': 6.022, 'electron_affinity': 0.50, 'valence': 3},
        'Er': {'Z': 68, 'period': 6, 'group': 3, 'electronegativity': 1.24, 'atomic_radius': 2.32, 'ionization_energy': 6.108, 'electron_affinity': 0.50, 'valence': 3},
        'Tm': {'Z': 69, 'period': 6, 'group': 3, 'electronegativity': 1.25, 'atomic_radius': 2.30, 'ionization_energy': 6.184, 'electron_affinity': 0.50, 'valence': 3},
        'Yb': {'Z': 70, 'period': 6, 'group': 3, 'electronegativity': 1.10, 'atomic_radius': 2.28, 'ionization_energy': 6.254, 'electron_affinity': 0.50, 'valence': 3},
        'Lu': {'Z': 71, 'period': 6, 'group': 3, 'electronegativity': 1.27, 'atomic_radius': 2.17, 'ionization_energy': 5.426, 'electron_affinity': 0.34, 'valence': 3},
        'Hf': {'Z': 72, 'period': 6, 'group': 4, 'electronegativity': 1.30, 'atomic_radius': 2.08, 'ionization_energy': 6.825, 'electron_affinity': 0.0, 'valence': 4},
        'Ta': {'Z': 73, 'period': 6, 'group': 5, 'electronegativity': 1.50, 'atomic_radius': 2.00, 'ionization_energy': 7.550, 'electron_affinity': 0.322, 'valence': 5},
        'W': {'Z': 74, 'period': 6, 'group': 6, 'electronegativity': 2.36, 'atomic_radius': 1.93, 'ionization_energy': 7.864, 'electron_affinity': 0.815, 'valence': 6},
        'Re': {'Z': 75, 'period': 6, 'group': 7, 'electronegativity': 1.90, 'atomic_radius': 1.88, 'ionization_energy': 7.833, 'electron_affinity': 0.15, 'valence': 7},
        'Os': {'Z': 76, 'period': 6, 'group': 8, 'electronegativity': 2.20, 'atomic_radius': 1.85, 'ionization_energy': 8.438, 'electron_affinity': 1.1, 'valence': 4},
        'Ir': {'Z': 77, 'period': 6, 'group': 9, 'electronegativity': 2.20, 'atomic_radius': 1.80, 'ionization_energy': 8.967, 'electron_affinity': 1.565, 'valence': 4},
        'Pt': {'Z': 78, 'period': 6, 'group': 10, 'electronegativity': 2.28, 'atomic_radius': 1.77, 'ionization_energy': 8.959, 'electron_affinity': 2.128, 'valence': 4},
        'Au': {'Z': 79, 'period': 6, 'group': 11, 'electronegativity': 2.54, 'atomic_radius': 1.74, 'ionization_energy': 9.226, 'electron_affinity': 2.309, 'valence': 3},
        'Hg': {'Z': 80, 'period': 6, 'group': 12, 'electronegativity': 2.00, 'atomic_radius': 1.71, 'ionization_energy': 10.438, 'electron_affinity': 0.0, 'valence': 2},
        'Tl': {'Z': 81, 'period': 6, 'group': 13, 'electronegativity': 1.62, 'atomic_radius': 1.56, 'ionization_energy': 6.108, 'electron_affinity': 0.20, 'valence': 3},
        'Pb': {'Z': 82, 'period': 6, 'group': 14, 'electronegativity': 2.33, 'atomic_radius': 1.54, 'ionization_energy': 7.417, 'electron_affinity': 0.364, 'valence': 4},
        'Bi': {'Z': 83, 'period': 6, 'group': 15, 'electronegativity': 2.02, 'atomic_radius': 1.43, 'ionization_energy': 7.286, 'electron_affinity': 0.946, 'valence': 5},
    }
    
    # 默认属性
    DEFAULT_PROPS = {'Z': 50, 'period': 5, 'group': 10, 'electronegativity': 1.5, 
                     'atomic_radius': 1.5, 'ionization_energy': 7.0, 'electron_affinity': 0.5, 'valence': 3}
    
    def __init__(self, n_radial_basis: int = 32, n_spherical_harmonics: int = 6, cutoff: float = 8.0):
        self.n_radial = n_radial_basis
        self.n_sph = n_spherical_harmonics
        self.cutoff = cutoff
        # 总特征维度: 512
        # 原子特征: 64, 径向特征: 128, 球谐特征: 192, 结构特征: 64, 对称性特征: 64
        self.feature_dim = 512
        
    def get_element_props(self, symbol: str) -> Dict:
        """获取元素属性"""
        return self.ELEMENT_PROPERTIES.get(symbol, self.DEFAULT_PROPS)
    
    def compute_radial_basis(self, distances: np.ndarray) -> np.ndarray:
        """计算径向基函数 - 多尺度高斯"""
        # 截断距离外的
        distances = np.clip(distances, 0.5, self.cutoff)
        
        # 多尺度中心和宽度
        centers = np.linspace(0.5, self.cutoff, self.n_radial)
        widths = np.linspace(0.3, 1.5, self.n_radial)
        
        # 计算所有径向基函数
        rbf = np.zeros((len(distances), self.n_radial))
        for i, (c, w) in enumerate(zip(centers, widths)):
            rbf[:, i] = np.exp(-((distances - c) ** 2) / (2 * w ** 2))
        
        # 应用截断函数
        cutoff_fn = 0.5 * (1 + np.cos(np.pi * distances / self.cutoff))
        rbf = rbf * cutoff_fn[:, np.newaxis]
        
        return rbf
    
    def compute_spherical_harmonics(self, vectors: np.ndarray) -> np.ndarray:
        """计算球谐函数特征 (简化版)"""
        if len(vectors) == 0:
            return np.zeros(self.n_sph * 4)
        
        # 归一化
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        unit_vectors = vectors / norms
        
        x, y, z = unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2]
        
        # 球谐函数 (l=0,1,2,3)
        Y = []
        # l=0
        Y.append(np.ones_like(x) * 0.282095)
        # l=1
        Y.append(0.488603 * y)
        Y.append(0.488603 * z)
        Y.append(0.488603 * x)
        # l=2
        Y.append(1.092548 * x * y)
        Y.append(1.092548 * y * z)
        Y.append(0.315392 * (3 * z**2 - 1))
        Y.append(1.092548 * x * z)
        Y.append(0.546274 * (x**2 - y**2))
        
        Y = np.array(Y).T  # shape: (n_pairs, 9)
        
        # 聚合统计
        features = []
        for i in range(min(Y.shape[1], self.n_sph)):
            features.extend([
                np.mean(Y[:, i]),
                np.std(Y[:, i]) if len(Y) > 1 else 0,
                np.max(Y[:, i]),
                np.min(Y[:, i])
            ])
        
        # 补充到目标维度
        while len(features) < self.n_sph * 4:
            features.append(0.0)
            
        return np.array(features[:self.n_sph * 4])
    
    def extract_structure_info(self, structure: Dict) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """从结构中提取原子坐标和晶格信息"""
        lattice = np.array(structure.get('lattice', {}).get('matrix', np.eye(3) * 5.0))
        sites = structure.get('sites', [])
        
        coords = []
        elements = []
        for site in sites:
            if 'xyz' in site:
                coords.append(site['xyz'])
            elif 'abc' in site:
                frac = np.array(site['abc'])
                cart = np.dot(frac, lattice)
                coords.append(cart.tolist())
            
            species = site.get('species', [])
            if species:
                elem = species[0].get('element', 'X')
            else:
                elem = site.get('label', 'X')
            elements.append(elem)
        
        coords = np.array(coords) if coords else np.zeros((1, 3))
        return coords, lattice, elements
    
    def compute_pair_features(self, coords: np.ndarray, lattice: np.ndarray, 
                             elements: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """计算原子对特征"""
        n_atoms = len(coords)
        if n_atoms < 2:
            return np.zeros(128), np.zeros(192)
        
        # 计算距离矩阵
        distances = []
        vectors = []
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                diff = coords[j] - coords[i]
                # 考虑周期性边界条件
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        for c in range(-1, 2):
                            shift = a * lattice[0] + b * lattice[1] + c * lattice[2]
                            vec = diff + shift
                            d = np.linalg.norm(vec)
                            if d < self.cutoff and d > 0.5:
                                distances.append(d)
                                vectors.append(vec)
        
        distances = np.array(distances) if distances else np.array([2.0])
        vectors = np.array(vectors) if vectors else np.zeros((1, 3))
        
        # 径向基函数特征
        rbf = self.compute_radial_basis(distances)
        rbf_features = np.concatenate([
            np.mean(rbf, axis=0),
            np.std(rbf, axis=0) if len(rbf) > 1 else np.zeros(self.n_radial),
            np.max(rbf, axis=0),
            np.min(rbf, axis=0)
        ])
        
        # 球谐函数特征
        sph_features = self.compute_spherical_harmonics(vectors)
        
        # 扩展到目标维度
        rbf_out = np.zeros(128)
        rbf_out[:min(len(rbf_features), 128)] = rbf_features[:128]
        
        sph_out = np.zeros(192)
        sph_out[:min(len(sph_features), 192)] = sph_features[:192]
        
        return rbf_out, sph_out
    
    def compute_structural_features(self, coords: np.ndarray, lattice: np.ndarray) -> np.ndarray:
        """计算结构特征"""
        features = []
        
        # 晶格参数
        a = np.linalg.norm(lattice[0])
        b = np.linalg.norm(lattice[1])
        c = np.linalg.norm(lattice[2])
        volume = abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))
        
        features.extend([a, b, c, volume])
        features.extend([a/b if b > 0 else 1, b/c if c > 0 else 1, a/c if c > 0 else 1])
        
        # 晶格角度
        cos_alpha = np.dot(lattice[1], lattice[2]) / (np.linalg.norm(lattice[1]) * np.linalg.norm(lattice[2]) + 1e-10)
        cos_beta = np.dot(lattice[0], lattice[2]) / (np.linalg.norm(lattice[0]) * np.linalg.norm(lattice[2]) + 1e-10)
        cos_gamma = np.dot(lattice[0], lattice[1]) / (np.linalg.norm(lattice[0]) * np.linalg.norm(lattice[1]) + 1e-10)
        
        features.extend([cos_alpha, cos_beta, cos_gamma])
        features.extend([np.arccos(np.clip(cos_alpha, -1, 1)),
                        np.arccos(np.clip(cos_beta, -1, 1)),
                        np.arccos(np.clip(cos_gamma, -1, 1))])
        
        # 原子密度
        n_atoms = len(coords)
        density = n_atoms / (volume + 1e-10)
        features.append(density)
        
        # 原子分布统计
        if n_atoms > 1:
            centroid = np.mean(coords, axis=0)
            distances_to_centroid = np.linalg.norm(coords - centroid, axis=1)
            features.extend([np.mean(distances_to_centroid), 
                           np.std(distances_to_centroid),
                           np.max(distances_to_centroid),
                           np.min(distances_to_centroid)])
        else:
            features.extend([0, 0, 0, 0])
        
        # 补充到64维
        while len(features) < 64:
            features.append(0.0)
        
        return np.array(features[:64])
    
    def compute_symmetry_features(self, coords: np.ndarray, elements: List[str]) -> np.ndarray:
        """计算对称性相关特征"""
        features = []
        
        n_atoms = len(coords)
        
        # 元素统计
        unique_elements = list(set(elements))
        n_unique = len(unique_elements)
        features.append(n_atoms)
        features.append(n_unique)
        features.append(n_atoms / (n_unique + 1e-10))
        
        # 元素属性统计
        props = [self.get_element_props(e) for e in elements]
        
        for prop_name in ['Z', 'electronegativity', 'atomic_radius', 'ionization_energy']:
            values = [p[prop_name] for p in props]
            features.extend([np.mean(values), np.std(values) if len(values) > 1 else 0,
                           np.max(values), np.min(values)])
        
        # 电负性差异 (铁电性相关)
        en_values = [p['electronegativity'] for p in props]
        if len(en_values) >= 2:
            en_diff = np.max(en_values) - np.min(en_values)
            features.append(en_diff)
        else:
            features.append(0)
        
        # 原子半径变化
        rad_values = [p['atomic_radius'] for p in props]
        if len(rad_values) >= 2:
            rad_var = np.var(rad_values)
            features.append(rad_var)
        else:
            features.append(0)
        
        # 反演对称性检测 (简化)
        if n_atoms > 1:
            centroid = np.mean(coords, axis=0)
            centered = coords - centroid
            inversion_score = 0
            for i, c in enumerate(centered):
                # 检查是否存在关于中心对称的原子
                for j, c2 in enumerate(centered):
                    if i != j:
                        if np.linalg.norm(c + c2) < 0.5:  # 近似反演对称
                            inversion_score += 1
            features.append(inversion_score / (n_atoms + 1e-10))
        else:
            features.append(0)
        
        # 极性检测
        if n_atoms > 1:
            # 计算偶极矩方向
            dipole = np.zeros(3)
            for i, (coord, elem) in enumerate(zip(coords, elements)):
                en = self.get_element_props(elem)['electronegativity']
                dipole += en * (coord - np.mean(coords, axis=0))
            dipole_mag = np.linalg.norm(dipole)
            features.append(dipole_mag)
            features.extend(dipole / (dipole_mag + 1e-10))
        else:
            features.extend([0, 0, 0, 0])
        
        # 补充到64维
        while len(features) < 64:
            features.append(0.0)
        
        return np.array(features[:64])
    
    def compute_atomic_features(self, elements: List[str]) -> np.ndarray:
        """计算原子特征统计"""
        features = []
        
        props = [self.get_element_props(e) for e in elements]
        
        # 各属性的统计
        for prop_name in ['Z', 'period', 'group', 'electronegativity', 
                         'atomic_radius', 'ionization_energy', 'electron_affinity', 'valence']:
            values = [p[prop_name] for p in props]
            features.extend([
                np.mean(values),
                np.std(values) if len(values) > 1 else 0,
                np.max(values),
                np.min(values),
                np.median(values),
                np.sum(values),
                np.max(values) - np.min(values),
                np.var(values) if len(values) > 1 else 0
            ])
        
        # 64维
        return np.array(features[:64])
    
    def extract_features(self, structure: Dict) -> np.ndarray:
        """提取完整的512维特征向量"""
        try:
            coords, lattice, elements = self.extract_structure_info(structure)
            
            # 各类特征
            atomic_features = self.compute_atomic_features(elements)  # 64维
            rbf_features, sph_features = self.compute_pair_features(coords, lattice, elements)  # 128 + 192维
            structural_features = self.compute_structural_features(coords, lattice)  # 64维
            symmetry_features = self.compute_symmetry_features(coords, elements)  # 64维
            
            # 拼接所有特征
            features = np.concatenate([
                atomic_features,
                rbf_features,
                sph_features,
                structural_features,
                symmetry_features
            ])
            
            # 处理NaN和Inf
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return features
            
        except Exception as e:
            return np.zeros(self.feature_dim)


# ============================================================================
# 第二部分: 深度学习模型
# ============================================================================

class AttentionBlock(nn.Module):
    """自注意力模块"""
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN
        x = self.norm2(x + self.ffn(x))
        return x


class DeepNequIPClassifier(nn.Module):
    """
    深度NequIP风格分类器
    
    特点:
    1. 多层特征变换学习高维表示
    2. 自注意力机制捕捉特征间关系
    3. 残差连接防止梯度消失
    4. 多尺度特征融合
    """
    
    def __init__(self, input_dim: int = 512, hidden_dims: List[int] = [512, 256, 128, 64],
                 num_attention_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 注意力层
        self.attention_layers = nn.ModuleList([
            AttentionBlock(hidden_dims[0], num_heads=8, dropout=dropout)
            for _ in range(num_attention_layers)
        ])
        
        # 深度特征提取
        self.feature_extractors = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.feature_extractors.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[i+1], hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.GELU()
            ))
        
        # 多尺度特征融合
        self.fusion = nn.Linear(sum(hidden_dims), hidden_dims[-1] * 2)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),
            nn.LayerNorm(hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # 对比学习投影头
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),
            nn.GELU(),
            nn.Linear(hidden_dims[-1], 64)
        )
        
    def forward(self, x, return_features: bool = False):
        # 输入投影
        h = self.input_proj(x)
        
        # 注意力层 (增加序列维度)
        h = h.unsqueeze(1)
        for attn in self.attention_layers:
            h = attn(h)
        h = h.squeeze(1)
        
        # 多尺度特征提取
        features = [h]
        for extractor in self.feature_extractors:
            h = extractor(h)
            features.append(h)
        
        # 特征融合
        fused = torch.cat(features, dim=-1)
        fused = self.fusion(fused)
        
        # 分类
        logits = self.classifier(fused)
        
        if return_features:
            proj = self.projection_head(fused)
            return logits, proj
        
        return logits


class FocalLoss(nn.Module):
    """Focal Loss - 处理极端类别不平衡"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: float = 30.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # 计算 focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # 计算 alpha weight
        alpha_weight = torch.where(targets == 1, self.alpha * self.pos_weight, 1 - self.alpha)
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Final loss
        loss = focal_weight * alpha_weight * bce
        return loss.mean()


class SupConLoss(nn.Module):
    """监督对比学习损失"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签mask
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # 对角线置零 (排除自身)
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=mask.device)
        mask = mask * logits_mask
        
        # 计算log_softmax
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)
        
        # 计算对比损失
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-10)
        loss = -mean_log_prob_pos.mean()
        
        return loss


# ============================================================================
# 第三部分: 数据集和训练
# ============================================================================

class CrystalDataset(Dataset):
    """晶体结构数据集"""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data(data_dir: str, extractor: AdvancedFeatureExtractor) -> Tuple[np.ndarray, np.ndarray]:
    """加载并处理数据"""
    print("\n" + "=" * 70)
    print("加载数据")
    print("=" * 70)
    
    seen_ids = set()
    
    def get_structure_id(structure: Dict) -> str:
        """生成结构唯一标识"""
        s = json.dumps(structure, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()
    
    # 正样本文件
    positive_files = [
        ('dataset_original_ferroelectric.jsonl', 1),
        ('dataset_known_FE_rest.jsonl', 1),
    ]
    
    # 负样本文件
    negative_files = [
        ('dataset_nonFE.jsonl', 0),
        ('dataset_nonFE_cleaned.jsonl', 0),
        ('dataset_nonFE_expanded.jsonl', 0),
    ]
    
    all_features = []
    all_labels = []
    
    # 处理正样本
    print("\n处理正样本 (铁电材料)...")
    for filename, label in positive_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r') as f:
            structures = [json.loads(line) for line in f]
        
        unique_count = 0
        for s in tqdm(structures, desc=f"  {filename}"):
            struct = s.get('structure', s)
            sid = get_structure_id(struct)
            
            if sid not in seen_ids:
                seen_ids.add(sid)
                features = extractor.extract_features(struct)
                all_features.append(features)
                all_labels.append(label)
                unique_count += 1
        
        print(f"  {filename}: {unique_count} 个唯一样本")
    
    # 处理负样本
    print("\n处理负样本 (非铁电材料)...")
    for filename, label in negative_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, 'r') as f:
            structures = [json.loads(line) for line in f]
        
        unique_count = 0
        for s in tqdm(structures, desc=f"  {filename}"):
            struct = s.get('structure', s)
            sid = get_structure_id(struct)
            
            if sid not in seen_ids:
                seen_ids.add(sid)
                features = extractor.extract_features(struct)
                all_features.append(features)
                all_labels.append(label)
                unique_count += 1
        
        print(f"  {filename}: {unique_count} 个唯一样本")
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n数据集统计:")
    print(f"  正样本 (FE): {np.sum(y == 1)}")
    print(f"  负样本 (non-FE): {np.sum(y == 0)}")
    print(f"  特征维度: {X.shape[1]}")
    
    return X, y


def train_epoch(model, train_loader, optimizer, focal_loss, contrastive_loss, 
                device, use_contrastive: bool = True):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device).view(-1, 1)
        
        optimizer.zero_grad()
        
        if use_contrastive:
            logits, proj = model(features, return_features=True)
            loss_cls = focal_loss(logits, labels)
            loss_con = contrastive_loss(proj, labels.view(-1))
            loss = loss_cls + 0.1 * loss_con
        else:
            logits = model(features)
            loss = focal_loss(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """评估模型"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    
    all_probs = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()
    
    # 计算ROC-AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.5
    
    # 不同阈值下的指标
    results = {'roc_auc': roc_auc}
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        acc = accuracy_score(all_labels, preds)
        prec = precision_score(all_labels, preds, zero_division=0)
        rec = recall_score(all_labels, preds, zero_division=0)
        f1 = f1_score(all_labels, preds, zero_division=0)
        
        results[f'thresh_{thresh}'] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    results['best_threshold'] = best_threshold
    results['best_f1'] = best_f1
    
    return results, all_probs, all_labels


def cross_validate(X: np.ndarray, y: np.ndarray, n_folds: int = 5, 
                   epochs: int = 100, batch_size: int = 64, lr: float = 1e-3):
    """5折交叉验证"""
    print("\n" + "=" * 70)
    print("5折交叉验证训练")
    print("=" * 70)
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    
    all_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n{'=' * 70}")
        print(f"Fold {fold + 1}/{n_folds}")
        print("=" * 70)
        
        # 划分数据
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # 创建数据集
        train_dataset = CrystalDataset(X_train, y_train)
        val_dataset = CrystalDataset(X_val, y_val)
        
        # 加权采样器处理类别不平衡
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"训练集: {len(train_dataset)} 样本 (正:{np.sum(y_train)}, 负:{np.sum(1-y_train)})")
        print(f"验证集: {len(val_dataset)} 样本 (正:{np.sum(y_val)}, 负:{np.sum(1-y_val)})")
        
        # 初始化模型
        model = DeepNequIPClassifier(
            input_dim=X.shape[1],
            hidden_dims=[512, 256, 128, 64],
            num_attention_layers=2,
            dropout=0.3
        ).to(DEVICE)
        
        # 损失函数
        pos_weight = class_counts[0] / class_counts[1]
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
        contrastive_loss = SupConLoss(temperature=0.07)
        
        # 优化器和调度器
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        # 训练
        best_val_auc = 0
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        print("训练中...")
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, focal_loss, 
                                    contrastive_loss, DEVICE, use_contrastive=True)
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                results, _, _ = evaluate(model, val_loader, DEVICE)
                print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, ROC-AUC={results['roc_auc']:.4f}")
                
                if results['roc_auc'] > best_val_auc:
                    best_val_auc = results['roc_auc']
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience // 10:
                    print(f"  早停 at epoch {epoch+1}")
                    break
        
        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 最终评估
        results, probs, labels = evaluate(model, val_loader, DEVICE,
                                         thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
        
        print(f"\nFold {fold + 1} 结果:")
        print(f"  ROC-AUC: {results['roc_auc']:.4f}")
        print(f"  最佳F1阈值: {results['best_threshold']}")
        
        for thresh in [0.1, 0.2, 0.3]:
            if f'thresh_{thresh}' in results:
                r = results[f'thresh_{thresh}']
                print(f"  阈值{thresh}: Acc={r['accuracy']:.4f}, Recall={r['recall']:.4f}, F1={r['f1']:.4f}")
        
        all_results.append(results)
    
    return all_results


def main():
    """主函数"""
    print("=" * 70)
    print("NequIP增强型深度学习铁电分类器 v6 - 深度版本")
    print("目标: 使用深度神经网络自动学习高维特征表示")
    print("=" * 70)
    
    # 数据目录
    data_dir = '/home/ubuntu/ai_wh/wh-ai/new_data'
    report_dir = '/home/ubuntu/ai_wh/wh-ai/reports_nequip_v6'
    model_dir = '/home/ubuntu/ai_wh/wh-ai/model_nequip_v6'
    
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # 特征提取器
    extractor = AdvancedFeatureExtractor(
        n_radial_basis=32,
        n_spherical_harmonics=6,
        cutoff=8.0
    )
    
    # 加载数据
    X, y = load_data(data_dir, extractor)
    
    # 交叉验证
    results = cross_validate(
        X, y,
        n_folds=5,
        epochs=100,
        batch_size=64,
        lr=1e-3
    )
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("交叉验证总结")
    print("=" * 70)
    
    auc_scores = [r['roc_auc'] for r in results]
    print(f"ROC-AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    
    # 不同阈值的平均性能
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    print("\n不同阈值的平均性能:")
    print("-" * 70)
    print(f"{'阈值':>8} {'准确率':>10} {'精确率':>10} {'召回率':>10} {'F1':>10}")
    print("-" * 70)
    
    for thresh in thresholds:
        key = f'thresh_{thresh}'
        accs = [r[key]['accuracy'] for r in results if key in r]
        recs = [r[key]['recall'] for r in results if key in r]
        precs = [r[key]['precision'] for r in results if key in r]
        f1s = [r[key]['f1'] for r in results if key in r]
        
        if accs:
            print(f"{thresh:>8.2f} {np.mean(accs):>10.4f} {np.mean(precs):>10.4f} "
                  f"{np.mean(recs):>10.4f} {np.mean(f1s):>10.4f}")
    
    # 保存结果
    results_df = pd.DataFrame([
        {
            'fold': i + 1,
            'roc_auc': r['roc_auc'],
            'best_threshold': r['best_threshold'],
            'best_f1': r['best_f1']
        }
        for i, r in enumerate(results)
    ])
    results_df.to_csv(os.path.join(report_dir, 'cv_results_deep.csv'), index=False)
    
    print(f"\n结果已保存到: {report_dir}/cv_results_deep.csv")
    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
