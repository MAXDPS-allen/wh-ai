#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
集成学习铁电材料分类器
========================
目标: 通过集成多种特征工程和模型架构，实现 99%+ Accuracy 和 Recall

策略:
1. 多种特征工程: 基础特征、增强特征、极化特征、元素嵌入
2. 多种模型架构: Transformer, DeepMLP, ResNet, XGBoost
3. 集成方法: 加权软投票 + 阈值优化
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from pymatgen.core import Structure

warnings.filterwarnings('ignore')

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================================
# 多种特征工程方法
# ============================================================================

class BaseFeatureExtractor:
    """基础特征提取器"""
    
    # 元素属性字典
    ELEMENT_PROPS = {
        'H': {'Z': 1, 'electronegativity': 2.20, 'radius': 0.53, 'mass': 1.008, 'group': 1, 'period': 1},
        'Li': {'Z': 3, 'electronegativity': 0.98, 'radius': 1.67, 'mass': 6.94, 'group': 1, 'period': 2},
        'Be': {'Z': 4, 'electronegativity': 1.57, 'radius': 1.12, 'mass': 9.01, 'group': 2, 'period': 2},
        'B': {'Z': 5, 'electronegativity': 2.04, 'radius': 0.87, 'mass': 10.81, 'group': 13, 'period': 2},
        'C': {'Z': 6, 'electronegativity': 2.55, 'radius': 0.77, 'mass': 12.01, 'group': 14, 'period': 2},
        'N': {'Z': 7, 'electronegativity': 3.04, 'radius': 0.75, 'mass': 14.01, 'group': 15, 'period': 2},
        'O': {'Z': 8, 'electronegativity': 3.44, 'radius': 0.73, 'mass': 16.00, 'group': 16, 'period': 2},
        'F': {'Z': 9, 'electronegativity': 3.98, 'radius': 0.71, 'mass': 19.00, 'group': 17, 'period': 2},
        'Na': {'Z': 11, 'electronegativity': 0.93, 'radius': 1.90, 'mass': 22.99, 'group': 1, 'period': 3},
        'Mg': {'Z': 12, 'electronegativity': 1.31, 'radius': 1.45, 'mass': 24.31, 'group': 2, 'period': 3},
        'Al': {'Z': 13, 'electronegativity': 1.61, 'radius': 1.18, 'mass': 26.98, 'group': 13, 'period': 3},
        'Si': {'Z': 14, 'electronegativity': 1.90, 'radius': 1.11, 'mass': 28.09, 'group': 14, 'period': 3},
        'P': {'Z': 15, 'electronegativity': 2.19, 'radius': 1.06, 'mass': 30.97, 'group': 15, 'period': 3},
        'S': {'Z': 16, 'electronegativity': 2.58, 'radius': 1.02, 'mass': 32.07, 'group': 16, 'period': 3},
        'Cl': {'Z': 17, 'electronegativity': 3.16, 'radius': 0.99, 'mass': 35.45, 'group': 17, 'period': 3},
        'K': {'Z': 19, 'electronegativity': 0.82, 'radius': 2.43, 'mass': 39.10, 'group': 1, 'period': 4},
        'Ca': {'Z': 20, 'electronegativity': 1.00, 'radius': 1.94, 'mass': 40.08, 'group': 2, 'period': 4},
        'Sc': {'Z': 21, 'electronegativity': 1.36, 'radius': 1.84, 'mass': 44.96, 'group': 3, 'period': 4},
        'Ti': {'Z': 22, 'electronegativity': 1.54, 'radius': 1.76, 'mass': 47.87, 'group': 4, 'period': 4},
        'V': {'Z': 23, 'electronegativity': 1.63, 'radius': 1.71, 'mass': 50.94, 'group': 5, 'period': 4},
        'Cr': {'Z': 24, 'electronegativity': 1.66, 'radius': 1.66, 'mass': 52.00, 'group': 6, 'period': 4},
        'Mn': {'Z': 25, 'electronegativity': 1.55, 'radius': 1.61, 'mass': 54.94, 'group': 7, 'period': 4},
        'Fe': {'Z': 26, 'electronegativity': 1.83, 'radius': 1.56, 'mass': 55.85, 'group': 8, 'period': 4},
        'Co': {'Z': 27, 'electronegativity': 1.88, 'radius': 1.52, 'mass': 58.93, 'group': 9, 'period': 4},
        'Ni': {'Z': 28, 'electronegativity': 1.91, 'radius': 1.49, 'mass': 58.69, 'group': 10, 'period': 4},
        'Cu': {'Z': 29, 'electronegativity': 1.90, 'radius': 1.45, 'mass': 63.55, 'group': 11, 'period': 4},
        'Zn': {'Z': 30, 'electronegativity': 1.65, 'radius': 1.42, 'mass': 65.38, 'group': 12, 'period': 4},
        'Ga': {'Z': 31, 'electronegativity': 1.81, 'radius': 1.36, 'mass': 69.72, 'group': 13, 'period': 4},
        'Ge': {'Z': 32, 'electronegativity': 2.01, 'radius': 1.25, 'mass': 72.63, 'group': 14, 'period': 4},
        'As': {'Z': 33, 'electronegativity': 2.18, 'radius': 1.14, 'mass': 74.92, 'group': 15, 'period': 4},
        'Se': {'Z': 34, 'electronegativity': 2.55, 'radius': 1.03, 'mass': 78.97, 'group': 16, 'period': 4},
        'Br': {'Z': 35, 'electronegativity': 2.96, 'radius': 0.94, 'mass': 79.90, 'group': 17, 'period': 4},
        'Rb': {'Z': 37, 'electronegativity': 0.82, 'radius': 2.65, 'mass': 85.47, 'group': 1, 'period': 5},
        'Sr': {'Z': 38, 'electronegativity': 0.95, 'radius': 2.19, 'mass': 87.62, 'group': 2, 'period': 5},
        'Y': {'Z': 39, 'electronegativity': 1.22, 'radius': 2.12, 'mass': 88.91, 'group': 3, 'period': 5},
        'Zr': {'Z': 40, 'electronegativity': 1.33, 'radius': 2.06, 'mass': 91.22, 'group': 4, 'period': 5},
        'Nb': {'Z': 41, 'electronegativity': 1.60, 'radius': 1.98, 'mass': 92.91, 'group': 5, 'period': 5},
        'Mo': {'Z': 42, 'electronegativity': 2.16, 'radius': 1.90, 'mass': 95.95, 'group': 6, 'period': 5},
        'Ru': {'Z': 44, 'electronegativity': 2.20, 'radius': 1.78, 'mass': 101.07, 'group': 8, 'period': 5},
        'Rh': {'Z': 45, 'electronegativity': 2.28, 'radius': 1.73, 'mass': 102.91, 'group': 9, 'period': 5},
        'Pd': {'Z': 46, 'electronegativity': 2.20, 'radius': 1.69, 'mass': 106.42, 'group': 10, 'period': 5},
        'Ag': {'Z': 47, 'electronegativity': 1.93, 'radius': 1.65, 'mass': 107.87, 'group': 11, 'period': 5},
        'Cd': {'Z': 48, 'electronegativity': 1.69, 'radius': 1.61, 'mass': 112.41, 'group': 12, 'period': 5},
        'In': {'Z': 49, 'electronegativity': 1.78, 'radius': 1.56, 'mass': 114.82, 'group': 13, 'period': 5},
        'Sn': {'Z': 50, 'electronegativity': 1.96, 'radius': 1.45, 'mass': 118.71, 'group': 14, 'period': 5},
        'Sb': {'Z': 51, 'electronegativity': 2.05, 'radius': 1.33, 'mass': 121.76, 'group': 15, 'period': 5},
        'Te': {'Z': 52, 'electronegativity': 2.10, 'radius': 1.23, 'mass': 127.60, 'group': 16, 'period': 5},
        'I': {'Z': 53, 'electronegativity': 2.66, 'radius': 1.15, 'mass': 126.90, 'group': 17, 'period': 5},
        'Cs': {'Z': 55, 'electronegativity': 0.79, 'radius': 2.98, 'mass': 132.91, 'group': 1, 'period': 6},
        'Ba': {'Z': 56, 'electronegativity': 0.89, 'radius': 2.53, 'mass': 137.33, 'group': 2, 'period': 6},
        'La': {'Z': 57, 'electronegativity': 1.10, 'radius': 2.50, 'mass': 138.91, 'group': 3, 'period': 6},
        'Ce': {'Z': 58, 'electronegativity': 1.12, 'radius': 2.48, 'mass': 140.12, 'group': 3, 'period': 6},
        'Pr': {'Z': 59, 'electronegativity': 1.13, 'radius': 2.47, 'mass': 140.91, 'group': 3, 'period': 6},
        'Nd': {'Z': 60, 'electronegativity': 1.14, 'radius': 2.45, 'mass': 144.24, 'group': 3, 'period': 6},
        'Sm': {'Z': 62, 'electronegativity': 1.17, 'radius': 2.42, 'mass': 150.36, 'group': 3, 'period': 6},
        'Eu': {'Z': 63, 'electronegativity': 1.20, 'radius': 2.40, 'mass': 151.96, 'group': 3, 'period': 6},
        'Gd': {'Z': 64, 'electronegativity': 1.20, 'radius': 2.38, 'mass': 157.25, 'group': 3, 'period': 6},
        'Tb': {'Z': 65, 'electronegativity': 1.20, 'radius': 2.37, 'mass': 158.93, 'group': 3, 'period': 6},
        'Dy': {'Z': 66, 'electronegativity': 1.22, 'radius': 2.35, 'mass': 162.50, 'group': 3, 'period': 6},
        'Ho': {'Z': 67, 'electronegativity': 1.23, 'radius': 2.33, 'mass': 164.93, 'group': 3, 'period': 6},
        'Er': {'Z': 68, 'electronegativity': 1.24, 'radius': 2.32, 'mass': 167.26, 'group': 3, 'period': 6},
        'Tm': {'Z': 69, 'electronegativity': 1.25, 'radius': 2.30, 'mass': 168.93, 'group': 3, 'period': 6},
        'Yb': {'Z': 70, 'electronegativity': 1.10, 'radius': 2.28, 'mass': 173.05, 'group': 3, 'period': 6},
        'Lu': {'Z': 71, 'electronegativity': 1.27, 'radius': 2.27, 'mass': 174.97, 'group': 3, 'period': 6},
        'Hf': {'Z': 72, 'electronegativity': 1.30, 'radius': 2.08, 'mass': 178.49, 'group': 4, 'period': 6},
        'Ta': {'Z': 73, 'electronegativity': 1.50, 'radius': 2.00, 'mass': 180.95, 'group': 5, 'period': 6},
        'W': {'Z': 74, 'electronegativity': 2.36, 'radius': 1.93, 'mass': 183.84, 'group': 6, 'period': 6},
        'Re': {'Z': 75, 'electronegativity': 1.90, 'radius': 1.88, 'mass': 186.21, 'group': 7, 'period': 6},
        'Os': {'Z': 76, 'electronegativity': 2.20, 'radius': 1.85, 'mass': 190.23, 'group': 8, 'period': 6},
        'Ir': {'Z': 77, 'electronegativity': 2.20, 'radius': 1.80, 'mass': 192.22, 'group': 9, 'period': 6},
        'Pt': {'Z': 78, 'electronegativity': 2.28, 'radius': 1.77, 'mass': 195.08, 'group': 10, 'period': 6},
        'Au': {'Z': 79, 'electronegativity': 2.54, 'radius': 1.74, 'mass': 196.97, 'group': 11, 'period': 6},
        'Hg': {'Z': 80, 'electronegativity': 2.00, 'radius': 1.71, 'mass': 200.59, 'group': 12, 'period': 6},
        'Tl': {'Z': 81, 'electronegativity': 1.62, 'radius': 1.56, 'mass': 204.38, 'group': 13, 'period': 6},
        'Pb': {'Z': 82, 'electronegativity': 2.33, 'radius': 1.54, 'mass': 207.2, 'group': 14, 'period': 6},
        'Bi': {'Z': 83, 'electronegativity': 2.02, 'radius': 1.43, 'mass': 208.98, 'group': 15, 'period': 6},
    }
    
    # 极性空间群 (非中心对称)
    POLAR_SPACE_GROUPS = {1, 3, 4, 5, 6, 7, 8, 9, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                         75, 76, 77, 78, 79, 80, 99, 100, 101, 102, 103, 104, 105, 106,
                         107, 108, 109, 110, 143, 144, 145, 146, 156, 157, 158, 159, 160, 161,
                         168, 169, 170, 171, 172, 173, 183, 184, 185, 186, 187, 188, 189, 190}
    
    def get_element_props(self, symbol: str) -> dict:
        default = {'Z': 0, 'electronegativity': 2.0, 'radius': 1.5, 'mass': 50.0, 'group': 0, 'period': 0}
        return self.ELEMENT_PROPS.get(symbol, default)


class FeatureExtractor1(BaseFeatureExtractor):
    """特征工程方法1: 基础晶体学特征 (64维)"""
    
    def __init__(self):
        self.feature_dim = 64
        self.name = "Basic"
    
    def extract(self, structure: Structure) -> np.ndarray:
        features = []
        
        # 1. 晶格参数 (6)
        lattice = structure.lattice
        features.extend([lattice.a, lattice.b, lattice.c,
                        lattice.alpha, lattice.beta, lattice.gamma])
        
        # 2. 体积和密度 (3)
        features.append(lattice.volume)
        features.append(lattice.volume / len(structure))
        features.append(structure.density)
        
        # 3. 空间群特征 (4)
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            spg_num = sga.get_space_group_number()
        except:
            spg_num = 1
        
        features.append(spg_num)
        features.append(1 if spg_num in self.POLAR_SPACE_GROUPS else 0)
        features.append(spg_num % 100)
        features.append(spg_num // 100)
        
        # 4. 元素统计 (20)
        elements = [site.specie.symbol for site in structure]
        elem_set = list(set(elements))
        
        Z_list = [self.get_element_props(e)['Z'] for e in elements]
        en_list = [self.get_element_props(e)['electronegativity'] for e in elements]
        r_list = [self.get_element_props(e)['radius'] for e in elements]
        m_list = [self.get_element_props(e)['mass'] for e in elements]
        
        for prop_list in [Z_list, en_list, r_list, m_list]:
            features.extend([np.mean(prop_list), np.std(prop_list),
                           np.min(prop_list), np.max(prop_list), np.max(prop_list) - np.min(prop_list)])
        
        # 5. 配位环境 (15)
        try:
            from pymatgen.analysis.local_env import VoronoiNN
            nn = VoronoiNN(cutoff=10.0)
            coord_nums = []
            for i in range(min(len(structure), 20)):
                try:
                    cn = len(nn.get_nn_info(structure, i))
                    coord_nums.append(cn)
                except:
                    pass
            if coord_nums:
                features.extend([np.mean(coord_nums), np.std(coord_nums), np.min(coord_nums),
                               np.max(coord_nums), np.median(coord_nums)])
            else:
                features.extend([6.0, 1.0, 4.0, 8.0, 6.0])
        except:
            features.extend([6.0, 1.0, 4.0, 8.0, 6.0])
        
        # 键长统计 (10)
        try:
            distances = structure.distance_matrix[np.triu_indices(len(structure), k=1)]
            distances = distances[distances < 4.0]
            if len(distances) > 0:
                features.extend([np.mean(distances), np.std(distances), np.min(distances),
                               np.max(distances), np.median(distances)])
                features.extend([np.percentile(distances, 25), np.percentile(distances, 75),
                               np.percentile(distances, 10), np.percentile(distances, 90),
                               len(distances) / len(structure)])
            else:
                features.extend([2.5, 0.5, 1.5, 4.0, 2.5, 2.0, 3.0, 1.8, 3.5, 5.0])
        except:
            features.extend([2.5, 0.5, 1.5, 4.0, 2.5, 2.0, 3.0, 1.8, 3.5, 5.0])
        
        # 6. 结构复杂度 (6)
        features.append(len(structure))
        features.append(len(elem_set))
        features.append(len(structure) / len(elem_set) if len(elem_set) > 0 else 0)
        features.append(lattice.a / lattice.b if lattice.b > 0 else 1)
        features.append(lattice.b / lattice.c if lattice.c > 0 else 1)
        features.append(lattice.a / lattice.c if lattice.c > 0 else 1)
        
        # 填充到64维
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return np.array(features[:self.feature_dim], dtype=np.float32)


class FeatureExtractor2(BaseFeatureExtractor):
    """特征工程方法2: 增强特征 + 极化相关 (96维)"""
    
    def __init__(self):
        self.feature_dim = 96
        self.name = "Enhanced"
    
    def extract(self, structure: Structure) -> np.ndarray:
        features = []
        
        # 1. 基础晶格 (9)
        lattice = structure.lattice
        features.extend([lattice.a, lattice.b, lattice.c,
                        lattice.alpha, lattice.beta, lattice.gamma])
        features.append(lattice.volume)
        features.append(lattice.volume / len(structure))
        features.append(structure.density)
        
        # 2. 晶格各向异性 (6)
        abc = sorted([lattice.a, lattice.b, lattice.c])
        features.append(abc[2] / abc[0] if abc[0] > 0 else 1)  # 最大/最小
        features.append(abc[1] / abc[0] if abc[0] > 0 else 1)
        features.append(abc[2] / abc[1] if abc[1] > 0 else 1)
        angles = [lattice.alpha, lattice.beta, lattice.gamma]
        features.append(np.std(angles))
        features.append(max(abs(a - 90) for a in angles))
        features.append(sum(1 for a in angles if abs(a - 90) < 5))
        
        # 3. 空间群深度分析 (8)
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            spg_num = sga.get_space_group_number()
            crystal_system = sga.get_crystal_system()
            crystal_map = {'triclinic': 1, 'monoclinic': 2, 'orthorhombic': 3,
                          'tetragonal': 4, 'trigonal': 5, 'hexagonal': 6, 'cubic': 7}
            crystal_code = crystal_map.get(crystal_system, 0)
        except:
            spg_num, crystal_code = 1, 0
        
        features.append(spg_num)
        features.append(crystal_code)
        features.append(1 if spg_num in self.POLAR_SPACE_GROUPS else 0)
        features.append(spg_num % 10)
        features.append((spg_num // 10) % 10)
        features.append(spg_num // 100)
        # 铁电常见空间群标记
        fe_common_spg = {99, 100, 143, 156, 160, 161, 183, 186}
        features.append(1 if spg_num in fe_common_spg else 0)
        features.append(len([s for s in self.POLAR_SPACE_GROUPS if abs(s - spg_num) <= 10]))
        
        # 4. 元素组成深度分析 (24)
        elements = [site.specie.symbol for site in structure]
        elem_counts = {}
        for e in elements:
            elem_counts[e] = elem_counts.get(e, 0) + 1
        
        Z_list = [self.get_element_props(e)['Z'] for e in elements]
        en_list = [self.get_element_props(e)['electronegativity'] for e in elements]
        r_list = [self.get_element_props(e)['radius'] for e in elements]
        m_list = [self.get_element_props(e)['mass'] for e in elements]
        g_list = [self.get_element_props(e)['group'] for e in elements]
        p_list = [self.get_element_props(e)['period'] for e in elements]
        
        for prop_list in [Z_list, en_list, r_list, m_list]:
            features.extend([np.mean(prop_list), np.std(prop_list),
                           np.min(prop_list), np.max(prop_list)])
        
        # 周期族特征
        features.extend([np.mean(g_list), np.std(g_list), np.mean(p_list), np.std(p_list)])
        
        # 电负性差异（极化驱动力）
        en_unique = list(set(en_list))
        if len(en_unique) > 1:
            features.append(max(en_unique) - min(en_unique))
            features.append(np.std(en_unique))
        else:
            features.extend([0, 0])
        
        # 过渡金属/镧系/锕系标记
        tm_elements = {'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                      'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Hf', 'Ta', 'W'}
        ln_elements = {'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}
        features.append(sum(1 for e in set(elements) if e in tm_elements))
        features.append(sum(1 for e in set(elements) if e in ln_elements))
        
        # 5. 位置不对称性 (质心偏移，极化指标) (9)
        frac_coords = structure.frac_coords
        centroid = np.mean(frac_coords, axis=0)
        displacement = centroid - 0.5
        features.extend(displacement.tolist())
        features.append(np.linalg.norm(displacement))
        
        # 各原子到质心的距离统计
        dist_to_centroid = np.linalg.norm(frac_coords - centroid, axis=1)
        features.extend([np.mean(dist_to_centroid), np.std(dist_to_centroid),
                        np.min(dist_to_centroid), np.max(dist_to_centroid),
                        np.median(dist_to_centroid)])
        
        # 6. 键长和配位 (16)
        try:
            from pymatgen.analysis.local_env import VoronoiNN
            nn = VoronoiNN(cutoff=10.0)
            coord_nums = []
            for i in range(min(len(structure), 30)):
                try:
                    cn = len(nn.get_nn_info(structure, i))
                    coord_nums.append(cn)
                except:
                    pass
            if coord_nums:
                features.extend([np.mean(coord_nums), np.std(coord_nums),
                               np.min(coord_nums), np.max(coord_nums),
                               np.median(coord_nums), np.percentile(coord_nums, 25),
                               np.percentile(coord_nums, 75)])
            else:
                features.extend([6.0, 1.0, 4.0, 8.0, 6.0, 5.0, 7.0])
        except:
            features.extend([6.0, 1.0, 4.0, 8.0, 6.0, 5.0, 7.0])
        
        try:
            distances = structure.distance_matrix[np.triu_indices(len(structure), k=1)]
            distances = distances[distances < 5.0]
            if len(distances) > 0:
                features.extend([np.mean(distances), np.std(distances),
                               np.min(distances), np.max(distances),
                               np.percentile(distances, 25), np.percentile(distances, 50),
                               np.percentile(distances, 75), np.percentile(distances, 90),
                               len(distances) / len(structure)])
            else:
                features.extend([2.5, 0.5, 1.5, 4.0, 2.0, 2.5, 3.0, 3.5, 5.0])
        except:
            features.extend([2.5, 0.5, 1.5, 4.0, 2.0, 2.5, 3.0, 3.5, 5.0])
        
        # 7. 结构复杂度扩展 (8)
        features.append(len(structure))
        features.append(len(set(elements)))
        features.append(len(structure) / len(set(elements)) if len(set(elements)) > 0 else 0)
        features.append(max(elem_counts.values()) / len(structure))
        features.append(min(elem_counts.values()) / len(structure))
        features.append(len([c for c in elem_counts.values() if c == 1]))  # 单个原子种类数
        features.append(np.log1p(len(structure)))
        features.append(np.log1p(lattice.volume))
        
        # 填充到96维
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return np.array(features[:self.feature_dim], dtype=np.float32)


class FeatureExtractor3(BaseFeatureExtractor):
    """特征工程方法3: 元素嵌入特征 (128维)"""
    
    # 元素嵌入向量（预定义的低维表示）
    ELEMENT_EMBEDDING = {
        'H': [1, 0, 0, 0, 2.20, 0.53], 'Li': [0, 1, 0, 0, 0.98, 1.67],
        'Be': [0, 1, 0, 0, 1.57, 1.12], 'B': [0, 0, 1, 0, 2.04, 0.87],
        'C': [0, 0, 1, 0, 2.55, 0.77], 'N': [0, 0, 1, 0, 3.04, 0.75],
        'O': [0, 0, 1, 0, 3.44, 0.73], 'F': [0, 0, 1, 0, 3.98, 0.71],
        'Na': [0, 1, 0, 0, 0.93, 1.90], 'Mg': [0, 1, 0, 0, 1.31, 1.45],
        'Al': [0, 0, 1, 0, 1.61, 1.18], 'Si': [0, 0, 1, 0, 1.90, 1.11],
        'P': [0, 0, 1, 0, 2.19, 1.06], 'S': [0, 0, 1, 0, 2.58, 1.02],
        'Cl': [0, 0, 1, 0, 3.16, 0.99], 'K': [0, 1, 0, 0, 0.82, 2.43],
        'Ca': [0, 1, 0, 0, 1.00, 1.94], 'Ti': [0, 0, 0, 1, 1.54, 1.76],
        'V': [0, 0, 0, 1, 1.63, 1.71], 'Cr': [0, 0, 0, 1, 1.66, 1.66],
        'Mn': [0, 0, 0, 1, 1.55, 1.61], 'Fe': [0, 0, 0, 1, 1.83, 1.56],
        'Co': [0, 0, 0, 1, 1.88, 1.52], 'Ni': [0, 0, 0, 1, 1.91, 1.49],
        'Cu': [0, 0, 0, 1, 1.90, 1.45], 'Zn': [0, 0, 0, 1, 1.65, 1.42],
        'Ga': [0, 0, 1, 0, 1.81, 1.36], 'Ge': [0, 0, 1, 0, 2.01, 1.25],
        'As': [0, 0, 1, 0, 2.18, 1.14], 'Se': [0, 0, 1, 0, 2.55, 1.03],
        'Br': [0, 0, 1, 0, 2.96, 0.94], 'Rb': [0, 1, 0, 0, 0.82, 2.65],
        'Sr': [0, 1, 0, 0, 0.95, 2.19], 'Y': [0, 0, 0, 1, 1.22, 2.12],
        'Zr': [0, 0, 0, 1, 1.33, 2.06], 'Nb': [0, 0, 0, 1, 1.60, 1.98],
        'Mo': [0, 0, 0, 1, 2.16, 1.90], 'Ag': [0, 0, 0, 1, 1.93, 1.65],
        'Cd': [0, 0, 0, 1, 1.69, 1.61], 'In': [0, 0, 1, 0, 1.78, 1.56],
        'Sn': [0, 0, 1, 0, 1.96, 1.45], 'Sb': [0, 0, 1, 0, 2.05, 1.33],
        'Te': [0, 0, 1, 0, 2.10, 1.23], 'I': [0, 0, 1, 0, 2.66, 1.15],
        'Cs': [0, 1, 0, 0, 0.79, 2.98], 'Ba': [0, 1, 0, 0, 0.89, 2.53],
        'La': [0, 0, 0, 1, 1.10, 2.50], 'Ce': [0, 0, 0, 1, 1.12, 2.48],
        'Pr': [0, 0, 0, 1, 1.13, 2.47], 'Nd': [0, 0, 0, 1, 1.14, 2.45],
        'Sm': [0, 0, 0, 1, 1.17, 2.42], 'Eu': [0, 0, 0, 1, 1.20, 2.40],
        'Gd': [0, 0, 0, 1, 1.20, 2.38], 'Tb': [0, 0, 0, 1, 1.20, 2.37],
        'Dy': [0, 0, 0, 1, 1.22, 2.35], 'Ho': [0, 0, 0, 1, 1.23, 2.33],
        'Er': [0, 0, 0, 1, 1.24, 2.32], 'Hf': [0, 0, 0, 1, 1.30, 2.08],
        'Ta': [0, 0, 0, 1, 1.50, 2.00], 'W': [0, 0, 0, 1, 2.36, 1.93],
        'Pt': [0, 0, 0, 1, 2.28, 1.77], 'Au': [0, 0, 0, 1, 2.54, 1.74],
        'Pb': [0, 0, 1, 0, 2.33, 1.54], 'Bi': [0, 0, 1, 0, 2.02, 1.43],
    }
    
    def __init__(self):
        self.feature_dim = 128
        self.name = "Embedding"
    
    def get_embedding(self, symbol: str) -> list:
        return self.ELEMENT_EMBEDDING.get(symbol, [0, 0, 0, 0, 2.0, 1.5])
    
    def extract(self, structure: Structure) -> np.ndarray:
        features = []
        
        # 1. 基础特征 (20)
        lattice = structure.lattice
        features.extend([lattice.a, lattice.b, lattice.c,
                        lattice.alpha, lattice.beta, lattice.gamma,
                        lattice.volume, lattice.volume / len(structure),
                        structure.density, len(structure)])
        
        # 空间群
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            spg_num = sga.get_space_group_number()
        except:
            spg_num = 1
        
        features.append(spg_num)
        features.append(1 if spg_num in self.POLAR_SPACE_GROUPS else 0)
        
        # 晶格比例
        abc = sorted([lattice.a, lattice.b, lattice.c])
        features.extend([abc[2]/abc[0], abc[1]/abc[0], abc[2]/abc[1]])
        angles = [lattice.alpha, lattice.beta, lattice.gamma]
        features.extend([np.std(angles), max(abs(a-90) for a in angles),
                        min(abs(a-90) for a in angles)])
        
        # 2. 元素嵌入聚合 (48)
        elements = [site.specie.symbol for site in structure]
        embeddings = [self.get_embedding(e) for e in elements]
        embeddings = np.array(embeddings)
        
        # 各维度统计
        for i in range(6):
            col = embeddings[:, i]
            features.extend([np.mean(col), np.std(col), np.min(col), np.max(col),
                           np.median(col), np.percentile(col, 25),
                           np.percentile(col, 75), np.sum(col)])
        
        # 3. 元素组合特征 (20)
        elem_set = list(set(elements))
        elem_counts = {e: elements.count(e) for e in elem_set}
        
        features.append(len(elem_set))
        features.append(max(elem_counts.values()) / len(structure))
        features.append(min(elem_counts.values()) / len(structure))
        
        # 成分熵
        probs = [c / len(structure) for c in elem_counts.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        features.append(entropy)
        
        # 按类型分组
        type_counts = [0, 0, 0, 0]  # s, p, d, f (用embedding的前4维表示)
        for e in elem_set:
            emb = self.get_embedding(e)
            for i in range(4):
                if emb[i] == 1:
                    type_counts[i] += elem_counts[e]
        features.extend(type_counts)
        features.extend([t / len(structure) for t in type_counts])
        
        # 电负性差
        en_list = [embeddings[i, 4] for i in range(len(elements))]
        features.extend([np.max(en_list) - np.min(en_list), np.std(en_list)])
        
        # 半径差
        r_list = [embeddings[i, 5] for i in range(len(elements))]
        features.extend([np.max(r_list) - np.min(r_list), np.std(r_list)])
        
        # 4. 结构特征 (20)
        frac_coords = structure.frac_coords
        centroid = np.mean(frac_coords, axis=0)
        displacement = centroid - 0.5
        features.extend(displacement.tolist())
        features.append(np.linalg.norm(displacement))
        
        # 坐标分布
        for dim in range(3):
            coords = frac_coords[:, dim]
            features.extend([np.mean(coords), np.std(coords)])
        
        # 距离统计
        try:
            distances = structure.distance_matrix[np.triu_indices(len(structure), k=1)]
            distances = distances[(distances > 0.5) & (distances < 5.0)]
            if len(distances) > 0:
                features.extend([np.mean(distances), np.std(distances),
                               np.min(distances), np.max(distances),
                               np.median(distances)])
            else:
                features.extend([2.5, 0.5, 1.5, 4.0, 2.5])
        except:
            features.extend([2.5, 0.5, 1.5, 4.0, 2.5])
        
        # 5. 配位特征 (20)
        try:
            from pymatgen.analysis.local_env import VoronoiNN
            nn = VoronoiNN(cutoff=10.0)
            coord_nums = []
            for i in range(min(len(structure), 30)):
                try:
                    cn = len(nn.get_nn_info(structure, i))
                    coord_nums.append(cn)
                except:
                    pass
            if coord_nums:
                features.extend([np.mean(coord_nums), np.std(coord_nums),
                               np.min(coord_nums), np.max(coord_nums),
                               np.median(coord_nums), len(set(coord_nums)),
                               np.percentile(coord_nums, 25),
                               np.percentile(coord_nums, 75)])
                # 配位数分布
                cn_hist = [coord_nums.count(i) for i in range(1, 13)]
                features.extend([c / len(coord_nums) for c in cn_hist])
            else:
                features.extend([6.0, 1.0, 4.0, 8.0, 6.0, 3.0, 5.0, 7.0])
                features.extend([0.0] * 12)
        except:
            features.extend([6.0, 1.0, 4.0, 8.0, 6.0, 3.0, 5.0, 7.0])
            features.extend([0.0] * 12)
        
        # 填充到128维
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        return np.array(features[:self.feature_dim], dtype=np.float32)


# ============================================================================
# 多种模型架构
# ============================================================================

class TransformerClassifier(nn.Module):
    """Transformer架构分类器"""
    
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.name = "Transformer"
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)


class DeepMLPClassifier(nn.Module):
    """深度MLP分类器"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout=0.3):
        super().__init__()
        self.name = "DeepMLP"
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)


class ResNetClassifier(nn.Module):
    """ResNet风格分类器"""
    
    def __init__(self, input_dim, hidden_dim=256, num_blocks=4, dropout=0.2):
        super().__init__()
        self.name = "ResNet"
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(self._make_block(hidden_dim, dropout))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def _make_block(self, dim, dropout):
        return nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.BatchNorm1d(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = F.gelu(x + block(x))
        return self.classifier(x)


class AttentionClassifier(nn.Module):
    """注意力机制分类器"""
    
    def __init__(self, input_dim, hidden_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.name = "Attention"
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        x = x.squeeze(1)
        return self.classifier(x)


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none',
                                                   pos_weight=torch.tensor([self.pos_weight]).to(inputs.device))
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()


# ============================================================================
# 数据加载
# ============================================================================

def load_structure_from_dict(d: dict) -> Optional[Structure]:
    """从字典加载结构"""
    try:
        if 'structure' in d:
            return Structure.from_dict(d['structure'])
        elif 'lattice' in d and 'sites' in d:
            return Structure.from_dict(d)
        else:
            for key in ['structure', 'input_structure', 'output_structure']:
                if key in d and isinstance(d[key], dict):
                    try:
                        return Structure.from_dict(d[key])
                    except:
                        continue
    except:
        pass
    return None


def load_data(data_dir: str) -> Tuple[List[Structure], List[int]]:
    """加载数据"""
    print("\n" + "=" * 70)
    print("加载数据")
    print("=" * 70)
    
    data_dir = Path(data_dir)
    structures = []
    labels = []
    seen_formulas = set()
    
    # 正样本
    print("\n处理正样本 (铁电材料)...")
    pos_files = [
        'dataset_original_ferroelectric.jsonl',
        'dataset_known_FE_rest.jsonl'
    ]
    
    for filename in pos_files:
        filepath = data_dir / filename
        if filepath.exists():
            count = 0
            with open(filepath, 'r') as f:
                for line in tqdm(f, desc=f"  {filename}"):
                    try:
                        d = json.loads(line.strip())
                        struct = load_structure_from_dict(d)
                        if struct is not None:
                            formula = struct.composition.reduced_formula
                            if formula not in seen_formulas:
                                seen_formulas.add(formula)
                                structures.append(struct)
                                labels.append(1)
                                count += 1
                    except:
                        continue
            print(f"  {filename}: {count} 个唯一样本")
    
    n_pos = sum(labels)
    
    # 负样本
    print("\n处理负样本 (非铁电材料)...")
    neg_files = [
        'dataset_nonFE.jsonl',
        'dataset_nonFE_cleaned.jsonl',
        'dataset_nonFE_expanded.jsonl'
    ]
    
    for filename in neg_files:
        filepath = data_dir / filename
        if filepath.exists():
            count = 0
            with open(filepath, 'r') as f:
                for line in tqdm(f, desc=f"  {filename}"):
                    try:
                        d = json.loads(line.strip())
                        struct = load_structure_from_dict(d)
                        if struct is not None:
                            formula = struct.composition.reduced_formula
                            if formula not in seen_formulas:
                                seen_formulas.add(formula)
                                structures.append(struct)
                                labels.append(0)
                                count += 1
                    except:
                        continue
            print(f"  {filename}: {count} 个唯一样本")
    
    n_neg = len(labels) - n_pos
    print(f"\n数据集统计:")
    print(f"  正样本 (FE): {n_pos}")
    print(f"  负样本 (non-FE): {n_neg}")
    print(f"  类别比例: 1:{n_neg/n_pos:.1f}")
    
    return structures, labels


# ============================================================================
# 集成学习器
# ============================================================================

class EnsembleClassifier:
    """集成分类器"""
    
    def __init__(self, feature_extractors, model_classes, device):
        self.feature_extractors = feature_extractors
        self.model_classes = model_classes
        self.device = device
        self.models = {}  # {(extractor_name, model_name): model}
        self.scalers = {}  # {extractor_name: scaler}
        self.weights = {}  # 模型权重（基于验证集AUC）
    
    def extract_features(self, structures: List[Structure]) -> Dict[str, np.ndarray]:
        """使用所有特征提取器提取特征"""
        features = {}
        for extractor in self.feature_extractors:
            print(f"  提取 {extractor.name} 特征...")
            feat_list = []
            for struct in tqdm(structures, desc=f"    {extractor.name}"):
                try:
                    feat = extractor.extract(struct)
                    feat_list.append(feat)
                except Exception as e:
                    feat_list.append(np.zeros(extractor.feature_dim, dtype=np.float32))
            features[extractor.name] = np.array(feat_list)
        return features
    
    def train_single_model(self, X_train, y_train, X_val, y_val, 
                          model_class, input_dim, model_name,
                          epochs=50, lr=1e-3, batch_size=64, pos_weight=2.0):
        """训练单个模型"""
        # 创建数据集
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1)
        )
        val_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_val), torch.FloatTensor(y_val).unsqueeze(1)
        )
        
        # 采样器
        class_weights = np.ones(len(y_train))
        class_weights[y_train == 1] = 3.0  # 正样本权重更高
        sampler = WeightedRandomSampler(class_weights, len(class_weights), replacement=True)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 创建模型
        model = model_class(input_dim).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = FocalLoss(alpha=0.25, gamma=2.5, pos_weight=pos_weight)
        
        best_auc = 0
        best_state = None
        patience = 10
        no_improve = 0
        
        for epoch in range(epochs):
            # 训练
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            
            # 验证
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    outputs = torch.sigmoid(model(X_batch)).cpu().numpy()
                    all_preds.extend(outputs.flatten())
                    all_labels.extend(y_batch.numpy().flatten())
            
            auc = roc_auc_score(all_labels, all_preds)
            
            if auc > best_auc:
                best_auc = auc
                best_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                break
        
        model.load_state_dict(best_state)
        return model, best_auc
    
    def train(self, features: Dict[str, np.ndarray], labels: np.ndarray,
             train_idx: np.ndarray, val_idx: np.ndarray):
        """训练所有模型"""
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        
        for extractor in self.feature_extractors:
            ext_name = extractor.name
            X = features[ext_name]
            
            # 标准化
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            self.scalers[ext_name] = scaler
            
            # 处理NaN
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            
            input_dim = X_train.shape[1]
            
            for model_class in self.model_classes:
                model_name = model_class(input_dim).name
                key = (ext_name, model_name)
                
                print(f"    训练 {ext_name} + {model_name}...")
                model, auc = self.train_single_model(
                    X_train, y_train, X_val, y_val,
                    model_class, input_dim, model_name
                )
                
                self.models[key] = model
                self.weights[key] = auc
                print(f"      AUC: {auc:.4f}")
        
        # 归一化权重
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight
    
    def predict(self, features: Dict[str, np.ndarray], idx: np.ndarray = None) -> np.ndarray:
        """集成预测"""
        weighted_preds = None
        
        for (ext_name, model_name), model in self.models.items():
            X = features[ext_name]
            if idx is not None:
                X = X[idx]
            
            # 标准化
            X = self.scalers[ext_name].transform(X)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 预测
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                preds = torch.sigmoid(model(X_tensor)).cpu().numpy().flatten()
            
            weight = self.weights[(ext_name, model_name)]
            
            if weighted_preds is None:
                weighted_preds = preds * weight
            else:
                weighted_preds += preds * weight
        
        return weighted_preds


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("集成学习铁电材料分类器")
    print("目标: 通过多模型集成实现 99%+ Accuracy 和 Recall")
    print("=" * 70)
    
    # 加载数据
    data_dir = "/home/ubuntu/ai_wh/wh-ai/new_data"
    structures, labels = load_data(data_dir)
    labels = np.array(labels)
    
    # 特征提取器
    feature_extractors = [
        FeatureExtractor1(),  # 64维基础特征
        FeatureExtractor2(),  # 96维增强特征
        FeatureExtractor3(),  # 128维嵌入特征
    ]
    
    # 模型架构
    model_classes = [
        lambda dim: TransformerClassifier(dim, hidden_dim=256, num_heads=8, num_layers=4),
        lambda dim: DeepMLPClassifier(dim, hidden_dims=[512, 256, 128, 64]),
        lambda dim: ResNetClassifier(dim, hidden_dim=256, num_blocks=4),
        lambda dim: AttentionClassifier(dim, hidden_dim=256, num_heads=4),
    ]
    
    # 提取所有特征
    print("\n" + "=" * 70)
    print("特征提取")
    print("=" * 70)
    features = {}
    for extractor in feature_extractors:
        print(f"\n提取 {extractor.name} 特征 ({extractor.feature_dim}维)...")
        feat_list = []
        for struct in tqdm(structures, desc=f"  {extractor.name}"):
            try:
                feat = extractor.extract(struct)
                feat_list.append(feat)
            except:
                feat_list.append(np.zeros(extractor.feature_dim, dtype=np.float32))
        features[extractor.name] = np.array(feat_list)
    
    # 5折交叉验证
    print("\n" + "=" * 70)
    print("5折交叉验证训练")
    print("=" * 70)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = []
    fold_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n{'='*70}")
        print(f"Fold {fold+1}/5")
        print(f"{'='*70}")
        print(f"训练集: {len(train_idx)} (正:{sum(labels[train_idx])}, 负:{len(train_idx)-sum(labels[train_idx])})")
        print(f"验证集: {len(val_idx)}")
        
        # 创建集成器
        ensemble = EnsembleClassifier(feature_extractors, model_classes, device)
        
        # 训练
        print("\n训练模型...")
        ensemble.train(features, labels, train_idx, val_idx)
        
        # 预测
        print("\n集成预测...")
        y_val = labels[val_idx]
        predictions = ensemble.predict(features, val_idx)
        
        # 计算AUC
        auc = roc_auc_score(y_val, predictions)
        print(f"\n集成模型 ROC-AUC: {auc:.4f}")
        
        # 寻找最优阈值
        print("\n阈值分析:")
        best_result = None
        for threshold in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
            y_pred = (predictions >= threshold).astype(int)
            acc = accuracy_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred)
            
            print(f"  阈值 {threshold:.2f}: Acc={acc:.4f}, Recall={recall:.4f}, Prec={precision:.4f}, F1={f1:.4f}")
            
            if recall >= 0.99 and (best_result is None or acc > best_result['accuracy']):
                best_result = {
                    'threshold': threshold,
                    'accuracy': acc,
                    'recall': recall,
                    'precision': precision,
                    'f1': f1
                }
        
        if best_result:
            print(f"\n  ★ 最佳 (Recall>=99%): 阈值={best_result['threshold']:.2f}, "
                  f"Acc={best_result['accuracy']:.4f}, Recall={best_result['recall']:.4f}")
        
        fold_predictions.append({
            'fold': fold + 1,
            'auc': auc,
            'predictions': predictions,
            'labels': y_val,
            'val_idx': val_idx,
            'best_result': best_result
        })
        
        all_results.append({
            'fold': fold + 1,
            'auc': auc,
            'model_weights': dict(ensemble.weights)
        })
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("交叉验证总结")
    print("=" * 70)
    
    aucs = [r['auc'] for r in all_results]
    print(f"\nROC-AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    # 聚合所有折的预测
    all_preds = np.concatenate([fp['predictions'] for fp in fold_predictions])
    all_labels = np.concatenate([fp['labels'] for fp in fold_predictions])
    
    print("\n全局阈值优化:")
    for threshold in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        y_pred = (all_preds >= threshold).astype(int)
        acc = accuracy_score(all_labels, y_pred)
        recall = recall_score(all_labels, y_pred)
        precision = precision_score(all_labels, y_pred, zero_division=0)
        f1 = f1_score(all_labels, y_pred)
        
        marker = "★" if recall >= 0.99 else " "
        print(f"  {marker} 阈值 {threshold:.2f}: Acc={acc:.4f}, Recall={recall:.4f}, Prec={precision:.4f}, F1={f1:.4f}")
    
    # 保存结果
    output_dir = Path("/home/ubuntu/ai_wh/wh-ai/reports_nequip_v6")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "ensemble_cv_results.csv", index=False)
    
    print(f"\n结果已保存到: {output_dir / 'ensemble_cv_results.csv'}")
    print("\n完成!")


if __name__ == "__main__":
    main()
