#!/usr/bin/env python3
"""
复合决策模型 - Composite Decision Model for Ferroelectric Classification

策略架构:
=========
Layer 0: 专家规则层 (Expert Rule Layer)
    - 铁电材料必须是极性材料 (非极性空间群直接排除)
    - 68个极性空间群筛选

Layer 1: 高召回筛选层 (High-Recall Screening)
    - 宽松阈值，确保不遗漏潜在FE
    - 使用简单快速模型
    
Layer 2: 多模型集成层 (Multi-Model Ensemble)
    - Model A: Transformer-based (结构特征)
    - Model B: Attention Network (电子特征)  
    - Model C: Wide ResNet (综合特征)
    - 加权投票决策

Layer 3: 专家确认层 (Expert Confirmation)
    - 对边界样本使用物理化学规则验证
    - 基于已知铁电材料的统计特征

数据集: 仅使用极性材料进行训练 (大幅减少类别不平衡)
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

warnings.filterwarnings('ignore')

# 尝试导入SMOTE
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    SMOTE_AVAILABLE = True
    print("✓ SMOTE/ADASYN可用")
except ImportError:
    SMOTE_AVAILABLE = False
    print("✗ SMOTE不可用")

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 极性空间群定义 (专家知识)
# ============================================================
# 10个极性点群: C1, C2, Cs, C2v, C3, C3v, C4, C4v, C6, C6v
# 对应68个极性空间群

POLAR_SPACE_GROUPS = {
    # C1 (1)
    1,
    # C2 (3-5)
    3, 4, 5,
    # Cs (6-9)
    6, 7, 8, 9,
    # C2v (25-46)
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    # C3 (143-146)
    143, 144, 145, 146,
    # C3v (156-161)
    156, 157, 158, 159, 160, 161,
    # C4 (75-80)
    75, 76, 77, 78, 79, 80,
    # C4v (99-110)
    99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    # C6 (168-173)
    168, 169, 170, 171, 172, 173,
    # C6v (183-186)
    183, 184, 185, 186
}

# 已知铁电材料的典型空间群分布 (基于文献)
FE_COMMON_SPACE_GROUPS = {
    99,   # P4mm (BaTiO3 tetragonal)
    38,   # Amm2 
    160,  # R3m (BaTiO3 rhombohedral)
    161,  # R3c (LiNbO3, BiFeO3)
    31,   # Pmn21
    33,   # Pna21
    1,    # P1
    4,    # P21
    26,   # Pmc21
    29,   # Pca21
    36,   # Cmc21
    185,  # P63cm
    186,  # P63mc
}

print(f"极性空间群数量: {len(POLAR_SPACE_GROUPS)}")
print(f"常见铁电空间群: {len(FE_COMMON_SPACE_GROUPS)}")


# ============================================================
# 特征提取器 - 多种特征工程策略
# ============================================================
class AdvancedFeatureExtractor:
    """综合特征提取器"""
    
    # 元素属性表
    ELEMENT_PROPERTIES = {
        'H': {'electronegativity': 2.20, 'ionic_radius': 0.25, 'atomic_mass': 1.008, 'valence': 1},
        'Li': {'electronegativity': 0.98, 'ionic_radius': 0.76, 'atomic_mass': 6.94, 'valence': 1},
        'Be': {'electronegativity': 1.57, 'ionic_radius': 0.45, 'atomic_mass': 9.01, 'valence': 2},
        'B': {'electronegativity': 2.04, 'ionic_radius': 0.27, 'atomic_mass': 10.81, 'valence': 3},
        'C': {'electronegativity': 2.55, 'ionic_radius': 0.16, 'atomic_mass': 12.01, 'valence': 4},
        'N': {'electronegativity': 3.04, 'ionic_radius': 1.46, 'atomic_mass': 14.01, 'valence': 5},
        'O': {'electronegativity': 3.44, 'ionic_radius': 1.40, 'atomic_mass': 16.00, 'valence': 6},
        'F': {'electronegativity': 3.98, 'ionic_radius': 1.33, 'atomic_mass': 19.00, 'valence': 7},
        'Na': {'electronegativity': 0.93, 'ionic_radius': 1.02, 'atomic_mass': 22.99, 'valence': 1},
        'Mg': {'electronegativity': 1.31, 'ionic_radius': 0.72, 'atomic_mass': 24.31, 'valence': 2},
        'Al': {'electronegativity': 1.61, 'ionic_radius': 0.54, 'atomic_mass': 26.98, 'valence': 3},
        'Si': {'electronegativity': 1.90, 'ionic_radius': 0.40, 'atomic_mass': 28.09, 'valence': 4},
        'P': {'electronegativity': 2.19, 'ionic_radius': 0.38, 'atomic_mass': 30.97, 'valence': 5},
        'S': {'electronegativity': 2.58, 'ionic_radius': 1.84, 'atomic_mass': 32.07, 'valence': 6},
        'Cl': {'electronegativity': 3.16, 'ionic_radius': 1.81, 'atomic_mass': 35.45, 'valence': 7},
        'K': {'electronegativity': 0.82, 'ionic_radius': 1.38, 'atomic_mass': 39.10, 'valence': 1},
        'Ca': {'electronegativity': 1.00, 'ionic_radius': 1.00, 'atomic_mass': 40.08, 'valence': 2},
        'Sc': {'electronegativity': 1.36, 'ionic_radius': 0.75, 'atomic_mass': 44.96, 'valence': 3},
        'Ti': {'electronegativity': 1.54, 'ionic_radius': 0.61, 'atomic_mass': 47.87, 'valence': 4},
        'V': {'electronegativity': 1.63, 'ionic_radius': 0.54, 'atomic_mass': 50.94, 'valence': 5},
        'Cr': {'electronegativity': 1.66, 'ionic_radius': 0.52, 'atomic_mass': 52.00, 'valence': 6},
        'Mn': {'electronegativity': 1.55, 'ionic_radius': 0.53, 'atomic_mass': 54.94, 'valence': 7},
        'Fe': {'electronegativity': 1.83, 'ionic_radius': 0.55, 'atomic_mass': 55.85, 'valence': 3},
        'Co': {'electronegativity': 1.88, 'ionic_radius': 0.55, 'atomic_mass': 58.93, 'valence': 3},
        'Ni': {'electronegativity': 1.91, 'ionic_radius': 0.69, 'atomic_mass': 58.69, 'valence': 2},
        'Cu': {'electronegativity': 1.90, 'ionic_radius': 0.73, 'atomic_mass': 63.55, 'valence': 2},
        'Zn': {'electronegativity': 1.65, 'ionic_radius': 0.74, 'atomic_mass': 65.38, 'valence': 2},
        'Ga': {'electronegativity': 1.81, 'ionic_radius': 0.62, 'atomic_mass': 69.72, 'valence': 3},
        'Ge': {'electronegativity': 2.01, 'ionic_radius': 0.53, 'atomic_mass': 72.63, 'valence': 4},
        'As': {'electronegativity': 2.18, 'ionic_radius': 0.58, 'atomic_mass': 74.92, 'valence': 5},
        'Se': {'electronegativity': 2.55, 'ionic_radius': 1.98, 'atomic_mass': 78.97, 'valence': 6},
        'Br': {'electronegativity': 2.96, 'ionic_radius': 1.96, 'atomic_mass': 79.90, 'valence': 7},
        'Rb': {'electronegativity': 0.82, 'ionic_radius': 1.52, 'atomic_mass': 85.47, 'valence': 1},
        'Sr': {'electronegativity': 0.95, 'ionic_radius': 1.18, 'atomic_mass': 87.62, 'valence': 2},
        'Y': {'electronegativity': 1.22, 'ionic_radius': 0.90, 'atomic_mass': 88.91, 'valence': 3},
        'Zr': {'electronegativity': 1.33, 'ionic_radius': 0.72, 'atomic_mass': 91.22, 'valence': 4},
        'Nb': {'electronegativity': 1.60, 'ionic_radius': 0.64, 'atomic_mass': 92.91, 'valence': 5},
        'Mo': {'electronegativity': 2.16, 'ionic_radius': 0.59, 'atomic_mass': 95.95, 'valence': 6},
        'Ru': {'electronegativity': 2.20, 'ionic_radius': 0.68, 'atomic_mass': 101.1, 'valence': 4},
        'Rh': {'electronegativity': 2.28, 'ionic_radius': 0.67, 'atomic_mass': 102.9, 'valence': 3},
        'Pd': {'electronegativity': 2.20, 'ionic_radius': 0.86, 'atomic_mass': 106.4, 'valence': 2},
        'Ag': {'electronegativity': 1.93, 'ionic_radius': 1.15, 'atomic_mass': 107.9, 'valence': 1},
        'Cd': {'electronegativity': 1.69, 'ionic_radius': 0.95, 'atomic_mass': 112.4, 'valence': 2},
        'In': {'electronegativity': 1.78, 'ionic_radius': 0.80, 'atomic_mass': 114.8, 'valence': 3},
        'Sn': {'electronegativity': 1.96, 'ionic_radius': 0.69, 'atomic_mass': 118.7, 'valence': 4},
        'Sb': {'electronegativity': 2.05, 'ionic_radius': 0.76, 'atomic_mass': 121.8, 'valence': 5},
        'Te': {'electronegativity': 2.10, 'ionic_radius': 2.21, 'atomic_mass': 127.6, 'valence': 6},
        'I': {'electronegativity': 2.66, 'ionic_radius': 2.20, 'atomic_mass': 126.9, 'valence': 7},
        'Cs': {'electronegativity': 0.79, 'ionic_radius': 1.67, 'atomic_mass': 132.9, 'valence': 1},
        'Ba': {'electronegativity': 0.89, 'ionic_radius': 1.35, 'atomic_mass': 137.3, 'valence': 2},
        'La': {'electronegativity': 1.10, 'ionic_radius': 1.03, 'atomic_mass': 138.9, 'valence': 3},
        'Ce': {'electronegativity': 1.12, 'ionic_radius': 1.01, 'atomic_mass': 140.1, 'valence': 3},
        'Pr': {'electronegativity': 1.13, 'ionic_radius': 0.99, 'atomic_mass': 140.9, 'valence': 3},
        'Nd': {'electronegativity': 1.14, 'ionic_radius': 0.98, 'atomic_mass': 144.2, 'valence': 3},
        'Sm': {'electronegativity': 1.17, 'ionic_radius': 0.96, 'atomic_mass': 150.4, 'valence': 3},
        'Eu': {'electronegativity': 1.20, 'ionic_radius': 0.95, 'atomic_mass': 152.0, 'valence': 3},
        'Gd': {'electronegativity': 1.20, 'ionic_radius': 0.94, 'atomic_mass': 157.3, 'valence': 3},
        'Tb': {'electronegativity': 1.20, 'ionic_radius': 0.92, 'atomic_mass': 158.9, 'valence': 3},
        'Dy': {'electronegativity': 1.22, 'ionic_radius': 0.91, 'atomic_mass': 162.5, 'valence': 3},
        'Ho': {'electronegativity': 1.23, 'ionic_radius': 0.90, 'atomic_mass': 164.9, 'valence': 3},
        'Er': {'electronegativity': 1.24, 'ionic_radius': 0.89, 'atomic_mass': 167.3, 'valence': 3},
        'Tm': {'electronegativity': 1.25, 'ionic_radius': 0.88, 'atomic_mass': 168.9, 'valence': 3},
        'Yb': {'electronegativity': 1.10, 'ionic_radius': 0.87, 'atomic_mass': 173.0, 'valence': 3},
        'Lu': {'electronegativity': 1.27, 'ionic_radius': 0.86, 'atomic_mass': 175.0, 'valence': 3},
        'Hf': {'electronegativity': 1.30, 'ionic_radius': 0.71, 'atomic_mass': 178.5, 'valence': 4},
        'Ta': {'electronegativity': 1.50, 'ionic_radius': 0.64, 'atomic_mass': 180.9, 'valence': 5},
        'W': {'electronegativity': 2.36, 'ionic_radius': 0.60, 'atomic_mass': 183.8, 'valence': 6},
        'Re': {'electronegativity': 1.90, 'ionic_radius': 0.53, 'atomic_mass': 186.2, 'valence': 7},
        'Os': {'electronegativity': 2.20, 'ionic_radius': 0.63, 'atomic_mass': 190.2, 'valence': 4},
        'Ir': {'electronegativity': 2.20, 'ionic_radius': 0.68, 'atomic_mass': 192.2, 'valence': 4},
        'Pt': {'electronegativity': 2.28, 'ionic_radius': 0.80, 'atomic_mass': 195.1, 'valence': 4},
        'Au': {'electronegativity': 2.54, 'ionic_radius': 1.37, 'atomic_mass': 197.0, 'valence': 3},
        'Hg': {'electronegativity': 2.00, 'ionic_radius': 1.02, 'atomic_mass': 200.6, 'valence': 2},
        'Tl': {'electronegativity': 1.62, 'ionic_radius': 1.50, 'atomic_mass': 204.4, 'valence': 3},
        'Pb': {'electronegativity': 1.87, 'ionic_radius': 1.19, 'atomic_mass': 207.2, 'valence': 4},
        'Bi': {'electronegativity': 2.02, 'ionic_radius': 1.03, 'atomic_mass': 209.0, 'valence': 5},
    }
    
    @classmethod
    def extract_structure_features(cls, structure_dict: dict) -> np.ndarray:
        """提取结构相关特征 (用于Model A)"""
        features = []
        
        try:
            from pymatgen.core import Structure
            structure = Structure.from_dict(structure_dict)
            
            # 空间群信息
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(structure, symprec=0.1)
                spg_number = sga.get_space_group_number()
                crystal_system_map = {
                    'triclinic': 1, 'monoclinic': 2, 'orthorhombic': 3,
                    'tetragonal': 4, 'trigonal': 5, 'hexagonal': 6, 'cubic': 7
                }
                crystal_system = crystal_system_map.get(sga.get_crystal_system(), 0)
            except:
                spg_number = 1
                crystal_system = 1
            
            # 归一化空间群号
            features.append(spg_number / 230.0)
            features.append(crystal_system / 7.0)
            
            # 是否为极性空间群
            features.append(1.0 if spg_number in POLAR_SPACE_GROUPS else 0.0)
            
            # 是否为常见铁电空间群
            features.append(1.0 if spg_number in FE_COMMON_SPACE_GROUPS else 0.0)
            
            # 晶格参数
            lattice = structure.lattice
            features.append(lattice.a / 20.0)
            features.append(lattice.b / 20.0)
            features.append(lattice.c / 20.0)
            features.append(lattice.alpha / 180.0)
            features.append(lattice.beta / 180.0)
            features.append(lattice.gamma / 180.0)
            
            # 晶格各向异性
            abc = sorted([lattice.a, lattice.b, lattice.c])
            features.append((abc[2] - abc[0]) / (abc[2] + 1e-6))  # 各向异性度
            features.append(abc[1] / (abc[0] + 1e-6))  # b/a ratio
            features.append(abc[2] / (abc[1] + 1e-6))  # c/b ratio
            
            # 体积相关
            volume = lattice.volume
            n_atoms = len(structure)
            features.append(np.log1p(volume) / 10.0)
            features.append(volume / n_atoms / 50.0)
            features.append(n_atoms / 100.0)
            
            # 密度
            features.append(structure.density / 10.0)
            
            # 原子位置统计
            frac_coords = structure.frac_coords
            features.append(np.mean(frac_coords))
            features.append(np.std(frac_coords))
            features.append(np.max(frac_coords) - np.min(frac_coords))
            
            # 填充到20维
            while len(features) < 20:
                features.append(0.0)
                
        except Exception as e:
            features = [0.0] * 20
        
        return np.array(features[:20], dtype=np.float32)
    
    @classmethod
    def extract_electronic_features(cls, structure_dict: dict) -> np.ndarray:
        """提取电子/化学特征 (用于Model B)"""
        features = []
        
        try:
            from pymatgen.core import Structure
            structure = Structure.from_dict(structure_dict)
            
            # 收集元素属性
            electronegativities = []
            ionic_radii = []
            atomic_masses = []
            valences = []
            
            for site in structure:
                elem = str(site.specie.element)
                if elem in cls.ELEMENT_PROPERTIES:
                    props = cls.ELEMENT_PROPERTIES[elem]
                    electronegativities.append(props['electronegativity'])
                    ionic_radii.append(props['ionic_radius'])
                    atomic_masses.append(props['atomic_mass'])
                    valences.append(props['valence'])
            
            if not electronegativities:
                electronegativities = [2.0]
                ionic_radii = [1.0]
                atomic_masses = [50.0]
                valences = [3]
            
            # 电负性特征
            features.append(np.mean(electronegativities) / 4.0)
            features.append(np.std(electronegativities) / 2.0)
            features.append(np.max(electronegativities) / 4.0)
            features.append(np.min(electronegativities) / 4.0)
            features.append((np.max(electronegativities) - np.min(electronegativities)) / 4.0)
            
            # 离子半径特征
            features.append(np.mean(ionic_radii) / 2.0)
            features.append(np.std(ionic_radii))
            features.append(np.max(ionic_radii) / 2.5)
            features.append(np.min(ionic_radii) / 2.0)
            if len(ionic_radii) >= 2:
                features.append(max(ionic_radii) / (min(ionic_radii) + 0.01))
            else:
                features.append(1.0)
            
            # 原子质量特征
            features.append(np.mean(atomic_masses) / 200.0)
            features.append(np.std(atomic_masses) / 50.0)
            features.append(np.sum(atomic_masses) / 1000.0)
            
            # 价电子特征
            features.append(np.mean(valences) / 7.0)
            features.append(np.std(valences) / 3.0)
            features.append(np.sum(valences) / 50.0)
            
            # 元素多样性
            unique_elements = len(set(str(s.specie.element) for s in structure))
            features.append(unique_elements / 10.0)
            
            # 离子性估计 (电负性差)
            if len(electronegativities) >= 2:
                ionicity = np.max(electronegativities) - np.min(electronegativities)
            else:
                ionicity = 0.0
            features.append(ionicity / 3.0)
            
            # 极化率估计 (基于离子半径和电负性)
            polarizability = np.mean(ionic_radii) * (4.0 - np.mean(electronegativities))
            features.append(polarizability / 5.0)
            
            # 填充到20维
            while len(features) < 20:
                features.append(0.0)
                
        except Exception as e:
            features = [0.0] * 20
        
        return np.array(features[:20], dtype=np.float32)
    
    @classmethod
    def extract_comprehensive_features(cls, structure_dict: dict) -> np.ndarray:
        """提取综合特征 (用于Model C)"""
        struct_features = cls.extract_structure_features(structure_dict)
        elec_features = cls.extract_electronic_features(structure_dict)
        
        # 额外的交叉特征
        cross_features = []
        
        try:
            from pymatgen.core import Structure
            structure = Structure.from_dict(structure_dict)
            
            # 结构-化学交叉特征
            volume_per_atom = structure.lattice.volume / len(structure)
            mean_en = np.mean([cls.ELEMENT_PROPERTIES.get(str(s.specie.element), {}).get('electronegativity', 2.0) 
                              for s in structure])
            
            cross_features.append(volume_per_atom * mean_en / 100.0)
            cross_features.append(structure.density * mean_en / 30.0)
            
            # 对称性-化学交叉
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(structure, symprec=0.1)
                spg = sga.get_space_group_number()
                
                is_polar = 1.0 if spg in POLAR_SPACE_GROUPS else 0.0
                cross_features.append(is_polar * mean_en)
                cross_features.append(is_polar * structure.density / 10.0)
            except:
                cross_features.extend([0.0, 0.0])
            
            # 晶格形变特征
            lattice = structure.lattice
            abc = [lattice.a, lattice.b, lattice.c]
            distortion = np.std(abc) / np.mean(abc)
            cross_features.append(distortion)
            
            # 角度偏离90度
            angles = [lattice.alpha, lattice.beta, lattice.gamma]
            angle_deviation = np.mean([abs(a - 90) for a in angles]) / 90.0
            cross_features.append(angle_deviation)
            
            # 填充到24维
            while len(cross_features) < 24:
                cross_features.append(0.0)
                
        except:
            cross_features = [0.0] * 24
        
        # 合并: 20 + 20 + 24 = 64维
        return np.concatenate([struct_features, elec_features, np.array(cross_features[:24], dtype=np.float32)])
    
    @classmethod
    def get_space_group(cls, structure_dict: dict) -> int:
        """获取空间群号"""
        try:
            from pymatgen.core import Structure
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            structure = Structure.from_dict(structure_dict)
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            return sga.get_space_group_number()
        except:
            return 1


# ============================================================
# 模型定义
# ============================================================

class ModelA_Transformer(nn.Module):
    """Model A: Transformer-based 结构特征模型"""
    
    def __init__(self, input_dim=20, hidden_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
            dropout=0.2, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # x: (batch, input_dim) -> (batch, 1, hidden)
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x).squeeze(-1)


class ModelB_Attention(nn.Module):
    """Model B: Self-Attention 电子特征模型"""
    
    def __init__(self, input_dim=20, hidden_dim=128):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.2, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        x = self.feature_extractor(x).unsqueeze(1)  # (batch, 1, hidden)
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.squeeze(1)
        return self.classifier(x).squeeze(-1)


class ModelC_WideResNet(nn.Module):
    """Model C: Wide ResNet 综合特征模型"""
    
    def __init__(self, input_dim=64, hidden_dim=256):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(hidden_dim) for _ in range(3)
        ])
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def _make_res_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = F.gelu(x + block(x))  # Residual connection
        return self.classifier(x).squeeze(-1)


class HighRecallScreener(nn.Module):
    """高召回率筛选模型 (Layer 1)"""
    
    def __init__(self, input_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# 损失函数
# ============================================================

class AsymmetricLoss(nn.Module):
    """非对称损失 - 严惩假阴性"""
    def __init__(self, gamma_neg=4.0, gamma_pos=0.5):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        pos_loss = -targets * ((1 - probs) ** self.gamma_pos) * torch.log(probs + 1e-8)
        neg_loss = -(1 - targets) * (probs ** self.gamma_neg) * torch.log(1 - probs + 1e-8)
        
        return (pos_loss + neg_loss).mean()


class FocalLoss(nn.Module):
    """Focal Loss - 关注难分类样本"""
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * ((1 - p_t) ** self.gamma)
        
        return (focal_weight * ce_loss).mean()


# ============================================================
# 复合决策系统
# ============================================================

class CompositeDecisionSystem:
    """复合决策系统"""
    
    def __init__(self, device='cuda'):
        self.device = device
        
        # 数据缓存
        self.polar_data = None
        self.fe_data = None
        self.non_fe_data = None
        
        # 模型
        self.screener = None  # Layer 1
        self.model_a = None   # Structure Transformer
        self.model_b = None   # Electronic Attention
        self.model_c = None   # Comprehensive ResNet
        
        # Scalers
        self.scaler_struct = StandardScaler()
        self.scaler_elec = StandardScaler()
        self.scaler_comp = StandardScaler()
        
        # 阈值
        self.threshold_screener = 0.05
        self.threshold_a = 0.5
        self.threshold_b = 0.5
        self.threshold_c = 0.5
        self.ensemble_threshold = 0.5
        
        # 模型权重
        self.model_weights = {'A': 1.0, 'B': 1.0, 'C': 1.0}
    
    def load_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """加载数据并提取特征，返回极性材料子集"""
        data_path = Path(data_dir)
        
        samples = []
        labels = []
        space_groups = []
        
        # 正样本文件
        pos_files = ['dataset_original_ferroelectric.jsonl', 'dataset_known_FE_rest.jsonl']
        # 负样本文件
        neg_files = ['dataset_nonFE.jsonl', 'dataset_nonFE_cleaned.jsonl', 'dataset_nonFE_expanded.jsonl']
        
        seen_formulas = set()
        
        print("\n加载正样本 (铁电材料)...")
        for fname in pos_files:
            fpath = data_path / fname
            if fpath.exists():
                with open(fpath) as f:
                    lines = f.readlines()
                for line in tqdm(lines, desc=f"  {fname}"):
                    try:
                        data = json.loads(line)
                        structure = data.get('structure', data)
                        
                        from pymatgen.core import Structure
                        struct = Structure.from_dict(structure)
                        formula = struct.composition.reduced_formula
                        
                        if formula not in seen_formulas:
                            seen_formulas.add(formula)
                            spg = AdvancedFeatureExtractor.get_space_group(structure)
                            samples.append(structure)
                            labels.append(1)
                            space_groups.append(spg)
                    except:
                        continue
                print(f"  {fname}: {sum(1 for l in labels if l==1)} 个正样本")
        
        print("\n加载负样本 (非铁电材料)...")
        for fname in neg_files:
            fpath = data_path / fname
            if fpath.exists():
                with open(fpath) as f:
                    lines = f.readlines()
                for line in tqdm(lines, desc=f"  {fname}"):
                    try:
                        data = json.loads(line)
                        structure = data.get('structure', data)
                        
                        from pymatgen.core import Structure
                        struct = Structure.from_dict(structure)
                        formula = struct.composition.reduced_formula
                        
                        if formula not in seen_formulas:
                            seen_formulas.add(formula)
                            spg = AdvancedFeatureExtractor.get_space_group(structure)
                            samples.append(structure)
                            labels.append(0)
                            space_groups.append(spg)
                    except:
                        continue
                neg_count = sum(1 for l in labels if l==0)
                print(f"  {fname}: {neg_count} 个负样本")
        
        # 打印总体统计
        total = len(labels)
        pos_count = sum(labels)
        neg_count = total - pos_count
        
        print(f"\n总样本数: {total}")
        print(f"FE样本: {pos_count} ({100*pos_count/total:.2f}%)")
        print(f"Non-FE样本: {neg_count} ({100*neg_count/total:.2f}%)")
        
        # 统计极性材料
        polar_count = sum(1 for spg in space_groups if spg in POLAR_SPACE_GROUPS)
        polar_fe = sum(1 for spg, lbl in zip(space_groups, labels) if spg in POLAR_SPACE_GROUPS and lbl == 1)
        polar_nonfe = polar_count - polar_fe
        
        print(f"\n极性材料统计 (共{len(POLAR_SPACE_GROUPS)}个极性空间群):")
        print(f"  极性材料总数: {polar_count} ({100*polar_count/total:.2f}%)")
        print(f"  极性FE: {polar_fe} ({100*polar_fe/pos_count:.2f}% of FE)")
        print(f"  极性Non-FE: {polar_nonfe}")
        
        # 保存原始数据
        self.all_samples = samples
        self.all_labels = labels
        self.all_space_groups = space_groups
        
        return samples, labels, space_groups
    
    def filter_polar_materials(self, samples, labels, space_groups):
        """Layer 0: 专家规则 - 筛选极性材料"""
        print("\n" + "="*60)
        print("Layer 0: 专家规则筛选 - 仅保留极性材料")
        print("="*60)
        
        polar_samples = []
        polar_labels = []
        polar_spgs = []
        
        for sample, label, spg in zip(samples, labels, space_groups):
            if spg in POLAR_SPACE_GROUPS:
                polar_samples.append(sample)
                polar_labels.append(label)
                polar_spgs.append(spg)
        
        pos_count = sum(polar_labels)
        neg_count = len(polar_labels) - pos_count
        
        print(f"极性材料子集: {len(polar_labels)} 样本")
        print(f"  FE: {pos_count} ({100*pos_count/len(polar_labels):.2f}%)")
        print(f"  Non-FE: {neg_count} ({100*neg_count/len(polar_labels):.2f}%)")
        print(f"  新类别比例: 1:{neg_count/pos_count:.1f}")
        
        return polar_samples, polar_labels, polar_spgs
    
    def extract_all_features(self, samples):
        """提取所有类型的特征"""
        print("\n提取特征...")
        
        features_struct = []
        features_elec = []
        features_comp = []
        
        for sample in tqdm(samples, desc="提取特征"):
            features_struct.append(AdvancedFeatureExtractor.extract_structure_features(sample))
            features_elec.append(AdvancedFeatureExtractor.extract_electronic_features(sample))
            features_comp.append(AdvancedFeatureExtractor.extract_comprehensive_features(sample))
        
        return (np.array(features_struct), 
                np.array(features_elec), 
                np.array(features_comp))
    
    def train_model(self, model, train_loader, val_loader, criterion, 
                   epochs=100, lr=0.001, model_name="Model"):
        """训练单个模型"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        best_auc = 0
        best_state = None
        patience = 20
        no_improve = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            scheduler.step()
            
            # Validation
            model.eval()
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    outputs = torch.sigmoid(model(X_batch))
                    val_preds.extend(outputs.cpu().numpy())
                    val_labels.extend(y_batch.numpy())
            
            val_auc = roc_auc_score(val_labels, val_preds)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve >= patience:
                break
        
        model.load_state_dict(best_state)
        return best_auc
    
    def train_composite_system(self, data_dir: str, n_folds: int = 5):
        """训练完整的复合决策系统"""
        print("\n" + "="*60)
        print("复合决策模型训练")
        print("="*60)
        
        # 1. 加载数据
        samples, labels, space_groups = self.load_data(data_dir)
        
        # 2. Layer 0: 筛选极性材料
        polar_samples, polar_labels, polar_spgs = self.filter_polar_materials(
            samples, labels, space_groups
        )
        
        # 3. 提取特征
        features_struct, features_elec, features_comp = self.extract_all_features(polar_samples)
        labels_array = np.array(polar_labels)
        
        print(f"\n特征维度:")
        print(f"  结构特征: {features_struct.shape}")
        print(f"  电子特征: {features_elec.shape}")
        print(f"  综合特征: {features_comp.shape}")
        
        # 4. 交叉验证训练
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        results = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(features_comp, labels_array)):
            print(f"\n{'='*60}")
            print(f"Fold {fold+1}/{n_folds}")
            print(f"{'='*60}")
            
            # 分割数据
            X_struct_train = features_struct[train_idx]
            X_struct_val = features_struct[val_idx]
            X_elec_train = features_elec[train_idx]
            X_elec_val = features_elec[val_idx]
            X_comp_train = features_comp[train_idx]
            X_comp_val = features_comp[val_idx]
            y_train = labels_array[train_idx]
            y_val = labels_array[val_idx]
            
            print(f"训练集: {len(y_train)} (正样本: {sum(y_train)})")
            print(f"验证集: {len(y_val)} (正样本: {sum(y_val)})")
            
            # 标准化
            scaler_struct = StandardScaler()
            scaler_elec = StandardScaler()
            scaler_comp = StandardScaler()
            
            X_struct_train_scaled = scaler_struct.fit_transform(X_struct_train)
            X_struct_val_scaled = scaler_struct.transform(X_struct_val)
            X_elec_train_scaled = scaler_elec.fit_transform(X_elec_train)
            X_elec_val_scaled = scaler_elec.transform(X_elec_val)
            X_comp_train_scaled = scaler_comp.fit_transform(X_comp_train)
            X_comp_val_scaled = scaler_comp.transform(X_comp_val)
            
            # SMOTE过采样
            if SMOTE_AVAILABLE and sum(y_train) < len(y_train) // 2:
                smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y_train)-1))
                X_struct_train_resampled, y_train_resampled = smote.fit_resample(X_struct_train_scaled, y_train)
                X_elec_train_resampled, _ = smote.fit_resample(X_elec_train_scaled, y_train)
                X_comp_train_resampled, _ = smote.fit_resample(X_comp_train_scaled, y_train)
                print(f"SMOTE: {sum(y_train)} -> {sum(y_train_resampled)} 正样本")
            else:
                X_struct_train_resampled = X_struct_train_scaled
                X_elec_train_resampled = X_elec_train_scaled
                X_comp_train_resampled = X_comp_train_scaled
                y_train_resampled = y_train
            
            # 创建数据加载器
            def create_loader(X, y, batch_size=64, shuffle=True):
                dataset = TensorDataset(
                    torch.FloatTensor(X),
                    torch.FloatTensor(y)
                )
                return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            
            train_loader_struct = create_loader(X_struct_train_resampled, y_train_resampled)
            val_loader_struct = create_loader(X_struct_val_scaled, y_val, shuffle=False)
            
            train_loader_elec = create_loader(X_elec_train_resampled, y_train_resampled)
            val_loader_elec = create_loader(X_elec_val_scaled, y_val, shuffle=False)
            
            train_loader_comp = create_loader(X_comp_train_resampled, y_train_resampled)
            val_loader_comp = create_loader(X_comp_val_scaled, y_val, shuffle=False)
            
            # ============ 训练 Layer 1: 高召回筛选器 ============
            print("\n--- Layer 1: 高召回筛选器训练 ---")
            screener = HighRecallScreener(input_dim=64).to(self.device)
            screener_auc = self.train_model(
                screener, train_loader_comp, val_loader_comp,
                AsymmetricLoss(gamma_neg=4, gamma_pos=0.5),
                epochs=80, model_name="Screener"
            )
            print(f"  Screener AUC: {screener_auc:.4f}")
            
            # ============ 训练 Layer 2: 多模型集成 ============
            print("\n--- Layer 2: 多模型集成训练 ---")
            
            # Model A: Transformer (结构特征)
            print("  训练 Model A (Transformer - 结构特征)...")
            model_a = ModelA_Transformer(input_dim=20).to(self.device)
            auc_a = self.train_model(
                model_a, train_loader_struct, val_loader_struct,
                FocalLoss(gamma=2, alpha=0.75),
                epochs=80, model_name="Model_A"
            )
            print(f"    Model A AUC: {auc_a:.4f}")
            
            # Model B: Attention (电子特征)
            print("  训练 Model B (Attention - 电子特征)...")
            model_b = ModelB_Attention(input_dim=20).to(self.device)
            auc_b = self.train_model(
                model_b, train_loader_elec, val_loader_elec,
                FocalLoss(gamma=2, alpha=0.75),
                epochs=80, model_name="Model_B"
            )
            print(f"    Model B AUC: {auc_b:.4f}")
            
            # Model C: WideResNet (综合特征)
            print("  训练 Model C (WideResNet - 综合特征)...")
            model_c = ModelC_WideResNet(input_dim=64).to(self.device)
            auc_c = self.train_model(
                model_c, train_loader_comp, val_loader_comp,
                FocalLoss(gamma=2, alpha=0.75),
                epochs=80, model_name="Model_C"
            )
            print(f"    Model C AUC: {auc_c:.4f}")
            
            # ============ 评估复合决策 ============
            print("\n--- 复合决策评估 ---")
            
            # 计算模型权重 (基于AUC)
            total_auc = auc_a + auc_b + auc_c
            weight_a = auc_a / total_auc
            weight_b = auc_b / total_auc
            weight_c = auc_c / total_auc
            print(f"  模型权重: A={weight_a:.3f}, B={weight_b:.3f}, C={weight_c:.3f}")
            
            # 获取验证集预测
            screener.eval()
            model_a.eval()
            model_b.eval()
            model_c.eval()
            
            with torch.no_grad():
                X_struct_t = torch.FloatTensor(X_struct_val_scaled).to(self.device)
                X_elec_t = torch.FloatTensor(X_elec_val_scaled).to(self.device)
                X_comp_t = torch.FloatTensor(X_comp_val_scaled).to(self.device)
                
                prob_screener = torch.sigmoid(screener(X_comp_t)).cpu().numpy()
                prob_a = torch.sigmoid(model_a(X_struct_t)).cpu().numpy()
                prob_b = torch.sigmoid(model_b(X_elec_t)).cpu().numpy()
                prob_c = torch.sigmoid(model_c(X_comp_t)).cpu().numpy()
            
            # 寻找最优阈值组合
            best_result = None
            best_f1 = 0
            
            for thresh_screen in [0.02, 0.05, 0.1]:
                for thresh_ensemble in [0.3, 0.4, 0.5, 0.6]:
                    # Layer 1: 筛选
                    screen_pass = prob_screener >= thresh_screen
                    
                    # Layer 2: 集成投票 (仅对通过筛选的样本)
                    ensemble_prob = weight_a * prob_a + weight_b * prob_b + weight_c * prob_c
                    
                    # 最终预测
                    final_pred = np.zeros(len(y_val))
                    final_pred[screen_pass] = (ensemble_prob[screen_pass] >= thresh_ensemble).astype(float)
                    
                    # 计算指标
                    if sum(final_pred) > 0:
                        acc = accuracy_score(y_val, final_pred)
                        recall = recall_score(y_val, final_pred, zero_division=0)
                        precision = precision_score(y_val, final_pred, zero_division=0)
                        f1 = f1_score(y_val, final_pred, zero_division=0)
                        
                        # 使用集成概率计算AUC
                        final_prob = np.zeros(len(y_val))
                        final_prob[screen_pass] = ensemble_prob[screen_pass]
                        auc = roc_auc_score(y_val, final_prob)
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_result = {
                                'thresh_screen': thresh_screen,
                                'thresh_ensemble': thresh_ensemble,
                                'accuracy': acc,
                                'recall': recall,
                                'precision': precision,
                                'f1': f1,
                                'auc': auc
                            }
            
            if best_result:
                print(f"\n  最优阈值: Screen={best_result['thresh_screen']}, Ensemble={best_result['thresh_ensemble']}")
                print(f"  Accuracy: {best_result['accuracy']:.4f}")
                print(f"  Recall: {best_result['recall']:.4f}")
                print(f"  Precision: {best_result['precision']:.4f}")
                print(f"  F1: {best_result['f1']:.4f}")
                print(f"  AUC: {best_result['auc']:.4f}")
                
                best_result['fold'] = fold + 1
                best_result['auc_a'] = auc_a
                best_result['auc_b'] = auc_b
                best_result['auc_c'] = auc_c
                results.append(best_result)
            
            # 保存最后一折的模型
            if fold == n_folds - 1:
                self.screener = screener
                self.model_a = model_a
                self.model_b = model_b
                self.model_c = model_c
                self.scaler_struct = scaler_struct
                self.scaler_elec = scaler_elec
                self.scaler_comp = scaler_comp
                self.model_weights = {'A': weight_a, 'B': weight_b, 'C': weight_c}
        
        # ============ 汇总结果 ============
        print("\n" + "="*60)
        print("复合决策系统 - 交叉验证汇总")
        print("="*60)
        
        if results:
            df_results = pd.DataFrame(results)
            
            print("\n各Fold结果:")
            print(df_results.to_string(index=False))
            
            print("\n平均结果:")
            for metric in ['accuracy', 'recall', 'precision', 'f1', 'auc']:
                mean_val = df_results[metric].mean()
                std_val = df_results[metric].std()
                print(f"  {metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
            
            print("\n各模型平均AUC:")
            print(f"  Model A (Transformer): {df_results['auc_a'].mean():.4f}")
            print(f"  Model B (Attention): {df_results['auc_b'].mean():.4f}")
            print(f"  Model C (WideResNet): {df_results['auc_c'].mean():.4f}")
        
        return results
    
    def save_models(self, save_dir: str):
        """保存所有模型"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        torch.save({
            'screener': self.screener.state_dict(),
            'model_a': self.model_a.state_dict(),
            'model_b': self.model_b.state_dict(),
            'model_c': self.model_c.state_dict(),
            'model_weights': self.model_weights,
        }, save_path / 'composite_models.pt')
        
        # 保存scalers
        joblib.dump({
            'scaler_struct': self.scaler_struct,
            'scaler_elec': self.scaler_elec,
            'scaler_comp': self.scaler_comp,
        }, save_path / 'scalers.pkl')
        
        print(f"\n模型已保存到: {save_path}")


# ============================================================
# 主程序
# ============================================================
def main():
    print("="*60)
    print("复合决策模型 - Composite Decision System")
    print("="*60)
    print("\n架构设计:")
    print("  Layer 0: 专家规则 (极性空间群筛选)")
    print("  Layer 1: 高召回筛选器 (AsymmetricLoss)")
    print("  Layer 2: 多模型集成")
    print("    - Model A: Transformer (结构特征)")
    print("    - Model B: Attention (电子特征)")
    print("    - Model C: WideResNet (综合特征)")
    print("  Layer 3: 加权投票决策")
    
    # 路径配置
    data_dir = Path(__file__).parent.parent / 'new_data'
    model_dir = Path(__file__).parent.parent / 'model_composite'
    report_dir = Path(__file__).parent.parent / 'reports_composite'
    
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    
    # 创建系统并训练
    system = CompositeDecisionSystem(device=device)
    results = system.train_composite_system(str(data_dir), n_folds=5)
    
    # 保存模型
    system.save_models(str(model_dir))
    
    # 保存结果
    if results:
        df = pd.DataFrame(results)
        df.to_csv(report_dir / 'composite_cv_results.csv', index=False)
        print(f"\n结果已保存到: {report_dir}")
    
    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)


if __name__ == "__main__":
    main()
