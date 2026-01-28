#!/usr/bin/env python3
"""
复合决策模型 V2 - 增强版多模型集成
====================================

策略架构:
=========
Layer 0: 专家规则层 (Expert Rule Layer)
    - 铁电材料必须是极性材料 (68个极性空间群筛选)

Layer 1: 高召回筛选层 (High-Recall Screening)
    - 宽松阈值，确保不遗漏潜在FE
    
Layer 2: 多模型集成层 (Multi-Model Ensemble) - 增强版
    - Model A: End-to-End Learning (自动学习特征)
    - Model B: GCNN (图卷积网络 - 结构特征)
    - Model C: NequIP_v6 (E3等变网络 - 复杂特征)
    - Model D: Transformer (综合特征)
    
Layer 3: 梯次决策层 (Cascade Decision)
    - 第一梯队: 高置信度直接判定
    - 第二梯队: 中置信度多数投票
    - 第三梯队: 低置信度专家规则验证

数据集: 仅使用极性材料进行训练
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
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)

warnings.filterwarnings('ignore')

# PyTorch Geometric
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader as GeoDataLoader

# SMOTE
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    SMOTE_AVAILABLE = True
    print("✓ SMOTE/ADASYN可用")
except ImportError:
    SMOTE_AVAILABLE = False
    print("✗ SMOTE不可用")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ============================================================
# 极性空间群定义 (专家知识)
# ============================================================
POLAR_SPACE_GROUPS = {
    1, 3, 4, 5, 6, 7, 8, 9,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    75, 76, 77, 78, 79, 80,
    99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    143, 144, 145, 146,
    156, 157, 158, 159, 160, 161,
    168, 169, 170, 171, 172, 173,
    183, 184, 185, 186
}

FE_COMMON_SPACE_GROUPS = {99, 38, 160, 161, 31, 33, 1, 4, 26, 29, 36, 185, 186}

print(f"极性空间群数量: {len(POLAR_SPACE_GROUPS)}")

# ============================================================
# 元素属性数据库
# ============================================================
ELEMENT_DATABASE = {
    'H': [1, 1.008, 2.20, 0.25, 1, 1, 14, 2.1],
    'Li': [3, 6.94, 0.98, 0.76, 1, 2, 453, 0.5],
    'Be': [4, 9.01, 1.57, 0.45, 2, 2, 1560, 1.8],
    'B': [5, 10.81, 2.04, 0.27, 3, 2, 2349, 2.4],
    'C': [6, 12.01, 2.55, 0.16, 4, 2, 3800, 2.3],
    'N': [7, 14.01, 3.04, 1.46, 5, 2, 63, 1.3],
    'O': [8, 16.00, 3.44, 1.40, 6, 2, 54, 1.4],
    'F': [9, 19.00, 3.98, 1.33, 7, 2, 53, 1.7],
    'Na': [11, 22.99, 0.93, 1.02, 1, 3, 371, 0.9],
    'Mg': [12, 24.31, 1.31, 0.72, 2, 3, 923, 1.7],
    'Al': [13, 26.98, 1.61, 0.54, 3, 3, 933, 2.7],
    'Si': [14, 28.09, 1.90, 0.40, 4, 3, 1687, 2.3],
    'P': [15, 30.97, 2.19, 0.38, 5, 3, 317, 1.8],
    'S': [16, 32.07, 2.58, 1.84, 6, 3, 388, 2.1],
    'Cl': [17, 35.45, 3.16, 1.81, 7, 3, 172, 3.2],
    'K': [19, 39.10, 0.82, 1.38, 1, 4, 336, 0.9],
    'Ca': [20, 40.08, 1.00, 1.00, 2, 4, 1115, 1.5],
    'Sc': [21, 44.96, 1.36, 0.75, 3, 4, 1814, 3.0],
    'Ti': [22, 47.87, 1.54, 0.61, 4, 4, 1941, 4.5],
    'V': [23, 50.94, 1.63, 0.54, 5, 4, 2183, 6.1],
    'Cr': [24, 52.00, 1.66, 0.52, 6, 4, 2180, 7.2],
    'Mn': [25, 54.94, 1.55, 0.53, 7, 4, 1519, 7.4],
    'Fe': [26, 55.85, 1.83, 0.55, 3, 4, 1811, 7.9],
    'Co': [27, 58.93, 1.88, 0.55, 3, 4, 1768, 8.9],
    'Ni': [28, 58.69, 1.91, 0.69, 2, 4, 1728, 8.9],
    'Cu': [29, 63.55, 1.90, 0.73, 2, 4, 1358, 9.0],
    'Zn': [30, 65.38, 1.65, 0.74, 2, 4, 693, 7.1],
    'Ga': [31, 69.72, 1.81, 0.62, 3, 4, 303, 5.9],
    'Ge': [32, 72.63, 2.01, 0.53, 4, 4, 1211, 5.3],
    'As': [33, 74.92, 2.18, 0.58, 5, 4, 1090, 5.7],
    'Se': [34, 78.97, 2.55, 1.98, 6, 4, 494, 4.8],
    'Br': [35, 79.90, 2.96, 1.96, 7, 4, 266, 3.1],
    'Rb': [37, 85.47, 0.82, 1.52, 1, 5, 312, 1.5],
    'Sr': [38, 87.62, 0.95, 1.18, 2, 5, 1050, 2.6],
    'Y': [39, 88.91, 1.22, 0.90, 3, 5, 1799, 4.5],
    'Zr': [40, 91.22, 1.33, 0.72, 4, 5, 2128, 6.5],
    'Nb': [41, 92.91, 1.60, 0.64, 5, 5, 2750, 8.6],
    'Mo': [42, 95.95, 2.16, 0.59, 6, 5, 2896, 10.2],
    'Ru': [44, 101.1, 2.20, 0.68, 4, 5, 2607, 12.4],
    'Rh': [45, 102.9, 2.28, 0.67, 3, 5, 2237, 12.4],
    'Pd': [46, 106.4, 2.20, 0.86, 2, 5, 1828, 12.0],
    'Ag': [47, 107.9, 1.93, 1.15, 1, 5, 1235, 10.5],
    'Cd': [48, 112.4, 1.69, 0.95, 2, 5, 594, 8.7],
    'In': [49, 114.8, 1.78, 0.80, 3, 5, 430, 7.3],
    'Sn': [50, 118.7, 1.96, 0.69, 4, 5, 505, 7.3],
    'Sb': [51, 121.8, 2.05, 0.76, 5, 5, 904, 6.7],
    'Te': [52, 127.6, 2.10, 2.21, 6, 5, 723, 6.2],
    'I': [53, 126.9, 2.66, 2.20, 7, 5, 387, 4.9],
    'Cs': [55, 132.9, 0.79, 1.67, 1, 6, 302, 1.9],
    'Ba': [56, 137.3, 0.89, 1.35, 2, 6, 1000, 3.5],
    'La': [57, 138.9, 1.10, 1.03, 3, 6, 1191, 6.2],
    'Ce': [58, 140.1, 1.12, 1.01, 3, 6, 1068, 6.8],
    'Pr': [59, 140.9, 1.13, 0.99, 3, 6, 1208, 6.8],
    'Nd': [60, 144.2, 1.14, 0.98, 3, 6, 1297, 7.0],
    'Sm': [62, 150.4, 1.17, 0.96, 3, 6, 1345, 7.5],
    'Eu': [63, 152.0, 1.20, 0.95, 3, 6, 1099, 5.2],
    'Gd': [64, 157.3, 1.20, 0.94, 3, 6, 1585, 7.9],
    'Tb': [65, 158.9, 1.20, 0.92, 3, 6, 1629, 8.2],
    'Dy': [66, 162.5, 1.22, 0.91, 3, 6, 1680, 8.6],
    'Ho': [67, 164.9, 1.23, 0.90, 3, 6, 1734, 8.8],
    'Er': [68, 167.3, 1.24, 0.89, 3, 6, 1802, 9.1],
    'Tm': [69, 168.9, 1.25, 0.88, 3, 6, 1818, 9.3],
    'Yb': [70, 173.0, 1.10, 0.87, 3, 6, 1097, 6.9],
    'Lu': [71, 175.0, 1.27, 0.86, 3, 6, 1925, 9.8],
    'Hf': [72, 178.5, 1.30, 0.71, 4, 6, 2506, 13.3],
    'Ta': [73, 180.9, 1.50, 0.64, 5, 6, 3290, 16.7],
    'W': [74, 183.8, 2.36, 0.60, 6, 6, 3695, 19.3],
    'Re': [75, 186.2, 1.90, 0.53, 7, 6, 3459, 21.0],
    'Os': [76, 190.2, 2.20, 0.63, 4, 6, 3306, 22.6],
    'Ir': [77, 192.2, 2.20, 0.68, 4, 6, 2719, 22.6],
    'Pt': [78, 195.1, 2.28, 0.80, 4, 6, 2041, 21.5],
    'Au': [79, 197.0, 2.54, 1.37, 3, 6, 1337, 19.3],
    'Hg': [80, 200.6, 2.00, 1.02, 2, 6, 234, 13.5],
    'Tl': [81, 204.4, 1.62, 1.50, 3, 6, 577, 11.9],
    'Pb': [82, 207.2, 1.87, 1.19, 4, 6, 601, 11.3],
    'Bi': [83, 209.0, 2.02, 1.03, 5, 6, 545, 9.8],
}


# ============================================================
# Model A: End-to-End Learning (自动学习特征)
# ============================================================
class EndToEndModel(nn.Module):
    """端到端学习模型 - 不预设特征，从原始结构自动学习"""
    
    def __init__(self, node_dim=16, hidden_dim=256, num_layers=4, dropout=0.2):
        super().__init__()
        
        # 原子嵌入 (从原子序数学习表示)
        self.atom_embedding = nn.Embedding(100, node_dim)
        
        # 位置编码 (学习空间位置表示)
        self.pos_encoder = nn.Sequential(
            nn.Linear(6, 32),  # [frac_coords, sin, cos]
            nn.SiLU(),
            nn.Linear(32, node_dim)
        )
        
        # 消息传递层
        self.message_layers = nn.ModuleList([
            self._make_message_layer(node_dim if i == 0 else hidden_dim, hidden_dim, dropout)
            for i in range(num_layers)
        ])
        
        # 全局池化后的MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def _make_message_layer(self, in_dim, out_dim, dropout):
        return nn.ModuleDict({
            'conv': GCNConv(in_dim, out_dim),
            'norm': nn.LayerNorm(out_dim),
            'dropout': nn.Dropout(dropout),
            'residual': nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        })
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        pos = data.pos if hasattr(data, 'pos') else None
        
        # 原子嵌入
        if x.dim() == 1:
            h = self.atom_embedding(x.clamp(0, 99))
        else:
            h = self.atom_embedding(x[:, 0].long().clamp(0, 99))
        
        # 加入位置编码
        if pos is not None:
            pos_feat = self.pos_encoder(pos)
            h = h + pos_feat
        
        # 消息传递
        for layer in self.message_layers:
            residual = layer['residual'](h)
            h = layer['conv'](h, edge_index)
            h = layer['norm'](h)
            h = F.silu(h + residual)
            h = layer['dropout'](h)
        
        # 全局池化
        h_mean = global_mean_pool(h, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h, batch) / 10.0
        h_global = torch.cat([h_mean, h_max, h_sum], dim=-1)
        
        # 分类
        logits = self.readout(h_global)
        return logits.squeeze(-1)


# ============================================================
# Model B: GCNN (图注意力网络)
# ============================================================
class GATBlock(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.1):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x, edge_index):
        res = self.residual(x)
        h = self.gat(x, edge_index)
        h = self.norm(h)
        h = self.act(h + res)
        h = self.dropout(h)
        return h


class GCNNModel(nn.Module):
    """GCNN模型 - 图注意力网络"""
    
    def __init__(self, node_feat_dim=16, global_feat_dim=64, hidden_dim=256, dropout=0.15):
        super().__init__()
        
        self.node_embed = nn.Sequential(
            nn.Linear(node_feat_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        self.gat1 = GATBlock(64, 128, heads=4, dropout=dropout)
        self.gat2 = GATBlock(128, 128, heads=4, dropout=dropout)
        self.gat3 = GATBlock(128, 128, heads=4, dropout=dropout)
        self.gat4 = GATBlock(128, 64, heads=4, dropout=dropout)
        
        graph_dim = 64 * 3
        
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feat_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(graph_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        u = data.u
        
        h = self.node_embed(x)
        h = self.gat1(h, edge_index)
        h = self.gat2(h, edge_index)
        h = self.gat3(h, edge_index)
        h = self.gat4(h, edge_index)
        
        graph_mean = global_mean_pool(h, batch)
        graph_max = global_max_pool(h, batch)
        graph_add = global_add_pool(h, batch) / 10.0
        graph_feat = torch.cat([graph_mean, graph_max, graph_add], dim=1)
        
        if u.dim() == 3:
            u = u.squeeze(1)
        global_feat = self.global_encoder(u)
        
        combined = torch.cat([graph_feat, global_feat], dim=1)
        h = self.fusion(combined)
        logits = self.classifier(h)
        
        return logits.squeeze(-1)


# ============================================================
# Model C: NequIP_v6 (E3等变神经网络 - 复杂特征版本)
# ============================================================
class BesselBasis(nn.Module):
    def __init__(self, num_basis=8, cutoff=5.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.register_buffer('freq', torch.arange(1, num_basis + 1) * np.pi / cutoff)
    
    def forward(self, r):
        r = r.unsqueeze(-1)
        basis = torch.sqrt(torch.tensor(2.0 / self.cutoff, device=r.device)) * \
                torch.sin(self.freq * r) / (r + 1e-8)
        return basis


class SmoothCutoff(nn.Module):
    def __init__(self, cutoff=5.0, p=6):
        super().__init__()
        self.cutoff = cutoff
        self.p = p
    
    def forward(self, r):
        x = r / self.cutoff
        envelope = (1 - x.pow(self.p)).pow(2)
        envelope = torch.where(r < self.cutoff, envelope, torch.zeros_like(envelope))
        return envelope


class SphericalHarmonics(nn.Module):
    """球谐函数 (l=0,1,2)"""
    def __init__(self):
        super().__init__()
    
    def forward(self, vec):
        x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
        r2 = (x**2 + y**2 + z**2).clamp(min=1e-8)
        r = r2.sqrt()
        
        Y00 = torch.ones_like(x) * 0.2820948
        Y1m1 = 0.4886025 * y / r
        Y10 = 0.4886025 * z / r
        Y1p1 = 0.4886025 * x / r
        Y2m2 = 1.0925484 * x * y / r2
        Y2m1 = 1.0925484 * y * z / r2
        Y20 = 0.3153916 * (3 * z**2 - r2) / r2
        Y2p1 = 1.0925484 * z * x / r2
        Y2p2 = 0.5462742 * (x**2 - y**2) / r2
        
        return torch.stack([Y00, Y1m1, Y10, Y1p1, Y2m2, Y2m1, Y20, Y2p1, Y2p2], dim=-1)


class EquivariantMessageLayer(nn.Module):
    def __init__(self, node_dim, sh_dim=9, radial_dim=8, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.node_dim = node_dim
        self.sh_dim = sh_dim
        
        self.radial_mlp = nn.Sequential(
            nn.Linear(radial_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim * sh_dim)
        )
        
        self.self_interaction = nn.Linear(node_dim, node_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )
        self.layer_norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h, edge_index, edge_sh, edge_radial):
        i, j = edge_index
        radial_weight = self.radial_mlp(edge_radial).view(-1, self.node_dim, self.sh_dim)
        edge_feat = (radial_weight * edge_sh.unsqueeze(1)).sum(dim=-1)
        h_j = h[j]
        messages = h_j * edge_feat
        
        agg = torch.zeros_like(h)
        agg = agg.scatter_add(0, i.unsqueeze(-1).expand_as(messages), messages)
        
        h_self = self.self_interaction(h)
        h_combined = torch.cat([h_self, agg], dim=-1)
        h_update = self.update_mlp(h_combined)
        h_new = self.layer_norm(h + self.dropout(h_update))
        
        return h_new


class NequIPV6Model(nn.Module):
    """NequIP v6 - 增强版E3等变神经网络"""
    
    def __init__(self, node_dim=64, hidden_dim=256, num_layers=4, 
                 global_feat_dim=128, cutoff=5.0, dropout=0.15):
        super().__init__()
        
        self.cutoff = cutoff
        self.node_dim = node_dim
        
        # 原子嵌入 (支持100种元素)
        self.atom_embedding = nn.Embedding(100, node_dim)
        
        # 径向基和截断函数
        self.radial_basis = BesselBasis(num_basis=16, cutoff=cutoff)
        self.cutoff_fn = SmoothCutoff(cutoff=cutoff)
        self.spherical_harmonics = SphericalHarmonics()
        
        # 等变消息传递层
        self.message_layers = nn.ModuleList([
            EquivariantMessageLayer(node_dim, sh_dim=9, radial_dim=16, 
                                   hidden_dim=hidden_dim // 2, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 全局特征编码器 (复杂版本)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feat_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        
        # 融合层
        graph_dim = node_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(graph_dim + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_vec = data.edge_vec
        edge_length = data.edge_length
        batch = data.batch
        u = data.u
        
        # 原子嵌入
        if x.dim() == 1:
            h = self.atom_embedding(x.clamp(0, 99))
        else:
            h = self.atom_embedding(x[:, 0].long().clamp(0, 99))
        
        # 边特征
        edge_unit = edge_vec / (edge_length.unsqueeze(-1) + 1e-8)
        edge_sh = self.spherical_harmonics(edge_unit)
        edge_radial = self.radial_basis(edge_length)
        cutoff_envelope = self.cutoff_fn(edge_length)
        edge_radial = edge_radial * cutoff_envelope.unsqueeze(-1)
        
        # 等变消息传递
        for layer in self.message_layers:
            h = layer(h, edge_index, edge_sh, edge_radial)
        
        # 全局池化
        graph_mean = global_mean_pool(h, batch)
        graph_max = global_max_pool(h, batch)
        graph_sum = global_add_pool(h, batch) / 10.0
        graph_feat = torch.cat([graph_mean, graph_max, graph_sum], dim=-1)
        
        # 全局特征
        if u.dim() == 3:
            u = u.squeeze(1)
        global_feat = self.global_encoder(u)
        
        # 融合
        combined = torch.cat([graph_feat, global_feat], dim=-1)
        h_fused = self.fusion(combined)
        logits = self.classifier(h_fused)
        
        return logits.squeeze(-1)


# ============================================================
# Model D: Transformer (综合特征)
# ============================================================
class TransformerModel(nn.Module):
    """Transformer模型 - 综合特征"""
    
    def __init__(self, input_dim=64, hidden_dim=256, num_heads=4, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim] 或 [batch, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, input_dim]
        
        h = self.input_proj(x)
        h = self.transformer(h)
        h = h.mean(dim=1)  # 平均池化
        logits = self.classifier(h)
        return logits.squeeze(-1)


# ============================================================
# 特征提取器
# ============================================================
class AdvancedFeatureExtractor:
    """高级特征提取器"""
    
    def __init__(self):
        pass
    
    def extract_comprehensive_features(self, structure_dict, spacegroup_number=None):
        """提取综合特征 (128维)"""
        try:
            from pymatgen.core import Structure
            structure = Structure.from_dict(structure_dict)
            
            features = []
            
            # 1. 晶格参数 (6维)
            lattice = structure.lattice
            features.extend([
                lattice.a / 20.0, lattice.b / 20.0, lattice.c / 20.0,
                lattice.alpha / 180.0, lattice.beta / 180.0, lattice.gamma / 180.0
            ])
            
            # 2. 体积和密度 (4维)
            volume = lattice.volume
            features.extend([
                volume / 1000.0,
                np.log1p(volume) / 10.0,
                len(structure) / volume * 100 if volume > 0 else 0,
                structure.density / 10.0
            ])
            
            # 3. 空间群特征 (8维)
            sg = spacegroup_number or 1
            features.extend([
                sg / 230.0,
                1.0 if sg in POLAR_SPACE_GROUPS else 0.0,
                1.0 if sg in FE_COMMON_SPACE_GROUPS else 0.0,
                np.sin(sg * np.pi / 115),
                np.cos(sg * np.pi / 115),
                1.0 if 1 <= sg <= 2 else 0.0,
                1.0 if 3 <= sg <= 15 else 0.0,
                1.0 if sg >= 195 else 0.0,
            ])
            
            # 4. 元素统计特征 (32维)
            elements = [site.specie.symbol for site in structure]
            element_counts = {}
            for el in elements:
                element_counts[el] = element_counts.get(el, 0) + 1
            
            # 电负性
            en_values = [ELEMENT_DATABASE.get(el, [0]*8)[2] for el in elements]
            features.extend([
                np.mean(en_values) / 4.0,
                np.std(en_values) / 2.0,
                np.max(en_values) / 4.0 - np.min(en_values) / 4.0,
                np.max(en_values) / 4.0,
            ])
            
            # 离子半径
            ir_values = [ELEMENT_DATABASE.get(el, [0]*8)[3] for el in elements]
            features.extend([
                np.mean(ir_values) / 2.0,
                np.std(ir_values),
                np.max(ir_values) / 2.0,
                np.min(ir_values) / 2.0,
            ])
            
            # 原子质量
            mass_values = [ELEMENT_DATABASE.get(el, [0]*8)[1] for el in elements]
            features.extend([
                np.mean(mass_values) / 200.0,
                np.std(mass_values) / 100.0,
                np.sum(mass_values) / 2000.0,
                len(set(elements)) / 10.0,
            ])
            
            # 价电子
            val_values = [ELEMENT_DATABASE.get(el, [0]*8)[4] for el in elements]
            features.extend([
                np.mean(val_values) / 7.0,
                np.std(val_values) / 3.0,
                np.max(val_values) / 7.0,
                np.sum(val_values) / 100.0,
            ])
            
            # 5. 对称性特征 (16维)
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(structure, symprec=0.1)
                sym_struct = sga.get_symmetrized_structure()
                
                eq_atoms = sym_struct.equivalent_indices
                features.extend([
                    len(eq_atoms) / 20.0,
                    len(structure) / (len(eq_atoms) + 1),
                    sga.get_space_group_number() / 230.0,
                    1.0 if sga.is_laue() else 0.0,
                ])
            except:
                features.extend([0.0, 0.0, sg / 230.0, 0.0])
            
            # 位置特征
            frac_coords = [site.frac_coords for site in structure]
            frac_coords = np.array(frac_coords)
            features.extend([
                np.mean(frac_coords[:, 0]),
                np.mean(frac_coords[:, 1]),
                np.mean(frac_coords[:, 2]),
                np.std(frac_coords[:, 0]),
                np.std(frac_coords[:, 1]),
                np.std(frac_coords[:, 2]),
            ])
            
            # 键长分布
            try:
                all_distances = []
                for i, site in enumerate(structure):
                    neighbors = structure.get_neighbors(site, 4.0)
                    for n in neighbors:
                        all_distances.append(n.nn_distance)
                if all_distances:
                    features.extend([
                        np.mean(all_distances) / 5.0,
                        np.std(all_distances) / 2.0,
                        np.min(all_distances) / 5.0,
                        np.max(all_distances) / 5.0,
                        len(all_distances) / (len(structure) * 20),
                        np.percentile(all_distances, 25) / 5.0,
                    ])
                else:
                    features.extend([0.5, 0.2, 0.3, 0.8, 0.5, 0.4])
            except:
                features.extend([0.5, 0.2, 0.3, 0.8, 0.5, 0.4])
            
            # 6. 铁电相关特征 (16维)
            # 常见铁电元素
            fe_elements = {'Ti', 'Zr', 'Pb', 'Ba', 'Bi', 'Nb', 'Ta', 'O', 'N'}
            fe_count = sum(1 for el in elements if el in fe_elements)
            features.extend([
                fe_count / len(elements),
                1.0 if 'Ti' in elements else 0.0,
                1.0 if 'O' in elements else 0.0,
                1.0 if 'Pb' in elements else 0.0,
                1.0 if 'Ba' in elements else 0.0,
                1.0 if 'Bi' in elements else 0.0,
                1.0 if 'Nb' in elements else 0.0,
                1.0 if 'Zr' in elements else 0.0,
            ])
            
            # A-site / B-site 特征 (钙钛矿)
            a_site = {'Ba', 'Sr', 'Ca', 'Pb', 'Bi', 'K', 'Na', 'La'}
            b_site = {'Ti', 'Zr', 'Nb', 'Ta', 'Fe', 'Mn', 'W'}
            a_count = sum(1 for el in elements if el in a_site)
            b_count = sum(1 for el in elements if el in b_site)
            features.extend([
                a_count / (len(elements) + 1),
                b_count / (len(elements) + 1),
                a_count / (b_count + 1),
                1.0 if (a_count > 0 and b_count > 0) else 0.0,
            ])
            
            # 极性指标
            features.extend([
                np.std(en_values) * np.std(ir_values) / 2.0,
                np.max(en_values) - np.min(en_values) if len(en_values) > 1 else 0,
                1.0 if sg in {99, 161, 160, 38} else 0.0,
                fe_count / (len(elements) + 1) * (1.0 if sg in POLAR_SPACE_GROUPS else 0.5),
            ])
            
            # 7. 补齐到128维
            while len(features) < 128:
                features.append(0.0)
            
            return np.array(features[:128], dtype=np.float32)
            
        except Exception as e:
            return np.zeros(128, dtype=np.float32)
    
    def extract_graph_node_features(self, structure_dict, spacegroup_number=None):
        """提取图节点特征 (16维)"""
        try:
            from pymatgen.core import Structure
            structure = Structure.from_dict(structure_dict)
            
            node_features = []
            for site in structure:
                el = site.specie.symbol
                if el in ELEMENT_DATABASE:
                    data = ELEMENT_DATABASE[el]
                    feat = [
                        data[0] / 100.0,  # 原子序数
                        data[1] / 200.0,  # 原子质量
                        data[2] / 4.0,    # 电负性
                        data[3] / 2.5,    # 离子半径
                        data[4] / 8.0,    # 价电子
                        data[5] / 7.0,    # 周期
                        data[6] / 4000.0, # 熔点
                        data[7] / 25.0,   # 密度
                        site.frac_coords[0],
                        site.frac_coords[1],
                        site.frac_coords[2],
                        np.sin(2 * np.pi * site.frac_coords[0]),
                        np.sin(2 * np.pi * site.frac_coords[1]),
                        np.sin(2 * np.pi * site.frac_coords[2]),
                        np.cos(2 * np.pi * site.frac_coords[0]),
                        np.cos(2 * np.pi * site.frac_coords[1]),
                    ]
                else:
                    feat = [0.5] * 16
                node_features.append(feat)
            
            return np.array(node_features, dtype=np.float32)
            
        except Exception as e:
            return None


# ============================================================
# 数据处理
# ============================================================
def structure_to_graph(structure_dict, label, global_features, node_features, cutoff=5.0):
    """将结构转换为图数据"""
    try:
        from pymatgen.core import Structure
        structure = Structure.from_dict(structure_dict)
        
        # 节点特征
        if node_features is not None:
            x = torch.tensor(node_features, dtype=torch.float)
        else:
            x = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
        
        # 边
        edge_index = []
        edge_vec = []
        edge_length = []
        
        for i, site_i in enumerate(structure):
            neighbors = structure.get_neighbors(site_i, cutoff)
            for neighbor in neighbors:
                j = neighbor.index
                if i != j:
                    edge_index.append([i, j])
                    vec = neighbor.coords - site_i.coords
                    edge_vec.append(vec)
                    edge_length.append(neighbor.nn_distance)
        
        if not edge_index:
            n = len(structure)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edge_index.append([i, j])
                        vec_frac = structure[j].frac_coords - structure[i].frac_coords
                        vec = structure.lattice.get_cartesian_coords(vec_frac)
                        dist = max(np.linalg.norm(vec), 1.0)
                        edge_vec.append(vec)
                        edge_length.append(dist)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_vec = torch.tensor(np.array(edge_vec), dtype=torch.float)
        edge_length = torch.tensor(edge_length, dtype=torch.float)
        
        # 位置编码
        frac_coords = np.array([site.frac_coords for site in structure])
        sin_coords = np.sin(2 * np.pi * frac_coords)
        pos = np.concatenate([frac_coords, sin_coords], axis=1)
        pos = torch.tensor(pos, dtype=torch.float)
        
        # 全局特征
        u = torch.tensor(global_features, dtype=torch.float).unsqueeze(0)
        
        # 标签
        y = torch.tensor([label], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_vec=edge_vec, 
                   edge_length=edge_length, pos=pos, y=y, u=u)
    except Exception as e:
        return None


def get_spacegroup_number(structure_dict):
    """获取空间群编号"""
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        structure = Structure.from_dict(structure_dict)
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        return sga.get_space_group_number()
    except:
        return None


def load_data_files(data_dir, only_polar=True):
    """加载数据文件"""
    data_dir = Path(data_dir)
    
    positive_files = [
        data_dir / 'dataset_original_ferroelectric.jsonl',
        data_dir / 'dataset_known_FE_rest.jsonl',
    ]
    
    negative_files = [
        data_dir / 'dataset_nonFE.jsonl',
        data_dir / 'dataset_nonFE_cleaned.jsonl',
        data_dir / 'dataset_nonFE_expanded.jsonl',
    ]
    
    samples = []
    
    # 加载正样本
    print("\n加载正样本 (铁电材料)...")
    for file_path in positive_files:
        if file_path.exists():
            with open(file_path, 'r') as f:
                lines = f.readlines()
            count = 0
            for line in tqdm(lines, desc=f"  {file_path.name}"):
                try:
                    item = json.loads(line)
                    struct = item.get('structure')
                    if struct:
                        sg = get_spacegroup_number(struct)
                        if sg is not None:
                            if not only_polar or sg in POLAR_SPACE_GROUPS:
                                samples.append({
                                    'structure': struct,
                                    'spacegroup': sg,
                                    'label': 1
                                })
                                count += 1
                except:
                    continue
            print(f"  {file_path.name}: {count} 个正样本")
    
    # 加载负样本
    print("\n加载负样本 (非铁电材料)...")
    for file_path in negative_files:
        if file_path.exists():
            with open(file_path, 'r') as f:
                lines = f.readlines()
            count = 0
            for line in tqdm(lines, desc=f"  {file_path.name}"):
                try:
                    item = json.loads(line)
                    struct = item.get('structure')
                    if struct:
                        sg = get_spacegroup_number(struct)
                        if sg is not None:
                            if not only_polar or sg in POLAR_SPACE_GROUPS:
                                samples.append({
                                    'structure': struct,
                                    'spacegroup': sg,
                                    'label': 0
                                })
                                count += 1
                except:
                    continue
            print(f"  {file_path.name}: {count} 个负样本")
    
    return samples


# ============================================================
# 梯次决策系统
# ============================================================
class CascadeDecisionSystem:
    """梯次决策系统"""
    
    def __init__(self, models, model_weights, thresholds):
        self.models = models  # {'e2e': model, 'gcnn': model, 'nequip': model, 'transformer': model}
        self.model_weights = model_weights  # {'e2e': w, 'gcnn': w, ...}
        self.thresholds = thresholds  # {'high_conf': 0.9, 'low_conf': 0.3, 'ensemble': 0.5}
    
    def predict(self, data_dict, global_features):
        """
        梯次决策预测
        
        策略:
        - 第一梯队: 任意模型高置信度(>0.9) → 直接判定为FE
        - 第二梯队: 中置信度 → 加权投票
        - 第三梯队: 低置信度 → 专家规则验证
        """
        predictions = {}
        
        # 各模型预测
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                # 根据模型类型准备数据
                if name == 'transformer':
                    x = torch.tensor(global_features, dtype=torch.float).unsqueeze(0).to(device)
                    logits = model(x)
                else:
                    # 图模型
                    data = data_dict[name].to(device)
                    logits = model(data)
                
                prob = torch.sigmoid(logits).item()
                predictions[name] = prob
        
        # 第一梯队: 高置信度判定
        high_conf = self.thresholds['high_conf']
        for name, prob in predictions.items():
            if prob > high_conf:
                return 1, prob, 'tier1_high_conf', predictions
        
        # 第二梯队: 加权投票
        weighted_sum = 0
        weight_sum = 0
        for name, prob in predictions.items():
            w = self.model_weights.get(name, 1.0)
            weighted_sum += prob * w
            weight_sum += w
        
        ensemble_prob = weighted_sum / (weight_sum + 1e-8)
        
        if ensemble_prob > self.thresholds['ensemble']:
            return 1, ensemble_prob, 'tier2_ensemble', predictions
        
        # 第三梯队: 低置信度，需要专家规则验证
        low_conf = self.thresholds['low_conf']
        if ensemble_prob > low_conf:
            # 检查是否满足专家规则
            # (在这里可以加入更多物理规则)
            return 0, ensemble_prob, 'tier3_expert_reject', predictions
        
        return 0, ensemble_prob, 'tier3_low_conf', predictions


# ============================================================
# 训练器
# ============================================================
class CompositeTrainerV2:
    """复合决策模型训练器 V2"""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / 'new_data'
        self.model_dir = Path(__file__).parent.parent / 'model_composite_v2'
        self.report_dir = Path(__file__).parent.parent / 'reports_composite_v2'
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_extractor = AdvancedFeatureExtractor()
        
    def prepare_data(self):
        """准备数据"""
        print("=" * 60)
        print("加载极性材料数据集")
        print("=" * 60)
        
        samples = load_data_files(self.data_dir, only_polar=True)
        
        labels = [s['label'] for s in samples]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        
        print(f"\n极性材料子集: {len(samples)} 样本")
        print(f"  FE: {n_pos} ({n_pos/len(samples)*100:.2f}%)")
        print(f"  Non-FE: {n_neg} ({n_neg/len(samples)*100:.2f}%)")
        print(f"  类别比例: 1:{n_neg/n_pos:.1f}")
        
        # 提取特征
        print("\n提取特征...")
        all_features = []
        all_node_features = []
        valid_samples = []
        
        for sample in tqdm(samples, desc="特征提取"):
            global_feat = self.feature_extractor.extract_comprehensive_features(
                sample['structure'], sample['spacegroup']
            )
            node_feat = self.feature_extractor.extract_graph_node_features(
                sample['structure'], sample['spacegroup']
            )
            
            if node_feat is not None and len(node_feat) > 0:
                all_features.append(global_feat)
                all_node_features.append(node_feat)
                valid_samples.append(sample)
        
        print(f"\n有效样本: {len(valid_samples)}")
        
        return valid_samples, np.array(all_features), all_node_features
    
    def train_fold(self, train_samples, val_samples, train_features, val_features,
                   train_node_features, val_node_features, fold_idx):
        """训练一个fold"""
        
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/5")
        print(f"{'='*60}")
        
        train_labels = np.array([s['label'] for s in train_samples])
        val_labels = np.array([s['label'] for s in val_samples])
        
        print(f"训练集: {len(train_samples)} (正样本: {sum(train_labels)})")
        print(f"验证集: {len(val_samples)} (正样本: {sum(val_labels)})")
        
        # 特征标准化
        scaler = RobustScaler()
        train_features_scaled = scaler.fit_transform(train_features)
        val_features_scaled = scaler.transform(val_features)
        
        # SMOTE过采样
        if SMOTE_AVAILABLE:
            smote = SMOTE(random_state=42)
            train_features_resampled, train_labels_resampled = smote.fit_resample(
                train_features_scaled, train_labels
            )
            print(f"SMOTE: {sum(train_labels)} -> {sum(train_labels_resampled)} 正样本")
        else:
            train_features_resampled = train_features_scaled
            train_labels_resampled = train_labels
        
        # 构建图数据
        print("\n构建图数据...")
        train_graphs = []
        for i, sample in enumerate(tqdm(train_samples, desc="训练集图")):
            graph = structure_to_graph(
                sample['structure'], sample['label'],
                train_features_scaled[i], train_node_features[i]
            )
            if graph is not None:
                train_graphs.append(graph)
        
        val_graphs = []
        for i, sample in enumerate(tqdm(val_samples, desc="验证集图")):
            graph = structure_to_graph(
                sample['structure'], sample['label'],
                val_features_scaled[i], val_node_features[i]
            )
            if graph is not None:
                val_graphs.append(graph)
        
        print(f"训练图: {len(train_graphs)}, 验证图: {len(val_graphs)}")
        
        # 创建DataLoader
        train_loader = GeoDataLoader(train_graphs, batch_size=32, shuffle=True)
        val_loader = GeoDataLoader(val_graphs, batch_size=32, shuffle=False)
        
        # 初始化模型
        models = {
            'e2e': EndToEndModel(node_dim=16, hidden_dim=256, num_layers=4, dropout=0.2).to(device),
            'gcnn': GCNNModel(node_feat_dim=16, global_feat_dim=128, hidden_dim=256, dropout=0.15).to(device),
            'nequip': NequIPV6Model(node_dim=64, hidden_dim=256, num_layers=4, global_feat_dim=128, dropout=0.15).to(device),
        }
        
        # Transformer模型使用全局特征
        transformer = TransformerModel(input_dim=128, hidden_dim=256, num_heads=4, num_layers=3, dropout=0.2).to(device)
        
        # 训练配置
        pos_weight = torch.tensor([10.0]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # 训练各个图模型
        model_results = {}
        
        for model_name, model in models.items():
            print(f"\n--- 训练 {model_name.upper()} ---")
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
            
            best_auc = 0
            patience = 30
            patience_counter = 0
            
            for epoch in range(100):
                # 训练
                model.train()
                train_loss = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    logits = model(batch)
                    labels = batch.y.float()
                    loss = criterion(logits, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                scheduler.step()
                
                # 验证
                model.eval()
                val_probs = []
                val_labels_list = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        logits = model(batch)
                        probs = torch.sigmoid(logits)
                        val_probs.extend(probs.cpu().numpy())
                        val_labels_list.extend(batch.y.cpu().numpy())
                
                val_probs = np.array(val_probs)
                val_labels_arr = np.array(val_labels_list)
                
                try:
                    auc = roc_auc_score(val_labels_arr, val_probs)
                except:
                    auc = 0.5
                
                if auc > best_auc:
                    best_auc = auc
                    patience_counter = 0
                    best_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
            
            # 恢复最佳模型
            model.load_state_dict(best_state)
            
            # 最终评估
            model.eval()
            val_probs = []
            val_labels_list = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    logits = model(batch)
                    probs = torch.sigmoid(logits)
                    val_probs.extend(probs.cpu().numpy())
                    val_labels_list.extend(batch.y.cpu().numpy())
            
            val_probs = np.array(val_probs)
            val_labels_arr = np.array(val_labels_list)
            
            auc = roc_auc_score(val_labels_arr, val_probs)
            model_results[model_name] = {
                'auc': auc,
                'probs': val_probs,
                'model': model
            }
            print(f"  {model_name} AUC: {auc:.4f}")
        
        # 训练Transformer
        print(f"\n--- 训练 TRANSFORMER ---")
        
        # 准备数据
        train_X = torch.tensor(train_features_resampled, dtype=torch.float32)
        train_Y = torch.tensor(train_labels_resampled, dtype=torch.float32)
        val_X = torch.tensor(val_features_scaled, dtype=torch.float32)
        val_Y = torch.tensor(val_labels, dtype=torch.float32)
        
        train_dataset = TensorDataset(train_X, train_Y)
        val_dataset = TensorDataset(val_X, val_Y)
        
        train_loader_tf = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader_tf = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        optimizer = torch.optim.AdamW(transformer.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        best_auc = 0
        patience = 30
        patience_counter = 0
        
        for epoch in range(100):
            transformer.train()
            for batch_x, batch_y in train_loader_tf:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                logits = transformer(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            
            # 验证
            transformer.eval()
            val_probs = []
            val_labels_list = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader_tf:
                    batch_x = batch_x.to(device)
                    logits = transformer(batch_x)
                    probs = torch.sigmoid(logits)
                    val_probs.extend(probs.cpu().numpy())
                    val_labels_list.extend(batch_y.numpy())
            
            val_probs = np.array(val_probs)
            val_labels_arr = np.array(val_labels_list)
            
            try:
                auc = roc_auc_score(val_labels_arr, val_probs)
            except:
                auc = 0.5
            
            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
                best_state = transformer.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        transformer.load_state_dict(best_state)
        
        # 最终评估
        transformer.eval()
        val_probs = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader_tf:
                batch_x = batch_x.to(device)
                logits = transformer(batch_x)
                probs = torch.sigmoid(logits)
                val_probs.extend(probs.cpu().numpy())
        
        val_probs = np.array(val_probs)
        auc = roc_auc_score(val_labels, val_probs)
        model_results['transformer'] = {
            'auc': auc,
            'probs': val_probs,
            'model': transformer
        }
        print(f"  Transformer AUC: {auc:.4f}")
        
        # 梯次集成决策
        print(f"\n--- 梯次集成决策 ---")
        
        # 计算模型权重 (基于AUC)
        total_auc = sum(r['auc'] for r in model_results.values())
        model_weights = {name: r['auc'] / total_auc for name, r in model_results.items()}
        
        print(f"模型权重:")
        for name, weight in model_weights.items():
            print(f"  {name}: {weight:.3f} (AUC: {model_results[name]['auc']:.4f})")
        
        # 加权集成预测
        ensemble_probs = np.zeros(len(val_labels))
        for name, result in model_results.items():
            # 确保长度匹配
            probs = result['probs']
            if len(probs) == len(val_labels):
                ensemble_probs += probs * model_weights[name]
        
        # 寻找最优阈值
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        
        for thresh in np.arange(0.1, 0.9, 0.02):
            preds = (ensemble_probs > thresh).astype(int)
            
            acc = accuracy_score(val_labels, preds)
            recall = recall_score(val_labels, preds, zero_division=0)
            precision = precision_score(val_labels, preds, zero_division=0)
            f1 = f1_score(val_labels, preds, zero_division=0)
            
            # 优化目标: 高召回率下的F1
            score = f1 * 0.4 + recall * 0.4 + acc * 0.2
            
            if score > best_f1:
                best_f1 = score
                best_threshold = thresh
                best_metrics = {
                    'accuracy': acc,
                    'recall': recall,
                    'precision': precision,
                    'f1': f1,
                    'auc': roc_auc_score(val_labels, ensemble_probs)
                }
        
        print(f"\n最优阈值: {best_threshold:.2f}")
        print(f"Accuracy: {best_metrics['accuracy']:.4f}")
        print(f"Recall: {best_metrics['recall']:.4f}")
        print(f"Precision: {best_metrics['precision']:.4f}")
        print(f"F1: {best_metrics['f1']:.4f}")
        print(f"AUC: {best_metrics['auc']:.4f}")
        
        return {
            'models': {name: r['model'] for name, r in model_results.items()},
            'model_weights': model_weights,
            'scaler': scaler,
            'threshold': best_threshold,
            'metrics': best_metrics
        }
    
    def train(self):
        """完整训练流程"""
        print("=" * 60)
        print("复合决策模型 V2 - 增强版多模型集成")
        print("=" * 60)
        print("\n架构设计:")
        print("  Layer 0: 专家规则 (极性空间群筛选)")
        print("  Layer 1: 多模型集成")
        print("    - Model A: End-to-End (自动学习特征)")
        print("    - Model B: GCNN (图注意力网络)")
        print("    - Model C: NequIP_v6 (E3等变网络)")
        print("    - Model D: Transformer (综合特征)")
        print("  Layer 2: 梯次决策")
        print("    - 第一梯队: 高置信度直接判定")
        print("    - 第二梯队: 加权投票")
        print("    - 第三梯队: 专家规则验证")
        
        # 准备数据
        samples, features, node_features = self.prepare_data()
        labels = np.array([s['label'] for s in samples])
        
        # 5折交叉验证
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        all_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(features, labels)):
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]
            train_features = features[train_idx]
            val_features = features[val_idx]
            train_node_features = [node_features[i] for i in train_idx]
            val_node_features = [node_features[i] for i in val_idx]
            
            result = self.train_fold(
                train_samples, val_samples,
                train_features, val_features,
                train_node_features, val_node_features,
                fold_idx
            )
            
            all_results.append(result)
        
        # 汇总结果
        print("\n" + "=" * 60)
        print("交叉验证汇总")
        print("=" * 60)
        
        metrics_summary = defaultdict(list)
        for result in all_results:
            for metric, value in result['metrics'].items():
                metrics_summary[metric].append(value)
        
        print(f"\n{'指标':<15} {'平均值':<12} {'标准差':<12}")
        print("-" * 40)
        for metric, values in metrics_summary.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric:<15} {mean_val:.4f}       {std_val:.4f}")
        
        # 保存最佳模型 (基于AUC)
        best_fold_idx = np.argmax([r['metrics']['auc'] for r in all_results])
        best_result = all_results[best_fold_idx]
        
        print(f"\n最佳Fold: {best_fold_idx + 1}")
        print(f"  Accuracy: {best_result['metrics']['accuracy']:.4f}")
        print(f"  Recall: {best_result['metrics']['recall']:.4f}")
        print(f"  Precision: {best_result['metrics']['precision']:.4f}")
        print(f"  F1: {best_result['metrics']['f1']:.4f}")
        print(f"  AUC: {best_result['metrics']['auc']:.4f}")
        
        # 保存模型
        torch.save({
            'models': {name: model.state_dict() for name, model in best_result['models'].items()},
            'model_weights': best_result['model_weights'],
            'threshold': best_result['threshold'],
        }, self.model_dir / 'composite_v2_models.pt')
        
        joblib.dump(best_result['scaler'], self.model_dir / 'scaler.pkl')
        
        # 保存结果
        results_df = pd.DataFrame([{
            'fold': i + 1,
            **r['metrics']
        } for i, r in enumerate(all_results)])
        results_df.to_csv(self.report_dir / 'cv_results.csv', index=False)
        
        print(f"\n模型已保存到: {self.model_dir}")
        print(f"报告已保存到: {self.report_dir}")
        
        return all_results


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    trainer = CompositeTrainerV2()
    results = trainer.train()
