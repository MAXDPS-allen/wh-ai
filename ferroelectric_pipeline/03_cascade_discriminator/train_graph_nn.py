#!/usr/bin/env python3
"""
图神经网络铁电分类器 - 让模型自己学习晶体结构表示

关键技术:
1. 将晶体结构转换为图表示 (节点=原子, 边=键)
2. 消息传递神经网络 (MPNN) 学习原子间相互作用
3. 全局注意力池化
4. Focal Loss + 对比学习
5. 多任务学习增强泛化
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score)
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ============================================================================
# 元素属性
# ============================================================================

ELEMENT_PROPERTIES = {
    'H': [1, 1, 1, 2.20, 0.53], 'He': [2, 1, 18, 0.0, 0.31],
    'Li': [3, 2, 1, 0.98, 1.67], 'Be': [4, 2, 2, 1.57, 1.12],
    'B': [5, 2, 13, 2.04, 0.87], 'C': [6, 2, 14, 2.55, 0.67],
    'N': [7, 2, 15, 3.04, 0.56], 'O': [8, 2, 16, 3.44, 0.48],
    'F': [9, 2, 17, 3.98, 0.42], 'Ne': [10, 2, 18, 0.0, 0.38],
    'Na': [11, 3, 1, 0.93, 1.90], 'Mg': [12, 3, 2, 1.31, 1.45],
    'Al': [13, 3, 13, 1.61, 1.18], 'Si': [14, 3, 14, 1.90, 1.11],
    'P': [15, 3, 15, 2.19, 0.98], 'S': [16, 3, 16, 2.58, 0.88],
    'Cl': [17, 3, 17, 3.16, 0.79], 'Ar': [18, 3, 18, 0.0, 0.71],
    'K': [19, 4, 1, 0.82, 2.43], 'Ca': [20, 4, 2, 1.00, 1.94],
    'Sc': [21, 4, 3, 1.36, 1.84], 'Ti': [22, 4, 4, 1.54, 1.76],
    'V': [23, 4, 5, 1.63, 1.71], 'Cr': [24, 4, 6, 1.66, 1.66],
    'Mn': [25, 4, 7, 1.55, 1.61], 'Fe': [26, 4, 8, 1.83, 1.56],
    'Co': [27, 4, 9, 1.88, 1.52], 'Ni': [28, 4, 10, 1.91, 1.49],
    'Cu': [29, 4, 11, 1.90, 1.45], 'Zn': [30, 4, 12, 1.65, 1.42],
    'Ga': [31, 4, 13, 1.81, 1.36], 'Ge': [32, 4, 14, 2.01, 1.25],
    'As': [33, 4, 15, 2.18, 1.14], 'Se': [34, 4, 16, 2.55, 1.03],
    'Br': [35, 4, 17, 2.96, 0.94], 'Kr': [36, 4, 18, 0.0, 0.88],
    'Rb': [37, 5, 1, 0.82, 2.65], 'Sr': [38, 5, 2, 0.95, 2.19],
    'Y': [39, 5, 3, 1.22, 2.12], 'Zr': [40, 5, 4, 1.33, 2.06],
    'Nb': [41, 5, 5, 1.60, 1.98], 'Mo': [42, 5, 6, 2.16, 1.90],
    'Tc': [43, 5, 7, 1.90, 1.83], 'Ru': [44, 5, 8, 2.20, 1.78],
    'Rh': [45, 5, 9, 2.28, 1.73], 'Pd': [46, 5, 10, 2.20, 1.69],
    'Ag': [47, 5, 11, 1.93, 1.65], 'Cd': [48, 5, 12, 1.69, 1.61],
    'In': [49, 5, 13, 1.78, 1.56], 'Sn': [50, 5, 14, 1.96, 1.45],
    'Sb': [51, 5, 15, 2.05, 1.33], 'Te': [52, 5, 16, 2.10, 1.23],
    'I': [53, 5, 17, 2.66, 1.15], 'Xe': [54, 5, 18, 0.0, 1.08],
    'Cs': [55, 6, 1, 0.79, 2.98], 'Ba': [56, 6, 2, 0.89, 2.53],
    'La': [57, 6, 3, 1.10, 2.50], 'Ce': [58, 6, 3, 1.12, 2.48],
    'Pr': [59, 6, 3, 1.13, 2.47], 'Nd': [60, 6, 3, 1.14, 2.45],
    'Pm': [61, 6, 3, 1.13, 2.43], 'Sm': [62, 6, 3, 1.17, 2.42],
    'Eu': [63, 6, 3, 1.20, 2.40], 'Gd': [64, 6, 3, 1.20, 2.38],
    'Tb': [65, 6, 3, 1.20, 2.37], 'Dy': [66, 6, 3, 1.22, 2.35],
    'Ho': [67, 6, 3, 1.23, 2.33], 'Er': [68, 6, 3, 1.24, 2.32],
    'Tm': [69, 6, 3, 1.25, 2.30], 'Yb': [70, 6, 3, 1.10, 2.28],
    'Lu': [71, 6, 3, 1.27, 2.17], 'Hf': [72, 6, 4, 1.30, 2.08],
    'Ta': [73, 6, 5, 1.50, 2.00], 'W': [74, 6, 6, 2.36, 1.93],
    'Re': [75, 6, 7, 1.90, 1.88], 'Os': [76, 6, 8, 2.20, 1.85],
    'Ir': [77, 6, 9, 2.20, 1.80], 'Pt': [78, 6, 10, 2.28, 1.77],
    'Au': [79, 6, 11, 2.54, 1.74], 'Hg': [80, 6, 12, 2.00, 1.71],
    'Tl': [81, 6, 13, 1.62, 1.56], 'Pb': [82, 6, 14, 2.33, 1.54],
    'Bi': [83, 6, 15, 2.02, 1.43], 'Po': [84, 6, 16, 2.00, 1.35],
}

DEFAULT_PROPS = [50, 5, 10, 1.5, 1.5]  # [Z, period, group, electronegativity, radius]


# ============================================================================
# 图数据结构
# ============================================================================

class CrystalGraph:
    """晶体图表示"""
    def __init__(self, node_features: np.ndarray, edge_index: np.ndarray, 
                 edge_attr: np.ndarray, global_features: np.ndarray):
        self.node_features = node_features  # (n_atoms, node_dim)
        self.edge_index = edge_index  # (2, n_edges)
        self.edge_attr = edge_attr  # (n_edges, edge_dim)
        self.global_features = global_features  # (global_dim,)


def structure_to_graph(structure: Dict, cutoff: float = 6.0) -> CrystalGraph:
    """将晶体结构转换为图"""
    # 提取晶格和原子信息
    lattice = np.array(structure.get('lattice', {}).get('matrix', np.eye(3) * 5.0))
    sites = structure.get('sites', [])
    
    if not sites:
        # 返回空图
        return CrystalGraph(
            node_features=np.zeros((1, 32)),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_attr=np.zeros((0, 16)),
            global_features=np.zeros(32)
        )
    
    # 提取坐标和元素
    coords = []
    elements = []
    for site in sites:
        if 'xyz' in site:
            coords.append(site['xyz'])
        elif 'abc' in site:
            frac = np.array(site['abc'])
            cart = np.dot(frac, lattice)
            coords.append(cart.tolist())
        else:
            coords.append([0, 0, 0])
        
        species = site.get('species', [])
        if species:
            elem = species[0].get('element', 'X')
        else:
            elem = site.get('label', 'X')[:2].strip()
        elements.append(elem)
    
    coords = np.array(coords)
    n_atoms = len(coords)
    
    # 节点特征 (原子属性)
    node_features = []
    for elem in elements:
        props = ELEMENT_PROPERTIES.get(elem, DEFAULT_PROPS)
        # 扩展特征: Z, period, group, electronegativity, radius + one-hot编码
        feat = list(props)
        # 添加周期的one-hot (1-7)
        period_onehot = [0] * 7
        if 1 <= props[1] <= 7:
            period_onehot[props[1] - 1] = 1
        feat.extend(period_onehot)
        # 添加族的one-hot (简化为几类)
        group = props[2]
        group_onehot = [0] * 5  # 碱金属, 碱土, 过渡, 主族, 稀有气体
        if group == 1:
            group_onehot[0] = 1
        elif group == 2:
            group_onehot[1] = 1
        elif 3 <= group <= 12:
            group_onehot[2] = 1
        elif 13 <= group <= 17:
            group_onehot[3] = 1
        elif group == 18:
            group_onehot[4] = 1
        feat.extend(group_onehot)
        # 补充到32维
        while len(feat) < 32:
            feat.append(0.0)
        node_features.append(feat[:32])
    
    node_features = np.array(node_features, dtype=np.float32)
    
    # 构建边 (原子间连接)
    edge_src = []
    edge_dst = []
    edge_attr = []
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                continue
            
            # 计算距离 (考虑周期性)
            diff = coords[j] - coords[i]
            min_dist = float('inf')
            min_vec = diff
            
            for a in range(-1, 2):
                for b in range(-1, 2):
                    for c in range(-1, 2):
                        shift = a * lattice[0] + b * lattice[1] + c * lattice[2]
                        vec = diff + shift
                        d = np.linalg.norm(vec)
                        if d < min_dist:
                            min_dist = d
                            min_vec = vec
            
            if min_dist < cutoff and min_dist > 0.5:
                edge_src.append(i)
                edge_dst.append(j)
                
                # 边特征
                feat = [min_dist]
                # 距离的高斯基函数
                centers = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
                for c in centers:
                    feat.append(np.exp(-(min_dist - c)**2 / 0.5))
                # 方向特征
                unit_vec = min_vec / (min_dist + 1e-10)
                feat.extend(unit_vec.tolist())
                # 截断函数
                feat.append(0.5 * (1 + np.cos(np.pi * min_dist / cutoff)))
                # 补充到16维
                while len(feat) < 16:
                    feat.append(0.0)
                edge_attr.append(feat[:16])
    
    if len(edge_src) == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, 16), dtype=np.float32)
    else:
        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        edge_attr = np.array(edge_attr, dtype=np.float32)
    
    # 全局特征 (晶格特征)
    a = np.linalg.norm(lattice[0])
    b = np.linalg.norm(lattice[1])
    c = np.linalg.norm(lattice[2])
    volume = abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))
    
    cos_alpha = np.dot(lattice[1], lattice[2]) / (np.linalg.norm(lattice[1]) * np.linalg.norm(lattice[2]) + 1e-10)
    cos_beta = np.dot(lattice[0], lattice[2]) / (np.linalg.norm(lattice[0]) * np.linalg.norm(lattice[2]) + 1e-10)
    cos_gamma = np.dot(lattice[0], lattice[1]) / (np.linalg.norm(lattice[0]) * np.linalg.norm(lattice[1]) + 1e-10)
    
    global_features = [
        a, b, c, volume,
        a/b if b > 0 else 1, b/c if c > 0 else 1, a/c if c > 0 else 1,
        cos_alpha, cos_beta, cos_gamma,
        n_atoms, n_atoms / (volume + 1e-10),
        len(set(elements)),  # 元素种类
        np.mean([ELEMENT_PROPERTIES.get(e, DEFAULT_PROPS)[3] for e in elements]),  # 平均电负性
        np.std([ELEMENT_PROPERTIES.get(e, DEFAULT_PROPS)[3] for e in elements]) if len(elements) > 1 else 0,
        np.max([ELEMENT_PROPERTIES.get(e, DEFAULT_PROPS)[3] for e in elements]) - 
        np.min([ELEMENT_PROPERTIES.get(e, DEFAULT_PROPS)[3] for e in elements]),  # 电负性差
    ]
    
    # 补充到32维
    while len(global_features) < 32:
        global_features.append(0.0)
    
    global_features = np.array(global_features[:32], dtype=np.float32)
    
    return CrystalGraph(node_features, edge_index, edge_attr, global_features)


# ============================================================================
# 图神经网络模型
# ============================================================================

class MessagePassingLayer(nn.Module):
    """消息传递层"""
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        # 消息网络
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # 更新网络
        self.update_net = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # 残差连接
        self.residual = nn.Linear(node_dim, node_dim) if node_dim != hidden_dim else nn.Identity()
        
    def forward(self, node_features, edge_index, edge_attr):
        """
        node_features: (n_nodes, node_dim)
        edge_index: (2, n_edges)
        edge_attr: (n_edges, edge_dim)
        """
        n_nodes = node_features.size(0)
        
        if edge_index.size(1) == 0:
            # 没有边，直接返回
            return node_features
        
        src, dst = edge_index[0], edge_index[1]
        
        # 构建消息
        src_features = node_features[src]
        dst_features = node_features[dst]
        messages = self.message_net(torch.cat([src_features, dst_features, edge_attr], dim=-1))
        
        # 聚合消息 (求和)
        aggregated = torch.zeros(n_nodes, messages.size(-1), device=node_features.device)
        aggregated.index_add_(0, dst, messages)
        
        # 更新节点
        updated = self.update_net(torch.cat([node_features, aggregated], dim=-1))
        
        # 残差连接
        return updated + self.residual(node_features)


class GlobalAttentionPool(nn.Module):
    """全局注意力池化"""
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.transform = nn.Linear(node_dim, hidden_dim)
        
    def forward(self, node_features, batch_idx=None):
        """
        node_features: (n_nodes, node_dim)
        batch_idx: (n_nodes,) - 批次索引，如果是单个图则为None
        """
        # 计算注意力权重
        attn_weights = self.attention(node_features)
        attn_weights = F.softmax(attn_weights, dim=0)
        
        # 加权求和
        transformed = self.transform(node_features)
        pooled = (attn_weights * transformed).sum(dim=0)
        
        return pooled


class CrystalGraphNN(nn.Module):
    """晶体图神经网络分类器"""
    def __init__(self, node_dim: int = 32, edge_dim: int = 16, global_dim: int = 32,
                 hidden_dim: int = 128, num_mp_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        
        # 节点嵌入
        self.node_embed = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # 边嵌入
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim // 2),
            nn.SiLU()
        )
        
        # 消息传递层
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, hidden_dim // 2, hidden_dim)
            for _ in range(num_mp_layers)
        ])
        
        # 全局注意力池化
        self.pool = GlobalAttentionPool(hidden_dim, hidden_dim)
        
        # 全局特征处理
        self.global_net = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 投影头 (对比学习)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 64)
        )
        
    def forward(self, graph: CrystalGraph, return_features: bool = False):
        # 转换为tensor
        node_features = torch.FloatTensor(graph.node_features).to(next(self.parameters()).device)
        edge_index = torch.LongTensor(graph.edge_index).to(next(self.parameters()).device)
        edge_attr = torch.FloatTensor(graph.edge_attr).to(next(self.parameters()).device)
        global_features = torch.FloatTensor(graph.global_features).to(next(self.parameters()).device)
        
        # 节点嵌入
        h = self.node_embed(node_features)
        
        # 边嵌入
        if edge_attr.size(0) > 0:
            e = self.edge_embed(edge_attr)
        else:
            e = torch.zeros((0, self.edge_embed[0].out_features), device=h.device)
        
        # 消息传递
        for mp_layer in self.mp_layers:
            h = mp_layer(h, edge_index, e)
        
        # 全局池化
        graph_repr = self.pool(h)
        
        # 全局特征
        global_repr = self.global_net(global_features)
        
        # 融合
        combined = torch.cat([graph_repr, global_repr], dim=-1)
        
        # 分类
        logits = self.classifier(combined)
        
        if return_features:
            proj = self.projection(combined)
            return logits, proj
        
        return logits


# ============================================================================
# 损失函数
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: float = 30.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha * self.pos_weight, 1 - self.alpha)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        return (focal_weight * alpha_weight * bce).mean()


# ============================================================================
# 数据加载
# ============================================================================

def load_graphs(data_dir: str, max_samples: int = None) -> Tuple[List[CrystalGraph], np.ndarray]:
    """加载数据并转换为图"""
    print("\n" + "=" * 70)
    print("加载数据并构建图表示")
    print("=" * 70)
    
    seen_ids = set()
    
    def get_structure_id(structure: Dict) -> str:
        s = json.dumps(structure, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()
    
    # 文件列表
    files = [
        ('dataset_original_ferroelectric.jsonl', 1),
        ('dataset_known_FE_rest.jsonl', 1),
        ('dataset_nonFE.jsonl', 0),
        ('dataset_nonFE_cleaned.jsonl', 0),
        ('dataset_nonFE_expanded.jsonl', 0),
    ]
    
    all_graphs = []
    all_labels = []
    
    for filename, label in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            continue
        
        with open(filepath, 'r') as f:
            structures = [json.loads(line) for line in f]
        
        if max_samples and label == 0:
            # 限制负样本数量以加快测试
            structures = structures[:min(len(structures), max_samples)]
        
        unique_count = 0
        for s in tqdm(structures, desc=f"处理 {filename}"):
            struct = s.get('structure', s)
            sid = get_structure_id(struct)
            
            if sid not in seen_ids:
                seen_ids.add(sid)
                graph = structure_to_graph(struct)
                all_graphs.append(graph)
                all_labels.append(label)
                unique_count += 1
        
        label_str = "铁电" if label == 1 else "非铁电"
        print(f"  {filename}: {unique_count} 个唯一{label_str}样本")
    
    labels = np.array(all_labels)
    print(f"\n数据集统计:")
    print(f"  正样本 (FE): {np.sum(labels == 1)}")
    print(f"  负样本 (non-FE): {np.sum(labels == 0)}")
    
    return all_graphs, labels


# ============================================================================
# 训练和评估
# ============================================================================

def train_epoch(model, graphs, labels, optimizer, focal_loss, device, 
                pos_indices, neg_indices, batch_size: int = 32):
    """训练一个epoch (平衡采样)"""
    model.train()
    total_loss = 0
    n_batches = 0
    
    # 平衡采样
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    # 每个batch中正负样本1:1
    half_batch = batch_size // 2
    n_batches_per_epoch = max(1, n_pos // half_batch)
    
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)
    
    for i in range(n_batches_per_epoch):
        # 采样
        pos_batch = pos_indices[i * half_batch: (i + 1) * half_batch]
        neg_batch = neg_indices[np.random.choice(n_neg, half_batch, replace=False)]
        batch_indices = np.concatenate([pos_batch, neg_batch])
        np.random.shuffle(batch_indices)
        
        batch_loss = 0
        for idx in batch_indices:
            graph = graphs[idx]
            label = torch.FloatTensor([labels[idx]]).to(device)
            
            optimizer.zero_grad()
            logits = model(graph)
            loss = focal_loss(logits, label)
            loss.backward()
            optimizer.step()
            
            batch_loss += loss.item()
        
        total_loss += batch_loss / len(batch_indices)
        n_batches += 1
    
    return total_loss / max(1, n_batches)


def evaluate(model, graphs, labels, device, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    """评估模型"""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for graph in tqdm(graphs, desc="评估中", leave=False):
            logits = model(graph)
            prob = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(prob.item())
    
    all_probs = np.array(all_probs)
    
    try:
        roc_auc = roc_auc_score(labels, all_probs)
    except:
        roc_auc = 0.5
    
    results = {'roc_auc': roc_auc}
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        preds = (all_probs >= thresh).astype(int)
        acc = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        results[f'thresh_{thresh}'] = {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    results['best_threshold'] = best_threshold
    results['best_f1'] = best_f1
    
    return results, all_probs


def cross_validate(graphs: List[CrystalGraph], labels: np.ndarray, 
                   n_folds: int = 5, epochs: int = 50):
    """交叉验证"""
    print("\n" + "=" * 70)
    print("5折交叉验证训练")
    print("=" * 70)
    
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    all_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.zeros(len(labels)), labels)):
        print(f"\n{'=' * 70}")
        print(f"Fold {fold + 1}/{n_folds}")
        print("=" * 70)
        
        train_graphs = [graphs[i] for i in train_idx]
        val_graphs = [graphs[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        pos_indices = np.where(train_labels == 1)[0]
        neg_indices = np.where(train_labels == 0)[0]
        
        print(f"训练集: {len(train_graphs)} (正:{len(pos_indices)}, 负:{len(neg_indices)})")
        print(f"验证集: {len(val_graphs)}")
        
        # 初始化模型
        model = CrystalGraphNN(
            node_dim=32, edge_dim=16, global_dim=32,
            hidden_dim=128, num_mp_layers=4, dropout=0.3
        ).to(DEVICE)
        
        # 损失函数
        pos_weight = len(neg_indices) / len(pos_indices)
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=pos_weight)
        
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # 训练
        best_auc = 0
        best_state = None
        patience = 10
        patience_counter = 0
        
        print("训练中...")
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_graphs, train_labels, optimizer, 
                                    focal_loss, DEVICE, pos_indices.copy(), neg_indices.copy())
            scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                results, _ = evaluate(model, val_graphs, val_labels, DEVICE)
                print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, AUC={results['roc_auc']:.4f}")
                
                if results['roc_auc'] > best_auc:
                    best_auc = results['roc_auc']
                    best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"  早停 at epoch {epoch+1}")
                    break
        
        if best_state:
            model.load_state_dict(best_state)
        
        # 最终评估
        results, _ = evaluate(model, val_graphs, val_labels, DEVICE,
                             thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
        
        print(f"\nFold {fold + 1} 结果:")
        print(f"  ROC-AUC: {results['roc_auc']:.4f}")
        for thresh in [0.1, 0.2, 0.3]:
            if f'thresh_{thresh}' in results:
                r = results[f'thresh_{thresh}']
                print(f"  阈值{thresh}: Acc={r['accuracy']:.4f}, Recall={r['recall']:.4f}")
        
        all_results.append(results)
    
    return all_results


def main():
    print("=" * 70)
    print("图神经网络铁电分类器")
    print("目标: 让模型自己学习晶体结构的高维表示")
    print("=" * 70)
    
    data_dir = '/home/ubuntu/ai_wh/wh-ai/new_data'
    report_dir = '/home/ubuntu/ai_wh/wh-ai/reports_nequip_v6'
    os.makedirs(report_dir, exist_ok=True)
    
    # 加载数据 (限制样本数以加快测试)
    graphs, labels = load_graphs(data_dir, max_samples=5000)
    
    # 交叉验证
    results = cross_validate(graphs, labels, n_folds=5, epochs=50)
    
    # 汇总
    print("\n" + "=" * 70)
    print("交叉验证总结")
    print("=" * 70)
    
    auc_scores = [r['roc_auc'] for r in results]
    print(f"ROC-AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    
    # 保存结果
    results_df = pd.DataFrame([
        {'fold': i+1, 'roc_auc': r['roc_auc'], 'best_threshold': r['best_threshold']}
        for i, r in enumerate(results)
    ])
    results_df.to_csv(os.path.join(report_dir, 'cv_results_gnn.csv'), index=False)
    
    print(f"\n结果已保存到: {report_dir}/cv_results_gnn.csv")
    print("完成!")


if __name__ == '__main__':
    main()
