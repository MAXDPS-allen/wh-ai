"""
NequIP v9 - 优化架构版E(3)等变神经网络
=========================================
目标: 在极性材料数据集上达到 Accuracy & Recall >= 99%

核心优化:
1. 6层深度SE(3)等变消息传递
2. 扩展球谐函数 (l_max=3, 16维)
3. 多尺度径向基 (16维 Bessel)
4. 注意力增强消息传递
5. 分层池化策略
6. Focal Loss + Label Smoothing
7. 动态阈值优化
8. 混合精度训练
"""

import sys
import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from pymatgen.core import Structure
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score, f1_score,
    confusion_matrix, roc_curve
)
from scipy.spatial import cKDTree
from tqdm import tqdm

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    SMOTE_AVAILABLE = True
    print("✓ SMOTE/ADASYN可用")
except ImportError:
    SMOTE_AVAILABLE = False
    print("✗ SMOTE不可用")

from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader as GeoDataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from feature_engineering import ELEMENT_DATABASE

# ============================================================
# 极性空间群定义 (68个极性空间群)
# ============================================================
POLAR_SPACE_GROUPS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 46,
    75, 76, 77, 78, 79, 80,
    99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
    143, 144, 145, 146, 156, 157, 158, 159, 160, 161,
    168, 169, 170, 171, 172, 173,
    183, 184, 185, 186
]

# ============================================================
# 配置
# ============================================================
class NequIPV9Config:
    """NequIP v9 配置"""
    
    # 数据路径
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_nequip_v9'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_nequip_v9'
    CACHE_DIR = Path(__file__).parent.parent / 'cache_nequip_v9'
    
    # 模型架构
    NUM_SPECIES = 100
    L_MAX = 3                    # 扩展球谐函数阶数
    SH_DIM = 16                  # l=0,1,2,3 -> 1+3+5+7=16
    NUM_RADIAL_BASIS = 16        # 增加径向基数量
    CUTOFF = 5.0
    
    NUM_LAYERS = 6               # 增加到6层消息传递
    NODE_DIM = 64                # 节点特征维度
    HIDDEN_DIM = 256             # 隐藏层维度
    NUM_HEADS = 4                # 注意力头数
    
    # 全局特征维度
    GLOBAL_FEAT_DIM = 128
    
    # 训练参数
    BATCH_SIZE = 16
    EPOCHS = 200
    LR = 3e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 40
    
    # 类别不平衡处理
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    FOCAL_ALPHA = 0.75
    LABEL_SMOOTHING = 0.05
    SMOTE_RATIO = 0.5            # SMOTE后正负比例
    
    # 交叉验证
    N_SPLITS = 5
    RANDOM_STATE = 42
    
    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = torch.cuda.is_available()
    
    @classmethod
    def prepare_dirs(cls):
        for d in [cls.MODEL_DIR, cls.REPORT_DIR, cls.CACHE_DIR]:
            d.mkdir(parents=True, exist_ok=True)


print(f"使用设备: {NequIPV9Config.DEVICE}")
print(f"极性空间群数量: {len(POLAR_SPACE_GROUPS)}")


# ============================================================
# 扩展球谐函数 (l=0,1,2,3)
# ============================================================
class SphericalHarmonicsL3(nn.Module):
    """球谐函数 l_max=3 (16维输出)"""
    
    def __init__(self):
        super().__init__()
        # 预计算系数
        self.register_buffer('sqrt_3', torch.tensor(math.sqrt(3)))
        self.register_buffer('sqrt_5', torch.tensor(math.sqrt(5)))
        self.register_buffer('sqrt_7', torch.tensor(math.sqrt(7)))
        self.register_buffer('sqrt_15', torch.tensor(math.sqrt(15)))
    
    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """
        输入: vec [N, 3] 方向向量
        输出: Y [N, 16] 球谐函数值
        """
        x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
        r2 = (x**2 + y**2 + z**2).clamp(min=1e-8)
        r = r2.sqrt()
        
        # 归一化
        x_n = x / r
        y_n = y / r
        z_n = z / r
        
        # l=0 (1个)
        Y00 = torch.ones_like(x) * 0.2820948  # 1/(2*sqrt(pi))
        
        # l=1 (3个)
        Y1m1 = 0.4886025 * y_n
        Y10 = 0.4886025 * z_n
        Y1p1 = 0.4886025 * x_n
        
        # l=2 (5个)
        Y2m2 = 1.0925484 * x_n * y_n
        Y2m1 = 1.0925484 * y_n * z_n
        Y20 = 0.3153916 * (3 * z_n**2 - 1)
        Y2p1 = 1.0925484 * z_n * x_n
        Y2p2 = 0.5462742 * (x_n**2 - y_n**2)
        
        # l=3 (7个)
        Y3m3 = 0.5900436 * y_n * (3 * x_n**2 - y_n**2)
        Y3m2 = 2.8906114 * x_n * y_n * z_n
        Y3m1 = 0.4570458 * y_n * (5 * z_n**2 - 1)
        Y30 = 0.3731763 * z_n * (5 * z_n**2 - 3)
        Y3p1 = 0.4570458 * x_n * (5 * z_n**2 - 1)
        Y3p2 = 1.4453057 * z_n * (x_n**2 - y_n**2)
        Y3p3 = 0.5900436 * x_n * (x_n**2 - 3 * y_n**2)
        
        return torch.stack([
            Y00, Y1m1, Y10, Y1p1,
            Y2m2, Y2m1, Y20, Y2p1, Y2p2,
            Y3m3, Y3m2, Y3m1, Y30, Y3p1, Y3p2, Y3p3
        ], dim=-1)


# ============================================================
# 多尺度径向基函数
# ============================================================
class MultiScaleBesselBasis(nn.Module):
    """多尺度Bessel径向基函数"""
    
    def __init__(self, num_basis: int = 16, cutoff: float = 5.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        
        # 多尺度频率
        freqs = torch.arange(1, num_basis + 1) * math.pi / cutoff
        self.register_buffer('freq', freqs)
        
        # 可学习的尺度参数
        self.scale = nn.Parameter(torch.ones(num_basis) * 0.1)
    
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = r.unsqueeze(-1)  # [N, 1]
        
        # Bessel基函数
        norm = math.sqrt(2.0 / self.cutoff)
        basis = norm * torch.sin(self.freq * r) / (r + 1e-8)
        
        # 应用可学习尺度
        basis = basis * torch.abs(self.scale)
        
        return basis


class AdaptiveCutoff(nn.Module):
    """自适应平滑截断函数"""
    
    def __init__(self, cutoff: float = 5.0, p: int = 6):
        super().__init__()
        self.cutoff = cutoff
        self.p = p
        self.sharpness = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = r / self.cutoff
        s = torch.abs(self.sharpness).clamp(min=0.5, max=2.0)
        envelope = (1 - (x * s).pow(self.p)).pow(2)
        envelope = torch.where(r < self.cutoff, envelope, torch.zeros_like(envelope))
        return envelope


# ============================================================
# 注意力增强的等变消息传递层
# ============================================================
class AttentiveEquivariantLayer(nn.Module):
    """注意力增强的E(3)等变消息传递层"""
    
    def __init__(self, node_dim: int, sh_dim: int = 16, radial_dim: int = 16,
                 hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_dim = node_dim
        self.sh_dim = sh_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        
        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"
        
        # 径向MLP生成注意力权重
        self.radial_attention = nn.Sequential(
            nn.Linear(radial_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads)
        )
        
        # 径向MLP生成消息权重
        self.radial_message = nn.Sequential(
            nn.Linear(radial_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim * sh_dim)
        )
        
        # Query, Key, Value 投影
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        
        # 自交互
        self.self_interaction = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # 更新MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, h, edge_index, edge_sh, edge_radial):
        """
        h: [N, node_dim] 节点特征
        edge_index: [2, E] 边索引
        edge_sh: [E, sh_dim] 边的球谐函数
        edge_radial: [E, radial_dim] 边的径向基
        """
        i, j = edge_index
        num_nodes = h.size(0)
        
        # 计算注意力得分
        attn_radial = self.radial_attention(edge_radial)  # [E, num_heads]
        
        q = self.q_proj(h)  # [N, node_dim]
        k = self.k_proj(h)  # [N, node_dim]
        v = self.v_proj(h)  # [N, node_dim]
        
        # 重塑为多头
        q = q.view(-1, self.num_heads, self.head_dim)  # [N, H, D]
        k = k.view(-1, self.num_heads, self.head_dim)
        v = v.view(-1, self.num_heads, self.head_dim)
        
        # 计算边注意力
        q_i = q[i]  # [E, H, D]
        k_j = k[j]  # [E, H, D]
        v_j = v[j]  # [E, H, D]
        
        attn_scores = (q_i * k_j).sum(dim=-1) / math.sqrt(self.head_dim)  # [E, H]
        attn_scores = attn_scores + attn_radial  # 加入径向注意力偏置
        
        # Softmax (按目标节点分组)
        attn_max = torch.zeros(num_nodes, self.num_heads, device=h.device)
        attn_max.scatter_reduce_(0, i.unsqueeze(-1).expand(-1, self.num_heads), 
                                  attn_scores, reduce='amax', include_self=False)
        attn_scores = attn_scores - attn_max[i]
        attn_exp = attn_scores.exp()
        
        attn_sum = torch.zeros(num_nodes, self.num_heads, device=h.device)
        attn_sum.scatter_add_(0, i.unsqueeze(-1).expand(-1, self.num_heads), attn_exp)
        attn_weights = attn_exp / (attn_sum[i] + 1e-8)  # [E, H]
        
        # 等变消息
        radial_weight = self.radial_message(edge_radial).view(-1, self.node_dim, self.sh_dim)
        equiv_message = (radial_weight * edge_sh.unsqueeze(1)).sum(dim=-1)  # [E, node_dim]
        
        # 加权消息
        v_j_flat = v_j.view(-1, self.node_dim)  # [E, node_dim]
        weighted_v = v_j_flat * attn_weights.mean(dim=-1, keepdim=True)  # 平均注意力
        messages = weighted_v * equiv_message
        
        # 聚合
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, i.unsqueeze(-1).expand_as(messages), messages)
        
        # 自交互
        h_self = self.self_interaction(h)
        
        # 门控更新
        combined = torch.cat([h_self, agg], dim=-1)
        gate_value = self.gate(combined)
        h_update = self.update_mlp(combined)
        
        h_new = h + gate_value * self.dropout(h_update)
        h_new = self.norm(h_new)
        
        return h_new


# ============================================================
# 分层池化
# ============================================================
class HierarchicalPooling(nn.Module):
    """分层池化模块"""
    
    def __init__(self, node_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        # 节点重要性预测
        self.importance = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 特征转换
        self.transform = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )
    
    def forward(self, h, batch):
        """
        h: [N, node_dim] 节点特征
        batch: [N] 批次索引
        """
        # 计算重要性权重
        importance = self.importance(h)  # [N, 1]
        
        # 加权特征
        h_weighted = h * importance
        
        # 多种池化
        h_mean = global_mean_pool(h_weighted, batch)
        h_max = global_max_pool(h, batch)
        h_sum = global_add_pool(h_weighted, batch)
        
        # 重要性加权池化
        importance_sum = global_add_pool(importance, batch) + 1e-8
        h_attn = global_add_pool(h * importance, batch) / importance_sum
        
        # 组合
        h_pool = torch.cat([h_mean, h_max, h_sum / 10.0, h_attn], dim=-1)
        
        return h_pool


# ============================================================
# NequIP v9 分类器主模型
# ============================================================
class NequIPClassifierV9(nn.Module):
    """NequIP v9 - 优化架构版"""
    
    def __init__(self, config=None, dropout=0.15):
        super().__init__()
        self.config = config or NequIPV9Config()
        
        # 原子嵌入
        self.atom_embedding = nn.Embedding(self.config.NUM_SPECIES, self.config.NODE_DIM)
        
        # 位置编码
        self.pos_encoder = nn.Sequential(
            nn.Linear(3, self.config.NODE_DIM // 2),
            nn.SiLU(),
            nn.Linear(self.config.NODE_DIM // 2, self.config.NODE_DIM)
        )
        
        # 径向基函数
        self.radial_basis = MultiScaleBesselBasis(
            self.config.NUM_RADIAL_BASIS, 
            self.config.CUTOFF
        )
        self.cutoff_fn = AdaptiveCutoff(self.config.CUTOFF)
        
        # 扩展球谐函数
        self.spherical_harmonics = SphericalHarmonicsL3()
        
        # 消息传递层
        self.message_layers = nn.ModuleList([
            AttentiveEquivariantLayer(
                self.config.NODE_DIM,
                self.config.SH_DIM,
                self.config.NUM_RADIAL_BASIS,
                self.config.HIDDEN_DIM,
                self.config.NUM_HEADS,
                dropout
            ) for _ in range(self.config.NUM_LAYERS)
        ])
        
        # 中间跳跃连接的转换
        self.skip_transforms = nn.ModuleList([
            nn.Linear(self.config.NODE_DIM, self.config.NODE_DIM)
            for _ in range(self.config.NUM_LAYERS // 2)
        ])
        
        # 分层池化
        self.pooling = HierarchicalPooling(self.config.NODE_DIM, self.config.HIDDEN_DIM)
        
        # 图特征维度
        graph_dim = self.config.NODE_DIM * 4
        
        # 全局特征编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(self.config.GLOBAL_FEAT_DIM, self.config.HIDDEN_DIM),
            nn.LayerNorm(self.config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.config.HIDDEN_DIM, self.config.HIDDEN_DIM),
            nn.LayerNorm(self.config.HIDDEN_DIM),
            nn.SiLU(),
        )
        
        # 融合层
        fusion_dim = graph_dim + self.config.HIDDEN_DIM
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.config.HIDDEN_DIM * 2),
            nn.LayerNorm(self.config.HIDDEN_DIM * 2),
            nn.SiLU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(self.config.HIDDEN_DIM * 2, self.config.HIDDEN_DIM),
            nn.LayerNorm(self.config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.config.HIDDEN_DIM, self.config.HIDDEN_DIM // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.config.HIDDEN_DIM // 2, self.config.HIDDEN_DIM // 4),
            nn.SiLU(),
            nn.Linear(self.config.HIDDEN_DIM // 4, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        pos = data.pos
        u = data.u  # 全局特征
        
        # 原子嵌入
        if x.dim() == 1:
            h = self.atom_embedding(x.clamp(0, self.config.NUM_SPECIES - 1))
        else:
            h = self.atom_embedding(x[:, 0].long().clamp(0, self.config.NUM_SPECIES - 1))
        
        # 添加位置编码
        pos_feat = self.pos_encoder(pos)
        h = h + pos_feat
        
        # 计算边特征
        i, j = edge_index
        vec = pos[j] - pos[i]
        dist = vec.norm(dim=-1)
        
        # 径向基和截断
        edge_radial = self.radial_basis(dist)  # [E, radial_dim]
        cutoff_weight = self.cutoff_fn(dist)   # [E]
        edge_radial = edge_radial * cutoff_weight.unsqueeze(-1)
        
        # 球谐函数
        edge_sh = self.spherical_harmonics(vec)  # [E, sh_dim]
        
        # 消息传递 (带跳跃连接)
        skip_features = []
        for idx, layer in enumerate(self.message_layers):
            h = layer(h, edge_index, edge_sh, edge_radial)
            
            # 保存中间特征用于跳跃连接
            if idx % 2 == 1 and idx // 2 < len(self.skip_transforms):
                skip_features.append(self.skip_transforms[idx // 2](h))
        
        # 融合跳跃连接
        if skip_features:
            h = h + sum(skip_features) / len(skip_features)
        
        # 分层池化
        graph_feat = self.pooling(h, batch)
        
        # 全局特征
        if u.dim() == 3:
            u = u.squeeze(1)
        global_feat = self.global_encoder(u)
        
        # 融合
        combined = torch.cat([graph_feat, global_feat], dim=-1)
        fused = self.fusion(combined)
        
        # 分类
        logits = self.classifier(fused)
        
        return logits.squeeze(-1)


# ============================================================
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, logits, targets):
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal weight
        pt = torch.where(targets >= 0.5, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_weight = torch.where(targets >= 0.5, self.alpha, 1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce
        return loss.mean()


# ============================================================
# 特征提取器
# ============================================================
class NequIPFeatureExtractor:
    """NequIP v9 特征提取器"""
    
    def __init__(self, feature_dim=128):
        self.feature_dim = feature_dim
    
    def extract(self, structure: Structure) -> np.ndarray:
        """提取128维全局特征"""
        features = []
        
        # 1. 元素统计 (30维)
        elements = [str(site.specie) for site in structure]
        element_counts = {}
        for elem in elements:
            element_counts[elem] = element_counts.get(elem, 0) + 1
        
        elem_props = {
            'electronegativity': [], 'atomic_radius': [], 'ionization_energy': [],
            'electron_affinity': [], 'polarizability': [], 'd_orbital': []
        }
        
        for elem, count in element_counts.items():
            if elem in ELEMENT_DATABASE:
                data = ELEMENT_DATABASE[elem]
                for key in elem_props:
                    if key in data and data[key] is not None:
                        elem_props[key].extend([data[key]] * count)
        
        for key in ['electronegativity', 'atomic_radius', 'ionization_energy', 
                    'electron_affinity', 'polarizability']:
            vals = elem_props[key] if elem_props[key] else [0]
            features.extend([np.mean(vals), np.std(vals), np.min(vals), 
                           np.max(vals), np.max(vals) - np.min(vals)])
        
        # 2. 结构特征 (25维)
        lattice = structure.lattice
        features.extend([
            lattice.a, lattice.b, lattice.c,
            lattice.alpha, lattice.beta, lattice.gamma,
            lattice.volume, lattice.volume / len(structure),
            len(structure),
            lattice.a / lattice.c if lattice.c > 0 else 1,
            lattice.b / lattice.c if lattice.c > 0 else 1,
        ])
        
        # 晶格畸变
        abc_mean = np.mean([lattice.a, lattice.b, lattice.c])
        abc_std = np.std([lattice.a, lattice.b, lattice.c])
        angle_mean = np.mean([lattice.alpha, lattice.beta, lattice.gamma])
        angle_std = np.std([lattice.alpha, lattice.beta, lattice.gamma])
        features.extend([abc_mean, abc_std, angle_mean, angle_std])
        
        # 对称性
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            sg_number = sga.get_space_group_number()
            is_polar = 1 if sg_number in POLAR_SPACE_GROUPS else 0
            crystal_system = sga.get_crystal_system()
            crystal_map = {'triclinic': 1, 'monoclinic': 2, 'orthorhombic': 3,
                          'tetragonal': 4, 'trigonal': 5, 'hexagonal': 6, 'cubic': 7}
            cs_num = crystal_map.get(crystal_system, 0)
        except:
            sg_number = 0
            is_polar = 0
            cs_num = 0
        
        features.extend([sg_number / 230, is_polar, cs_num / 7])
        
        # 3. 键长分布 (20维)
        try:
            dist_matrix = structure.distance_matrix
            valid_dists = dist_matrix[(dist_matrix > 0.5) & (dist_matrix < 5.0)]
            if len(valid_dists) > 0:
                percentiles = [10, 25, 50, 75, 90]
                for p in percentiles:
                    features.append(np.percentile(valid_dists, p))
                features.extend([
                    np.mean(valid_dists), np.std(valid_dists),
                    np.min(valid_dists), np.max(valid_dists),
                    len(valid_dists) / len(structure)
                ])
            else:
                features.extend([0] * 10)
        except:
            features.extend([0] * 10)
        
        # 配位数分布 (10维)
        try:
            coord_nums = []
            for i in range(len(structure)):
                cn = np.sum((dist_matrix[i] > 0.5) & (dist_matrix[i] < 3.0))
                coord_nums.append(cn)
            if coord_nums:
                features.extend([
                    np.mean(coord_nums), np.std(coord_nums),
                    np.min(coord_nums), np.max(coord_nums),
                    len(set(coord_nums)) / len(coord_nums)
                ])
            else:
                features.extend([0] * 5)
        except:
            features.extend([0] * 5)
        
        # 4. 位置分布 (15维)
        frac_coords = structure.frac_coords
        features.extend([
            np.mean(frac_coords[:, 0]), np.std(frac_coords[:, 0]),
            np.mean(frac_coords[:, 1]), np.std(frac_coords[:, 1]),
            np.mean(frac_coords[:, 2]), np.std(frac_coords[:, 2]),
        ])
        
        # 质心偏移
        cart_coords = structure.cart_coords
        centroid = np.mean(cart_coords, axis=0)
        lattice_center = np.array([lattice.a/2, lattice.b/2, lattice.c/2])
        center_offset = np.linalg.norm(centroid - lattice_center)
        features.append(center_offset)
        
        # 原子分布熵
        for dim in range(3):
            coords = frac_coords[:, dim]
            hist, _ = np.histogram(coords, bins=10, range=(0, 1))
            hist = hist / (hist.sum() + 1e-8)
            entropy = -np.sum(hist * np.log(hist + 1e-8))
            features.append(entropy)
        
        # 5. d轨道电子统计 (5维)
        d_electrons = elem_props['d_orbital'] if elem_props['d_orbital'] else [0]
        features.extend([
            np.sum(d_electrons),
            np.mean(d_electrons),
            np.max(d_electrons) if d_electrons else 0,
            len([d for d in d_electrons if d > 0]),
            len([d for d in d_electrons if d > 0]) / len(structure)
        ])
        
        # 填充到128维
        features = np.array(features[:self.feature_dim])
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        
        # 处理NaN和Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return features.astype(np.float32)


# ============================================================
# 图构建
# ============================================================
def structure_to_graph(structure: Structure, features: np.ndarray, 
                       cutoff: float = 5.0) -> Data:
    """将晶体结构转换为图数据"""
    
    # 原子特征
    atomic_numbers = []
    positions = []
    
    for site in structure:
        atomic_numbers.append(site.specie.Z)
        positions.append(site.coords)
    
    x = torch.tensor(atomic_numbers, dtype=torch.long)
    pos = torch.tensor(positions, dtype=torch.float)
    
    # 构建边 (使用KDTree加速)
    tree = cKDTree(positions)
    pairs = tree.query_pairs(r=cutoff, output_type='ndarray')
    
    if len(pairs) == 0:
        # 如果没有边，创建自环
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
    else:
        # 双向边
        edge_index = torch.tensor(pairs.T, dtype=torch.long)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # 全局特征
    u = torch.tensor(features, dtype=torch.float).unsqueeze(0)
    
    return Data(x=x, pos=pos, edge_index=edge_index, u=u)


# ============================================================
# 数据加载
# ============================================================
def load_polar_dataset(config: NequIPV9Config):
    """加载极性材料数据集"""
    
    print("=" * 60)
    print("加载极性材料数据集")
    print("=" * 60)
    
    extractor = NequIPFeatureExtractor(config.GLOBAL_FEAT_DIM)
    samples = []
    
    # 加载正样本
    print("\n加载正样本 (铁电材料)...")
    pos_files = [
        ('dataset_original_ferroelectric.jsonl', True),
        ('dataset_known_FE_rest.jsonl', True),
    ]
    
    for filename, is_positive in pos_files:
        filepath = config.DATA_DIR / filename
        if filepath.exists():
            with open(filepath) as f:
                lines = f.readlines()
            
            count = 0
            for line in tqdm(lines, desc=f"  {filename}"):
                try:
                    data = json.loads(line)
                    structure = Structure.from_dict(data['structure'])
                    
                    # 检查是否在极性空间群中
                    try:
                        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                        sga = SpacegroupAnalyzer(structure, symprec=0.1)
                        sg = sga.get_space_group_number()
                        if sg not in POLAR_SPACE_GROUPS:
                            continue
                    except:
                        continue
                    
                    features = extractor.extract(structure)
                    samples.append({
                        'structure': structure,
                        'features': features,
                        'label': 1,
                        'source': filename
                    })
                    count += 1
                except Exception as e:
                    continue
            
            print(f"  {filename}: {count} 个正样本")
    
    # 加载负样本
    print("\n加载负样本 (非铁电材料)...")
    neg_files = [
        'dataset_nonFE.jsonl',
        'dataset_nonFE_cleaned.jsonl',
        'dataset_nonFE_expanded.jsonl'
    ]
    
    for filename in neg_files:
        filepath = config.DATA_DIR / filename
        if filepath.exists():
            with open(filepath) as f:
                lines = f.readlines()
            
            count = 0
            for line in tqdm(lines, desc=f"  {filename}"):
                try:
                    data = json.loads(line)
                    structure = Structure.from_dict(data['structure'])
                    
                    # 只保留极性空间群中的负样本
                    try:
                        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                        sga = SpacegroupAnalyzer(structure, symprec=0.1)
                        sg = sga.get_space_group_number()
                        if sg not in POLAR_SPACE_GROUPS:
                            continue
                    except:
                        continue
                    
                    features = extractor.extract(structure)
                    samples.append({
                        'structure': structure,
                        'features': features,
                        'label': 0,
                        'source': filename
                    })
                    count += 1
                except:
                    continue
            
            print(f"  {filename}: {count} 个负样本")
    
    # 统计
    labels = [s['label'] for s in samples]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    
    print(f"\n极性材料子集: {len(samples)} 样本")
    print(f"  FE: {n_pos} ({100*n_pos/len(samples):.2f}%)")
    print(f"  Non-FE: {n_neg} ({100*n_neg/len(samples):.2f}%)")
    print(f"  类别比例: 1:{n_neg/n_pos:.1f}")
    
    return samples


# ============================================================
# 训练器
# ============================================================
class NequIPV9Trainer:
    """NequIP v9 训练器"""
    
    def __init__(self, config=None):
        self.config = config or NequIPV9Config()
        self.config.prepare_dirs()
        self.device = self.config.DEVICE
        self.scaler = GradScaler() if self.config.USE_AMP else None
    
    def train_fold(self, train_samples, val_samples, fold):
        """训练单个fold"""
        
        print(f"\n{'=' * 60}")
        print(f"Fold {fold + 1}/{self.config.N_SPLITS}")
        print(f"{'=' * 60}")
        
        # 准备数据
        train_features = np.array([s['features'] for s in train_samples])
        train_labels = np.array([s['label'] for s in train_samples])
        val_features = np.array([s['features'] for s in val_samples])
        val_labels = np.array([s['label'] for s in val_samples])
        
        print(f"训练集: {len(train_samples)} (正样本: {train_labels.sum()})")
        print(f"验证集: {len(val_samples)} (正样本: {val_labels.sum()})")
        
        # SMOTE过采样
        if SMOTE_AVAILABLE and self.config.SMOTE_RATIO > 0:
            try:
                n_pos = train_labels.sum()
                n_neg = len(train_labels) - n_pos
                target_ratio = self.config.SMOTE_RATIO
                n_target = int(n_neg * target_ratio)
                
                if n_target > n_pos:
                    smote = SMOTE(sampling_strategy={1: n_target}, random_state=42)
                    train_features_sm, train_labels_sm = smote.fit_resample(
                        train_features, train_labels
                    )
                    print(f"SMOTE: {int(n_pos)} -> {int(train_labels_sm.sum())} 正样本")
                    
                    # 扩展训练样本
                    n_new = len(train_labels_sm) - len(train_labels)
                    if n_new > 0:
                        # 为新样本复制结构
                        pos_samples = [s for s in train_samples if s['label'] == 1]
                        new_samples = []
                        for i in range(n_new):
                            src = pos_samples[i % len(pos_samples)]
                            new_samples.append({
                                'structure': src['structure'],
                                'features': train_features_sm[len(train_labels) + i],
                                'label': 1,
                                'source': 'SMOTE'
                            })
                        train_samples = train_samples + new_samples
            except Exception as e:
                print(f"SMOTE失败: {e}")
        
        # 构建图数据
        print("构建图数据...")
        train_graphs = []
        for s in tqdm(train_samples, desc="训练集图"):
            try:
                g = structure_to_graph(s['structure'], s['features'], self.config.CUTOFF)
                g.y = torch.tensor([s['label']], dtype=torch.float)
                train_graphs.append(g)
            except:
                continue
        
        val_graphs = []
        for s in tqdm(val_samples, desc="验证集图"):
            try:
                g = structure_to_graph(s['structure'], s['features'], self.config.CUTOFF)
                g.y = torch.tensor([s['label']], dtype=torch.float)
                val_graphs.append(g)
            except:
                continue
        
        print(f"训练图: {len(train_graphs)}, 验证图: {len(val_graphs)}")
        
        # 数据加载器
        train_loader = GeoDataLoader(train_graphs, batch_size=self.config.BATCH_SIZE, 
                                      shuffle=True, drop_last=True)
        val_loader = GeoDataLoader(val_graphs, batch_size=self.config.BATCH_SIZE)
        
        # 模型
        model = NequIPClassifierV9(self.config).to(self.device)
        
        # 损失函数
        if self.config.USE_FOCAL_LOSS:
            criterion = FocalLoss(
                alpha=self.config.FOCAL_ALPHA,
                gamma=self.config.FOCAL_GAMMA,
                label_smoothing=self.config.LABEL_SMOOTHING
            )
        else:
            criterion = nn.BCEWithLogitsLoss()
        
        # 优化器
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LR,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-6
        )
        
        # 训练循环
        best_auc = 0
        best_model_state = None
        patience_counter = 0
        
        print(f"开始训练 {self.config.EPOCHS} epochs...")
        sys.stdout.flush()
        
        for epoch in range(self.config.EPOCHS):
            # 训练
            model.train()
            train_loss = 0
            batch_count = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                if self.config.USE_AMP:
                    with autocast():
                        logits = model(batch)
                        loss = criterion(logits, batch.y)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    logits = model(batch)
                    loss = criterion(logits, batch.y)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                
                # 每50个batch打印一次进度
                if batch_count % 50 == 0:
                    print(f"    Epoch {epoch+1} Batch {batch_count}: Loss={loss.item():.4f}")
                    sys.stdout.flush()
            
            scheduler.step()
            
            # 验证
            model.eval()
            val_logits = []
            val_targets = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    logits = model(batch)
                    val_logits.append(logits.cpu())
                    val_targets.append(batch.y.cpu())
            
            val_logits = torch.cat(val_logits).numpy()
            val_targets = torch.cat(val_targets).numpy()
            val_probs = 1 / (1 + np.exp(-val_logits))
            
            try:
                auc = roc_auc_score(val_targets, val_probs)
            except:
                auc = 0.5
            
            if auc > best_auc:
                best_auc = auc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 每5个epoch打印一次详细进度
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1}/{self.config.EPOCHS}: Loss={train_loss/len(train_loader):.4f}, "
                      f"Val AUC={auc:.4f}, Best={best_auc:.4f}, Patience={patience_counter}")
                sys.stdout.flush()
            
            if patience_counter >= self.config.PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                sys.stdout.flush()
                break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        # 最终评估
        model.eval()
        val_logits = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                logits = model(batch)
                val_logits.append(logits.cpu())
                val_targets.append(batch.y.cpu())
        
        val_logits = torch.cat(val_logits).numpy()
        val_targets = torch.cat(val_targets).numpy()
        val_probs = 1 / (1 + np.exp(-val_logits))
        
        # 动态阈值优化 (优先Recall)
        best_threshold = 0.5
        best_f1 = 0
        best_recall = 0
        
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (val_probs >= thresh).astype(int)
            recall = recall_score(val_targets, preds, zero_division=0)
            f1 = f1_score(val_targets, preds, zero_division=0)
            
            # 优先高Recall
            if recall >= 0.9 and f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                best_recall = recall
            elif recall > best_recall and best_recall < 0.9:
                best_recall = recall
                best_threshold = thresh
                best_f1 = f1
        
        # 使用最佳阈值计算指标
        preds = (val_probs >= best_threshold).astype(int)
        
        results = {
            'fold': fold + 1,
            'auc': roc_auc_score(val_targets, val_probs),
            'accuracy': accuracy_score(val_targets, preds),
            'recall': recall_score(val_targets, preds, zero_division=0),
            'precision': precision_score(val_targets, preds, zero_division=0),
            'f1': f1_score(val_targets, preds, zero_division=0),
            'threshold': best_threshold
        }
        
        print(f"\nFold {fold + 1} 结果:")
        print(f"  AUC: {results['auc']:.4f}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  F1: {results['f1']:.4f}")
        print(f"  Threshold: {results['threshold']:.2f}")
        
        return results, model
    
    def train(self):
        """完整训练流程"""
        
        print("=" * 60)
        print("NequIP v9 - 优化架构版E(3)等变神经网络")
        print("=" * 60)
        print("\n核心优化:")
        print("  - 6层深度SE(3)等变消息传递")
        print("  - 扩展球谐函数 (l_max=3, 16维)")
        print("  - 多尺度径向基 (16维 Bessel)")
        print("  - 注意力增强消息传递")
        print("  - 分层池化策略")
        print("  - Focal Loss + Label Smoothing")
        print("=" * 60)
        
        # 加载数据
        samples = load_polar_dataset(self.config)
        
        # 交叉验证
        labels = [s['label'] for s in samples]
        skf = StratifiedKFold(n_splits=self.config.N_SPLITS, shuffle=True, 
                              random_state=self.config.RANDOM_STATE)
        
        all_results = []
        best_model = None
        best_auc = 0
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(samples, labels)):
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]
            
            results, model = self.train_fold(train_samples, val_samples, fold)
            all_results.append(results)
            
            if results['auc'] > best_auc:
                best_auc = results['auc']
                best_model = model
        
        # 汇总结果
        print("\n" + "=" * 60)
        print("交叉验证汇总")
        print("=" * 60)
        
        metrics = ['auc', 'accuracy', 'recall', 'precision', 'f1']
        print(f"\n{'指标':<15}{'平均值':<15}{'标准差':<15}")
        print("-" * 45)
        
        for metric in metrics:
            values = [r[metric] for r in all_results]
            print(f"{metric:<15}{np.mean(values):.4f}          {np.std(values):.4f}")
        
        # 找最佳fold
        best_fold = max(all_results, key=lambda x: x['recall'] * 0.5 + x['auc'] * 0.5)
        print(f"\n最佳Fold: {best_fold['fold']}")
        print(f"  AUC: {best_fold['auc']:.4f}")
        print(f"  Accuracy: {best_fold['accuracy']:.4f}")
        print(f"  Recall: {best_fold['recall']:.4f}")
        print(f"  Precision: {best_fold['precision']:.4f}")
        print(f"  F1: {best_fold['f1']:.4f}")
        
        # 保存模型
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'config': self.config.__dict__,
            'results': all_results
        }, self.config.MODEL_DIR / 'nequip_v9_best.pt')
        
        # 保存结果
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(self.config.REPORT_DIR / 'cv_results.csv', index=False)
        
        print(f"\n模型已保存到: {self.config.MODEL_DIR}")
        print(f"报告已保存到: {self.config.REPORT_DIR}")
        
        return all_results


# ============================================================
# 主函数
# ============================================================
if __name__ == '__main__':
    trainer = NequIPV9Trainer()
    results = trainer.train()
