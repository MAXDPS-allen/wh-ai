"""
NequIP-Based Ferroelectric Classifier
=============================================
使用 E(3)-等变神经网络进行铁电材料分类

目标: 
- 铁电准确率 (Precision) >= 95%
- 召回率 (Recall) >= 99%

架构特点:
1. 基于球谐函数的方向编码
2. 径向基函数编码距离信息
3. 消息传递网络聚合邻居信息
4. 结合全局特征进行最终预测

依赖:
pip install torch torch-geometric pymatgen
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加共享模块路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from feature_engineering import (
    UnifiedFeatureExtractor, FEATURE_DIM, FEATURE_NAMES, ELEMENT_DATABASE
)

from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score


class NequIPConfig:
    """NequIP 分类器配置"""
    
    # 特征维度
    GLOBAL_FEAT_DIM = FEATURE_DIM  # 64维全局特征
    NUM_SPECIES = 100  # 元素种类数
    
    # 球谐函数配置
    LMAX = 2  # 最大角动量 (l=0,1,2 -> 1+3+5=9维)
    SH_DIM = 9  # 球谐函数维度
    
    # 径向基配置
    NUM_RADIAL_BASIS = 8  # 径向基函数数量
    CUTOFF = 5.0  # 截断半径 (Å)
    
    # 网络结构
    NUM_LAYERS = 4  # 消息传递层数量
    NODE_DIM = 64  # 节点特征维度
    HIDDEN_DIM = 256  # MLP隐藏层维度
    
    # 训练配置
    BATCH_SIZE = 32
    EPOCHS = 150
    LR = 5e-4
    WEIGHT_DECAY = 5e-4
    PATIENCE = 60  # 增加耐心
    
    # 类别权重 (处理不平衡)
    POS_WEIGHT = 10.0  # 稍微降低，平衡 recall 和 precision
    
    # 过采样
    OVERSAMPLE_RATIO = 3.0  # 增加过采样
    
    # 路径
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_nequip'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_nequip'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 径向基函数
# ==========================================
class BesselBasis(nn.Module):
    """Bessel 径向基函数"""
    
    def __init__(self, num_basis: int = 8, cutoff: float = 5.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        
        # 预计算 bessel 频率
        self.register_buffer(
            'freq', 
            torch.arange(1, num_basis + 1) * np.pi / cutoff
        )
    
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        计算 Bessel 基函数值
        
        Args:
            r: 原子间距离 [num_edges]
            
        Returns:
            basis: 径向基函数值 [num_edges, num_basis]
        """
        r = r.unsqueeze(-1)  # [num_edges, 1]
        
        # Bessel 函数: sqrt(2/r_c) * sin(n*pi*r/r_c) / r
        basis = torch.sqrt(torch.tensor(2.0 / self.cutoff, device=r.device)) * \
                torch.sin(self.freq * r) / (r + 1e-8)
        
        return basis


class SmoothCutoff(nn.Module):
    """平滑截断函数"""
    
    def __init__(self, cutoff: float = 5.0, p: int = 6):
        super().__init__()
        self.cutoff = cutoff
        self.p = p
    
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """余弦平滑截断"""
        x = r / self.cutoff
        envelope = (1 - x.pow(self.p)).pow(2)
        envelope = torch.where(r < self.cutoff, envelope, torch.zeros_like(envelope))
        return envelope


class SphericalHarmonics(nn.Module):
    """球谐函数编码 (l=0, 1, 2)"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, vec: torch.Tensor) -> torch.Tensor:
        """
        计算球谐函数 (l=0, 1, 2)
        
        Args:
            vec: 归一化方向向量 [N, 3]
            
        Returns:
            sh: 球谐函数值 [N, 9]
        """
        x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
        
        # l=0: Y_0^0 = 1/sqrt(4*pi) ≈ 0.282
        y00 = torch.ones_like(x) * 0.28209479
        
        # l=1: Y_1^{-1}, Y_1^0, Y_1^1
        # sqrt(3/(4*pi)) ≈ 0.489
        y1_m1 = 0.4886025 * y
        y1_0 = 0.4886025 * z
        y1_1 = 0.4886025 * x
        
        # l=2: 5个分量
        # sqrt(15/(4*pi)) ≈ 1.093
        y2_m2 = 1.0925485 * x * y
        y2_m1 = 1.0925485 * y * z
        # sqrt(5/(16*pi)) ≈ 0.315
        y2_0 = 0.31539157 * (3*z*z - 1)
        y2_1 = 1.0925485 * x * z
        # sqrt(15/(16*pi)) ≈ 0.546
        y2_2 = 0.5462742 * (x*x - y*y)
        
        sh = torch.stack([y00, y1_m1, y1_0, y1_1, y2_m2, y2_m1, y2_0, y2_1, y2_2], dim=-1)
        return sh


# ==========================================
# 消息传递层
# ==========================================
class EquivariantMessageLayer(nn.Module):
    """
    基于球谐函数的消息传递层
    
    模拟 NequIP 的核心思想:
    1. 使用球谐函数编码方向信息
    2. 使用径向基编码距离信息
    3. 通过 MLP 组合并生成消息
    """
    
    def __init__(
        self, 
        node_dim: int = 64, 
        sh_dim: int = 9, 
        radial_dim: int = 8,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # 消息生成网络
        # 输入: 源节点特征 + 球谐函数 + 径向基
        msg_input_dim = node_dim + sh_dim + radial_dim
        
        self.message_net = nn.Sequential(
            nn.Linear(msg_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim)
        )
        
        # 更新网络
        # 输入: 当前节点特征 + 聚合消息
        self.update_net = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
            nn.LayerNorm(node_dim)
        )
        
        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_sh: torch.Tensor,
        edge_radial: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_feats: 节点特征 [num_nodes, node_dim]
            edge_index: 边索引 [2, num_edges]
            edge_sh: 边的球谐函数 [num_edges, sh_dim]
            edge_radial: 边的径向基 [num_edges, radial_dim]
            
        Returns:
            updated_feats: 更新后的节点特征
        """
        src, dst = edge_index
        
        # 构建消息输入
        msg_input = torch.cat([node_feats[src], edge_sh, edge_radial], dim=-1)
        
        # 生成消息
        messages = self.message_net(msg_input)
        
        # 聚合消息
        aggregated = torch.zeros_like(node_feats)
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        
        # 门控更新
        update_input = torch.cat([node_feats, aggregated], dim=-1)
        gate = self.gate(update_input)
        update = self.update_net(update_input)
        
        # 残差连接 + 门控
        out = node_feats + gate * update
        
        return out


# ==========================================
# NequIP 分类器
# ==========================================
class NequIPClassifier(nn.Module):
    """
    基于 NequIP 思想的铁电材料分类器
    
    核心特点:
    1. 球谐函数编码原子间方向 (保留旋转信息)
    2. Bessel 径向基编码距离
    3. 多层消息传递聚合邻域信息
    4. 结合全局结构特征进行分类
    """
    
    def __init__(self, config: NequIPConfig = None):
        super().__init__()
        self.config = config or NequIPConfig()
        
        # 1. 原子嵌入
        self.atom_embedding = nn.Embedding(self.config.NUM_SPECIES, self.config.NODE_DIM)
        
        # 2. 边特征编码
        self.spherical_harmonics = SphericalHarmonics()
        self.radial_basis = BesselBasis(
            num_basis=self.config.NUM_RADIAL_BASIS,
            cutoff=self.config.CUTOFF
        )
        self.cutoff_fn = SmoothCutoff(cutoff=self.config.CUTOFF)
        
        # 3. 消息传递层
        self.message_layers = nn.ModuleList()
        for i in range(self.config.NUM_LAYERS):
            self.message_layers.append(
                EquivariantMessageLayer(
                    node_dim=self.config.NODE_DIM,
                    sh_dim=self.config.SH_DIM,
                    radial_dim=self.config.NUM_RADIAL_BASIS,
                    hidden_dim=128
                )
            )
        
        # 4. 全局特征编码器
        self.global_encoder = nn.Sequential(
            nn.Linear(self.config.GLOBAL_FEAT_DIM, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
        )
        
        # 5. 特征融合
        # 图特征: mean + max + sum pooling
        graph_feat_dim = self.config.NODE_DIM * 3
        fusion_dim = graph_feat_dim + 96  # graph + global
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.config.HIDDEN_DIM),
            nn.LayerNorm(self.config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(self.config.HIDDEN_DIM, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.1),
        )
        
        # 6. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
    
    def forward(self, data) -> torch.Tensor:
        """
        前向传播
        
        Args:
            data: PyG Data 对象，包含:
                - x: 原子类型 (原子序数) [num_nodes]
                - edge_index: 边索引 [2, num_edges]
                - edge_vec: 边向量 (相对位置) [num_edges, 3]
                - edge_length: 边长度 [num_edges]
                - batch: 批次索引 [num_nodes]
                - u: 全局特征 [batch_size, global_feat_dim]
        """
        x = data.x  # [num_nodes]
        edge_index = data.edge_index
        edge_vec = data.edge_vec
        edge_length = data.edge_length
        batch = data.batch
        
        # 1. 原子嵌入
        x_clamped = x.clamp(0, self.config.NUM_SPECIES - 1).long()
        h = self.atom_embedding(x_clamped)
        
        # 2. 边特征
        # 归一化边向量
        edge_unit = edge_vec / (edge_length.unsqueeze(-1) + 1e-8)
        
        # 球谐函数
        edge_sh = self.spherical_harmonics(edge_unit)
        
        # 径向基 + 截断
        edge_radial = self.radial_basis(edge_length)
        cutoff_envelope = self.cutoff_fn(edge_length)
        edge_radial = edge_radial * cutoff_envelope.unsqueeze(-1)
        
        # 3. 消息传递
        for layer in self.message_layers:
            h = layer(h, edge_index, edge_sh, edge_radial)
        
        # 4. 图级池化
        graph_mean = global_mean_pool(h, batch)
        graph_max = global_max_pool(h, batch)
        graph_sum = global_add_pool(h, batch) / 10.0
        graph_feat = torch.cat([graph_mean, graph_max, graph_sum], dim=-1)
        
        # 5. 全局特征
        u = data.u
        if u.dim() == 3:
            u = u.squeeze(1)
        global_feat = self.global_encoder(u)
        
        # 6. 融合和分类
        combined = torch.cat([graph_feat, global_feat], dim=-1)
        h_fused = self.fusion(combined)
        logits = self.classifier(h_fused)
        
        return logits.squeeze(-1)


# ==========================================
# 数据处理
# ==========================================
def structure_to_nequip_graph(
    struct_dict: Dict, 
    label: int = 0, 
    global_features: np.ndarray = None,
    cutoff: float = 5.0
) -> Optional[Data]:
    """
    将结构字典转换为 NequIP 兼容的图数据
    
    关键: 需要保存边向量 (edge_vec) 用于球谐函数计算
    """
    try:
        from pymatgen.core import Structure
        structure = Structure.from_dict(struct_dict)
        
        # 节点特征: 原子序数
        atomic_numbers = []
        for site in structure:
            z = site.specie.Z
            atomic_numbers.append(z)
        
        x = torch.tensor(atomic_numbers, dtype=torch.long)
        
        # 边构建: 需要位置信息
        edge_index = []
        edge_vec = []
        edge_length = []
        
        for i, site_i in enumerate(structure):
            neighbors = structure.get_neighbors(site_i, cutoff)
            for neighbor in neighbors:
                j = neighbor.index
                if i != j:
                    edge_index.append([i, j])
                    
                    # 相对位置向量 (考虑周期性边界)
                    vec = neighbor.coords - site_i.coords
                    edge_vec.append(vec)
                    edge_length.append(neighbor.nn_distance)
        
        # 如果没有邻居，创建全连接图
        if not edge_index:
            n = len(atomic_numbers)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edge_index.append([i, j])
                        vec_frac = structure[j].frac_coords - structure[i].frac_coords
                        vec = structure.lattice.get_cartesian_coords(vec_frac)
                        dist = np.linalg.norm(vec)
                        edge_vec.append(vec)
                        edge_length.append(dist if dist > 0 else 1.0)
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_vec = torch.tensor(np.array(edge_vec), dtype=torch.float)
        edge_length = torch.tensor(edge_length, dtype=torch.float)
        
        # 全局特征
        if global_features is not None:
            u = torch.tensor(global_features, dtype=torch.float).unsqueeze(0)
        else:
            u = torch.zeros(1, FEATURE_DIM, dtype=torch.float)
        
        # 标签
        y = torch.tensor([label], dtype=torch.long)
        
        return Data(
            x=x, 
            edge_index=edge_index, 
            edge_vec=edge_vec,
            edge_length=edge_length,
            y=y, 
            u=u
        )
        
    except Exception as e:
        return None


class NequIPDataset(Dataset):
    """NequIP 数据集"""
    
    def __init__(
        self, 
        data_files: List[Tuple[str, int]], 
        extractor,
        cutoff: float = 5.0,
        oversample_ratio: float = 1.0
    ):
        self.graphs = []
        self.extractor = extractor
        self.cutoff = cutoff
        
        pos_graphs = []
        neg_graphs = []
        
        for file_path, label in data_files:
            if os.path.exists(file_path):
                graphs = self._load_file(file_path, label)
                if label == 1:
                    pos_graphs.extend(graphs)
                else:
                    neg_graphs.extend(graphs)
        
        # 过采样正样本
        if oversample_ratio > 1.0 and len(pos_graphs) > 0:
            n_oversample = int(len(pos_graphs) * (oversample_ratio - 1))
            oversampled = [pos_graphs[i % len(pos_graphs)] for i in range(n_oversample)]
            pos_graphs.extend(oversampled)
        
        self.graphs = pos_graphs + neg_graphs
        np.random.shuffle(self.graphs)
        
        print(f"Loaded {len(self.graphs)} samples (Pos: {len(pos_graphs)}, Neg: {len(neg_graphs)})")
    
    def _load_file(self, file_path: str, label: int) -> List[Data]:
        graphs = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    struct = item.get('structure')
                    sg = item.get('spacegroup_number')
                    
                    if struct:
                        global_feat = self.extractor.extract_from_structure_dict(struct, sg)
                        graph = structure_to_nequip_graph(
                            struct, label, global_feat, self.cutoff
                        )
                        if graph is not None:
                            graphs.append(graph)
                except:
                    continue
        return graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


# ==========================================
# 训练器
# ==========================================
class NequIPTrainer:
    """NequIP 分类器训练器"""
    
    def __init__(self, config: NequIPConfig = None):
        self.config = config or NequIPConfig()
        self.device = self.config.DEVICE
        
        # 创建目录
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 特征提取器
        self.extractor = UnifiedFeatureExtractor()
        
        # 模型
        print("Creating NequIP-style classifier with spherical harmonics encoding")
        self.model = NequIPClassifier(self.config).to(self.device)
        
        # 计算参数量
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LR,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2
        )
        
        # 损失函数 (加权)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.config.POS_WEIGHT]).to(self.device)
        )
        
        # 训练历史
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_recall': [],
            'val_precision': [],
            'val_f1': [],
            'val_auc': []
        }
        
        self.best_recall = 0
        self.best_precision = 0
        self.best_threshold = 0.5
        self.patience_counter = 0
    
    def load_data(self):
        """加载数据"""
        data_files = [
            (str(self.config.DATA_DIR / 'dataset_original_ferroelectric.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_known_FE_rest.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_nonFE.jsonl'), 0),
            (str(self.config.DATA_DIR / 'dataset_polar_non_ferroelectric_final.jsonl'), 0),
        ]
        
        dataset = NequIPDataset(
            data_files, 
            self.extractor,
            cutoff=self.config.CUTOFF,
            oversample_ratio=self.config.OVERSAMPLE_RATIO
        )
        
        # 划分数据集
        n = len(dataset)
        n_train = int(0.8 * n)
        n_val = n - n_train
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val]
        )
        
        self.train_loader = GeoDataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE,
            shuffle=True
        )
        
        self.val_loader = GeoDataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False
        )
        
        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    def find_optimal_threshold(
        self, 
        labels: np.ndarray, 
        probs: np.ndarray, 
        target_recall: float = 0.99
    ) -> Tuple[float, float, float]:
        """寻找最优阈值以达到目标召回率"""
        best_threshold = 0.5
        best_precision = 0
        achieved_recall = 0
        
        for threshold in np.linspace(0.01, 0.99, 99):
            preds = (probs >= threshold).astype(int)
            
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            
            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            
            if recall >= target_recall:
                if precision > best_precision:
                    best_threshold = threshold
                    best_precision = precision
                    achieved_recall = recall
        
        # 如果找不到满足条件的阈值
        if achieved_recall < target_recall:
            for threshold in np.linspace(0.01, 0.5, 50):
                preds = (probs >= threshold).astype(int)
                tp = np.sum((preds == 1) & (labels == 1))
                fn = np.sum((preds == 0) & (labels == 1))
                fp = np.sum((preds == 1) & (labels == 0))
                
                recall = tp / (tp + fn + 1e-8)
                precision = tp / (tp + fp + 1e-8)
                
                if recall > achieved_recall:
                    achieved_recall = recall
                    best_precision = precision
                    best_threshold = threshold
        
        return best_threshold, best_precision, achieved_recall
    
    def train_epoch(self, dataloader) -> float:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0
        
        for data in dataloader:
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(data)
            labels = data.y.float()
            
            loss = self.criterion(logits, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader) -> Tuple[float, np.ndarray, np.ndarray]:
        """验证"""
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                
                logits = self.model(data)
                labels = data.y.float()
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.cpu().numpy())
        
        return total_loss / len(dataloader), np.array(all_probs), np.array(all_labels)
    
    def train(self, epochs: int = None):
        """完整训练流程"""
        epochs = epochs or self.config.EPOCHS
        
        print("\n" + "="*60)
        print("NequIP Ferroelectric Classifier Training")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"L_max: {self.config.LMAX}")
        print(f"Radial Basis: {self.config.NUM_RADIAL_BASIS}")
        print(f"Message Layers: {self.config.NUM_LAYERS}")
        print(f"Positive Weight: {self.config.POS_WEIGHT}")
        print("="*60 + "\n")
        
        # 加载数据
        self.load_data()
        
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(self.train_loader)
            
            # 验证
            val_loss, probs, labels = self.validate(self.val_loader)
            
            # 寻找最优阈值
            threshold, precision, recall = self.find_optimal_threshold(
                labels, probs, target_recall=0.99
            )
            
            # 计算其他指标
            preds = (probs >= threshold).astype(int)
            f1 = f1_score(labels, preds)
            
            try:
                auc = roc_auc_score(labels, probs)
            except:
                auc = 0.5
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.history['epoch'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_recall'].append(recall)
            self.history['val_precision'].append(precision)
            self.history['val_f1'].append(f1)
            self.history['val_auc'].append(auc)
            
            # 检查是否达到目标
            target_met = recall >= 0.99 and precision >= 0.95
            
            # 打印进度
            if epoch % 5 == 0 or target_met or (recall >= 0.99 and precision >= 0.90):
                status = "✅ TARGET MET!" if target_met else ("🔥 CLOSE!" if precision >= 0.90 else "")
                print(f"Epoch {epoch:3d} | "
                      f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                      f"Recall: {recall:.4f} | Precision: {precision:.4f} | "
                      f"F1: {f1:.4f} | AUC: {auc:.4f} | "
                      f"Thr: {threshold:.3f} {status}")
            
            # 保存最佳模型 - 优化选择条件
            # 优先考虑同时满足高召回和高精确的模型
            score = recall * precision if recall >= 0.99 else recall * 0.5
            if recall >= self.best_recall:
                if precision > self.best_precision or recall > self.best_recall:
                    self.best_recall = recall
                    self.best_precision = precision
                    self.best_threshold = threshold
                    best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                    
                    if target_met:
                        print(f"\n🎯 Target achieved! Saving model...")
                        self.save_model("best")
            else:
                self.patience_counter += 1
            
            # 早停
            if self.patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # 保存最终模型
        self.save_model("final")
        
        # 生成报告
        self.generate_report()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Recall: {self.best_recall:.4f}")
        print(f"Best Precision: {self.best_precision:.4f}")
        print(f"Best Threshold: {self.best_threshold:.4f}")
        print("="*60)
    
    def save_model(self, suffix: str):
        """保存模型"""
        model_path = self.config.MODEL_DIR / f"nequip_classifier_{suffix}.pt"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.best_threshold,
            'history': self.history,
            'config': {
                'LMAX': self.config.LMAX,
                'NUM_RADIAL_BASIS': self.config.NUM_RADIAL_BASIS,
                'CUTOFF': self.config.CUTOFF,
                'NUM_LAYERS': self.config.NUM_LAYERS,
                'NODE_DIM': self.config.NODE_DIM,
            }
        }, model_path)
        
        print(f"Model saved to {model_path}")
    
    def generate_report(self):
        """生成训练报告"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # 最终评估
        _, probs, labels = self.validate(self.val_loader)
        preds = (probs >= self.best_threshold).astype(int)
        
        # 混淆矩阵
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.5
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 损失曲线
        ax1 = axes[0, 0]
        ax1.plot(self.history['epoch'], self.history['train_loss'], label='Train')
        ax1.plot(self.history['epoch'], self.history['val_loss'], label='Val')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Recall/Precision 曲线
        ax2 = axes[0, 1]
        ax2.plot(self.history['epoch'], self.history['val_recall'], 
                 label='Recall', color='blue')
        ax2.plot(self.history['epoch'], self.history['val_precision'], 
                 label='Precision', color='green')
        ax2.axhline(0.99, color='blue', linestyle='--', alpha=0.5, label='Target Recall')
        ax2.axhline(0.95, color='green', linestyle='--', alpha=0.5, label='Target Precision')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.set_title('Recall & Precision')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        # 3. 混淆矩阵
        ax3 = axes[1, 0]
        im = ax3.imshow(cm, cmap='Blues')
        ax3.set_xticks([0, 1])
        ax3.set_yticks([0, 1])
        ax3.set_xticklabels(['Non-FE', 'FE'])
        ax3.set_yticklabels(['Non-FE', 'FE'])
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_title('Confusion Matrix')
        
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, str(cm[i, j]), ha='center', va='center', 
                        fontsize=14, color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        # 4. 指标总结
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        target_status = '🎯 ACHIEVED!' if (recall >= 0.99 and precision >= 0.95) else '⏳ In Progress'
        
        metrics_text = f"""
        NequIP Classifier Performance
        ══════════════════════════════════════
        
        Accuracy:     {accuracy:.4f}
        Precision:    {precision:.4f}  {'✅' if precision >= 0.95 else '❌'}
        Recall:       {recall:.4f}  {'✅' if recall >= 0.99 else '❌'}
        F1 Score:     {f1:.4f}
        ROC-AUC:      {auc:.4f}
        
        Threshold:    {self.best_threshold:.4f}
        
        ══════════════════════════════════════
        Confusion Matrix:
          TP: {tp}  |  FP: {fp}
          FN: {fn}  |  TN: {tn}
        ══════════════════════════════════════
        
        Target: Recall >= 99%, Precision >= 95%
        Status: {target_status}
        """
        
        ax4.text(0.1, 0.5, metrics_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='center', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(self.config.REPORT_DIR / 'nequip_training_report.png', 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存文本报告
        with open(self.config.REPORT_DIR / 'nequip_report.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("NequIP Ferroelectric Classifier Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Configuration:\n")
            f.write(f"  - L_max: {self.config.LMAX}\n")
            f.write(f"  - Radial Basis: {self.config.NUM_RADIAL_BASIS}\n")
            f.write(f"  - Cutoff: {self.config.CUTOFF} Å\n")
            f.write(f"  - Layers: {self.config.NUM_LAYERS}\n")
            f.write(f"  - Node Dim: {self.config.NODE_DIM}\n\n")
            f.write("Performance:\n")
            f.write(f"  - Accuracy: {accuracy:.4f}\n")
            f.write(f"  - Precision: {precision:.4f}\n")
            f.write(f"  - Recall: {recall:.4f}\n")
            f.write(f"  - F1 Score: {f1:.4f}\n")
            f.write(f"  - ROC-AUC: {auc:.4f}\n")
            f.write(f"  - Threshold: {self.best_threshold:.4f}\n\n")
            f.write("="*60 + "\n")
        
        print(f"\nReport saved to {self.config.REPORT_DIR}")


# ==========================================
# 主函数
# ==========================================
def main():
    print("="*60)
    print("NequIP-Based Ferroelectric Classifier")
    print("Spherical Harmonics + Radial Basis Encoding")
    print("="*60)
    
    # 训练
    trainer = NequIPTrainer()
    trainer.train()


if __name__ == '__main__':
    main()
