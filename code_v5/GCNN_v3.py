"""
增强型图神经网络分类器 v3
=============================================
针对高Recall优化的铁电材料分类模型

改进点:
1. Focal Loss - 处理类别不平衡
2. 更高的正类权重
3. 阈值优化 - 针对高recall
4. 数据增强和正则化增强
5. 更深的网络架构
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 添加共享模块路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from feature_engineering import (
    UnifiedFeatureExtractor,
    FEATURE_DIM,
    FEATURE_NAMES,
    ELEMENT_DATABASE,
    extract_features
)

# PyTorch Geometric
try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool, global_add_pool, global_max_pool
    from torch_geometric.loader import DataLoader as GeoDataLoader
    HAS_GEOMETRIC = True
except ImportError:
    HAS_GEOMETRIC = False
    print("Warning: torch_geometric not found.")


# ==========================================
# 1. 配置
# ==========================================
class Config:
    # 特征维度 (使用统一特征)
    GLOBAL_FEAT_DIM = FEATURE_DIM  # 64
    NODE_FEAT_DIM = 16             # 节点特征
    EDGE_FEAT_DIM = 4              # 边特征
    HIDDEN_DIM = 512               # 增大隐藏层
    
    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 200
    LR = 5e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE = 40
    
    # 类别权重 - 大幅增加铁电材料权重
    POS_WEIGHT = 10.0  # 铁电材料权重 (进一步增加)
    FOCAL_ALPHA = 0.85
    FOCAL_GAMMA = 1.5
    
    # 阈值 - 降低阈值以提高recall
    THRESHOLD = 0.15
    
    # 路径
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_gcnn_v3'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_gcnn_v3'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 2. Focal Loss
# ==========================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.75, gamma=2.0, pos_weight=5.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        
        # 获取每个样本的目标类概率
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal factor
        focal_weight = (1 - p_t) ** self.gamma
        
        # 类别权重
        alpha_t = torch.where(targets == 1, 
                             torch.tensor(self.alpha * self.pos_weight, device=logits.device),
                             torch.tensor(1 - self.alpha, device=logits.device))
        
        loss = alpha_t * focal_weight * ce_loss
        return loss.mean()


# ==========================================
# 3. 数据处理
# ==========================================
def structure_to_graph(struct_dict: Dict, label: int = 0, 
                       global_features: np.ndarray = None) -> Optional[Data]:
    """
    将结构字典转换为图数据
    """
    if not HAS_GEOMETRIC:
        return None
    
    try:
        from pymatgen.core import Structure
        
        structure = Structure.from_dict(struct_dict)
        
        # 节点特征 - 更丰富的特征
        node_features = []
        for site in structure:
            el = site.specie.symbol
            if el in ELEMENT_DATABASE:
                data = ELEMENT_DATABASE[el]
                feat = [
                    data[0] / 100.0,   # 原子序数
                    data[1] / 200.0,   # 质量
                    data[2] / 2.5,     # 半径
                    data[3] / 4.0,     # 电负性
                    data[4] / 15.0,    # 电离能
                    data[5] / 8.0,     # 价电子
                    data[6] / 3500.0,  # 熔点
                    data[7] / 200.0,   # 热导率
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
        
        if not node_features:
            return None
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # 边和边特征
        edge_index = []
        edge_attr = []
        cutoff = 5.0
        
        for i, site_i in enumerate(structure):
            neighbors = structure.get_neighbors(site_i, cutoff)
            for neighbor in neighbors:
                j = neighbor.index
                if i != j:
                    edge_index.append([i, j])
                    
                    dist = neighbor.nn_distance if hasattr(neighbor, 'nn_distance') else 2.0
                    edge_feat = [
                        dist / cutoff,
                        1.0 / max(dist, 0.1) / 10.0,
                        np.exp(-dist / 2),
                        1.0 if dist < 3.0 else 0.0
                    ]
                    edge_attr.append(edge_feat)
        
        if not edge_index:
            n = len(node_features)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edge_index.append([i, j])
                        edge_attr.append([0.5, 0.5, 0.5, 0.5])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        if global_features is not None:
            u = torch.tensor(global_features, dtype=torch.float).unsqueeze(0)
        else:
            u = torch.zeros(1, FEATURE_DIM, dtype=torch.float)
        
        y = torch.tensor([label], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u)
    
    except Exception as e:
        return None


class MaterialGraphDataset(Dataset):
    """材料图数据集"""
    
    def __init__(self, data_files: List[Tuple[str, int]], extractor: UnifiedFeatureExtractor):
        self.graphs = []
        self.extractor = extractor
        
        for file_path, label in data_files:
            if os.path.exists(file_path):
                self._load_file(file_path, label)
        
        print(f"Loaded {len(self.graphs)} graph samples")
    
    def _load_file(self, file_path: str, label: int):
        """加载文件"""
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    struct = item.get('structure')
                    sg = item.get('spacegroup_number')
                    
                    if struct:
                        global_feat = self.extractor.extract_from_structure_dict(struct, sg)
                        graph = structure_to_graph(struct, label, global_feat)
                        if graph is not None:
                            self.graphs.append(graph)
                except:
                    continue
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


# ==========================================
# 4. 网络架构 - 更深更强
# ==========================================
class GATBlock(nn.Module):
    """增强版GAT块 with residual"""
    
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual projection if needed
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    
    def forward(self, x, edge_index):
        res = self.residual(x)
        h = self.gat(x, edge_index)
        h = self.norm(h)
        h = self.act(h + res)  # Residual connection
        h = self.dropout(h)
        return h


class MultiHeadAttention(nn.Module):
    """多头自注意力用于特征融合"""
    
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class GCNNClassifierV3(nn.Module):
    """增强版图神经网络分类器 - 针对高Recall优化"""
    
    def __init__(self, config: Config = None):
        super().__init__()
        
        config = config or Config()
        self.config = config
        
        if HAS_GEOMETRIC:
            # 图编码器 - 更深
            self.node_embed = nn.Sequential(
                nn.Linear(config.NODE_FEAT_DIM, 64),
                nn.LayerNorm(64),
                nn.GELU(),
            )
            
            self.gat1 = GATBlock(64, 128, heads=4, dropout=0.15)
            self.gat2 = GATBlock(128, 256, heads=4, dropout=0.15)
            self.gat3 = GATBlock(256, 256, heads=4, dropout=0.15)
            self.gat4 = GATBlock(256, 128, heads=4, dropout=0.15)
            
            graph_dim = 128 * 3  # mean + max + add pooling
        else:
            graph_dim = 0
        
        # 全局特征处理 - 更强
        self.global_encoder = nn.Sequential(
            nn.Linear(config.GLOBAL_FEAT_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.GELU(),
        )
        
        # 融合层
        fusion_dim = graph_dim + 192
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.25),
            
            nn.Linear(config.HIDDEN_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        
        # 分类头 - 使用sigmoid输出用于高recall
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )
    
    def forward(self, data):
        if HAS_GEOMETRIC:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            
            # 图编码
            h = self.node_embed(x)
            h = self.gat1(h, edge_index)
            h = self.gat2(h, edge_index)
            h = self.gat3(h, edge_index)
            h = self.gat4(h, edge_index)
            
            # 多种池化融合
            graph_mean = global_mean_pool(h, batch)
            graph_max = global_max_pool(h, batch)
            graph_add = global_add_pool(h, batch) / 10.0  # normalize
            graph_feat = torch.cat([graph_mean, graph_max, graph_add], dim=1)
            
            # 全局特征
            u = data.u
            if u.dim() == 3:
                u = u.squeeze(1)
            global_feat = self.global_encoder(u)
            
            # 融合
            combined = torch.cat([graph_feat, global_feat], dim=1)
        else:
            u = data.u
            if u.dim() == 3:
                u = u.squeeze(1)
            combined = self.global_encoder(u)
        
        h = self.fusion(combined)
        logits = self.classifier(h)
        
        return logits


# ==========================================
# 5. 训练器
# ==========================================
class GCNNTrainerV3:
    """GCNN训练器 - 高Recall优化"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        # 创建目录
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 特征提取器
        self.extractor = UnifiedFeatureExtractor()
        
        # 模型
        self.model = GCNNClassifierV3(self.config).to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LR,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2
        )
        
        # Focal Loss
        self.criterion = FocalLoss(
            alpha=self.config.FOCAL_ALPHA,
            gamma=self.config.FOCAL_GAMMA,
            pos_weight=self.config.POS_WEIGHT
        )
        
        # 训练历史
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'train_recall': [],
            'val_loss': [],
            'val_acc': [],
            'val_recall': [],
            'val_precision': [],
            'val_auc': [],
            'val_f1': []
        }
        
        self.best_recall = 0
        self.best_threshold = 0.5
    
    def load_data(self):
        """加载数据"""
        data_files = [
            (str(self.config.DATA_DIR / 'dataset_original_ferroelectric.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_known_FE_rest.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_nonFE.jsonl'), 0),
            (str(self.config.DATA_DIR / 'dataset_polar_non_ferroelectric_final.jsonl'), 0),
        ]
        
        dataset = MaterialGraphDataset(data_files, self.extractor)
        
        # 划分
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        
        train_loader = GeoDataLoader(train_set, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = GeoDataLoader(val_set, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        # 计算类别分布
        pos_count = sum(1 for g in dataset.graphs if g.y.item() == 1)
        neg_count = len(dataset) - pos_count
        print(f"Dataset: {len(dataset)} samples (Pos: {pos_count}, Neg: {neg_count})")
        
        return train_loader, val_loader
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for data in dataloader:
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(data)
            loss = self.criterion(logits, data.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1)[:, 1]
            pred = (probs > self.config.THRESHOLD).long()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        acc = (all_preds == all_labels).mean()
        
        # Recall for positive class
        pos_mask = all_labels == 1
        if pos_mask.sum() > 0:
            recall = (all_preds[pos_mask] == 1).mean()
        else:
            recall = 0.0
        
        return total_loss / len(dataloader), acc, recall
    
    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                
                logits = self.model(data)
                loss = self.criterion(logits, data.y)
                
                total_loss += loss.item()
                
                probs = F.softmax(logits, dim=1)[:, 1]
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 寻找最佳阈值以最大化recall同时保持precision
        best_threshold = self.config.THRESHOLD
        best_score = 0
        
        # 寻找最佳阈值以达到 recall >= 99%
        for thresh in np.arange(0.05, 0.5, 0.02):
            preds = (all_probs > thresh).astype(int)
            
            pos_mask = all_labels == 1
            neg_mask = all_labels == 0
            
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                recall = (preds[pos_mask] == 1).mean()
                precision = (all_labels[preds == 1] == 1).mean() if (preds == 1).sum() > 0 else 0
                
                # 优化目标: recall >= 0.99, 然后最大化precision
                if recall >= 0.99:
                    if precision > best_score:
                        best_score = precision
                        best_threshold = thresh
                elif recall >= 0.95 and precision > 0.5:
                    # 如果达不到99%，至少要95%
                    score = recall * 0.7 + precision * 0.3
                    if score > best_score * 0.8:
                        best_score = max(best_score, precision * 0.8)
                        best_threshold = thresh
        
        # 使用最佳阈值
        all_preds = (all_probs > best_threshold).astype(int)
        
        acc = (all_preds == all_labels).mean()
        
        pos_mask = all_labels == 1
        neg_mask = all_labels == 0
        
        recall = (all_preds[pos_mask] == 1).mean() if pos_mask.sum() > 0 else 0
        
        if (all_preds == 1).sum() > 0:
            precision = (all_labels[all_preds == 1] == 1).mean()
        else:
            precision = 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc_score = roc_auc_score(all_labels, all_probs)
        except:
            auc_score = 0.5
        
        return total_loss / len(dataloader), acc, recall, precision, f1, auc_score, best_threshold
    
    def train(self, epochs: int = None):
        """完整训练"""
        epochs = epochs or self.config.EPOCHS
        train_loader, val_loader = self.load_data()
        
        print(f"\n{'='*60}")
        print(f"GCNN v3 Training (High Recall Optimization)")
        print(f"Device: {self.device}")
        print(f"Focal Loss Alpha: {self.config.FOCAL_ALPHA}, Gamma: {self.config.FOCAL_GAMMA}")
        print(f"Positive Class Weight: {self.config.POS_WEIGHT}")
        print(f"Initial Threshold: {self.config.THRESHOLD}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        best_score = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc, train_recall = self.train_epoch(train_loader)
            val_loss, val_acc, val_recall, val_precision, val_f1, val_auc, optimal_thresh = self.validate(val_loader)
            
            self.scheduler.step()
            
            # 记录
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_recall'].append(train_recall)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_recall'].append(val_recall)
            self.history['val_precision'].append(val_precision)
            self.history['val_f1'].append(val_f1)
            self.history['val_auc'].append(val_auc)
            
            # 输出
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d} | "
                      f"Train: Loss={train_loss:.4f}, Acc={train_acc:.1%}, Rec={train_recall:.1%} | "
                      f"Val: Acc={val_acc:.1%}, Rec={val_recall:.1%}, Prec={val_precision:.1%}, F1={val_f1:.3f}")
            
            # 保存最佳 (优化recall同时保持precision)
            # 目标: recall >= 0.99, precision >= 0.95
            score = val_recall * 0.6 + val_precision * 0.3 + val_f1 * 0.1
            if val_recall >= 0.95:  # 只有当recall达到95%以上才考虑保存
                score += 0.5  # 奖励
            if val_precision >= 0.90:
                score += 0.3
            
            if score > best_score:
                best_score = score
                self.best_recall = val_recall
                self.best_threshold = optimal_thresh
                patience_counter = 0
                self.save_model('best', optimal_thresh)
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        print("\n✓ Training complete!")
        print(f"Best Recall: {self.best_recall:.1%}")
        print(f"Optimal Threshold: {self.best_threshold:.2f}")
        
        self.save_model('final', self.best_threshold)
        self.generate_report()
    
    def save_model(self, suffix: str = 'final', threshold: float = 0.3):
        """保存模型"""
        torch.save({
            'model': self.model.state_dict(),
            'config': {
                'global_feat_dim': self.config.GLOBAL_FEAT_DIM,
                'hidden_dim': self.config.HIDDEN_DIM,
                'pos_weight': self.config.POS_WEIGHT,
                'threshold': threshold
            },
            'threshold': threshold,
            'history': self.history
        }, self.config.MODEL_DIR / f'gcnn_v3_{suffix}.pt')
        
        print(f"✓ Model saved: gcnn_v3_{suffix}.pt (threshold={threshold:.2f})")
    
    def generate_report(self):
        """生成报告"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], label='Train', alpha=0.8)
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], label='Val', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Focal Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率
        axes[0, 1].plot(self.history['epoch'], self.history['train_acc'], label='Train', alpha=0.8)
        axes[0, 1].plot(self.history['epoch'], self.history['val_acc'], label='Val', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Recall
        axes[0, 2].plot(self.history['epoch'], self.history['train_recall'], label='Train', alpha=0.8)
        axes[0, 2].plot(self.history['epoch'], self.history['val_recall'], label='Val', alpha=0.8)
        axes[0, 2].axhline(0.99, color='red', linestyle='--', label='Target (99%)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Recall')
        axes[0, 2].legend()
        axes[0, 2].set_title('Recall (Ferroelectric)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history['epoch'], self.history['val_precision'], color='orange')
        axes[1, 0].axhline(0.95, color='red', linestyle='--', label='Target (95%)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].set_title('Precision (Ferroelectric)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 1].plot(self.history['epoch'], self.history['val_f1'], color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 2].plot(self.history['epoch'], self.history['val_auc'], color='green')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('AUC')
        axes[1, 2].set_title('Validation AUC')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.REPORT_DIR / f'training_report_{timestamp}.png', dpi=150)
        plt.close()
        
        # 文本报告
        report_path = self.config.REPORT_DIR / f'training_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("GCNN v3 Training Report (High Recall Optimization)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Feature Dimension: {self.config.GLOBAL_FEAT_DIM}\n")
            f.write(f"Focal Loss: alpha={self.config.FOCAL_ALPHA}, gamma={self.config.FOCAL_GAMMA}\n")
            f.write(f"Positive Weight: {self.config.POS_WEIGHT}\n")
            f.write(f"Optimal Threshold: {self.best_threshold:.2f}\n")
            f.write(f"Total Epochs: {len(self.history['epoch'])}\n\n")
            
            f.write("Final Metrics:\n")
            f.write(f"  Validation Accuracy: {self.history['val_acc'][-1]:.1%}\n")
            f.write(f"  Validation Recall: {self.history['val_recall'][-1]:.1%}\n")
            f.write(f"  Validation Precision: {self.history['val_precision'][-1]:.1%}\n")
            f.write(f"  Validation F1: {self.history['val_f1'][-1]:.4f}\n")
            f.write(f"  Validation AUC: {self.history['val_auc'][-1]:.4f}\n")
            f.write(f"\nBest Recall Achieved: {self.best_recall:.1%}\n")
        
        print(f"✓ Report saved: {report_path}")


# ==========================================
# 6. 主函数
# ==========================================
def main():
    print("="*60)
    print("GCNN Classifier v3 (High Recall Optimization)")
    print("="*60)
    
    trainer = GCNNTrainerV3()
    trainer.train(epochs=200)


if __name__ == '__main__':
    main()
