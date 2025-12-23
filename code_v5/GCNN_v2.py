"""
增强型图神经网络分类器 v2
=============================================
使用统一特征工程模块 (64维特征)
铁电材料分类模型

功能:
1. 图神经网络特征提取
2. 结合64维全局特征
3. 铁电/非铁电分类
4. 模型保存和报告生成
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
    from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool
    from torch_geometric.loader import DataLoader as GeoDataLoader
    HAS_GEOMETRIC = True
except ImportError:
    HAS_GEOMETRIC = False
    print("Warning: torch_geometric not found. Using MLP fallback.")


# ==========================================
# 1. 配置
# ==========================================
class Config:
    # 特征维度 (使用统一特征)
    GLOBAL_FEAT_DIM = FEATURE_DIM  # 64
    NODE_FEAT_DIM = 16             # 节点特征
    EDGE_FEAT_DIM = 4              # 边特征
    HIDDEN_DIM = 256               # 隐藏层
    
    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 150
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE = 25
    
    # 路径
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_gcnn_v2'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_gcnn_v2'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 2. 数据处理
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
        
        # 节点特征
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
                    # 额外特征
                    site.frac_coords[0],
                    site.frac_coords[1],
                    site.frac_coords[2],
                    0.0, 0.0, 0.0, 0.0, 0.0  # padding to 16
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
                    
                    # 边特征
                    dist = neighbor.nn_distance if hasattr(neighbor, 'nn_distance') else 2.0
                    edge_feat = [
                        dist / cutoff,
                        1.0 / max(dist, 0.1) / 10.0,
                        np.exp(-dist / 2),
                        1.0 if dist < 3.0 else 0.0
                    ]
                    edge_attr.append(edge_feat)
        
        if not edge_index:
            # 创建全连接图
            n = len(node_features)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edge_index.append([i, j])
                        edge_attr.append([0.5, 0.5, 0.5, 0.5])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # 全局特征
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
                        # 提取全局特征
                        global_feat = self.extractor.extract_from_structure_dict(struct, sg)
                        
                        # 转换为图
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
# 3. 网络架构
# ==========================================
class GATBlock(nn.Module):
    """GAT块"""
    
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.gat = GATConv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        h = self.norm(h)
        h = self.act(h)
        h = self.dropout(h)
        return h


class GCNNClassifier(nn.Module):
    """图神经网络分类器"""
    
    def __init__(self, config: Config = None):
        super().__init__()
        
        config = config or Config()
        self.config = config
        
        if HAS_GEOMETRIC:
            # 图编码器
            self.node_embed = nn.Linear(config.NODE_FEAT_DIM, 64)
            
            self.gat1 = GATBlock(64, 128, heads=4)
            self.gat2 = GATBlock(128, 128, heads=4)
            self.gat3 = GATBlock(128, 64, heads=4)
            
            graph_dim = 64
        else:
            graph_dim = 0
        
        # 全局特征处理
        self.global_encoder = nn.Sequential(
            nn.Linear(config.GLOBAL_FEAT_DIM, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        
        # 融合
        fusion_dim = graph_dim + 128
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(config.HIDDEN_DIM, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
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
            
            # 池化
            graph_feat = global_mean_pool(h, batch)
            
            # 全局特征
            u = data.u
            if u.dim() == 3:
                u = u.squeeze(1)
            global_feat = self.global_encoder(u)
            
            # 融合
            combined = torch.cat([graph_feat, global_feat], dim=1)
        else:
            # 仅使用全局特征
            u = data.u
            if u.dim() == 3:
                u = u.squeeze(1)
            combined = self.global_encoder(u)
        
        h = self.fusion(combined)
        logits = self.classifier(h)
        
        return logits


class MLPClassifier(nn.Module):
    """MLP分类器 (后备方案)"""
    
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        return self.net(x)


# ==========================================
# 4. 训练器
# ==========================================
class GCNNTrainer:
    """GCNN训练器"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        # 创建目录
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 特征提取器
        self.extractor = UnifiedFeatureExtractor()
        
        # 模型
        if HAS_GEOMETRIC:
            self.model = GCNNClassifier(self.config).to(self.device)
        else:
            self.model = MLPClassifier().to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LR,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.config.EPOCHS
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': []
        }
    
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
        
        if HAS_GEOMETRIC:
            train_loader = GeoDataLoader(train_set, batch_size=self.config.BATCH_SIZE, shuffle=True)
            val_loader = GeoDataLoader(val_set, batch_size=self.config.BATCH_SIZE, shuffle=False)
        else:
            train_loader = DataLoader(train_set, batch_size=self.config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in dataloader:
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(data)
            loss = self.criterion(logits, data.y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
        
        return total_loss / len(dataloader), correct / total
    
    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                
                logits = self.model(data)
                loss = self.criterion(logits, data.y)
                
                total_loss += loss.item()
                
                probs = F.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        acc = (all_preds == all_labels).mean()
        
        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        
        return total_loss / len(dataloader), acc, auc
    
    def train(self, epochs: int = None):
        """完整训练"""
        epochs = epochs or self.config.EPOCHS
        train_loader, val_loader = self.load_data()
        
        print(f"\n{'='*60}")
        print(f"GCNN Training (64-dim features)")
        print(f"Device: {self.device}")
        print(f"Model: {'GCNNClassifier' if HAS_GEOMETRIC else 'MLPClassifier'}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_auc = self.validate(val_loader)
            
            self.scheduler.step()
            
            # 记录
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            
            # 输出
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d} | "
                      f"Train: {train_loss:.4f} ({train_acc:.1%}) | "
                      f"Val: {val_loss:.4f} ({val_acc:.1%}) | "
                      f"AUC: {val_auc:.4f}")
            
            # 保存最佳
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model('best')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        print("\n✓ Training complete!")
        self.save_model('final')
        self.generate_report()
    
    def save_model(self, suffix: str = 'final'):
        """保存模型"""
        torch.save({
            'model': self.model.state_dict(),
            'config': {
                'global_feat_dim': self.config.GLOBAL_FEAT_DIM,
                'hidden_dim': self.config.HIDDEN_DIM,
                'has_geometric': HAS_GEOMETRIC
            }
        }, self.config.MODEL_DIR / f'gcnn_v2_{suffix}.pt')
        
        print(f"✓ Model saved: gcnn_v2_{suffix}.pt")
    
    def generate_report(self):
        """生成报告"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(self.history['epoch'], self.history['train_loss'], label='Train', alpha=0.8)
        axes[0].plot(self.history['epoch'], self.history['val_loss'], label='Val', alpha=0.8)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Training Loss')
        axes[0].grid(True, alpha=0.3)
        
        # 准确率
        axes[1].plot(self.history['epoch'], self.history['train_acc'], label='Train', alpha=0.8)
        axes[1].plot(self.history['epoch'], self.history['val_acc'], label='Val', alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].set_title('Accuracy')
        axes[1].grid(True, alpha=0.3)
        
        # AUC
        axes[2].plot(self.history['epoch'], self.history['val_auc'], color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].set_title('Validation AUC')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.REPORT_DIR / f'training_report_{timestamp}.png', dpi=150)
        plt.close()
        
        # 文本报告
        report_path = self.config.REPORT_DIR / f'training_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("GCNN Training Report (64-dim features)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Feature Dimension: {self.config.GLOBAL_FEAT_DIM}\n")
            f.write(f"Model Type: {'GCNN' if HAS_GEOMETRIC else 'MLP'}\n")
            f.write(f"Total Epochs: {len(self.history['epoch'])}\n\n")
            
            f.write("Final Metrics:\n")
            f.write(f"  Validation Accuracy: {self.history['val_acc'][-1]:.1%}\n")
            f.write(f"  Validation AUC: {self.history['val_auc'][-1]:.4f}\n")
            f.write(f"  Best Accuracy: {max(self.history['val_acc']):.1%}\n")
        
        print(f"✓ Report saved: {report_path}")


# ==========================================
# 5. 主函数
# ==========================================
def main():
    print("="*60)
    print("GCNN Classifier v2 (64-dim unified features)")
    print("="*60)
    
    trainer = GCNNTrainer()
    trainer.train(epochs=100)


if __name__ == '__main__':
    main()
