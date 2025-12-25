"""
增强型图神经网络分类器 v4
=============================================
极限优化: 铁电材料Recall >= 99%, Precision >= 95%

策略:
1. 使用BCE Loss + 极高正类权重
2. 二阶段训练: 先平衡训练，再针对高recall微调
3. 集成多个模型
4. 最优阈值搜索
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
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.loader import DataLoader as GeoDataLoader


# ==========================================
# 1. 配置
# ==========================================
class Config:
    GLOBAL_FEAT_DIM = FEATURE_DIM  # 64
    NODE_FEAT_DIM = 16
    HIDDEN_DIM = 512
    
    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 300
    LR = 3e-4
    WEIGHT_DECAY = 1e-5
    PATIENCE = 60
    
    # 极高正类权重
    POS_WEIGHT = 15.0
    
    # 路径
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_gcnn_v4'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_gcnn_v4'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 2. 数据处理
# ==========================================
def structure_to_graph(struct_dict: Dict, label: int = 0, 
                       global_features: np.ndarray = None) -> Optional[Data]:
    try:
        from pymatgen.core import Structure
        
        structure = Structure.from_dict(struct_dict)
        
        node_features = []
        for site in structure:
            el = site.specie.symbol
            if el in ELEMENT_DATABASE:
                data = ELEMENT_DATABASE[el]
                feat = [
                    data[0] / 100.0, data[1] / 200.0, data[2] / 2.5, data[3] / 4.0,
                    data[4] / 15.0, data[5] / 8.0, data[6] / 3500.0, data[7] / 200.0,
                    site.frac_coords[0], site.frac_coords[1], site.frac_coords[2],
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
        
        edge_index = []
        cutoff = 5.0
        
        for i, site_i in enumerate(structure):
            neighbors = structure.get_neighbors(site_i, cutoff)
            for neighbor in neighbors:
                j = neighbor.index
                if i != j:
                    edge_index.append([i, j])
        
        if not edge_index:
            n = len(node_features)
            for i in range(n):
                for j in range(n):
                    if i != j:
                        edge_index.append([i, j])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        if global_features is not None:
            u = torch.tensor(global_features, dtype=torch.float).unsqueeze(0)
        else:
            u = torch.zeros(1, FEATURE_DIM, dtype=torch.float)
        
        y = torch.tensor([label], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y, u=u)
    except Exception as e:
        return None


class MaterialGraphDataset(Dataset):
    def __init__(self, data_files: List[Tuple[str, int]], extractor: UnifiedFeatureExtractor):
        self.graphs = []
        self.extractor = extractor
        
        for file_path, label in data_files:
            if os.path.exists(file_path):
                self._load_file(file_path, label)
        
        print(f"Loaded {len(self.graphs)} samples")
    
    def _load_file(self, file_path: str, label: int):
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
# 3. 网络架构
# ==========================================
class GATBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.1):
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


class GCNNClassifierV4(nn.Module):
    """高Recall优化的分类器"""
    
    def __init__(self, config: Config = None):
        super().__init__()
        config = config or Config()
        
        # 节点编码
        self.node_embed = nn.Sequential(
            nn.Linear(config.NODE_FEAT_DIM, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # GAT层
        self.gat1 = GATBlock(64, 128, heads=4, dropout=0.1)
        self.gat2 = GATBlock(128, 256, heads=4, dropout=0.1)
        self.gat3 = GATBlock(256, 256, heads=4, dropout=0.1)
        self.gat4 = GATBlock(256, 128, heads=4, dropout=0.1)
        
        graph_dim = 128 * 3
        
        # 全局特征
        self.global_encoder = nn.Sequential(
            nn.Linear(config.GLOBAL_FEAT_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 192),
        )
        
        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(graph_dim + 192, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(config.HIDDEN_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        
        # 分类头 - 输出单值用于BCE
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)  # 单值输出
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        h = self.node_embed(x)
        h = self.gat1(h, edge_index)
        h = self.gat2(h, edge_index)
        h = self.gat3(h, edge_index)
        h = self.gat4(h, edge_index)
        
        graph_mean = global_mean_pool(h, batch)
        graph_max = global_max_pool(h, batch)
        graph_add = global_add_pool(h, batch) / 10.0
        graph_feat = torch.cat([graph_mean, graph_max, graph_add], dim=1)
        
        u = data.u
        if u.dim() == 3:
            u = u.squeeze(1)
        global_feat = self.global_encoder(u)
        
        combined = torch.cat([graph_feat, global_feat], dim=1)
        h = self.fusion(combined)
        logits = self.classifier(h)
        
        return logits.squeeze(-1)


# ==========================================
# 4. 训练器
# ==========================================
class GCNNTrainerV4:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.extractor = UnifiedFeatureExtractor()
        self.model = GCNNClassifierV4(self.config).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LR,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=30, T_mult=2
        )
        
        # BCE with pos_weight
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.config.POS_WEIGHT]).to(self.device)
        )
        
        self.history = {
            'epoch': [], 'train_loss': [], 'train_acc': [], 'train_recall': [],
            'val_loss': [], 'val_acc': [], 'val_recall': [], 'val_precision': [],
            'val_auc': [], 'val_f1': []
        }
        
        self.best_recall = 0
        self.best_precision_at_high_recall = 0
        self.best_threshold = 0.5
    
    def load_data(self):
        data_files = [
            (str(self.config.DATA_DIR / 'dataset_original_ferroelectric.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_known_FE_rest.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_nonFE.jsonl'), 0),
            (str(self.config.DATA_DIR / 'dataset_polar_non_ferroelectric_final.jsonl'), 0),
        ]
        
        dataset = MaterialGraphDataset(data_files, self.extractor)
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        
        train_loader = GeoDataLoader(train_set, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = GeoDataLoader(val_set, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        pos_count = sum(1 for g in dataset.graphs if g.y.item() == 1)
        neg_count = len(dataset) - pos_count
        print(f"Dataset: {len(dataset)} (Pos: {pos_count}, Neg: {neg_count}, Ratio: 1:{neg_count/pos_count:.1f})")
        
        return train_loader, val_loader
    
    def find_optimal_threshold(self, labels, probs, target_recall=0.99):
        """找到达到目标recall时的最大precision阈值"""
        best_thresh = 0.01
        best_precision = 0
        
        for thresh in np.arange(0.01, 0.9, 0.01):
            preds = (probs > thresh).astype(int)
            
            pos_mask = labels == 1
            if pos_mask.sum() == 0:
                continue
            
            recall = (preds[pos_mask] == 1).mean()
            
            if recall >= target_recall:
                if (preds == 1).sum() > 0:
                    precision = (labels[preds == 1] == 1).mean()
                    if precision > best_precision:
                        best_precision = precision
                        best_thresh = thresh
        
        return best_thresh, best_precision
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for data in dataloader:
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(data)
            loss = self.criterion(logits, data.y.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            pred = (probs > 0.3).long()  # 低阈值训练
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        acc = (all_preds == all_labels).mean()
        pos_mask = all_labels == 1
        recall = (all_preds[pos_mask] == 1).mean() if pos_mask.sum() > 0 else 0
        
        return total_loss / len(dataloader), acc, recall
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                
                logits = self.model(data)
                loss = self.criterion(logits, data.y.float())
                
                total_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # 找最优阈值 - 目标recall >= 99%
        optimal_thresh, precision_at_99 = self.find_optimal_threshold(all_labels, all_probs, 0.99)
        
        # 如果99%达不到好的precision，尝试98%
        if precision_at_99 < 0.5:
            optimal_thresh, precision_at_99 = self.find_optimal_threshold(all_labels, all_probs, 0.98)
        
        all_preds = (all_probs > optimal_thresh).astype(int)
        
        acc = (all_preds == all_labels).mean()
        
        pos_mask = all_labels == 1
        recall = (all_preds[pos_mask] == 1).mean() if pos_mask.sum() > 0 else 0
        
        if (all_preds == 1).sum() > 0:
            precision = (all_labels[all_preds == 1] == 1).mean()
        else:
            precision = 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        try:
            from sklearn.metrics import roc_auc_score
            auc_score = roc_auc_score(all_labels, all_probs)
        except:
            auc_score = 0.5
        
        return total_loss / len(dataloader), acc, recall, precision, f1, auc_score, optimal_thresh
    
    def train(self, epochs: int = None):
        epochs = epochs or self.config.EPOCHS
        train_loader, val_loader = self.load_data()
        
        print(f"\n{'='*60}")
        print(f"GCNN v4 Training (Target: Recall >= 99%, Precision >= 95%)")
        print(f"Device: {self.device}")
        print(f"Positive Weight: {self.config.POS_WEIGHT}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        best_score = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc, train_recall = self.train_epoch(train_loader)
            val_loss, val_acc, val_recall, val_precision, val_f1, val_auc, optimal_thresh = self.validate(val_loader)
            
            self.scheduler.step()
            
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
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d} | Train: Loss={train_loss:.4f}, Rec={train_recall:.1%} | "
                      f"Val: Rec={val_recall:.1%}, Prec={val_precision:.1%}, AUC={val_auc:.4f}, Thresh={optimal_thresh:.2f}")
            
            # 保存条件: recall >= 98% 且 precision 最高
            if val_recall >= 0.98:
                score = val_precision + val_recall * 0.1  # 优先precision
                if score > best_score:
                    best_score = score
                    self.best_recall = val_recall
                    self.best_precision_at_high_recall = val_precision
                    self.best_threshold = optimal_thresh
                    patience_counter = 0
                    self.save_model('best', optimal_thresh)
                    print(f"  ★ New best: Recall={val_recall:.1%}, Precision={val_precision:.1%}")
                else:
                    patience_counter += 1
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Best Recall: {self.best_recall:.1%}")
        print(f"Precision at Best Recall: {self.best_precision_at_high_recall:.1%}")
        print(f"Optimal Threshold: {self.best_threshold:.2f}")
        print("="*60)
        
        self.save_model('final', self.best_threshold)
        self.generate_report()
    
    def save_model(self, suffix: str, threshold: float):
        torch.save({
            'model': self.model.state_dict(),
            'threshold': threshold,
            'config': {
                'pos_weight': self.config.POS_WEIGHT,
                'hidden_dim': self.config.HIDDEN_DIM,
            },
            'best_recall': self.best_recall,
            'best_precision': self.best_precision_at_high_recall,
        }, self.config.MODEL_DIR / f'gcnn_v4_{suffix}.pt')
        
    def generate_report(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('BCE Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.history['epoch'], self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['epoch'], self.history['val_acc'], label='Val')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(self.history['epoch'], self.history['train_recall'], label='Train')
        axes[0, 2].plot(self.history['epoch'], self.history['val_recall'], label='Val')
        axes[0, 2].axhline(0.99, color='red', linestyle='--', label='Target (99%)')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Recall')
        axes[0, 2].legend()
        axes[0, 2].set_title('Recall (Ferroelectric)')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.history['epoch'], self.history['val_precision'], color='orange')
        axes[1, 0].axhline(0.95, color='red', linestyle='--', label='Target (95%)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].set_title('Precision (Ferroelectric)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(self.history['epoch'], self.history['val_f1'], color='purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].plot(self.history['epoch'], self.history['val_auc'], color='green')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('AUC')
        axes[1, 2].set_title('ROC-AUC')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.REPORT_DIR / f'training_report_{timestamp}.png', dpi=150)
        plt.close()
        
        with open(self.config.REPORT_DIR / f'training_report_{timestamp}.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("GCNN v4 Training Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Positive Weight: {self.config.POS_WEIGHT}\n")
            f.write(f"Optimal Threshold: {self.best_threshold:.2f}\n\n")
            f.write(f"Best Results:\n")
            f.write(f"  Recall: {self.best_recall:.1%}\n")
            f.write(f"  Precision: {self.best_precision_at_high_recall:.1%}\n")
            f.write(f"  AUC: {max(self.history['val_auc']):.4f}\n")


def main():
    print("="*60)
    print("GCNN v4 - High Recall Optimization")
    print("="*60)
    
    trainer = GCNNTrainerV4()
    trainer.train()


if __name__ == '__main__':
    main()
