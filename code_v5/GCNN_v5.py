"""
GCNN v5 - 最终优化版本
=============================================
使用多策略达成目标: Recall >= 99%, Precision >= 95%

策略组合:
1. SMOTE过采样平衡数据
2. 加权损失函数
3. 模型集成 (3个模型投票)
4. 动态阈值优化
5. 更强的正则化防止过拟合
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from feature_engineering import (
    UnifiedFeatureExtractor, FEATURE_DIM, FEATURE_NAMES, ELEMENT_DATABASE
)

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.metrics import roc_auc_score, confusion_matrix


class Config:
    GLOBAL_FEAT_DIM = FEATURE_DIM
    NODE_FEAT_DIM = 16
    HIDDEN_DIM = 384
    
    BATCH_SIZE = 32
    EPOCHS = 250
    LR = 2e-4
    WEIGHT_DECAY = 5e-5
    PATIENCE = 50
    
    # 类别权重 - 极端偏向正类
    POS_WEIGHT = 20.0
    
    # SMOTE过采样
    OVERSAMPLE_RATIO = 3.0  # 正样本过采样3倍
    
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_gcnn_v5'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_gcnn_v5'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    except:
        return None


class MaterialGraphDataset(Dataset):
    def __init__(self, data_files: List[Tuple[str, int]], extractor, oversample_ratio=1.0):
        self.graphs = []
        self.extractor = extractor
        self.oversample_ratio = oversample_ratio
        
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
        if oversample_ratio > 1.0:
            n_oversample = int(len(pos_graphs) * (oversample_ratio - 1))
            oversampled = [pos_graphs[i % len(pos_graphs)] for i in range(n_oversample)]
            pos_graphs.extend(oversampled)
        
        self.graphs = pos_graphs + neg_graphs
        np.random.shuffle(self.graphs)
        
        print(f"Loaded {len(self.graphs)} samples (Pos: {len(pos_graphs)}, Neg: {len(neg_graphs)})")
    
    def _load_file(self, file_path: str, label: int):
        graphs = []
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
                            graphs.append(graph)
                except:
                    continue
        return graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


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


class GCNNClassifierV5(nn.Module):
    """优化版分类器"""
    
    def __init__(self, config=None, dropout=0.15):
        super().__init__()
        config = config or Config()
        
        self.node_embed = nn.Sequential(
            nn.Linear(config.NODE_FEAT_DIM, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        self.gat1 = GATBlock(64, 128, heads=4, dropout=dropout)
        self.gat2 = GATBlock(128, 256, heads=4, dropout=dropout)
        self.gat3 = GATBlock(256, 256, heads=4, dropout=dropout)
        self.gat4 = GATBlock(256, 128, heads=4, dropout=dropout)
        
        graph_dim = 128 * 3
        
        self.global_encoder = nn.Sequential(
            nn.Linear(config.GLOBAL_FEAT_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 192),
            nn.LayerNorm(192),
            nn.GELU(),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(graph_dim + 192, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(config.HIDDEN_DIM, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Linear(64, 1)
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


class EnsembleClassifier:
    """集成分类器"""
    
    def __init__(self, models, device):
        self.models = models
        self.device = device
    
    def predict_proba(self, dataloader):
        all_probs = []
        all_labels = []
        
        for model in self.models:
            model.eval()
        
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                
                probs_list = []
                for model in self.models:
                    logits = model(data)
                    probs = torch.sigmoid(logits)
                    probs_list.append(probs.cpu().numpy())
                
                # 平均融合
                avg_probs = np.mean(probs_list, axis=0)
                all_probs.extend(avg_probs)
                all_labels.extend(data.y.cpu().numpy())
        
        return np.array(all_probs), np.array(all_labels)


class GCNNTrainerV5:
    def __init__(self, config=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.extractor = UnifiedFeatureExtractor()
        
        # 创建3个模型用于集成
        self.models = [
            GCNNClassifierV5(self.config, dropout=0.1).to(self.device),
            GCNNClassifierV5(self.config, dropout=0.15).to(self.device),
            GCNNClassifierV5(self.config, dropout=0.2).to(self.device),
        ]
        
        self.optimizers = [
            optim.AdamW(m.parameters(), lr=self.config.LR, weight_decay=self.config.WEIGHT_DECAY)
            for m in self.models
        ]
        
        self.schedulers = [
            optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=25, T_mult=2)
            for opt in self.optimizers
        ]
        
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.config.POS_WEIGHT]).to(self.device)
        )
        
        self.history = {
            'epoch': [], 'val_recall': [], 'val_precision': [], 'val_auc': [], 'val_f1': []
        }
        
        self.best_recall = 0
        self.best_precision = 0
        self.best_threshold = 0.5
    
    def load_data(self):
        data_files = [
            (str(self.config.DATA_DIR / 'dataset_original_ferroelectric.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_known_FE_rest.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_nonFE.jsonl'), 0),
            (str(self.config.DATA_DIR / 'dataset_polar_non_ferroelectric_final.jsonl'), 0),
        ]
        
        dataset = MaterialGraphDataset(
            data_files, self.extractor, 
            oversample_ratio=self.config.OVERSAMPLE_RATIO
        )
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        
        train_loader = GeoDataLoader(train_set, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = GeoDataLoader(val_set, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader
    
    def find_optimal_threshold(self, labels, probs, target_recall=0.99):
        best_thresh = 0.01
        best_precision = 0
        best_recall_achieved = 0
        
        for thresh in np.arange(0.01, 0.95, 0.01):
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
                        best_recall_achieved = recall
        
        # 如果达不到目标recall，找recall最高时的结果
        if best_precision == 0:
            best_recall = 0
            for thresh in np.arange(0.01, 0.5, 0.01):
                preds = (probs > thresh).astype(int)
                pos_mask = labels == 1
                if pos_mask.sum() == 0:
                    continue
                recall = (preds[pos_mask] == 1).mean()
                if recall > best_recall:
                    best_recall = recall
                    best_thresh = thresh
                    if (preds == 1).sum() > 0:
                        best_precision = (labels[preds == 1] == 1).mean()
            best_recall_achieved = best_recall
        
        return best_thresh, best_precision, best_recall_achieved
    
    def train_epoch(self, dataloader, model_idx):
        model = self.models[model_idx]
        optimizer = self.optimizers[model_idx]
        
        model.train()
        total_loss = 0
        
        for data in dataloader:
            data = data.to(self.device)
            
            optimizer.zero_grad()
            logits = model(data)
            loss = self.criterion(logits, data.y.float())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        ensemble = EnsembleClassifier(self.models, self.device)
        probs, labels = ensemble.predict_proba(dataloader)
        
        optimal_thresh, precision, recall = self.find_optimal_threshold(labels, probs, 0.99)
        
        preds = (probs > optimal_thresh).astype(int)
        
        try:
            auc_score = roc_auc_score(labels, probs)
        except:
            auc_score = 0.5
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return recall, precision, auc_score, f1, optimal_thresh
    
    def train(self, epochs=None):
        epochs = epochs or self.config.EPOCHS
        train_loader, val_loader = self.load_data()
        
        print(f"\n{'='*60}")
        print(f"GCNN v5 Ensemble Training")
        print(f"Device: {self.device}")
        print(f"Models: 3 (ensemble)")
        print(f"Positive Weight: {self.config.POS_WEIGHT}")
        print(f"Oversample Ratio: {self.config.OVERSAMPLE_RATIO}")
        print(f"{'='*60}\n")
        
        best_score = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练所有模型
            losses = []
            for i in range(len(self.models)):
                loss = self.train_epoch(train_loader, i)
                self.schedulers[i].step()
                losses.append(loss)
            
            # 集成验证
            recall, precision, auc_score, f1, optimal_thresh = self.validate(val_loader)
            
            self.history['epoch'].append(epoch + 1)
            self.history['val_recall'].append(recall)
            self.history['val_precision'].append(precision)
            self.history['val_auc'].append(auc_score)
            self.history['val_f1'].append(f1)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d} | Loss={np.mean(losses):.4f} | "
                      f"Rec={recall:.1%}, Prec={precision:.1%}, AUC={auc_score:.4f}, F1={f1:.3f}")
            
            # 保存最佳
            if recall >= 0.98:
                score = precision + recall * 0.1
                if score > best_score:
                    best_score = score
                    self.best_recall = recall
                    self.best_precision = precision
                    self.best_threshold = optimal_thresh
                    patience_counter = 0
                    self.save_model('best', optimal_thresh)
                    print(f"  ★ Best: Rec={recall:.1%}, Prec={precision:.1%}, Thresh={optimal_thresh:.2f}")
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
        print(f"Best Precision: {self.best_precision:.1%}")
        print(f"Optimal Threshold: {self.best_threshold:.2f}")
        print("="*60)
        
        self.save_model('final', self.best_threshold)
        self.generate_report()
    
    def save_model(self, suffix, threshold):
        for i, model in enumerate(self.models):
            torch.save({
                'model': model.state_dict(),
                'threshold': threshold,
                'best_recall': self.best_recall,
                'best_precision': self.best_precision,
            }, self.config.MODEL_DIR / f'gcnn_v5_model{i}_{suffix}.pt')
        
        # 保存集成配置
        torch.save({
            'n_models': len(self.models),
            'threshold': threshold,
            'best_recall': self.best_recall,
            'best_precision': self.best_precision,
        }, self.config.MODEL_DIR / f'gcnn_v5_ensemble_{suffix}.pt')
    
    def generate_report(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(self.history['epoch'], self.history['val_recall'])
        axes[0, 0].axhline(0.99, color='red', linestyle='--', label='Target (99%)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Recall')
        axes[0, 0].set_title('Recall (Ferroelectric)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.history['epoch'], self.history['val_precision'], color='orange')
        axes[0, 1].axhline(0.95, color='red', linestyle='--', label='Target (95%)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision (Ferroelectric)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.history['epoch'], self.history['val_f1'], color='purple')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(self.history['epoch'], self.history['val_auc'], color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].set_title('ROC-AUC')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.REPORT_DIR / f'training_report_{timestamp}.png', dpi=150)
        plt.close()
        
        with open(self.config.REPORT_DIR / f'training_report_{timestamp}.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("GCNN v5 Ensemble Training Report\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Models: 3 (ensemble)\n")
            f.write(f"Positive Weight: {self.config.POS_WEIGHT}\n")
            f.write(f"Oversample Ratio: {self.config.OVERSAMPLE_RATIO}\n")
            f.write(f"Optimal Threshold: {self.best_threshold:.2f}\n\n")
            f.write(f"Best Results:\n")
            f.write(f"  Recall: {self.best_recall:.1%}\n")
            f.write(f"  Precision: {self.best_precision:.1%}\n")
            f.write(f"  Max AUC: {max(self.history['val_auc']):.4f}\n")


def main():
    print("="*60)
    print("GCNN v5 - Ensemble + Oversampling")
    print("="*60)
    
    trainer = GCNNTrainerV5()
    trainer.train()


if __name__ == '__main__':
    main()
