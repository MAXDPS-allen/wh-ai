"""
GCNN v6 - 使用扩展负样本重新训练
================================
使用清理后的扩展负样本数据集重新训练分类模型
目标: 减少假阳性，提高精确度
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
from tqdm import tqdm


class Config:
    GLOBAL_FEAT_DIM = FEATURE_DIM
    NODE_FEAT_DIM = 16
    HIDDEN_DIM = 256  # 减小隐藏层维度
    
    BATCH_SIZE = 16  # 减小batch size防止OOM
    EPOCHS = 200
    LR = 2e-4
    WEIGHT_DECAY = 5e-5
    PATIENCE = 40
    
    # 类别权重 - 因为负样本更多，降低正类权重
    POS_WEIGHT = 10.0  # 从20降到10
    
    # 不需要过采样，因为负样本已经很多
    OVERSAMPLE_RATIO = 2.0
    
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_gcnn_v6'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_gcnn_v6'
    
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
                print(f"Loading {Path(file_path).name}...")
                graphs = self._load_file(file_path, label)
                print(f"  Loaded {len(graphs)} graphs")
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
        
        print(f"\nTotal: {len(self.graphs)} samples (Pos: {len(pos_graphs)}, Neg: {len(neg_graphs)})")
        print(f"Positive ratio: {len(pos_graphs)/len(self.graphs)*100:.1f}%")
    
    def _load_file(self, file_path: str, label: int):
        graphs = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"  Processing", leave=False):
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


class GCNNClassifierV6(nn.Module):
    """优化版分类器 - 减少复杂度防止OOM"""
    
    def __init__(self, config=None, dropout=0.15):
        super().__init__()
        config = config or Config()
        
        self.node_embed = nn.Sequential(
            nn.Linear(config.NODE_FEAT_DIM, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        
        # 简化GAT网络 - 减少头数和维度
        self.gat1 = GATBlock(64, 128, heads=2, dropout=dropout)
        self.gat2 = GATBlock(128, 128, heads=2, dropout=dropout)
        self.gat3 = GATBlock(128, 128, heads=2, dropout=dropout)
        self.gat4 = GATBlock(128, 64, heads=2, dropout=dropout)
        
        graph_dim = 64 * 3
        
        self.global_encoder = nn.Sequential(
            nn.Linear(config.GLOBAL_FEAT_DIM, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(graph_dim + 128, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(config.HIDDEN_DIM, 128),
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
                
                avg_probs = np.mean(probs_list, axis=0)
                all_probs.extend(avg_probs)
                all_labels.extend(data.y.cpu().numpy())
        
        return np.array(all_probs), np.array(all_labels)


class GCNNTrainerV6:
    def __init__(self, config=None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.extractor = UnifiedFeatureExtractor()
        
        # 创建3个模型用于集成
        self.models = [
            GCNNClassifierV6(self.config, dropout=0.1).to(self.device),
            GCNNClassifierV6(self.config, dropout=0.15).to(self.device),
            GCNNClassifierV6(self.config, dropout=0.2).to(self.device),
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
        
        self.best_f1 = 0
        self.best_precision = 0
        self.best_threshold = 0.5
    
    def load_data(self):
        """使用扩展后的负样本数据集"""
        data_files = [
            # 正样本 (铁电)
            (str(self.config.DATA_DIR / 'dataset_original_ferroelectric.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_known_FE_rest.jsonl'), 1),
            # 负样本 (使用扩展后的数据集)
            (str(self.config.DATA_DIR / 'dataset_nonFE_expanded.jsonl'), 0),
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
        """寻找满足目标recall的最佳阈值"""
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
        print(f"GCNN v6 Training with Expanded Negative Samples")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Models: 3 (ensemble)")
        print(f"Positive Weight: {self.config.POS_WEIGHT}")
        print(f"{'='*60}\n")
        
        no_improve = 0
        
        for epoch in range(epochs):
            # 训练所有模型
            losses = []
            for i in range(len(self.models)):
                loss = self.train_epoch(train_loader, i)
                losses.append(loss)
                self.schedulers[i].step()
            
            avg_loss = np.mean(losses)
            
            # 验证
            recall, precision, auc, f1, thresh = self.validate(val_loader)
            
            self.history['epoch'].append(epoch)
            self.history['val_recall'].append(recall)
            self.history['val_precision'].append(precision)
            self.history['val_auc'].append(auc)
            self.history['val_f1'].append(f1)
            
            # 根据F1和精确度综合判断
            score = f1 * 0.7 + precision * 0.3
            
            if score > self.best_f1:
                self.best_f1 = score
                self.best_precision = precision
                self.best_threshold = thresh
                no_improve = 0
                
                # 保存最佳模型
                for i, model in enumerate(self.models):
                    torch.save({
                        'model': model.state_dict(),
                        'threshold': thresh,
                        'recall': recall,
                        'precision': precision,
                        'f1': f1,
                    }, self.config.MODEL_DIR / f'gcnn_v6_model{i}_best.pt')
                
                print(f"[{epoch:3d}] Loss={avg_loss:.4f} | Recall={recall:.4f} | "
                      f"Precision={precision:.4f} | F1={f1:.4f} | AUC={auc:.4f} | "
                      f"Thresh={thresh:.2f} ⭐ BEST")
            else:
                no_improve += 1
                if epoch % 10 == 0:
                    print(f"[{epoch:3d}] Loss={avg_loss:.4f} | Recall={recall:.4f} | "
                          f"Precision={precision:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
            
            if no_improve >= self.config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # 保存训练配置
        config_info = {
            'best_threshold': self.best_threshold,
            'best_precision': self.best_precision,
            'best_f1': self.best_f1,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epochs_trained': epoch + 1,
        }
        with open(self.config.MODEL_DIR / 'training_config.json', 'w') as f:
            json.dump(config_info, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best F1: {self.best_f1:.4f}")
        print(f"Best Precision: {self.best_precision:.4f}")
        print(f"Best Threshold: {self.best_threshold:.2f}")
        print(f"Models saved to: {self.config.MODEL_DIR}")
        print(f"{'='*60}")
        
        return self.history


if __name__ == "__main__":
    trainer = GCNNTrainerV6()
    trainer.train()
