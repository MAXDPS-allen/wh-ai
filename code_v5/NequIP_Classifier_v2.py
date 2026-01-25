"""
NequIP Classifier v2 - 使用扩展负样本重新训练
==============================================
基于E(3)-等变神经网络的铁电材料分类器
使用清理后的扩展负样本数据集
"""

import sys
import os
import json
import torch
import torch.nn as nn
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

from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class NequIPConfigV2:
    """NequIP v2 分类器配置"""
    
    GLOBAL_FEAT_DIM = FEATURE_DIM
    NUM_SPECIES = 100
    
    LMAX = 2
    SH_DIM = 9
    NUM_RADIAL_BASIS = 8
    CUTOFF = 5.0
    
    NUM_LAYERS = 3  # 减少层数
    NODE_DIM = 48   # 减小节点维度
    HIDDEN_DIM = 192  # 减小隐藏层
    
    BATCH_SIZE = 16  # 减小batch size防止OOM
    EPOCHS = 150
    LR = 5e-4
    WEIGHT_DECAY = 5e-4
    PATIENCE = 50
    
    # 调整正类权重（因为负样本更多）
    POS_WEIGHT = 8.0
    OVERSAMPLE_RATIO = 2.0
    
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_nequip_v2'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_nequip_v2'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== 径向基函数 ====================
class BesselBasis(nn.Module):
    def __init__(self, num_basis: int = 8, cutoff: float = 5.0):
        super().__init__()
        self.num_basis = num_basis
        self.cutoff = cutoff
        self.register_buffer('freq', torch.arange(1, num_basis + 1) * np.pi / cutoff)
    
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = r.unsqueeze(-1)
        basis = torch.sqrt(torch.tensor(2.0 / self.cutoff, device=r.device)) * \
                torch.sin(self.freq * r) / (r + 1e-8)
        return basis


class SmoothCutoff(nn.Module):
    def __init__(self, cutoff: float = 5.0, p: int = 6):
        super().__init__()
        self.cutoff = cutoff
        self.p = p
    
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        x = r / self.cutoff
        envelope = (1 - x.pow(self.p)).pow(2)
        envelope = torch.where(r < self.cutoff, envelope, torch.zeros_like(envelope))
        return envelope


class SphericalHarmonics(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, vec: torch.Tensor) -> torch.Tensor:
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


# ==================== 消息传递层 ====================
class RadialMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class EquivariantMessageLayer(nn.Module):
    def __init__(self, node_dim: int, sh_dim: int, radial_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.node_dim = node_dim
        self.sh_dim = sh_dim
        
        self.radial_mlp = RadialMLP(radial_dim, hidden_dim, node_dim * sh_dim, dropout)
        self.self_interaction = nn.Linear(node_dim, node_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + node_dim, hidden_dim),
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


# ==================== NequIP 分类器 ====================
class NequIPClassifierV2(nn.Module):
    def __init__(self, config=None, dropout=0.15):
        super().__init__()
        self.config = config or NequIPConfigV2()
        
        self.atom_embedding = nn.Embedding(self.config.NUM_SPECIES, self.config.NODE_DIM)
        self.radial_basis = BesselBasis(self.config.NUM_RADIAL_BASIS, self.config.CUTOFF)
        self.cutoff_fn = SmoothCutoff(self.config.CUTOFF)
        self.spherical_harmonics = SphericalHarmonics()
        
        self.message_layers = nn.ModuleList([
            EquivariantMessageLayer(
                self.config.NODE_DIM, self.config.SH_DIM, 
                self.config.NUM_RADIAL_BASIS, self.config.HIDDEN_DIM // 2, dropout
            ) for _ in range(self.config.NUM_LAYERS)
        ])
        
        graph_dim = self.config.NODE_DIM * 3
        
        self.global_encoder = nn.Sequential(
            nn.Linear(self.config.GLOBAL_FEAT_DIM, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(graph_dim + 192, self.config.HIDDEN_DIM),
            nn.LayerNorm(self.config.HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(self.config.HIDDEN_DIM, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        
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
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)
    
    def forward(self, data) -> torch.Tensor:
        x = data.x
        edge_index = data.edge_index
        edge_vec = data.edge_vec
        edge_length = data.edge_length
        batch = data.batch
        
        x_clamped = x.clamp(0, self.config.NUM_SPECIES - 1).long()
        h = self.atom_embedding(x_clamped)
        
        edge_unit = edge_vec / (edge_length.unsqueeze(-1) + 1e-8)
        edge_sh = self.spherical_harmonics(edge_unit)
        edge_radial = self.radial_basis(edge_length)
        cutoff_envelope = self.cutoff_fn(edge_length)
        edge_radial = edge_radial * cutoff_envelope.unsqueeze(-1)
        
        for layer in self.message_layers:
            h = layer(h, edge_index, edge_sh, edge_radial)
        
        graph_mean = global_mean_pool(h, batch)
        graph_max = global_max_pool(h, batch)
        graph_sum = global_add_pool(h, batch) / 10.0
        graph_feat = torch.cat([graph_mean, graph_max, graph_sum], dim=-1)
        
        u = data.u
        if u.dim() == 3:
            u = u.squeeze(1)
        global_feat = self.global_encoder(u)
        
        combined = torch.cat([graph_feat, global_feat], dim=-1)
        h_fused = self.fusion(combined)
        logits = self.classifier(h_fused)
        
        return logits.squeeze(-1)


# ==================== 数据处理 ====================
def structure_to_nequip_graph(struct_dict, label=0, global_features=None, cutoff=5.0):
    try:
        from pymatgen.core import Structure
        structure = Structure.from_dict(struct_dict)
        
        atomic_numbers = [site.specie.Z for site in structure]
        x = torch.tensor(atomic_numbers, dtype=torch.long)
        
        edge_index, edge_vec, edge_length = [], [], []
        
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
            n = len(atomic_numbers)
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
        
        if global_features is not None:
            u = torch.tensor(global_features, dtype=torch.float).unsqueeze(0)
        else:
            u = torch.zeros(1, FEATURE_DIM, dtype=torch.float)
        
        y = torch.tensor([label], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, edge_vec=edge_vec, edge_length=edge_length, y=y, u=u)
    except:
        return None


class NequIPDatasetV2(Dataset):
    def __init__(self, data_files, extractor, cutoff=5.0, oversample_ratio=1.0):
        self.graphs = []
        self.extractor = extractor
        self.cutoff = cutoff
        
        pos_graphs, neg_graphs = [], []
        
        for file_path, label in data_files:
            if os.path.exists(file_path):
                print(f"Loading {Path(file_path).name}...")
                graphs = self._load_file(file_path, label)
                print(f"  Loaded {len(graphs)} graphs")
                if label == 1:
                    pos_graphs.extend(graphs)
                else:
                    neg_graphs.extend(graphs)
        
        if oversample_ratio > 1.0 and len(pos_graphs) > 0:
            n_oversample = int(len(pos_graphs) * (oversample_ratio - 1))
            oversampled = [pos_graphs[i % len(pos_graphs)] for i in range(n_oversample)]
            pos_graphs.extend(oversampled)
        
        self.graphs = pos_graphs + neg_graphs
        np.random.shuffle(self.graphs)
        
        print(f"\nTotal: {len(self.graphs)} samples (Pos: {len(pos_graphs)}, Neg: {len(neg_graphs)})")
    
    def _load_file(self, file_path, label):
        graphs = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="  Processing", leave=False):
                try:
                    item = json.loads(line)
                    struct = item.get('structure')
                    sg = item.get('spacegroup_number')
                    if struct:
                        global_feat = self.extractor.extract_from_structure_dict(struct, sg)
                        graph = structure_to_nequip_graph(struct, label, global_feat, self.cutoff)
                        if graph is not None:
                            graphs.append(graph)
                except:
                    continue
        return graphs
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]


# ==================== 训练器 ====================
class NequIPTrainerV2:
    def __init__(self, config=None):
        self.config = config or NequIPConfigV2()
        self.device = self.config.DEVICE
        
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.extractor = UnifiedFeatureExtractor()
        self.model = NequIPClassifierV2(self.config, dropout=0.15).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.LR, weight_decay=self.config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=25, T_mult=2)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.config.POS_WEIGHT]).to(self.device)
        )
        
        self.best_f1 = 0
        self.best_threshold = 0.5
    
    def load_data(self):
        data_files = [
            (str(self.config.DATA_DIR / 'dataset_original_ferroelectric.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_known_FE_rest.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_nonFE_expanded.jsonl'), 0),
        ]
        
        dataset = NequIPDatasetV2(
            data_files, self.extractor, 
            cutoff=self.config.CUTOFF,
            oversample_ratio=self.config.OVERSAMPLE_RATIO
        )
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        
        train_loader = GeoDataLoader(train_set, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = GeoDataLoader(val_set, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader
    
    def find_optimal_threshold(self, labels, probs, target_recall=0.99):
        best_thresh, best_precision, best_recall_achieved = 0.01, 0, 0
        
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
            for thresh in np.arange(0.01, 0.5, 0.01):
                preds = (probs > thresh).astype(int)
                pos_mask = labels == 1
                if pos_mask.sum() == 0:
                    continue
                recall = (preds[pos_mask] == 1).mean()
                if recall > best_recall_achieved:
                    best_recall_achieved = recall
                    best_thresh = thresh
                    if (preds == 1).sum() > 0:
                        best_precision = (labels[preds == 1] == 1).mean()
        
        return best_thresh, best_precision, best_recall_achieved
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for data in dataloader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = self.criterion(logits, data.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        all_probs, all_labels = [], []
        
        with torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                logits = self.model(data)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(data.y.cpu().numpy())
        
        probs, labels = np.array(all_probs), np.array(all_labels)
        thresh, precision, recall = self.find_optimal_threshold(labels, probs, 0.99)
        
        try:
            auc = roc_auc_score(labels, probs)
        except:
            auc = 0.5
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return recall, precision, auc, f1, thresh
    
    def train(self, epochs=None):
        epochs = epochs or self.config.EPOCHS
        train_loader, val_loader = self.load_data()
        
        print(f"\n{'='*60}")
        print(f"NequIP v2 Training with Expanded Negative Samples")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Positive Weight: {self.config.POS_WEIGHT}")
        print(f"{'='*60}\n")
        
        no_improve = 0
        
        for epoch in range(epochs):
            loss = self.train_epoch(train_loader)
            self.scheduler.step()
            
            recall, precision, auc, f1, thresh = self.validate(val_loader)
            
            score = f1 * 0.7 + precision * 0.3
            
            if score > self.best_f1:
                self.best_f1 = score
                self.best_threshold = thresh
                no_improve = 0
                
                torch.save({
                    'model': self.model.state_dict(),
                    'threshold': thresh,
                    'recall': recall,
                    'precision': precision,
                    'f1': f1,
                }, self.config.MODEL_DIR / 'nequip_v2_best.pt')
                
                print(f"[{epoch:3d}] Loss={loss:.4f} | Recall={recall:.4f} | "
                      f"Precision={precision:.4f} | F1={f1:.4f} | AUC={auc:.4f} ⭐ BEST")
            else:
                no_improve += 1
                if epoch % 10 == 0:
                    print(f"[{epoch:3d}] Loss={loss:.4f} | Recall={recall:.4f} | "
                          f"Precision={precision:.4f} | F1={f1:.4f} | AUC={auc:.4f}")
            
            if no_improve >= self.config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        config_info = {
            'best_threshold': self.best_threshold,
            'best_f1': self.best_f1,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epochs_trained': epoch + 1,
        }
        with open(self.config.MODEL_DIR / 'training_config.json', 'w') as f:
            json.dump(config_info, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"Best F1: {self.best_f1:.4f}")
        print(f"Best Threshold: {self.best_threshold:.2f}")
        print(f"Model saved to: {self.config.MODEL_DIR}")
        print(f"{'='*60}")


if __name__ == "__main__":
    trainer = NequIPTrainerV2()
    trainer.train()
