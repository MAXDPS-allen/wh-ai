#!/usr/bin/env python3
"""
终极版铁电分类器 v2 - 优化版本

目标: 准确率和Recall都达到99%以上

核心策略:
1. 多模型集成 (Transformer, DeepResNet, AttentionMLP)
2. 级联分类器策略 - 先高召回筛选，再高精度确认
3. 多阈值策略 - 自动寻找最优阈值组合
4. SMOTE过采样
5. 模型保存和加载
"""

import os
import sys
import json
import warnings
import hashlib
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR

# 尝试导入SMOTE，如果没有则使用简单过采样
try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    HAS_SMOTE = True
    print("✓ SMOTE可用")
except ImportError:
    HAS_SMOTE = False
    print("⚠ SMOTE不可用，使用简单过采样")

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 路径配置
BASE_DIR = Path('/home/ubuntu/ai_wh/wh-ai')
DATA_DIR = BASE_DIR / 'new_data'
MODEL_DIR = BASE_DIR / 'model_ultimate_v2'
REPORT_DIR = BASE_DIR / 'reports_ultimate_v2'

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def get_structure_hash(structure: Structure) -> str:
    """获取结构的唯一哈希"""
    sorted_species = sorted([str(s) for s in structure.species])
    lattice_params = tuple(round(x, 4) for x in structure.lattice.abc + structure.lattice.angles)
    unique_str = f"{sorted_species}_{lattice_params}"
    return hashlib.md5(unique_str.encode()).hexdigest()


class AdvancedFeatureExtractor:
    """增强版 NequIP 特征提取器 - 64维"""
    
    def __init__(self, num_radial_basis=16, cutoff=5.0):
        self.num_radial_basis = num_radial_basis
        self.cutoff = cutoff
        self.feature_dim = 64
        
    def extract_features(self, structure: Structure) -> np.ndarray:
        """提取64维物理特征"""
        try:
            features = np.zeros(self.feature_dim)
            
            # 1. 原子特征 (0-15)
            atomic_numbers = [site.specie.Z for site in structure]
            features[0] = np.mean(atomic_numbers)
            features[1] = np.std(atomic_numbers) if len(atomic_numbers) > 1 else 0
            features[2] = len(atomic_numbers)
            features[3] = len(set(atomic_numbers))
            
            electronegativities = []
            ionic_radii = []
            for site in structure:
                try:
                    electronegativities.append(site.specie.X)
                except:
                    electronegativities.append(2.0)
                try:
                    ionic_radii.append(float(site.specie.ionic_radius or 1.0))
                except:
                    ionic_radii.append(1.0)
            
            features[4] = np.mean(electronegativities)
            features[5] = np.std(electronegativities) if len(electronegativities) > 1 else 0
            features[6] = np.max(electronegativities) - np.min(electronegativities)
            features[7] = np.mean(ionic_radii)
            features[8] = np.std(ionic_radii) if len(ionic_radii) > 1 else 0
            
            # 2. 晶格特征 (9-20)
            lattice = structure.lattice
            features[9] = lattice.a
            features[10] = lattice.b
            features[11] = lattice.c
            features[12] = lattice.alpha
            features[13] = lattice.beta
            features[14] = lattice.gamma
            features[15] = lattice.volume
            features[16] = lattice.volume / len(structure)
            
            abc = np.array([lattice.a, lattice.b, lattice.c])
            features[17] = np.std(abc) / np.mean(abc) if np.mean(abc) > 0 else 0
            
            angles = np.array([lattice.alpha, lattice.beta, lattice.gamma])
            features[18] = np.mean(np.abs(angles - 90))
            features[19] = np.std(angles)
            features[20] = np.max(np.abs(angles - 90))
            
            # 3. 径向分布特征 (21-36)
            try:
                neighbors = structure.get_all_neighbors(self.cutoff)
                all_distances = []
                for neighbor_list in neighbors:
                    all_distances.extend([n.nn_distance for n in neighbor_list])
                
                if all_distances:
                    all_distances = np.array(all_distances)
                    features[21] = np.mean(all_distances)
                    features[22] = np.std(all_distances)
                    features[23] = np.min(all_distances)
                    features[24] = np.max(all_distances)
                    features[25] = np.median(all_distances)
                    features[26] = len(all_distances) / len(structure)
                    
                    hist, bin_edges = np.histogram(all_distances, bins=self.num_radial_basis, range=(0, self.cutoff))
                    hist = hist / (np.sum(hist) + 1e-8)
                    peak_idx = np.argmax(hist)
                    features[27] = bin_edges[peak_idx]
                    features[28] = hist[peak_idx]
                    features[29] = stats.entropy(hist + 1e-8)
                    
                    coord_numbers = [len(n) for n in neighbors]
                    features[30] = np.mean(coord_numbers)
                    features[31] = np.std(coord_numbers) if len(coord_numbers) > 1 else 0
                    features[32] = np.max(coord_numbers)
                    features[33] = np.min(coord_numbers)
            except Exception:
                pass
            
            # 4. 对称性特征 (37-48)
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sga = SpacegroupAnalyzer(structure, symprec=0.1)
                sym_dataset = sga.get_symmetry_dataset()
                
                features[37] = sym_dataset.get('number', 1)
                features[38] = len(sym_dataset.get('equivalent_atoms', []))
                features[39] = len(sym_dataset.get('wyckoffs', []))
                
                pg = sga.get_point_group_symbol()
                pg_map = {
                    '1': 1, '-1': 2, '2': 3, 'm': 4, '2/m': 5,
                    '222': 6, 'mm2': 7, 'mmm': 8, '4': 9, '-4': 10,
                    '4/m': 11, '422': 12, '4mm': 13, '-42m': 14, '4/mmm': 15,
                    '3': 16, '-3': 17, '32': 18, '3m': 19, '-3m': 20,
                    '6': 21, '-6': 22, '6/m': 23, '622': 24, '6mm': 25,
                    '-6m2': 26, '6/mmm': 27, '23': 28, 'm-3': 29, '432': 30,
                    '-43m': 31, 'm-3m': 32
                }
                features[40] = pg_map.get(pg, 0)
                
                polar_groups = {'1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm'}
                features[41] = 1.0 if pg in polar_groups else 0.0
                features[42] = float(sga.is_laue())
                features[43] = 1.0 if features[37] in range(1, 11) else 0.0
                features[44] = 1.0 if features[37] in range(75, 143) else 0.0
                features[45] = 1.0 if features[37] in range(143, 195) else 0.0
            except Exception:
                pass
            
            # 5. 结构失真和极化特征 (46-63)
            coords = structure.frac_coords
            center = np.mean(coords, axis=0)
            features[46] = np.linalg.norm(center - 0.5)
            features[47] = np.std(np.linalg.norm(coords - center, axis=1))
            
            try:
                cations = [i for i, s in enumerate(structure) if s.specie.is_metal]
                anions = [i for i, s in enumerate(structure) if not s.specie.is_metal]
                if cations and anions:
                    cation_center = np.mean([coords[i] for i in cations], axis=0)
                    anion_center = np.mean([coords[i] for i in anions], axis=0)
                    displacement = cation_center - anion_center
                    features[48] = np.linalg.norm(displacement)
                    features[49:52] = displacement
            except Exception:
                pass
            
            try:
                if neighbors:
                    for i, nlist in enumerate(neighbors):
                        if len(nlist) >= 4:
                            nn_distances = sorted([n.nn_distance for n in nlist])[:6]
                            if len(nn_distances) >= 4:
                                features[52] = np.std(nn_distances[:4])
                            if len(nn_distances) == 6:
                                features[53] = np.std(nn_distances)
                            break
            except Exception:
                pass
            
            try:
                groups = []
                periods = []
                for site in structure:
                    el = site.specie
                    groups.append(el.group)
                    periods.append(el.row)
                features[54] = np.mean(groups)
                features[55] = np.std(groups) if len(groups) > 1 else 0
                features[56] = np.mean(periods)
                features[57] = np.std(periods) if len(periods) > 1 else 0
            except Exception:
                pass
            
            try:
                d_electrons = []
                for site in structure:
                    try:
                        ec = site.specie.electronic_structure
                        d_count = ec.count('d')
                        d_electrons.append(d_count)
                    except:
                        d_electrons.append(0)
                features[58] = np.mean(d_electrons)
                features[59] = np.max(d_electrons)
            except Exception:
                pass
            
            try:
                oxygen_count = sum(1 for s in structure if s.specie.symbol == 'O')
                features[60] = oxygen_count / len(structure)
                tm_count = sum(1 for s in structure if s.specie.is_transition_metal)
                features[61] = tm_count / len(structure)
                metal_count = sum(1 for s in structure if s.specie.is_metal)
                features[62] = metal_count / len(structure)
                features[63] = features[3] / len(structure)
            except Exception:
                pass
            
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            return features
            
        except Exception as e:
            return np.zeros(self.feature_dim)


# ==================== 模型定义 ====================

class TransformerClassifier(nn.Module):
    """Transformer分类器"""
    def __init__(self, input_dim=64, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x):
        # x: (batch_size, input_dim)
        # 将特征reshape成序列形式
        batch_size = x.size(0)
        x = self.input_proj(x)  # (batch_size, d_model)
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        x = self.transformer(x)  # (batch_size, 1, d_model)
        x = x.squeeze(1)  # (batch_size, d_model)
        return self.classifier(x)


class DeepResNet(nn.Module):
    """深度残差网络"""
    def __init__(self, input_dim=64, hidden_dims=[256, 512, 256, 128], dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_norm = nn.LayerNorm(hidden_dims[0])
        
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.blocks.append(self._make_block(hidden_dims[i], hidden_dims[i+1], dropout))
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )
        
    def _make_block(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.input_norm(self.input_proj(x))
        for block in self.blocks:
            identity = x
            x = block(x)
            if x.size(-1) == identity.size(-1):
                x = x + identity
        return self.classifier(x)


class AttentionMLP(nn.Module):
    """注意力MLP"""
    def __init__(self, input_dim=64, hidden_dim=256, num_heads=4, dropout=0.2):
        super().__init__()
        self.feature_embed = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        x = self.feature_embed(x).unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = x.squeeze(1)
        x = x + self.mlp(x)
        return self.classifier(x)


class EnsembleClassifier(nn.Module):
    """集成分类器 - 组合多个子模型"""
    def __init__(self, input_dim=64):
        super().__init__()
        self.models = nn.ModuleList([
            TransformerClassifier(input_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1),
            TransformerClassifier(input_dim, d_model=96, nhead=6, num_layers=3, dropout=0.15),
            DeepResNet(input_dim, hidden_dims=[256, 512, 256, 128], dropout=0.2),
            DeepResNet(input_dim, hidden_dims=[128, 256, 128], dropout=0.15),
            AttentionMLP(input_dim, hidden_dim=256, num_heads=4, dropout=0.2),
        ])
        self.weights = nn.Parameter(torch.ones(len(self.models)) / len(self.models))
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=1)
        weights = F.softmax(self.weights, dim=0)
        weighted_output = (outputs.squeeze(-1) * weights).sum(dim=1, keepdim=True)
        return weighted_output


class FocalLoss(nn.Module):
    """Focal Loss - 处理类别不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ==================== 数据加载 ====================

def load_data_from_jsonl(file_path: str, extractor, seen_hashes: set, limit: int = None) -> np.ndarray:
    """从JSONL文件加载数据并提取特征"""
    features_list = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if limit:
            lines = lines[:limit]
        
        for line in tqdm(lines, desc=f"  {Path(file_path).name}"):
            try:
                data = json.loads(line)
                structure = Structure.from_dict(data['structure'])
                struct_hash = get_structure_hash(structure)
                
                if struct_hash not in seen_hashes:
                    seen_hashes.add(struct_hash)
                    features = extractor.extract_features(structure)
                    features_list.append(features)
            except Exception:
                continue
    
    return np.array(features_list) if features_list else np.array([])


def load_data(extractor):
    """加载并预处理数据"""
    print("\n" + "="*60)
    print("加载数据...")
    print("="*60)
    
    # 加载正样本
    seen_hashes_fe = set()
    print("\n处理正样本 (铁电材料)...")
    
    fe_files = [
        DATA_DIR / 'dataset_original_ferroelectric.jsonl',
        DATA_DIR / 'dataset_known_FE_rest.jsonl'
    ]
    
    all_fe_features = []
    for fe_file in fe_files:
        if fe_file.exists():
            features = load_data_from_jsonl(str(fe_file), extractor, seen_hashes_fe)
            if len(features) > 0:
                all_fe_features.append(features)
                print(f"  {fe_file.name}: {len(features)} 个唯一样本")
    
    X_fe = np.vstack(all_fe_features) if all_fe_features else np.array([])
    y_fe = np.ones(len(X_fe))
    
    # 加载负样本
    seen_hashes_nfe = set()
    print("\n处理负样本 (非铁电材料)...")
    
    nfe_files = [
        (DATA_DIR / 'dataset_nonFE.jsonl', 5000),
        (DATA_DIR / 'dataset_nonFE_cleaned.jsonl', None),
        (DATA_DIR / 'dataset_nonFE_expanded.jsonl', None)
    ]
    
    all_nfe_features = []
    for nfe_file, limit in nfe_files:
        if nfe_file.exists():
            features = load_data_from_jsonl(str(nfe_file), extractor, seen_hashes_nfe, limit)
            if len(features) > 0:
                all_nfe_features.append(features)
                print(f"  {nfe_file.name}: {len(features)} 个唯一样本")
    
    X_nfe = np.vstack(all_nfe_features) if all_nfe_features else np.array([])
    y_nfe = np.zeros(len(X_nfe))
    
    # 合并
    X = np.vstack([X_fe, X_nfe])
    y = np.concatenate([y_fe, y_nfe])
    
    print(f"\n总样本数: {len(X)}")
    print(f"FE样本: {len(X_fe)} ({100*len(X_fe)/len(X):.2f}%)")
    print(f"Non-FE样本: {len(X_nfe)} ({100*len(X_nfe)/len(X):.2f}%)")
    print(f"类别比例: 1:{len(X_nfe)/len(X_fe):.1f}")
    
    return X, y


def apply_smote(X_train, y_train, target_ratio=0.5):
    """应用SMOTE或简单过采样"""
    n_pos = int(np.sum(y_train))
    n_neg = len(y_train) - n_pos
    target_pos = int(n_neg * target_ratio)
    
    if HAS_SMOTE:
        try:
            smote = BorderlineSMOTE(
                sampling_strategy=min(target_ratio, 1.0),
                k_neighbors=min(5, n_pos - 1),
                random_state=42
            )
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            print(f"  SMOTE: {n_pos} -> {int(np.sum(y_resampled))} 正样本")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"  SMOTE失败: {e}, 使用简单过采样")
    
    # 简单过采样
    pos_indices = np.where(y_train == 1)[0]
    oversample_times = min(target_pos // n_pos, 20)
    
    X_pos_oversampled = np.tile(X_train[pos_indices], (oversample_times, 1))
    y_pos_oversampled = np.ones(len(X_pos_oversampled))
    
    # 添加小噪声
    noise = np.random.normal(0, 0.01, X_pos_oversampled.shape)
    X_pos_oversampled = X_pos_oversampled + noise
    
    X_resampled = np.vstack([X_train, X_pos_oversampled])
    y_resampled = np.hstack([y_train, y_pos_oversampled])
    
    print(f"  过采样: {n_pos} -> {int(np.sum(y_resampled))} 正样本")
    return X_resampled, y_resampled


# ==================== 训练和评估 ====================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100):
    """训练模型"""
    best_val_auc = 0
    best_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        if scheduler:
            scheduler.step()
        
        # 验证
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                val_preds.extend(probs if isinstance(probs, np.ndarray) else [probs])
                val_targets.extend(y_batch.numpy())
        
        val_auc = roc_auc_score(val_targets, val_preds)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_val_auc


def find_optimal_threshold(y_true, y_pred, target_recall=0.99):
    """寻找在目标召回率下的最优阈值"""
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_threshold = 0.5
    best_acc = 0
    best_recall = 0
    
    for thresh in thresholds:
        preds = (np.array(y_pred) >= thresh).astype(int)
        acc = accuracy_score(y_true, preds)
        recall = recall_score(y_true, preds, zero_division=0)
        
        if recall >= target_recall and acc > best_acc:
            best_threshold = thresh
            best_acc = acc
            best_recall = recall
    
    return best_threshold, best_acc, best_recall


def evaluate_at_threshold(y_true, y_pred, threshold):
    """在指定阈值下评估"""
    preds = (np.array(y_pred) >= threshold).astype(int)
    
    return {
        'accuracy': accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1': f1_score(y_true, preds, zero_division=0),
        'threshold': threshold
    }


# ==================== 主训练流程 ====================

def main():
    print("\n" + "="*60)
    print("终极版铁电分类器 v2")
    print("目标: 准确率和Recall都达到99%以上")
    print("="*60)
    
    # 特征提取器
    extractor = AdvancedFeatureExtractor(num_radial_basis=16, cutoff=5.0)
    
    # 加载数据
    features, labels = load_data(extractor)
    print(f"特征维度: {features.shape}")
    
    # 5折交叉验证
    n_folds = 5
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_results = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(features, labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print("="*60)
        
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        print(f"训练集: {len(X_train)} (正样本: {sum(y_train)})")
        print(f"验证集: {len(X_val)} (正样本: {sum(y_val)})")
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 应用SMOTE
        X_train_resampled, y_train_resampled = apply_smote(X_train_scaled, y_train, target_ratio=0.5)
        
        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_resampled),
            torch.FloatTensor(y_train_resampled)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val)
        )
        
        # 加权采样
        class_weights = torch.FloatTensor([1.0, 10.0])
        sample_weights = [class_weights[int(y)] for y in y_train_resampled]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        # 创建集成模型
        model = EnsembleClassifier(input_dim=64).to(device)
        
        # 损失函数 - Focal Loss
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=100, steps_per_epoch=len(train_loader))
        
        # 训练
        print("训练中...")
        model, best_auc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=100)
        print(f"最佳验证AUC: {best_auc:.4f}")
        
        # 预测
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                val_preds.extend(probs if isinstance(probs, np.ndarray) else [probs])
        
        val_preds = np.array(val_preds)
        auc = roc_auc_score(y_val, val_preds)
        
        # 多阈值评估
        print("\n不同阈值下的结果:")
        fold_result = {'fold': fold + 1, 'auc': auc}
        
        for thresh in [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]:
            metrics = evaluate_at_threshold(y_val, val_preds, thresh)
            print(f"  阈值 {thresh:.2f}: Acc={metrics['accuracy']:.4f}, Recall={metrics['recall']:.4f}, Prec={metrics['precision']:.4f}")
            fold_result[f'acc_{thresh}'] = metrics['accuracy']
            fold_result[f'recall_{thresh}'] = metrics['recall']
            fold_result[f'precision_{thresh}'] = metrics['precision']
        
        # 寻找最优阈值
        for target_recall in [0.99, 0.995, 1.0]:
            opt_thresh, opt_acc, opt_recall = find_optimal_threshold(y_val, val_preds, target_recall)
            print(f"  目标Recall≥{target_recall}: 阈值={opt_thresh:.2f}, Acc={opt_acc:.4f}, Recall={opt_recall:.4f}")
            fold_result[f'opt_thresh_{target_recall}'] = opt_thresh
            fold_result[f'opt_acc_{target_recall}'] = opt_acc
            fold_result[f'opt_recall_{target_recall}'] = opt_recall
        
        all_results.append(fold_result)
        
        # 保存模型
        model_path = MODEL_DIR / f'ensemble_fold{fold+1}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'fold': fold + 1,
            'auc': auc,
        }, model_path)
        print(f"模型已保存: {model_path}")
        
        fold_models.append((model, scaler, auc))
    
    # 汇总结果
    print("\n" + "="*60)
    print("交叉验证汇总")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(REPORT_DIR / 'cv_results.csv', index=False)
    
    print(f"\n平均 ROC-AUC: {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    
    for thresh in [0.5, 0.3, 0.2, 0.1, 0.05]:
        mean_acc = results_df[f'acc_{thresh}'].mean()
        mean_recall = results_df[f'recall_{thresh}'].mean()
        print(f"阈值 {thresh}: 平均Acc={mean_acc:.4f}, 平均Recall={mean_recall:.4f}")
    
    # 找出最佳折
    best_fold_idx = np.argmax([r['auc'] for r in all_results])
    best_model, best_scaler, best_auc = fold_models[best_fold_idx]
    
    # 保存最佳模型
    final_model_path = MODEL_DIR / 'best_ensemble_model.pt'
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'scaler': best_scaler,
        'auc': best_auc,
        'fold': best_fold_idx + 1,
    }, final_model_path)
    print(f"\n最佳模型已保存: {final_model_path}")
    
    # 保存scaler
    with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(best_scaler, f)
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"模型保存目录: {MODEL_DIR}")
    print(f"报告保存目录: {REPORT_DIR}")
    print("="*60)
    
    # 检查是否达到目标
    best_result = None
    for thresh in [0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.05]:
        mean_acc = results_df[f'acc_{thresh}'].mean()
        mean_recall = results_df[f'recall_{thresh}'].mean()
        if mean_acc >= 0.99 and mean_recall >= 0.99:
            best_result = {'threshold': thresh, 'accuracy': mean_acc, 'recall': mean_recall}
            break
    
    if best_result:
        print(f"\n✓ 达到目标! 阈值={best_result['threshold']}, Acc={best_result['accuracy']:.4f}, Recall={best_result['recall']:.4f}")
    else:
        print("\n⚠ 未达到99%+双目标，请查看详细结果进行进一步优化")


if __name__ == '__main__':
    main()
