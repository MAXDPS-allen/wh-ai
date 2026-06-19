#!/usr/bin/env python3
"""
级联分类器 - Cascade Classifier for Ferroelectric Classification

策略:
1. 第一阶段 (Stage 1): 高召回率筛选器
   - 目标: 召回率 > 99.5%，尽可能不遗漏真正的铁电材料
   - 使用低阈值，宁可多选，不可漏选
   
2. 第二阶段 (Stage 2): 高精度确认器
   - 目标: 高准确率，过滤掉假阳性
   - 只对第一阶段筛选出的样本进行分类

最终目标: 准确率和Recall都达到99%以上
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
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
    HAS_SMOTE = True
    print("✓ SMOTE可用")
except ImportError:
    HAS_SMOTE = False
    print("⚠ SMOTE不可用")

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 路径配置
BASE_DIR = Path('/home/ubuntu/ai_wh/wh-ai')
DATA_DIR = BASE_DIR / 'new_data'
MODEL_DIR = BASE_DIR / 'model_cascade'
REPORT_DIR = BASE_DIR / 'reports_cascade'

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def get_structure_hash(structure: Structure) -> str:
    """获取结构的唯一哈希"""
    sorted_species = sorted([str(s) for s in structure.species])
    lattice_params = tuple(round(x, 4) for x in structure.lattice.abc + structure.lattice.angles)
    unique_str = f"{sorted_species}_{lattice_params}"
    return hashlib.md5(unique_str.encode()).hexdigest()


class AdvancedFeatureExtractor:
    """增强版特征提取器 - 64维"""
    
    def __init__(self, num_radial_basis=16, cutoff=5.0):
        self.num_radial_basis = num_radial_basis
        self.cutoff = cutoff
        self.feature_dim = 64
        
    def extract_features(self, structure: Structure) -> np.ndarray:
        """提取64维物理特征"""
        try:
            features = np.zeros(self.feature_dim)
            
            # 1. 原子特征 (0-8)
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
            
            # 4. 对称性特征 (37-45)
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

class Stage1HighRecallModel(nn.Module):
    """第一阶段: 高召回率筛选模型
    
    设计原则:
    - 较宽的网络，捕获更多特征
    - 使用更激进的dropout防止过拟合
    - 倾向于将样本分类为正样本
    """
    def __init__(self, input_dim=64, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
        # 初始化偏置，倾向于预测正样本
        self.network[-1].bias.data.fill_(1.0)
        
    def forward(self, x):
        return self.network(x)


class Stage2HighPrecisionModel(nn.Module):
    """第二阶段: 高精度确认模型
    
    设计原则:
    - 更深的网络，学习更精细的特征
    - 使用注意力机制聚焦关键特征
    - 严格确认正样本
    """
    def __init__(self, input_dim=64, hidden_dim=256, num_heads=4, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # 自注意力层
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        x = self.input_norm(self.input_proj(x))
        x = x.unsqueeze(1)  # (batch, 1, hidden)
        
        for attn, norm, ffn in zip(self.attention_layers, self.layer_norms, self.ffn_layers):
            attn_out, _ = attn(x, x, x)
            x = norm(x + attn_out)
            x = x + ffn(x)
        
        x = x.squeeze(1)
        return self.classifier(x)


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


class AsymmetricLoss(nn.Module):
    """非对称损失 - 对假负样本惩罚更重（用于Stage1）"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        
    def forward(self, inputs, targets):
        probs = torch.sigmoid(inputs)
        
        # 正样本损失
        pos_loss = targets * torch.log(probs.clamp(min=1e-8))
        pos_loss = pos_loss * ((1 - probs) ** self.gamma_pos)
        
        # 负样本损失（带clip）
        neg_probs = (1 - probs).clamp(min=self.clip)
        neg_loss = (1 - targets) * torch.log(neg_probs.clamp(min=1e-8))
        neg_loss = neg_loss * (probs ** self.gamma_neg)
        
        loss = -pos_loss - neg_loss
        return loss.mean()


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
    """应用SMOTE过采样"""
    n_pos = int(np.sum(y_train))
    n_neg = len(y_train) - n_pos
    
    if HAS_SMOTE and n_pos >= 5:
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
            print(f"  SMOTE失败: {e}")
    
    # 简单过采样
    pos_indices = np.where(y_train == 1)[0]
    target_pos = int(n_neg * target_ratio)
    oversample_times = min(target_pos // n_pos, 20)
    
    X_pos_oversampled = np.tile(X_train[pos_indices], (oversample_times, 1))
    y_pos_oversampled = np.ones(len(X_pos_oversampled))
    
    noise = np.random.normal(0, 0.01, X_pos_oversampled.shape)
    X_pos_oversampled = X_pos_oversampled + noise
    
    X_resampled = np.vstack([X_train, X_pos_oversampled])
    y_resampled = np.hstack([y_train, y_pos_oversampled])
    
    print(f"  过采样: {n_pos} -> {int(np.sum(y_resampled))} 正样本")
    return X_resampled, y_resampled


# ==================== 训练函数 ====================

def train_stage1(model, train_loader, val_loader, epochs=80):
    """训练第一阶段模型 - 目标高召回率"""
    print("\n训练 Stage 1 (高召回率筛选器)...")
    
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0.5)  # 对漏检惩罚更重
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_recall = 0
    best_state = None
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs if isinstance(probs, np.ndarray) else [probs])
                val_labels.extend(y_batch.numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        # 使用低阈值评估召回率
        for thresh in [0.1, 0.05, 0.02]:
            preds = (val_preds >= thresh).astype(int)
            recall = recall_score(val_labels, preds, zero_division=0)
            if recall >= 0.995 and recall > best_recall:
                best_recall = recall
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
                break
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_recall


def train_stage2(model, train_loader, val_loader, epochs=100):
    """训练第二阶段模型 - 目标高精度"""
    print("\n训练 Stage 2 (高精度确认器)...")
    
    criterion = FocalLoss(alpha=0.5, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
    scheduler = OneCycleLR(optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader))
    
    best_f1 = 0
    best_state = None
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs if isinstance(probs, np.ndarray) else [probs])
                val_labels.extend(y_batch.numpy())
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        preds = (val_preds >= 0.5).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_f1


def cascade_predict(stage1_model, stage2_model, X, stage1_threshold=0.05, stage2_threshold=0.5):
    """级联预测"""
    stage1_model.eval()
    stage2_model.eval()
    
    X_tensor = torch.FloatTensor(X).to(device)
    
    with torch.no_grad():
        # Stage 1: 筛选
        stage1_probs = torch.sigmoid(stage1_model(X_tensor).squeeze()).cpu().numpy()
        stage1_pass = stage1_probs >= stage1_threshold
        
        # 初始化最终预测
        final_preds = np.zeros(len(X))
        final_probs = stage1_probs.copy()
        
        # Stage 2: 只对通过第一阶段的样本进行精确分类
        if np.sum(stage1_pass) > 0:
            X_stage2 = torch.FloatTensor(X[stage1_pass]).to(device)
            stage2_probs = torch.sigmoid(stage2_model(X_stage2).squeeze()).cpu().numpy()
            
            # 更新最终预测
            stage2_idx = np.where(stage1_pass)[0]
            for i, idx in enumerate(stage2_idx):
                final_probs[idx] = stage1_probs[idx] * stage2_probs[i] if isinstance(stage2_probs, np.ndarray) else stage2_probs
                final_preds[idx] = 1 if (stage2_probs[i] if isinstance(stage2_probs, np.ndarray) else stage2_probs) >= stage2_threshold else 0
    
    return final_preds, final_probs


def find_optimal_thresholds(y_true, stage1_probs, stage2_model, X, target_recall=0.99, target_acc=0.99):
    """寻找最优的级联阈值组合"""
    best_result = None
    
    for s1_thresh in [0.02, 0.03, 0.05, 0.08, 0.1, 0.15]:
        stage1_pass = stage1_probs >= s1_thresh
        
        if np.sum(stage1_pass) == 0:
            continue
        
        # Stage 2 预测
        stage2_model.eval()
        X_stage2 = torch.FloatTensor(X[stage1_pass]).to(device)
        with torch.no_grad():
            stage2_probs = torch.sigmoid(stage2_model(X_stage2).squeeze()).cpu().numpy()
        
        for s2_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            final_preds = np.zeros(len(X))
            stage2_idx = np.where(stage1_pass)[0]
            
            for i, idx in enumerate(stage2_idx):
                s2_prob = stage2_probs[i] if isinstance(stage2_probs, np.ndarray) else stage2_probs
                final_preds[idx] = 1 if s2_prob >= s2_thresh else 0
            
            acc = accuracy_score(y_true, final_preds)
            recall = recall_score(y_true, final_preds, zero_division=0)
            precision = precision_score(y_true, final_preds, zero_division=0)
            f1 = f1_score(y_true, final_preds, zero_division=0)
            
            # 评分: 加权组合
            score = 0.4 * acc + 0.4 * recall + 0.2 * f1
            
            if best_result is None or score > best_result['score']:
                best_result = {
                    's1_thresh': s1_thresh,
                    's2_thresh': s2_thresh,
                    'accuracy': acc,
                    'recall': recall,
                    'precision': precision,
                    'f1': f1,
                    'score': score
                }
    
    return best_result


# ==================== 主训练流程 ====================

def main():
    print("\n" + "="*60)
    print("级联分类器 - Cascade Classifier")
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
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(features, labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold + 1}/{n_folds}")
        print("="*60)
        
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        print(f"训练集: {len(X_train)} (正样本: {int(sum(y_train))})")
        print(f"验证集: {len(X_val)} (正样本: {int(sum(y_val))})")
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # ==================== Stage 1 训练 ====================
        print("\n--- Stage 1: 高召回率筛选器 ---")
        
        # 对Stage1使用更激进的过采样
        X_train_s1, y_train_s1 = apply_smote(X_train_scaled, y_train, target_ratio=0.8)
        
        train_dataset_s1 = TensorDataset(
            torch.FloatTensor(X_train_s1),
            torch.FloatTensor(y_train_s1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_scaled),
            torch.FloatTensor(y_val)
        )
        
        # 加权采样
        class_weights = torch.FloatTensor([1.0, 15.0])  # 更强调正样本
        sample_weights = [class_weights[int(y)] for y in y_train_s1]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader_s1 = DataLoader(train_dataset_s1, batch_size=64, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        
        stage1_model = Stage1HighRecallModel(input_dim=64).to(device)
        stage1_model, s1_recall = train_stage1(stage1_model, train_loader_s1, val_loader)
        
        # Stage 1 评估
        stage1_model.eval()
        with torch.no_grad():
            s1_probs = torch.sigmoid(stage1_model(torch.FloatTensor(X_val_scaled).to(device)).squeeze()).cpu().numpy()
        
        print(f"\nStage 1 结果 (不同阈值):")
        for thresh in [0.1, 0.05, 0.02]:
            s1_preds = (s1_probs >= thresh).astype(int)
            s1_acc = accuracy_score(y_val, s1_preds)
            s1_rec = recall_score(y_val, s1_preds, zero_division=0)
            s1_prec = precision_score(y_val, s1_preds, zero_division=0)
            print(f"  阈值 {thresh}: Acc={s1_acc:.4f}, Recall={s1_rec:.4f}, Precision={s1_prec:.4f}")
        
        # ==================== Stage 2 训练 ====================
        print("\n--- Stage 2: 高精度确认器 ---")
        
        # 选择通过Stage1的样本来训练Stage2
        s1_thresh_for_s2 = 0.05
        s1_train_probs = torch.sigmoid(stage1_model(torch.FloatTensor(X_train_scaled).to(device)).squeeze()).cpu().detach().numpy()
        s1_pass_train = s1_train_probs >= s1_thresh_for_s2
        
        X_train_s2 = X_train_scaled[s1_pass_train]
        y_train_s2 = y_train[s1_pass_train]
        
        print(f"Stage 2 训练集: {len(X_train_s2)} (正样本: {int(sum(y_train_s2))})")
        
        if len(X_train_s2) > 0 and sum(y_train_s2) > 0:
            # Stage2 使用平衡采样
            X_train_s2_resampled, y_train_s2_resampled = apply_smote(X_train_s2, y_train_s2, target_ratio=0.5)
            
            train_dataset_s2 = TensorDataset(
                torch.FloatTensor(X_train_s2_resampled),
                torch.FloatTensor(y_train_s2_resampled)
            )
            
            train_loader_s2 = DataLoader(train_dataset_s2, batch_size=64, shuffle=True)
            
            stage2_model = Stage2HighPrecisionModel(input_dim=64).to(device)
            stage2_model, s2_f1 = train_stage2(stage2_model, train_loader_s2, val_loader)
        else:
            print("  Stage 2 训练样本不足，使用默认模型")
            stage2_model = Stage2HighPrecisionModel(input_dim=64).to(device)
        
        # ==================== 级联评估 ====================
        print("\n--- 级联分类结果 ---")
        
        # 寻找最优阈值组合
        best_result = find_optimal_thresholds(y_val, s1_probs, stage2_model, X_val_scaled)
        
        if best_result:
            print(f"最优阈值: Stage1={best_result['s1_thresh']}, Stage2={best_result['s2_thresh']}")
            print(f"  Accuracy: {best_result['accuracy']:.4f}")
            print(f"  Recall: {best_result['recall']:.4f}")
            print(f"  Precision: {best_result['precision']:.4f}")
            print(f"  F1: {best_result['f1']:.4f}")
        
        # 计算AUC
        final_preds, final_probs = cascade_predict(
            stage1_model, stage2_model, X_val_scaled,
            stage1_threshold=best_result['s1_thresh'] if best_result else 0.05,
            stage2_threshold=best_result['s2_thresh'] if best_result else 0.5
        )
        
        try:
            auc = roc_auc_score(y_val, final_probs)
        except:
            auc = 0.0
        
        print(f"  AUC: {auc:.4f}")
        
        # 记录结果
        fold_result = {
            'fold': fold + 1,
            'auc': auc,
            's1_thresh': best_result['s1_thresh'] if best_result else 0.05,
            's2_thresh': best_result['s2_thresh'] if best_result else 0.5,
            'accuracy': best_result['accuracy'] if best_result else 0,
            'recall': best_result['recall'] if best_result else 0,
            'precision': best_result['precision'] if best_result else 0,
            'f1': best_result['f1'] if best_result else 0,
        }
        all_results.append(fold_result)
        
        # 保存模型
        torch.save({
            'stage1_state_dict': stage1_model.state_dict(),
            'stage2_state_dict': stage2_model.state_dict(),
            'scaler': scaler,
            's1_thresh': fold_result['s1_thresh'],
            's2_thresh': fold_result['s2_thresh'],
            'fold': fold + 1,
        }, MODEL_DIR / f'cascade_fold{fold+1}.pt')
    
    # ==================== 汇总结果 ====================
    print("\n" + "="*60)
    print("交叉验证汇总")
    print("="*60)
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(REPORT_DIR / 'cascade_cv_results.csv', index=False)
    
    print(f"\n平均 ROC-AUC: {results_df['auc'].mean():.4f} ± {results_df['auc'].std():.4f}")
    print(f"平均 Accuracy: {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"平均 Recall: {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    print(f"平均 Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"平均 F1: {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    
    # 找出最佳折
    best_fold_idx = results_df['accuracy'].idxmax()
    best_fold = results_df.loc[best_fold_idx]
    print(f"\n最佳Fold: {int(best_fold['fold'])}")
    print(f"  Accuracy: {best_fold['accuracy']:.4f}")
    print(f"  Recall: {best_fold['recall']:.4f}")
    
    # 检查是否达到目标
    target_achieved = (results_df['accuracy'] >= 0.99) & (results_df['recall'] >= 0.99)
    if target_achieved.any():
        print(f"\n✓ 达到目标! Fold {results_df[target_achieved]['fold'].values}")
    else:
        # 找出最接近目标的结果
        results_df['gap'] = abs(results_df['accuracy'] - 0.99) + abs(results_df['recall'] - 0.99)
        closest = results_df.loc[results_df['gap'].idxmin()]
        print(f"\n最接近目标: Fold {int(closest['fold'])}")
        print(f"  Accuracy: {closest['accuracy']:.4f} (差距: {0.99 - closest['accuracy']:.4f})")
        print(f"  Recall: {closest['recall']:.4f} (差距: {0.99 - closest['recall']:.4f})")
    
    print("\n" + "="*60)
    print("训练完成!")
    print(f"模型保存目录: {MODEL_DIR}")
    print(f"报告保存目录: {REPORT_DIR}")
    print("="*60)


if __name__ == '__main__':
    main()
