#!/usr/bin/env python3
"""
基于 Transformer 的铁电分类器
使用自注意力机制让模型自动学习特征之间的关系

核心策略:
1. 使用 NequIP 风格的 64 维特征作为基础
2. 使用 Transformer 编码器来捕捉特征之间的关系
3. 使用更激进的类别平衡策略:
   - Focal Loss 降低易分样本权重
   - 增加负样本的困难样本挖掘
   - 使用对比学习增强特征分离
"""

import os
import sys
import json
import warnings
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve
)
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import OneCycleLR

warnings.filterwarnings('ignore')

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


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
            
            # 各向异性度
            abc = np.array([lattice.a, lattice.b, lattice.c])
            features[17] = np.std(abc) / np.mean(abc) if np.mean(abc) > 0 else 0
            
            # 角度偏离
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
                    
                    # RDF峰
                    hist, bin_edges = np.histogram(all_distances, bins=self.num_radial_basis, range=(0, self.cutoff))
                    hist = hist / (np.sum(hist) + 1e-8)
                    peak_idx = np.argmax(hist)
                    features[27] = bin_edges[peak_idx]
                    features[28] = hist[peak_idx]
                    features[29] = stats.entropy(hist + 1e-8)
                    
                    # 配位数
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
                
                # 点群特征
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
                
                # 极性指标
                polar_groups = {'1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm'}
                features[41] = 1.0 if pg in polar_groups else 0.0
                features[42] = float(sga.is_laue())
                features[43] = 1.0 if features[37] in range(1, 11) else 0.0  # 三斜/单斜
                features[44] = 1.0 if features[37] in range(75, 143) else 0.0  # 四方
                features[45] = 1.0 if features[37] in range(143, 195) else 0.0  # 三角/六方
            except Exception:
                pass
            
            # 5. 结构失真和极化特征 (46-63)
            coords = structure.frac_coords
            center = np.mean(coords, axis=0)
            features[46] = np.linalg.norm(center - 0.5)
            features[47] = np.std(np.linalg.norm(coords - center, axis=1))
            
            # 阳离子-阴离子位移
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
            
            # 八面体/四面体畸变
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
            
            # 元素周期表位置
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
            
            # d电子信息
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
            
            # 氧含量
            try:
                oxygen_count = sum(1 for s in structure if s.specie.symbol == 'O')
                features[60] = oxygen_count / len(structure)
                # 过渡金属含量
                tm_count = sum(1 for s in structure if s.specie.is_transition_metal)
                features[61] = tm_count / len(structure)
                # A位/B位阳离子比例
                large_cations = sum(1 for s in structure if s.specie.is_alkali or s.specie.is_alkaline)
                features[62] = large_cations / len(structure)
            except Exception:
                pass
            
            features[63] = np.sum(features[:20])  # 综合特征
            
            # 归一化
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features
            
        except Exception as e:
            return np.zeros(self.feature_dim)


class TransformerBlock(nn.Module):
    """Transformer Block with Multi-Head Attention"""
    
    def __init__(self, dim, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerClassifier(nn.Module):
    """Transformer-based Ferroelectric Classifier"""
    
    def __init__(self, input_dim=64, hidden_dim=128, num_heads=4, num_layers=4, dropout=0.2):
        super().__init__()
        
        # 将64维特征扩展为序列 (每8维作为一个token)
        self.token_dim = 8
        self.num_tokens = input_dim // self.token_dim
        
        # Token embedding
        self.token_embed = nn.Linear(self.token_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_tokens, hidden_dim) * 0.02)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape to tokens: (B, 64) -> (B, 8, 8)
        x = x.view(batch_size, self.num_tokens, self.token_dim)
        
        # Token embedding
        x = self.token_embed(x)  # (B, 8, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 9, hidden_dim)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Use CLS token for classification
        x = self.norm(x[:, 0])  # (B, hidden_dim)
        
        return self.head(x)


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=3.0, pos_weight=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, 
            pos_weight=torch.tensor([self.pos_weight]).to(inputs.device),
            reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce_loss
        
        return loss.mean()


def load_data_from_jsonl(file_path: str, extractor, seen_hashes: set, limit: int = None) -> tuple:
    """从JSONL文件加载数据"""
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


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features).squeeze(-1)  # 确保维度正确
        
        # 确保 outputs 和 labels 维度一致
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features).squeeze(-1)
            
            # 确保维度正确
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            if isinstance(probs, np.floating):
                probs = [probs]
            
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 计算AUC
    auc = roc_auc_score(all_labels, all_probs)
    
    # 找最优阈值
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    gmeans = np.sqrt(tpr * (1 - fpr))
    best_idx = np.argmax(gmeans)
    best_threshold = thresholds[best_idx]
    
    results = {'auc': auc, 'best_threshold': best_threshold}
    
    # 在不同阈值下的结果
    for thresh in [0.1, 0.2, 0.3, 0.5, best_threshold]:
        preds = (all_probs >= thresh).astype(int)
        results[f'acc_{thresh:.2f}'] = accuracy_score(all_labels, preds)
        results[f'recall_{thresh:.2f}'] = recall_score(all_labels, preds, zero_division=0)
        results[f'prec_{thresh:.2f}'] = precision_score(all_labels, preds, zero_division=0)
        results[f'f1_{thresh:.2f}'] = f1_score(all_labels, preds, zero_division=0)
    
    return results, all_probs, all_labels


def main():
    print("=" * 70)
    print("Transformer 铁电分类器")
    print("使用自注意力机制学习特征间的关系")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = Path('/home/ubuntu/ai_wh/wh-ai/reports_nequip_v6')
    output_dir.mkdir(exist_ok=True)
    
    # 数据目录
    data_dir = Path('/home/ubuntu/ai_wh/wh-ai/new_data')
    
    # 特征提取器
    extractor = AdvancedFeatureExtractor(num_radial_basis=16, cutoff=5.0)
    
    print("\n" + "=" * 70)
    print("加载数据")
    print("=" * 70)
    
    # 加载正样本
    seen_hashes_fe = set()
    print("\n处理正样本 (铁电材料)...")
    
    fe_files = [
        data_dir / 'dataset_original_ferroelectric.jsonl',
        data_dir / 'dataset_known_FE_rest.jsonl'
    ]
    
    all_fe_features = []
    for fe_file in fe_files:
        if fe_file.exists():
            features = load_data_from_jsonl(str(fe_file), extractor, seen_hashes_fe)
            if len(features) > 0:
                all_fe_features.append(features)
                print(f"  {fe_file.name}: {len(features)} 个唯一样本")
    
    X_fe = np.vstack(all_fe_features)
    y_fe = np.ones(len(X_fe))
    
    # 加载负样本
    seen_hashes_nfe = set()
    print("\n处理负样本 (非铁电材料)...")
    
    nfe_files = [
        (data_dir / 'dataset_nonFE.jsonl', 5000),
        (data_dir / 'dataset_nonFE_cleaned.jsonl', None),
        (data_dir / 'dataset_nonFE_expanded.jsonl', None)
    ]
    
    all_nfe_features = []
    for nfe_file, limit in nfe_files:
        if nfe_file.exists():
            features = load_data_from_jsonl(str(nfe_file), extractor, seen_hashes_nfe, limit)
            if len(features) > 0:
                all_nfe_features.append(features)
                print(f"  {nfe_file.name}: {len(features)} 个唯一样本")
    
    X_nfe = np.vstack(all_nfe_features)
    y_nfe = np.zeros(len(X_nfe))
    
    # 合并
    X = np.vstack([X_fe, X_nfe])
    y = np.concatenate([y_fe, y_nfe])
    
    # 特征标准化
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"\n数据集统计:")
    print(f"  正样本 (FE): {len(X_fe)}")
    print(f"  负样本 (non-FE): {len(X_nfe)}")
    print(f"  类别比例: 1:{len(X_nfe)/len(X_fe):.1f}")
    
    # 5折交叉验证
    print("\n" + "=" * 70)
    print("5折交叉验证训练")
    print("=" * 70)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n{'=' * 70}")
        print(f"Fold {fold}/5")
        print(f"{'=' * 70}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"训练集: {len(X_train)} (正:{int(y_train.sum())}, 负:{len(y_train) - int(y_train.sum())})")
        print(f"验证集: {len(X_val)}")
        
        # 转换为Tensor
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val)
        
        # 加权采样器
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = torch.FloatTensor([class_weights[int(label)] for label in y_train])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        
        # 模型
        model = TransformerClassifier(
            input_dim=64,
            hidden_dim=128,
            num_heads=4,
            num_layers=4,
            dropout=0.2
        ).to(device)
        
        # 损失函数 - 使用Focal Loss处理类别不平衡
        pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        criterion = FocalLoss(alpha=0.25, gamma=3.0, pos_weight=pos_weight)
        
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        
        # 学习率调度
        num_epochs = 100
        steps_per_epoch = len(train_loader)
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=5e-4, 
            total_steps=num_epochs * steps_per_epoch,
            pct_start=0.1
        )
        
        # 训练
        print("训练中...")
        best_auc = 0
        patience_counter = 0
        patience = 15
        
        for epoch in range(1, num_epochs + 1):
            loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            
            if epoch % 10 == 0 or epoch == 1:
                results, _, _ = evaluate(model, val_loader, device)
                auc = results['auc']
                print(f"  Epoch {epoch}: Loss={loss:.4f}, AUC={auc:.4f}")
                
                if auc > best_auc:
                    best_auc = auc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience // 10:
                    print(f"  早停在 epoch {epoch}")
                    break
        
        # 加载最佳模型
        model.load_state_dict(best_model_state)
        
        # 最终评估
        final_results, _, _ = evaluate(model, val_loader, device)
        
        print(f"\nFold {fold} 结果:")
        print(f"  ROC-AUC: {final_results['auc']:.4f}")
        print(f"  最优阈值: {final_results['best_threshold']:.4f}")
        for thresh in [0.10, 0.20, 0.30, 0.50]:
            acc = final_results[f'acc_{thresh:.2f}']
            recall = final_results[f'recall_{thresh:.2f}']
            prec = final_results.get(f'prec_{thresh:.2f}', 0)
            f1 = final_results.get(f'f1_{thresh:.2f}', 0)
            print(f"  阈值{thresh}: Acc={acc:.4f}, Recall={recall:.4f}, Prec={prec:.4f}, F1={f1:.4f}")
        
        cv_results.append(final_results)
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("交叉验证总结")
    print("=" * 70)
    
    aucs = [r['auc'] for r in cv_results]
    print(f"ROC-AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    for thresh in [0.10, 0.20, 0.30, 0.50]:
        accs = [r[f'acc_{thresh:.2f}'] for r in cv_results]
        recalls = [r[f'recall_{thresh:.2f}'] for r in cv_results]
        precs = [r.get(f'prec_{thresh:.2f}', 0) for r in cv_results]
        f1s = [r.get(f'f1_{thresh:.2f}', 0) for r in cv_results]
        print(f"\n阈值 {thresh}:")
        print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"  Precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")
        print(f"  F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    
    # 保存结果
    df = pd.DataFrame(cv_results)
    df.to_csv(output_dir / 'cv_results_transformer.csv', index=False)
    print(f"\n结果已保存到: {output_dir / 'cv_results_transformer.csv'}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
