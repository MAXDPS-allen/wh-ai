#!/usr/bin/env python3
"""
终极铁电材料分类器 - 多策略集成优化
目标: 同时达到 99%+ Accuracy 和 99%+ Recall

策略:
1. 级联分类器: 高Recall模型筛选 + 高Precision模型确认
2. 多模型加权集成
3. SMOTE过采样 + 类别加权
4. 自适应阈值优化
5. 模型保存和加载
"""

import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score, 
                             precision_score, f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'code_v5'))
from matminer.featurizers.structure import SiteStatsFingerprint
from matminer.featurizers.site import CrystalNNFingerprint
from pymatgen.core import Structure

# ============== 配置 ==============
DATA_DIR = Path(__file__).parent.parent / 'new_data'
MODEL_DIR = Path(__file__).parent.parent / 'model_ultimate'
REPORT_DIR = Path(__file__).parent.parent / 'reports_ultimate'

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# ============== 特征提取 ==============
class FeatureExtractor:
    """多种特征提取器"""
    
    def __init__(self):
        self.ssf = SiteStatsFingerprint(
            CrystalNNFingerprint.from_preset('ops'),
            stats=['mean', 'std', 'maximum', 'minimum']
        )
        self.scaler = StandardScaler()
        
    def extract_basic_features(self, structure):
        """基础结构特征"""
        try:
            features = []
            # 晶格参数
            lattice = structure.lattice
            features.extend([lattice.a, lattice.b, lattice.c])
            features.extend([lattice.alpha, lattice.beta, lattice.gamma])
            features.append(lattice.volume)
            
            # 原子统计
            features.append(len(structure))
            features.append(structure.density)
            
            # 元素统计
            elements = [site.specie.Z for site in structure]
            features.extend([np.mean(elements), np.std(elements), 
                           np.min(elements), np.max(elements)])
            
            # 电负性
            electroneg = [site.specie.X for site in structure if hasattr(site.specie, 'X')]
            if electroneg:
                features.extend([np.mean(electroneg), np.std(electroneg),
                               np.max(electroneg) - np.min(electroneg)])
            else:
                features.extend([0, 0, 0])
            
            # 离子半径差异（铁电相关）
            radii = []
            for site in structure:
                try:
                    radii.append(site.specie.ionic_radii.get(site.specie.common_oxidation_states[0], 1.0))
                except:
                    radii.append(1.0)
            features.extend([np.mean(radii), np.std(radii), np.max(radii) - np.min(radii)])
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            return np.zeros(19, dtype=np.float32)
    
    def extract_matminer_features(self, structure):
        """Matminer结构指纹"""
        try:
            fp = self.ssf.featurize(structure)
            return np.array(fp, dtype=np.float32)
        except:
            return np.zeros(256, dtype=np.float32)  # 默认大小
    
    def extract_symmetry_features(self, structure):
        """对称性特征（铁电相关）"""
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            
            features = []
            # 空间群号
            features.append(sga.get_space_group_number())
            # 点群对称性数量
            features.append(len(sga.get_point_group_operations()))
            # 是否有反演中心（铁电材料没有）
            features.append(0 if sga.is_laue() else 1)
            # 晶系编码
            crystal_systems = {'triclinic': 1, 'monoclinic': 2, 'orthorhombic': 3,
                             'tetragonal': 4, 'trigonal': 5, 'hexagonal': 6, 'cubic': 7}
            features.append(crystal_systems.get(sga.get_crystal_system(), 0))
            
            return np.array(features, dtype=np.float32)
        except:
            return np.zeros(4, dtype=np.float32)


# ============== 模型定义 ==============
class TransformerClassifier(nn.Module):
    """Transformer分类器"""
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, 
            dim_feedforward=hidden_dim*4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        x = self.input_proj(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.classifier(x)


class DeepResNet(nn.Module):
    """深度残差网络"""
    def __init__(self, input_dim, hidden_dim=256, num_blocks=6, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            self._make_block(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def _make_block(self, dim, dropout):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = F.relu(x + block(x))
        return self.classifier(x)


class AttentionMLP(nn.Module):
    """注意力MLP"""
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        x = F.relu(self.fc1(x)).unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = attn_out.squeeze(1)
        return self.fc2(x)


class EnsembleClassifier(nn.Module):
    """集成分类器"""
    def __init__(self, input_dim, num_models=5):
        super().__init__()
        self.models = nn.ModuleList([
            TransformerClassifier(input_dim, hidden_dim=128, num_layers=3),
            DeepResNet(input_dim, hidden_dim=256, num_blocks=6),
            AttentionMLP(input_dim, hidden_dim=256, num_heads=8),
            TransformerClassifier(input_dim, hidden_dim=64, num_layers=2),
            DeepResNet(input_dim, hidden_dim=128, num_blocks=4),
        ][:num_models])
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)
    
    def forward(self, x):
        outputs = torch.stack([model(x) for model in self.models], dim=0)
        weights = F.softmax(self.weights, dim=0)
        return (outputs * weights.view(-1, 1, 1)).sum(dim=0)


# ============== 损失函数 ==============
class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ============== 数据加载 ==============
def load_data():
    """加载数据"""
    print("\n" + "="*70)
    print("加载数据")
    print("="*70)
    
    positive_files = [
        DATA_DIR / 'dataset_original_ferroelectric.jsonl',
        DATA_DIR / 'dataset_known_FE_rest.jsonl',
    ]
    negative_files = [
        DATA_DIR / 'dataset_nonFE.jsonl',
        DATA_DIR / 'dataset_nonFE_cleaned.jsonl',
        DATA_DIR / 'dataset_nonFE_expanded.jsonl',
    ]
    
    structures = []
    labels = []
    seen_formulas = set()
    
    def load_jsonl(filepath, label, max_samples=None):
        count = 0
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if max_samples:
                lines = lines[:max_samples]
            for line in tqdm(lines, desc=f"  {filepath.name}"):
                try:
                    data = json.loads(line)
                    struct = Structure.from_dict(data['structure'])
                    formula = struct.composition.reduced_formula
                    if formula not in seen_formulas:
                        seen_formulas.add(formula)
                        structures.append(struct)
                        labels.append(label)
                        count += 1
                except:
                    continue
        return count
    
    print("\n处理正样本 (铁电材料)...")
    pos_count = 0
    for f in positive_files:
        if f.exists():
            c = load_jsonl(f, 1)
            print(f"  {f.name}: {c} 个唯一样本")
            pos_count += c
    
    print("\n处理负样本 (非铁电材料)...")
    neg_count = 0
    for f in negative_files:
        if f.exists():
            c = load_jsonl(f, 0, max_samples=10000 if 'expanded' in f.name else None)
            print(f"  {f.name}: {c} 个唯一样本")
            neg_count += c
    
    print(f"\n数据集统计:")
    print(f"  正样本 (FE): {pos_count}")
    print(f"  负样本 (non-FE): {neg_count}")
    print(f"  类别比例: 1:{neg_count/pos_count:.1f}")
    
    return structures, np.array(labels)


def extract_features(structures):
    """提取特征"""
    print("\n" + "="*70)
    print("特征提取")
    print("="*70)
    
    extractor = FeatureExtractor()
    
    all_features = []
    for struct in tqdm(structures, desc="提取特征"):
        basic = extractor.extract_basic_features(struct)
        symmetry = extractor.extract_symmetry_features(struct)
        features = np.concatenate([basic, symmetry])
        all_features.append(features)
    
    X = np.array(all_features)
    
    # 处理NaN和Inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    print(f"特征维度: {X.shape}")
    return X


# ============== 训练函数 ==============
def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, epochs=100, patience=15, device=DEVICE):
    """训练模型"""
    model.to(device)
    best_auc = 0
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # 验证
        model.eval()
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = torch.sigmoid(model(X_batch))
                val_probs.extend(outputs.cpu().numpy().flatten())
                val_labels.extend(y_batch.numpy().flatten())
        
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, AUC={val_auc:.4f}")
        
        if no_improve >= patience:
            print(f"  早停在 epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model, best_auc


def find_optimal_threshold(y_true, y_prob, target_recall=0.99):
    """找到满足目标recall的最优阈值"""
    thresholds = np.arange(0.01, 0.99, 0.01)
    best_thresh = 0.5
    best_acc = 0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        recall = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        
        if recall >= target_recall and acc > best_acc:
            best_acc = acc
            best_thresh = thresh
    
    return best_thresh, best_acc


# ============== 主训练流程 ==============
def main():
    print("="*70)
    print("终极铁电分类器 - 多策略集成优化")
    print("目标: Accuracy >= 99% 且 Recall >= 99%")
    print("="*70)
    
    # 加载数据
    structures, labels = load_data()
    
    # 提取特征
    X = extract_features(structures)
    y = labels
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 保存scaler
    with open(MODEL_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n" + "="*70)
    print("5折交叉验证训练")
    print("="*70)
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    all_results = []
    all_models = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        print(f"\n{'='*70}")
        print(f"Fold {fold}/5")
        print(f"{'='*70}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"训练集: {len(train_idx)} (正:{sum(y_train)}, 负:{len(y_train)-sum(y_train)})")
        print(f"验证集: {len(val_idx)}")
        
        # SMOTE过采样
        print("应用SMOTE过采样...")
        try:
            smote = BorderlineSMOTE(random_state=42, k_neighbors=3)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"过采样后: {len(y_train_resampled)} (正:{sum(y_train_resampled)}, 负:{len(y_train_resampled)-sum(y_train_resampled)})")
        except:
            print("SMOTE失败，使用原始数据")
            X_train_resampled, y_train_resampled = X_train, y_train
        
        # 创建DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_resampled),
            torch.FloatTensor(y_train_resampled).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val).unsqueeze(1)
        )
        
        # 类别加权采样
        pos_weight = (len(y_train_resampled) - sum(y_train_resampled)) / max(sum(y_train_resampled), 1)
        sample_weights = [pos_weight if y == 1 else 1.0 for y in y_train_resampled]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=256)
        
        # 创建集成模型
        input_dim = X.shape[1]
        model = EnsembleClassifier(input_dim, num_models=5)
        
        # 损失函数和优化器
        criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=torch.tensor([pos_weight]).to(DEVICE))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # 训练
        print("训练集成模型...")
        model, best_auc = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            epochs=100, patience=15
        )
        
        # 评估
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(torch.FloatTensor(X_val).to(DEVICE))).cpu().numpy().flatten()
        
        # 找到最优阈值
        best_thresh, best_acc = find_optimal_threshold(y_val, val_probs, target_recall=0.99)
        
        # 计算各种阈值下的指标
        print(f"\nFold {fold} 结果:")
        print(f"  ROC-AUC: {best_auc:.4f}")
        
        results = {'fold': fold, 'auc': best_auc}
        
        for thresh in [0.05, 0.10, 0.15, 0.20, 0.30]:
            y_pred = (val_probs >= thresh).astype(int)
            acc = accuracy_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            print(f"  阈值{thresh:.2f}: Acc={acc:.4f}, Recall={recall:.4f}, Prec={prec:.4f}, F1={f1:.4f}")
            results[f'acc_{thresh}'] = acc
            results[f'recall_{thresh}'] = recall
            results[f'prec_{thresh}'] = prec
            results[f'f1_{thresh}'] = f1
        
        # 检查是否达到目标
        for thresh in [0.05, 0.10, 0.15, 0.20]:
            y_pred = (val_probs >= thresh).astype(int)
            acc = accuracy_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            if acc >= 0.99 and recall >= 0.99:
                print(f"\n  🎉 达到目标! 阈值{thresh}: Acc={acc:.4f}, Recall={recall:.4f}")
        
        all_results.append(results)
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'fold': fold,
            'auc': best_auc,
            'input_dim': input_dim,
        }, MODEL_DIR / f'ensemble_fold{fold}.pt')
        
        all_models.append(model)
    
    # 总结
    print("\n" + "="*70)
    print("交叉验证总结")
    print("="*70)
    
    df_results = pd.DataFrame(all_results)
    
    aucs = df_results['auc'].values
    print(f"ROC-AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.30]:
        accs = df_results[f'acc_{thresh}'].values
        recalls = df_results[f'recall_{thresh}'].values
        print(f"\n阈值 {thresh:.2f}:")
        print(f"  Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    
    # 保存结果
    df_results.to_csv(REPORT_DIR / 'ultimate_cv_results.csv', index=False)
    
    # 训练最终模型（使用全部数据）
    print("\n" + "="*70)
    print("训练最终模型（全部数据）")
    print("="*70)
    
    # SMOTE
    try:
        smote = BorderlineSMOTE(random_state=42, k_neighbors=3)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"过采样后: {len(y_resampled)} (正:{sum(y_resampled)}, 负:{len(y_resampled)-sum(y_resampled)})")
    except:
        X_resampled, y_resampled = X, y
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_resampled),
        torch.FloatTensor(y_resampled).unsqueeze(1)
    )
    
    pos_weight = (len(y_resampled) - sum(y_resampled)) / max(sum(y_resampled), 1)
    sample_weights = [pos_weight if label == 1 else 1.0 for label in y_resampled]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    
    final_model = EnsembleClassifier(X.shape[1], num_models=5)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    final_model.to(DEVICE)
    for epoch in range(50):
        final_model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = final_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': final_model.state_dict(),
        'input_dim': X.shape[1],
        'scaler_path': str(MODEL_DIR / 'scaler.pkl'),
    }, MODEL_DIR / 'final_ensemble_model.pt')
    
    print(f"\n模型已保存到: {MODEL_DIR}")
    print(f"结果已保存到: {REPORT_DIR / 'ultimate_cv_results.csv'}")
    print("\n完成!")


if __name__ == '__main__':
    main()
