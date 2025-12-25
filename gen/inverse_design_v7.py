"""
增强型逆向设计模型 v7
=============================================
从64维特征向量预测材料组成和晶格参数

改进 (相对于 v6):
1. 对 b/a, c/a 比值应用 sqrt 变换进行预测
2. 预测 sqrt(b/a), sqrt(c/a) 而非直接预测比值
3. 添加 Parity Plot 可视化（包含 y=x 和 y=sqrt(x) 对比）
4. 增强网络深度和注意力机制
5. 添加残差连接
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
    ELEMENT_TO_IDX,
    IDX_TO_ELEMENT,
    NUM_ELEMENTS,
    MAX_ELEMENTS,
    LATTICE_TARGET_DIM,
    TOTAL_TARGET_DIM,
)


# ==========================================
# 1. 配置
# ==========================================
class Config:
    # 维度 (使用统一特征)
    INPUT_DIM = FEATURE_DIM  # 64
    HIDDEN_DIM = 512
    NUM_ELEMENTS = NUM_ELEMENTS
    MAX_ELEMENTS_OUTPUT = MAX_ELEMENTS  # 5
    
    # 输出维度
    # 改进: 预测 vol_root, sqrt(b/a), sqrt(c/a), alpha, beta, gamma
    LATTICE_DIM = 6
    SPACEGROUP_DIM = 1
    
    # 训练参数
    BATCH_SIZE = 32
    EPOCHS = 200
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    PATIENCE = 30
    
    # 路径
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'invs_dgn_model_v2'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_inverse_v3'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 2. 数据集 (使用 sqrt 变换)
# ==========================================
class InverseDesignDatasetV7(Dataset):
    """逆向设计数据集 - 使用 sqrt 变换"""
    
    def __init__(self, data_files: List[str], extractor: UnifiedFeatureExtractor):
        self.features = []
        self.targets_lattice = []
        self.targets_elements = []
        self.targets_fractions = []
        self.targets_spacegroup = []
        self.extractor = extractor
        
        # 保存原始比值用于验证
        self.raw_ratios = []  # [(b/a, c/a), ...]
        
        for file_path in data_files:
            if os.path.exists(file_path):
                self._load_file(file_path)
        
        if self.features:
            self.features = np.array(self.features, dtype=np.float32)
            self.targets_lattice = np.array(self.targets_lattice, dtype=np.float32)
            self.targets_elements = np.array(self.targets_elements, dtype=np.int64)
            self.targets_fractions = np.array(self.targets_fractions, dtype=np.float32)
            self.targets_spacegroup = np.array(self.targets_spacegroup, dtype=np.float32)
            self.raw_ratios = np.array(self.raw_ratios, dtype=np.float32)
        else:
            self.features = np.zeros((0, Config.INPUT_DIM), dtype=np.float32)
            self.targets_lattice = np.zeros((0, 6), dtype=np.float32)
            self.targets_elements = np.zeros((0, MAX_ELEMENTS), dtype=np.int64)
            self.targets_fractions = np.zeros((0, MAX_ELEMENTS), dtype=np.float32)
            self.targets_spacegroup = np.zeros((0, 1), dtype=np.float32)
            self.raw_ratios = np.zeros((0, 2), dtype=np.float32)
        
        print(f"Loaded {len(self.features)} samples for inverse design (v7 with sqrt transform)")
    
    def _load_file(self, file_path: str):
        """加载文件并提取特征和目标"""
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    struct = item.get('structure')
                    sg = item.get('spacegroup_number', 1)
                    
                    if struct:
                        # 提取特征
                        feat = self.extractor.extract_from_structure_dict(struct, sg)
                        if np.sum(np.abs(feat)) == 0:
                            continue
                        
                        # 提取目标
                        target_data = self._extract_targets(struct, sg)
                        if target_data is None:
                            continue
                        
                        lattice, elements, fractions, spacegroup, raw = target_data
                        
                        self.features.append(feat)
                        self.targets_lattice.append(lattice)
                        self.targets_elements.append(elements)
                        self.targets_fractions.append(fractions)
                        self.targets_spacegroup.append(spacegroup)
                        self.raw_ratios.append(raw)
                        
                except Exception as e:
                    continue
    
    def _extract_targets(self, struct_dict, spacegroup):
        """
        提取目标向量
        关键改进: 对 b/a 和 c/a 应用 sqrt 变换
        """
        try:
            lattice = struct_dict.get('lattice', {})
            sites = struct_dict.get('sites', [])
            
            a = lattice.get('a', 0)
            b = lattice.get('b', 0)
            c = lattice.get('c', 0)
            alpha = lattice.get('alpha', 90)
            beta = lattice.get('beta', 90)
            gamma = lattice.get('gamma', 90)
            vol = lattice.get('volume', 0)
            
            if vol <= 0 or a <= 0:
                return None
            
            # 晶格目标 - 关键改进: sqrt 变换
            vol_root = vol ** (1/3)
            ratio_ba = b / a
            ratio_ca = c / a
            
            # 保存原始比值
            raw_ratios = [ratio_ba, ratio_ca]
            
            # sqrt 变换: 预测 sqrt(b/a), sqrt(c/a)
            # 这样模型的线性预测会更接近真实分布
            sqrt_ratio_ba = np.sqrt(max(ratio_ba, 0.01))  # 避免负数
            sqrt_ratio_ca = np.sqrt(max(ratio_ca, 0.01))
            
            lattice_targets = [
                vol_root / 20.0,          # 归一化
                sqrt_ratio_ba,             # sqrt(b/a) - 改进
                sqrt_ratio_ca,             # sqrt(c/a) - 改进
                alpha / 180.0,
                beta / 180.0,
                gamma / 180.0
            ]
            
            # 成分
            comp = {}
            total_atoms = 0
            for site in sites:
                for species in site['species']:
                    el = species['element']
                    occu = species['occu']
                    comp[el] = comp.get(el, 0) + occu
                    total_atoms += occu
            
            if total_atoms == 0:
                return None
            
            # 编码成分
            el_list = []
            for el, count in comp.items():
                if el in ELEMENT_TO_IDX:
                    idx = ELEMENT_TO_IDX[el]
                    frac = count / total_atoms
                    el_list.append((idx, frac))
            
            el_list.sort(key=lambda x: x[1], reverse=True)
            
            elements = []
            fractions = []
            for i in range(MAX_ELEMENTS):
                if i < len(el_list):
                    elements.append(el_list[i][0])
                    fractions.append(el_list[i][1])
                else:
                    elements.append(0)
                    fractions.append(0)
            
            sg_norm = [spacegroup / 230.0]
            
            return (
                np.array(lattice_targets, dtype=np.float32),
                np.array(elements, dtype=np.int64),
                np.array(fractions, dtype=np.float32),
                np.array(sg_norm, dtype=np.float32),
                np.array(raw_ratios, dtype=np.float32)
            )
            
        except Exception as e:
            return None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': torch.from_numpy(self.features[idx]),
            'lattice': torch.from_numpy(self.targets_lattice[idx]),
            'elements': torch.from_numpy(self.targets_elements[idx]),
            'fractions': torch.from_numpy(self.targets_fractions[idx]),
            'spacegroup': torch.from_numpy(self.targets_spacegroup[idx]),
            'raw_ratios': torch.from_numpy(self.raw_ratios[idx])
        }


# ==========================================
# 3. 网络架构 (增强版)
# ==========================================
class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(x + self.block(x))


class AttentionBlock(nn.Module):
    """自注意力模块"""
    
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x.squeeze(1)


class LatticePredictor(nn.Module):
    """晶格参数预测网络 (预测 sqrt 变换后的比值)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # 残差块
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, 0.1),
            ResidualBlock(hidden_dim, 0.1),
            ResidualBlock(hidden_dim, 0.1),
        )
        
        # 分离的预测头
        self.vol_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 归一化到 [0, 1]
        )
        
        # sqrt(ratio) 预测头
        self.ratio_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Softplus()  # 确保 > 0
        )
        
        self.angle_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),
            nn.Sigmoid()  # 归一化到 [0, 1]
        )
    
    def forward(self, x):
        h = self.encoder(x)
        h = self.res_blocks(h)
        
        vol = self.vol_head(h)           # [B, 1]
        ratios = self.ratio_head(h)      # [B, 2] - sqrt(b/a), sqrt(c/a)
        angles = self.angle_head(h)      # [B, 3] - alpha, beta, gamma
        
        return torch.cat([vol, ratios, angles], dim=1)  # [B, 6]


class ElementPredictor(nn.Module):
    """元素预测网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, 
                 num_elements: int = NUM_ELEMENTS, max_output: int = MAX_ELEMENTS):
        super().__init__()
        
        self.max_output = max_output
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, 0.1),
            ResidualBlock(hidden_dim, 0.1),
        )
        
        self.attention = AttentionBlock(hidden_dim)
        
        # 每个位置的元素分类
        self.element_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_elements)
            ) for _ in range(max_output)
        ])
        
        # 比例预测
        self.fraction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, max_output),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        h = self.res_blocks(h)
        h = self.attention(h)
        
        # 元素分类
        element_logits = []
        for classifier in self.element_classifiers:
            logits = classifier(h)
            element_logits.append(logits)
        
        element_logits = torch.stack(element_logits, dim=1)
        fractions = self.fraction_head(h)
        
        return element_logits, fractions


class InverseDesignNetworkV7(nn.Module):
    """完整逆向设计网络 v7"""
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.lattice_predictor = LatticePredictor(config.INPUT_DIM, config.HIDDEN_DIM // 2)
        self.element_predictor = ElementPredictor(config.INPUT_DIM, config.HIDDEN_DIM // 2)
        
        # 空间群预测
        self.spacegroup_head = nn.Sequential(
            nn.Linear(config.INPUT_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        lattice = self.lattice_predictor(x)
        element_logits, fractions = self.element_predictor(x)
        spacegroup = self.spacegroup_head(x)
        
        return {
            'lattice': lattice,
            'element_logits': element_logits,
            'fractions': fractions,
            'spacegroup': spacegroup
        }


# ==========================================
# 4. 训练器
# ==========================================
class InverseDesignTrainerV7:
    """训练器"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.extractor = UnifiedFeatureExtractor()
        self.model = InverseDesignNetworkV7(self.config).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LR,
            weight_decay=self.config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=15
        )
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        
        self.history = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'lattice_mse': [], 'element_acc': [], 'fraction_mse': [],
            'top3_acc': [], 'ratio_mse': []
        }
        
        # 保存验证数据用于 parity plot
        self.val_predictions = None
        self.val_targets = None
    
    def load_data(self):
        """加载数据"""
        data_files = [
            str(self.config.DATA_DIR / 'dataset_original_ferroelectric.jsonl'),
            str(self.config.DATA_DIR / 'dataset_polar_non_ferroelectric_final.jsonl'),
            str(self.config.DATA_DIR / 'dataset_known_FE_rest.jsonl'),
        ]
        
        dataset = InverseDesignDatasetV7(data_files, self.extractor)
        
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader
    
    def compute_loss(self, outputs, batch):
        """计算损失"""
        # 晶格损失 (包括 sqrt 变换后的比值)
        lattice_loss = self.mse_loss(
            outputs['lattice'],
            batch['lattice'].to(self.device)
        )
        
        # 额外惩罚比值预测误差
        ratio_loss = self.mse_loss(
            outputs['lattice'][:, 1:3],  # sqrt(b/a), sqrt(c/a)
            batch['lattice'][:, 1:3].to(self.device)
        )
        
        # 元素分类损失
        element_loss = 0
        for i in range(MAX_ELEMENTS):
            element_loss += self.ce_loss(
                outputs['element_logits'][:, i],
                batch['elements'][:, i].to(self.device)
            )
        element_loss /= MAX_ELEMENTS
        
        # 比例损失
        fraction_loss = self.mse_loss(
            outputs['fractions'],
            batch['fractions'].to(self.device)
        )
        
        # 空间群损失
        sg_loss = self.mse_loss(
            outputs['spacegroup'],
            batch['spacegroup'].to(self.device)
        )
        
        total = lattice_loss + ratio_loss * 0.5 + element_loss + fraction_loss * 0.5 + sg_loss * 0.2
        
        return {
            'total': total,
            'lattice': lattice_loss,
            'ratio': ratio_loss,
            'element': element_loss,
            'fraction': fraction_loss,
            'spacegroup': sg_loss
        }
    
    def compute_metrics(self, outputs, batch):
        """计算指标"""
        # 晶格 MSE
        lattice_mse = self.mse_loss(
            outputs['lattice'],
            batch['lattice'].to(self.device)
        ).item()
        
        # 比值 MSE (sqrt 空间)
        ratio_mse = self.mse_loss(
            outputs['lattice'][:, 1:3],
            batch['lattice'][:, 1:3].to(self.device)
        ).item()
        
        # 元素准确率
        element_acc = 0
        top3_acc = 0
        for i in range(MAX_ELEMENTS):
            pred = outputs['element_logits'][:, i].argmax(dim=1)
            target = batch['elements'][:, i].to(self.device)
            element_acc += (pred == target).float().mean().item()
            
            top3_idx = outputs['element_logits'][:, i].topk(3, dim=1)[1]
            top3_acc += (top3_idx == target.unsqueeze(1)).any(dim=1).float().mean().item()
        
        element_acc /= MAX_ELEMENTS
        top3_acc /= MAX_ELEMENTS
        
        fraction_mse = self.mse_loss(
            outputs['fractions'],
            batch['fractions'].to(self.device)
        ).item()
        
        return {
            'lattice_mse': lattice_mse,
            'ratio_mse': ratio_mse,
            'element_acc': element_acc,
            'top3_acc': top3_acc,
            'fraction_mse': fraction_mse
        }
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            features = batch['features'].to(self.device)
            outputs = self.model(features)
            
            losses = self.compute_loss(outputs, batch)
            losses['total'].backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            total_loss += losses['total'].item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader, collect_predictions: bool = False):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_metrics = {
            'lattice_mse': [], 'ratio_mse': [], 
            'element_acc': [], 'top3_acc': [], 'fraction_mse': []
        }
        
        all_preds = {'lattice': [], 'raw_ratios': []}
        all_targets = {'lattice': [], 'raw_ratios': []}
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                outputs = self.model(features)
                
                losses = self.compute_loss(outputs, batch)
                total_loss += losses['total'].item()
                
                metrics = self.compute_metrics(outputs, batch)
                for k, v in metrics.items():
                    all_metrics[k].append(v)
                
                if collect_predictions:
                    all_preds['lattice'].append(outputs['lattice'].cpu().numpy())
                    all_preds['raw_ratios'].append(batch['raw_ratios'].numpy())
                    all_targets['lattice'].append(batch['lattice'].numpy())
                    all_targets['raw_ratios'].append(batch['raw_ratios'].numpy())
        
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        
        if collect_predictions:
            self.val_predictions = {k: np.concatenate(v) for k, v in all_preds.items()}
            self.val_targets = {k: np.concatenate(v) for k, v in all_targets.items()}
        
        return avg_loss, avg_metrics
    
    def train(self, epochs: int = None):
        """完整训练"""
        epochs = epochs or self.config.EPOCHS
        train_loader, val_loader = self.load_data()
        
        print(f"\n{'='*60}")
        print(f"Inverse Design v7 Training (sqrt transform for ratios)")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader, collect_predictions=(epoch == epochs - 1))
            
            self.scheduler.step(val_loss)
            
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lattice_mse'].append(val_metrics['lattice_mse'])
            self.history['ratio_mse'].append(val_metrics['ratio_mse'])
            self.history['element_acc'].append(val_metrics['element_acc'])
            self.history['fraction_mse'].append(val_metrics['fraction_mse'])
            self.history['top3_acc'].append(val_metrics['top3_acc'])
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d} | "
                      f"Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | "
                      f"Ratio MSE: {val_metrics['ratio_mse']:.6f} | "
                      f"Elem Acc: {val_metrics['element_acc']:.1%}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best')
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
        
        # 最终验证并收集预测
        self.validate(val_loader, collect_predictions=True)
        
        print("\n✓ Training complete!")
        self.save_model('final')
        self.generate_report()
    
    def save_model(self, suffix: str = 'final'):
        """保存模型"""
        torch.save({
            'model': self.model.state_dict(),
            'config': {
                'input_dim': self.config.INPUT_DIM,
                'hidden_dim': self.config.HIDDEN_DIM,
                'num_elements': self.config.NUM_ELEMENTS,
                'max_elements': MAX_ELEMENTS
            }
        }, self.config.MODEL_DIR / f'inverse_design_v7_{suffix}.pt')
        
        print(f"✓ Model saved: inverse_design_v7_{suffix}.pt")
    
    def generate_report(self):
        """生成报告 (包括 parity plots)"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 训练曲线
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], label='Train', alpha=0.8)
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], label='Val', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.history['epoch'], self.history['element_acc'], label='Top-1', alpha=0.8)
        axes[0, 1].plot(self.history['epoch'], self.history['top3_acc'], label='Top-3', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].set_title('Element Prediction Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.history['epoch'], self.history['ratio_mse'], color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].set_title('Ratio (sqrt) Prediction MSE')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(self.history['epoch'], self.history['fraction_mse'], color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].set_title('Fraction Prediction MSE')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.REPORT_DIR / f'training_curves_{timestamp}.png', dpi=150)
        plt.close()
        
        # Parity Plots (关键改进)
        if self.val_predictions is not None:
            self._generate_parity_plots(timestamp)
        
        # 文本报告
        report_path = self.config.REPORT_DIR / f'training_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Inverse Design v7 Training Report (sqrt transform)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Feature Dimension: {self.config.INPUT_DIM}\n")
            f.write(f"Total Epochs: {len(self.history['epoch'])}\n\n")
            f.write("Key Improvement:\n")
            f.write("  - Predicting sqrt(b/a), sqrt(c/a) instead of b/a, c/a\n")
            f.write("  - This linearizes the ratio prediction problem\n\n")
            f.write("Final Metrics:\n")
            f.write(f"  Element Top-1 Accuracy: {self.history['element_acc'][-1]:.1%}\n")
            f.write(f"  Element Top-3 Accuracy: {self.history['top3_acc'][-1]:.1%}\n")
            f.write(f"  Lattice MSE: {self.history['lattice_mse'][-1]:.6f}\n")
            f.write(f"  Ratio MSE (sqrt): {self.history['ratio_mse'][-1]:.6f}\n")
            f.write(f"  Fraction MSE: {self.history['fraction_mse'][-1]:.6f}\n")
        
        print(f"✓ Report saved: {report_path}")
    
    def _generate_parity_plots(self, timestamp: str):
        """生成 Parity Plots"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 获取预测和目标
        pred_lattice = self.val_predictions['lattice']
        target_lattice = self.val_targets['lattice']
        raw_ratios = self.val_targets['raw_ratios']
        
        labels = ['vol_root', 'sqrt(b/a)', 'sqrt(c/a)', 'alpha', 'beta', 'gamma']
        
        # 所有晶格参数的 parity plots
        for i in range(6):
            ax = axes.flat[i]
            pred = pred_lattice[:, i]
            target = target_lattice[:, i]
            
            ax.scatter(target, pred, alpha=0.5, s=10)
            
            # y = x 线
            min_val = min(target.min(), pred.min())
            max_val = max(target.max(), pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x', linewidth=2)
            
            # 对于比值，也画 y = sqrt(x)
            if i in [1, 2]:
                x_range = np.linspace(max(0.01, min_val), max_val, 100)
                ax.plot(x_range, np.sqrt(x_range), 'g:', label='y = √x', linewidth=2)
            
            ax.set_xlabel(f'True {labels[i]}')
            ax.set_ylabel(f'Predicted {labels[i]}')
            ax.set_title(f'{labels[i]} Parity Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.REPORT_DIR / f'parity_plots_sqrt_{timestamp}.png', dpi=150)
        plt.close()
        
        # 额外: 在原始比值空间的对比
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 预测的 sqrt(ratio)^2 = ratio vs 真实 ratio
        for i, (name, ax) in enumerate(zip(['b/a', 'c/a'], axes)):
            pred_sqrt = pred_lattice[:, i + 1]
            pred_ratio = pred_sqrt ** 2  # 反变换
            true_ratio = raw_ratios[:, i]
            
            ax.scatter(true_ratio, pred_ratio, alpha=0.5, s=10)
            
            min_val = min(true_ratio.min(), pred_ratio.min())
            max_val = max(true_ratio.max(), pred_ratio.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x', linewidth=2)
            
            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{name} Parity Plot (after inverse sqrt)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 计算 R²
            correlation = np.corrcoef(true_ratio, pred_ratio)[0, 1]
            ax.text(0.05, 0.95, f'R² = {correlation**2:.3f}', transform=ax.transAxes,
                   fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        plt.savefig(self.config.REPORT_DIR / f'parity_plots_original_space_{timestamp}.png', dpi=150)
        plt.close()
        
        print(f"✓ Parity plots saved")
    
    def predict(self, features: np.ndarray) -> List[Dict]:
        """预测材料组成 (使用 sqrt 反变换)"""
        self.model.eval()
        
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            outputs = self.model(x)
        
        results = []
        batch_size = x.size(0)
        
        for i in range(batch_size):
            lattice = outputs['lattice'][i].cpu().numpy()
            vol_root = lattice[0] * 20.0
            
            # 反变换: sqrt(ratio) -> ratio
            sqrt_ratio_ba = max(lattice[1], 0.1)
            sqrt_ratio_ca = max(lattice[2], 0.1)
            ratio_ba = sqrt_ratio_ba ** 2  # 反变换
            ratio_ca = sqrt_ratio_ca ** 2  # 反变换
            
            alpha = lattice[3] * 180.0
            beta = lattice[4] * 180.0
            gamma = lattice[5] * 180.0
            
            # 从比值还原 a, b, c
            a = vol_root / (ratio_ba * ratio_ca) ** (1/3)
            b = a * ratio_ba
            c = a * ratio_ca
            
            # 元素
            element_logits = outputs['element_logits'][i]
            element_preds = element_logits.argmax(dim=1).cpu().numpy()
            fractions = outputs['fractions'][i].cpu().numpy()
            
            elements = []
            for j in range(MAX_ELEMENTS):
                if fractions[j] > 0.02:
                    el = IDX_TO_ELEMENT.get(int(element_preds[j]), 'X')
                    elements.append({
                        'element': el,
                        'fraction': float(fractions[j])
                    })
            
            sg = int(outputs['spacegroup'][i].item() * 230)
            sg = max(1, min(sg, 230))
            
            results.append({
                'lattice': {
                    'a': float(a), 'b': float(b), 'c': float(c),
                    'alpha': float(alpha), 'beta': float(beta), 'gamma': float(gamma),
                    'volume': float(vol_root ** 3)
                },
                'composition': elements,
                'spacegroup': sg
            })
        
        return results


# ==========================================
# 5. 主函数
# ==========================================
def main():
    print("="*60)
    print("Inverse Design Model v7 (sqrt transform for lattice ratios)")
    print("="*60)
    
    trainer = InverseDesignTrainerV7()
    trainer.train(epochs=150)
    
    print("\n" + "="*60)
    print("Testing predictions...")
    
    test_file = trainer.config.DATA_DIR / 'dataset_original_ferroelectric.jsonl'
    test_features = []
    
    with open(test_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            item = json.loads(line)
            feat = trainer.extractor.extract_from_structure_dict(
                item['structure'], 
                item.get('spacegroup_number')
            )
            test_features.append(feat)
    
    test_features = np.array(test_features)
    predictions = trainer.predict(test_features)
    
    print("\nSample predictions:")
    for i, pred in enumerate(predictions[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Lattice: a={pred['lattice']['a']:.3f}, b={pred['lattice']['b']:.3f}, c={pred['lattice']['c']:.3f}")
        print(f"  Composition: {pred['composition']}")
        print(f"  Spacegroup: {pred['spacegroup']}")


if __name__ == '__main__':
    main()
