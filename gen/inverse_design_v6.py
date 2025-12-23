"""
增强型逆向设计模型 v6
=============================================
从64维特征向量预测材料组成和晶格参数

改进:
1. 使用统一特征工程模块 (64维)
2. 分离晶格预测和成分预测网络
3. 添加注意力机制
4. 使用更深的网络结构
5. 改进元素预测为分类问题
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
    extract_features,
    extract_target,
    decode_composition,
    decode_lattice
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
    LATTICE_DIM = 6  # vol_root, b/a, c/a, alpha, beta, gamma
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
    REPORT_DIR = Path(__file__).parent.parent / 'reports_inverse_v2'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 2. 数据集
# ==========================================
class InverseDesignDataset(Dataset):
    """逆向设计数据集"""
    
    def __init__(self, data_files: List[str], extractor: UnifiedFeatureExtractor):
        self.features = []
        self.targets_lattice = []
        self.targets_elements = []  # 元素索引 (分类)
        self.targets_fractions = []  # 元素比例 (回归)
        self.targets_spacegroup = []
        self.extractor = extractor
        
        for file_path in data_files:
            if os.path.exists(file_path):
                self._load_file(file_path)
        
        if self.features:
            self.features = np.array(self.features, dtype=np.float32)
            self.targets_lattice = np.array(self.targets_lattice, dtype=np.float32)
            self.targets_elements = np.array(self.targets_elements, dtype=np.int64)
            self.targets_fractions = np.array(self.targets_fractions, dtype=np.float32)
            self.targets_spacegroup = np.array(self.targets_spacegroup, dtype=np.float32)
        else:
            self.features = np.zeros((0, Config.INPUT_DIM), dtype=np.float32)
            self.targets_lattice = np.zeros((0, 6), dtype=np.float32)
            self.targets_elements = np.zeros((0, MAX_ELEMENTS), dtype=np.int64)
            self.targets_fractions = np.zeros((0, MAX_ELEMENTS), dtype=np.float32)
            self.targets_spacegroup = np.zeros((0, 1), dtype=np.float32)
        
        print(f"Loaded {len(self.features)} samples for inverse design")
    
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
                        
                        lattice, elements, fractions, spacegroup = target_data
                        
                        self.features.append(feat)
                        self.targets_lattice.append(lattice)
                        self.targets_elements.append(elements)
                        self.targets_fractions.append(fractions)
                        self.targets_spacegroup.append(spacegroup)
                        
                except Exception as e:
                    continue
    
    def _extract_targets(self, struct_dict, spacegroup):
        """提取目标向量"""
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
            
            # 晶格目标
            vol_root = vol ** (1/3)
            lattice_targets = [
                vol_root / 20.0,  # 归一化
                b / a,
                c / a,
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
            
            # 按比例排序（从高到低）
            el_list.sort(key=lambda x: x[1], reverse=True)
            
            # 填充到MAX_ELEMENTS
            elements = []
            fractions = []
            for i in range(MAX_ELEMENTS):
                if i < len(el_list):
                    elements.append(el_list[i][0])
                    fractions.append(el_list[i][1])
                else:
                    elements.append(0)  # padding
                    fractions.append(0)
            
            # 空间群
            sg_norm = [spacegroup / 230.0]
            
            return (
                np.array(lattice_targets, dtype=np.float32),
                np.array(elements, dtype=np.int64),
                np.array(fractions, dtype=np.float32),
                np.array(sg_norm, dtype=np.float32)
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
            'spacegroup': torch.from_numpy(self.targets_spacegroup[idx])
        }


# ==========================================
# 3. 网络架构
# ==========================================
class AttentionBlock(nn.Module):
    """自注意力模块"""
    
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # x: [B, D] -> [B, 1, D]
        x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x.squeeze(1)


class LatticePredictor(nn.Module):
    """晶格参数预测网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        
        # 分别预测不同的晶格参数
        self.head_volume = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.head_ratios = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # b/a, c/a
            nn.Softplus()
        )
        
        self.head_angles = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.GELU(),
            nn.Linear(32, 3),  # alpha, beta, gamma (归一化)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        h = self.encoder(x)
        volume = self.head_volume(h)
        ratios = self.head_ratios(h)
        angles = self.head_angles(h)
        return torch.cat([volume, ratios, angles], dim=1)


class CompositionPredictor(nn.Module):
    """化学成分预测网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_elements: int = NUM_ELEMENTS, max_output: int = MAX_ELEMENTS):
        super().__init__()
        
        self.num_elements = num_elements
        self.max_output = max_output
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.attention = AttentionBlock(hidden_dim, num_heads=4)
        
        # 元素分类器 (每个位置独立分类)
        self.element_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, num_elements)
            ) for _ in range(max_output)
        ])
        
        # 比例回归器
        self.fraction_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, max_output),
            nn.Softmax(dim=1)  # 确保比例和为1
        )
    
    def forward(self, x):
        h = self.encoder(x)
        h = self.attention(h)
        
        # 元素分类 (logits)
        element_logits = []
        for classifier in self.element_classifiers:
            logits = classifier(h)
            element_logits.append(logits)
        element_logits = torch.stack(element_logits, dim=1)  # [B, MAX, NUM_ELEM]
        
        # 比例预测
        fractions = self.fraction_head(h)
        
        return element_logits, fractions


class SpacegroupPredictor(nn.Module):
    """空间群预测网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


class InverseDesignModel(nn.Module):
    """完整逆向设计模型"""
    
    def __init__(self, config: Config = None):
        super().__init__()
        
        config = config or Config()
        
        # 共享编码器
        self.shared_encoder = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.GELU(),
        )
        
        # 专用预测头
        self.lattice_predictor = LatticePredictor(config.HIDDEN_DIM, 256)
        self.composition_predictor = CompositionPredictor(config.HIDDEN_DIM, 256)
        self.spacegroup_predictor = SpacegroupPredictor(config.HIDDEN_DIM, 128)
    
    def forward(self, x):
        h = self.shared_encoder(x)
        
        lattice = self.lattice_predictor(h)
        element_logits, fractions = self.composition_predictor(h)
        spacegroup = self.spacegroup_predictor(h)
        
        return {
            'lattice': lattice,
            'element_logits': element_logits,
            'fractions': fractions,
            'spacegroup': spacegroup
        }


# ==========================================
# 4. 训练器
# ==========================================
class InverseDesignTrainer:
    """逆向设计训练器"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        # 创建目录
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 特征提取器
        self.extractor = UnifiedFeatureExtractor()
        
        # 模型
        self.model = InverseDesignModel(self.config).to(self.device)
        
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LR,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # 学习率调度
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 训练历史
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'lattice_mse': [],
            'element_acc': [],
            'fraction_mse': [],
            'top3_acc': []
        }
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """加载数据"""
        data_files = [
            str(self.config.DATA_DIR / 'dataset_original_ferroelectric.jsonl'),
            str(self.config.DATA_DIR / 'dataset_known_FE_rest.jsonl'),
        ]
        
        dataset = InverseDesignDataset(data_files, self.extractor)
        
        # 划分训练/验证
        train_size = int(0.85 * len(dataset))
        val_size = len(dataset) - train_size
        train_set, val_set = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_set, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader
    
    def compute_loss(self, outputs, batch):
        """计算损失"""
        # 晶格损失
        lattice_loss = self.mse_loss(outputs['lattice'], batch['lattice'].to(self.device))
        
        # 元素分类损失
        element_logits = outputs['element_logits']  # [B, MAX, NUM_ELEM]
        element_targets = batch['elements'].to(self.device)  # [B, MAX]
        
        element_loss = 0
        for i in range(MAX_ELEMENTS):
            element_loss += self.ce_loss(element_logits[:, i, :], element_targets[:, i])
        element_loss /= MAX_ELEMENTS
        
        # 比例损失
        fraction_loss = self.mse_loss(outputs['fractions'], batch['fractions'].to(self.device))
        
        # 空间群损失
        sg_loss = self.mse_loss(outputs['spacegroup'], batch['spacegroup'].to(self.device))
        
        # 总损失
        total_loss = lattice_loss + 2.0 * element_loss + fraction_loss + 0.5 * sg_loss
        
        return {
            'total': total_loss,
            'lattice': lattice_loss.item(),
            'element': element_loss.item(),
            'fraction': fraction_loss.item(),
            'spacegroup': sg_loss.item()
        }
    
    def compute_metrics(self, outputs, batch):
        """计算评估指标"""
        # 晶格 MSE
        lattice_mse = self.mse_loss(
            outputs['lattice'], 
            batch['lattice'].to(self.device)
        ).item()
        
        # 元素准确率
        element_logits = outputs['element_logits']
        element_targets = batch['elements'].to(self.device)
        
        # Top-1 准确率
        element_preds = element_logits.argmax(dim=2)  # [B, MAX]
        element_acc = (element_preds == element_targets).float().mean().item()
        
        # Top-3 准确率
        top3_acc = 0
        for i in range(MAX_ELEMENTS):
            _, top3_idx = element_logits[:, i, :].topk(3, dim=1)
            target = element_targets[:, i].unsqueeze(1)
            top3_acc += (top3_idx == target).any(dim=1).float().mean().item()
        top3_acc /= MAX_ELEMENTS
        
        # 比例 MSE
        fraction_mse = self.mse_loss(
            outputs['fractions'],
            batch['fractions'].to(self.device)
        ).item()
        
        return {
            'lattice_mse': lattice_mse,
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
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            total_loss += losses['total'].item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_metrics = {'lattice_mse': [], 'element_acc': [], 'top3_acc': [], 'fraction_mse': []}
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                outputs = self.model(features)
                
                losses = self.compute_loss(outputs, batch)
                total_loss += losses['total'].item()
                
                metrics = self.compute_metrics(outputs, batch)
                for k, v in metrics.items():
                    all_metrics[k].append(v)
        
        avg_loss = total_loss / len(dataloader)
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def train(self, epochs: int = None):
        """完整训练"""
        epochs = epochs or self.config.EPOCHS
        train_loader, val_loader = self.load_data()
        
        print(f"\n{'='*60}")
        print(f"Inverse Design Training (64-dim features)")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate(val_loader)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            # 记录历史
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lattice_mse'].append(val_metrics['lattice_mse'])
            self.history['element_acc'].append(val_metrics['element_acc'])
            self.history['fraction_mse'].append(val_metrics['fraction_mse'])
            self.history['top3_acc'].append(val_metrics['top3_acc'])
            
            # 输出
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d} | "
                      f"Train: {train_loss:.4f} | "
                      f"Val: {val_loss:.4f} | "
                      f"Elem Acc: {val_metrics['element_acc']:.1%} | "
                      f"Top3: {val_metrics['top3_acc']:.1%}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best')
            else:
                patience_counter += 1
            
            # 早停
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
                'input_dim': self.config.INPUT_DIM,
                'hidden_dim': self.config.HIDDEN_DIM,
                'num_elements': self.config.NUM_ELEMENTS,
                'max_elements': MAX_ELEMENTS
            }
        }, self.config.MODEL_DIR / f'inverse_design_v6_{suffix}.pt')
        
        print(f"✓ Model saved: inverse_design_v6_{suffix}.pt")
    
    def generate_report(self):
        """生成报告"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.history['epoch'], self.history['train_loss'], label='Train', alpha=0.8)
        axes[0, 0].plot(self.history['epoch'], self.history['val_loss'], label='Val', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 元素准确率
        axes[0, 1].plot(self.history['epoch'], self.history['element_acc'], label='Top-1', alpha=0.8)
        axes[0, 1].plot(self.history['epoch'], self.history['top3_acc'], label='Top-3', alpha=0.8)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].set_title('Element Prediction Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 晶格MSE
        axes[1, 0].plot(self.history['epoch'], self.history['lattice_mse'], color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].set_title('Lattice Prediction MSE')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 比例MSE
        axes[1, 1].plot(self.history['epoch'], self.history['fraction_mse'], color='orange')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MSE')
        axes[1, 1].set_title('Fraction Prediction MSE')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.REPORT_DIR / f'training_report_{timestamp}.png', dpi=150)
        plt.close()
        
        # 文本报告
        report_path = self.config.REPORT_DIR / f'training_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Inverse Design Training Report (64-dim features)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Feature Dimension: {self.config.INPUT_DIM}\n")
            f.write(f"Total Epochs: {len(self.history['epoch'])}\n\n")
            
            f.write("Final Metrics:\n")
            f.write(f"  Element Top-1 Accuracy: {self.history['element_acc'][-1]:.1%}\n")
            f.write(f"  Element Top-3 Accuracy: {self.history['top3_acc'][-1]:.1%}\n")
            f.write(f"  Lattice MSE: {self.history['lattice_mse'][-1]:.6f}\n")
            f.write(f"  Fraction MSE: {self.history['fraction_mse'][-1]:.6f}\n")
        
        print(f"✓ Report saved: {report_path}")
    
    def predict(self, features: np.ndarray) -> List[Dict]:
        """预测材料组成"""
        self.model.eval()
        
        with torch.no_grad():
            x = torch.from_numpy(features).float().to(self.device)
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            outputs = self.model(x)
        
        results = []
        batch_size = x.size(0)
        
        for i in range(batch_size):
            # 解码晶格
            lattice = outputs['lattice'][i].cpu().numpy()
            vol_root = lattice[0] * 20.0
            ratio_ba = max(lattice[1], 0.1)
            ratio_ca = max(lattice[2], 0.1)
            alpha = lattice[3] * 180.0
            beta = lattice[4] * 180.0
            gamma = lattice[5] * 180.0
            
            a = vol_root / (ratio_ba * ratio_ca) ** (1/3)
            b = a * ratio_ba
            c = a * ratio_ca
            
            # 解码元素
            element_logits = outputs['element_logits'][i]  # [MAX, NUM_ELEM]
            element_preds = element_logits.argmax(dim=1).cpu().numpy()
            fractions = outputs['fractions'][i].cpu().numpy()
            
            elements = []
            for j in range(MAX_ELEMENTS):
                if fractions[j] > 0.02:  # 过滤小于2%
                    el = IDX_TO_ELEMENT.get(int(element_preds[j]), 'X')
                    elements.append({
                        'element': el,
                        'fraction': float(fractions[j])
                    })
            
            # 空间群
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
    print("Inverse Design Model v6 (64-dim unified features)")
    print("="*60)
    
    trainer = InverseDesignTrainer()
    trainer.train(epochs=150)
    
    # 测试预测
    print("\n" + "="*60)
    print("Testing predictions...")
    
    # 加载一些真实数据作为测试
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
