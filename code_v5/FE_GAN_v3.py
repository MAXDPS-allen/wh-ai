"""
高质量生成对抗网络 (GAN) v3 - 铁电材料生成
=============================================
改进策略:
1. 标准GAN训练 (更稳定)
2. 强力分布匹配损失
3. 逐特征统计约束
4. 梯度裁剪防止NaN
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
)


# ==========================================
# 1. 配置
# ==========================================
class Config:
    FEATURE_DIM = FEATURE_DIM  # 64
    LATENT_DIM = 128
    HIDDEN_DIM = 512
    
    BATCH_SIZE = 32
    EPOCHS = 300
    LR = 2e-4
    
    # 损失权重 - 分布匹配为主
    DIST_WEIGHT = 20.0   # 分布匹配权重 (很高)
    ADV_WEIGHT = 1.0     # 对抗损失权重
    
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_gan_v3'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_gan_v3'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 2. 数据集
# ==========================================
class MaterialDataset(Dataset):
    def __init__(self, data_files, extractor):
        self.features = []
        self.labels = []
        
        for file_path, label in data_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            struct = item.get('structure')
                            if struct:
                                sg = item.get('spacegroup_number', None)
                                feat = extractor.extract_from_structure_dict(struct, sg)
                                if np.sum(np.abs(feat)) > 0:
                                    self.features.append(feat)
                                    self.labels.append(label)
                        except:
                            continue
        
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        
        # 分离FE和Non-FE
        self.fe_features = self.features[self.labels == 1]
        self.nonfe_features = self.features[self.labels == 0]
        
        # 计算统计量
        self.fe_stats = {
            'mean': torch.tensor(self.fe_features.mean(0), dtype=torch.float32),
            'std': torch.tensor(self.fe_features.std(0) + 1e-6, dtype=torch.float32),
        }
        self.nonfe_stats = {
            'mean': torch.tensor(self.nonfe_features.mean(0), dtype=torch.float32),
            'std': torch.tensor(self.nonfe_features.std(0) + 1e-6, dtype=torch.float32),
        }
        
        print(f"Loaded {len(self.features)} samples (FE: {len(self.fe_features)}, Non-FE: {len(self.nonfe_features)})")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx])


# ==========================================
# 3. 生成器
# ==========================================
class Generator(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=512, output_dim=64):
        super().__init__()
        
        self.cond_embed = nn.Embedding(2, 32)
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()  # 输出[0, 1]
        )
    
    def forward(self, z, labels):
        cond = self.cond_embed(labels)
        x = torch.cat([z, cond], dim=1)
        return self.net(x)


# ==========================================
# 4. 判别器
# ==========================================
class Discriminator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=512):
        super().__init__()
        
        self.cond_embed = nn.Embedding(2, 32)
        
        self.features = nn.Sequential(
            nn.Linear(input_dim + 32, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )
        
        self.validity = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x, labels):
        cond = self.cond_embed(labels)
        h = torch.cat([x, cond], dim=1)
        feat = self.features(h)
        return self.validity(feat), feat


# ==========================================
# 5. 训练器
# ==========================================
class Trainer:
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.extractor = UnifiedFeatureExtractor()
        
        self.G = Generator(self.config.LATENT_DIM, self.config.HIDDEN_DIM, self.config.FEATURE_DIM).to(self.device)
        self.D = Discriminator(self.config.FEATURE_DIM, self.config.HIDDEN_DIM).to(self.device)
        
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.config.LR, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.config.LR * 0.5, betas=(0.5, 0.999))
        
        self.bce = nn.BCEWithLogitsLoss()
        
        self.history = {'epoch': [], 'g_loss': [], 'd_loss': [], 'dist_loss': [], 'mean_diff': [], 'corr': []}
        self.dataset = None
    
    def load_data(self):
        data_files = [
            (str(self.config.DATA_DIR / 'dataset_original_ferroelectric.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_known_FE_rest.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_nonFE.jsonl'), 0),
            (str(self.config.DATA_DIR / 'dataset_polar_non_ferroelectric_final.jsonl'), 0),
        ]
        self.dataset = MaterialDataset(data_files, self.extractor)
        
        # 将统计量移到设备
        self.fe_mean = self.dataset.fe_stats['mean'].to(self.device)
        self.fe_std = self.dataset.fe_stats['std'].to(self.device)
        self.nonfe_mean = self.dataset.nonfe_stats['mean'].to(self.device)
        self.nonfe_std = self.dataset.nonfe_stats['std'].to(self.device)
        
        return DataLoader(self.dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    def distribution_loss(self, fake_data, labels):
        """强力分布匹配损失"""
        loss = torch.tensor(0.0, device=self.device)
        
        fe_mask = (labels == 1)
        nonfe_mask = (labels == 0)
        
        if fe_mask.sum() > 1:
            fe_fake = fake_data[fe_mask]
            # 均值匹配
            loss = loss + F.mse_loss(fe_fake.mean(0), self.fe_mean) * 10
            # 方差匹配
            loss = loss + F.mse_loss(fe_fake.std(0), self.fe_std) * 5
            # 逐特征约束
            for i in range(self.config.FEATURE_DIM):
                loss = loss + F.mse_loss(fe_fake[:, i].mean(), self.fe_mean[i])
        
        if nonfe_mask.sum() > 1:
            nonfe_fake = fake_data[nonfe_mask]
            loss = loss + F.mse_loss(nonfe_fake.mean(0), self.nonfe_mean) * 10
            loss = loss + F.mse_loss(nonfe_fake.std(0), self.nonfe_std) * 5
        
        return loss
    
    def train_step(self, real_data, real_labels):
        bs = real_data.size(0)
        real_data = real_data.to(self.device)
        real_labels = real_labels.long().to(self.device)
        
        ones = torch.ones(bs, 1, device=self.device)
        zeros = torch.zeros(bs, 1, device=self.device)
        
        # === Train D ===
        self.opt_D.zero_grad()
        
        real_pred, _ = self.D(real_data, real_labels)
        d_real = self.bce(real_pred, ones * 0.9)  # label smoothing
        
        z = torch.randn(bs, self.config.LATENT_DIM, device=self.device)
        fake_data = self.G(z, real_labels)
        fake_pred, _ = self.D(fake_data.detach(), real_labels)
        d_fake = self.bce(fake_pred, zeros + 0.1)
        
        d_loss = d_real + d_fake
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
        self.opt_D.step()
        
        # === Train G ===
        self.opt_G.zero_grad()
        
        z = torch.randn(bs, self.config.LATENT_DIM, device=self.device)
        fake_data = self.G(z, real_labels)
        fake_pred, _ = self.D(fake_data, real_labels)
        
        g_adv = self.bce(fake_pred, ones)
        g_dist = self.distribution_loss(fake_data, real_labels)
        
        g_loss = self.config.ADV_WEIGHT * g_adv + self.config.DIST_WEIGHT * g_dist
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
        self.opt_G.step()
        
        return {'d_loss': d_loss.item(), 'g_loss': g_loss.item(), 'dist_loss': g_dist.item()}
    
    def evaluate(self, n=500):
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n, self.config.LATENT_DIM, device=self.device)
            labels = torch.ones(n, dtype=torch.long, device=self.device)
            fake = self.G(z, labels).cpu().numpy()
            real = self.dataset.fe_features[:n]
            
            mean_diff = np.abs(fake.mean(0) - real.mean(0)).mean()
            corr = np.corrcoef(fake.mean(0), real.mean(0))[0, 1]
        self.G.train()
        return {'mean_diff': mean_diff, 'corr': corr}
    
    def train(self, epochs=None):
        epochs = epochs or self.config.EPOCHS
        loader = self.load_data()
        
        print(f"\n{'='*60}")
        print("GAN v3 Training (Distribution-Focused)")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Distribution Weight: {self.config.DIST_WEIGHT}")
        print(f"{'='*60}\n")
        
        best_corr = 0
        
        for epoch in range(epochs):
            metrics = {'d_loss': [], 'g_loss': [], 'dist_loss': []}
            
            for real_data, real_labels in loader:
                m = self.train_step(real_data, real_labels)
                for k, v in m.items():
                    metrics[k].append(v)
            
            self.history['epoch'].append(epoch + 1)
            self.history['d_loss'].append(np.mean(metrics['d_loss']))
            self.history['g_loss'].append(np.mean(metrics['g_loss']))
            self.history['dist_loss'].append(np.mean(metrics['dist_loss']))
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                ev = self.evaluate()
                self.history['mean_diff'].append(ev['mean_diff'])
                self.history['corr'].append(ev['corr'])
                
                print(f"Epoch {epoch+1:3d} | D: {np.mean(metrics['d_loss']):.4f} | "
                      f"G: {np.mean(metrics['g_loss']):.4f} | "
                      f"Dist: {np.mean(metrics['dist_loss']):.4f} | "
                      f"Mean_diff: {ev['mean_diff']:.4f} | Corr: {ev['corr']:.4f}")
                
                if ev['corr'] > best_corr:
                    best_corr = ev['corr']
                    self.save('best')
        
        print("\n✓ Training complete!")
        self.save('final')
        self.report()
    
    def save(self, suffix):
        torch.save({
            'generator': self.G.state_dict(),
            'discriminator': self.D.state_dict(),
            'config': {'latent_dim': self.config.LATENT_DIM, 'hidden_dim': self.config.HIDDEN_DIM}
        }, self.config.MODEL_DIR / f'gan_v3_{suffix}.pt')
        torch.save(self.G.state_dict(), self.config.MODEL_DIR / f'generator_v3_{suffix}.pt')
        print(f"✓ Saved gan_v3_{suffix}.pt")
    
    def report(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        ev = self.evaluate(1000)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        axes[0, 0].plot(self.history['epoch'], self.history['d_loss'], label='D Loss')
        axes[0, 0].plot(self.history['epoch'], self.history['g_loss'], label='G Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.history['epoch'], self.history['dist_loss'], 'g-')
        axes[0, 1].set_title('Distribution Matching Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 生成vs真实对比
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(500, self.config.LATENT_DIM, device=self.device)
            labels = torch.ones(500, dtype=torch.long, device=self.device)
            fake = self.G(z, labels).cpu().numpy()
        real = self.dataset.fe_features[:500]
        
        # 特征均值对比
        gen_mean = fake.mean(0)[:20]
        real_mean = real.mean(0)[:20]
        x = np.arange(20)
        width = 0.35
        axes[0, 2].bar(x - width/2, real_mean, width, label='Real FE')
        axes[0, 2].bar(x + width/2, gen_mean, width, label='Generated FE')
        axes[0, 2].set_title('Feature Mean Comparison')
        axes[0, 2].legend()
        
        # 特征分布对比
        for i, feat_idx in enumerate([0, 12, 24]):
            axes[1, i].hist(real[:, feat_idx], bins=30, alpha=0.5, label='Real', density=True)
            axes[1, i].hist(fake[:, feat_idx], bins=30, alpha=0.5, label='Generated', density=True)
            axes[1, i].set_title(f'Feature {feat_idx} Distribution')
            axes[1, i].legend()
        
        plt.tight_layout()
        plt.savefig(self.config.REPORT_DIR / f'training_report_{ts}.png', dpi=150)
        plt.close()
        
        with open(self.config.REPORT_DIR / f'training_report_{ts}.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("GAN v3 Training Report (Distribution-Focused)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Epochs: {len(self.history['epoch'])}\n\n")
            f.write("Final Metrics:\n")
            f.write(f"  Feature Mean Difference: {ev['mean_diff']:.4f}\n")
            f.write(f"  Feature Correlation: {ev['corr']:.4f}\n")
            f.write(f"  Final D Loss: {self.history['d_loss'][-1]:.4f}\n")
            f.write(f"  Final G Loss: {self.history['g_loss'][-1]:.4f}\n")
            f.write(f"  Final Dist Loss: {self.history['dist_loss'][-1]:.4f}\n")
        
        print(f"✓ Report saved")


def main():
    print("="*60)
    print("High Quality GAN v3 - Distribution Focused")
    print("="*60)
    trainer = Trainer()
    trainer.train(epochs=300)


if __name__ == '__main__':
    main()
