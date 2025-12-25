"""
高质量生成对抗网络 (GAN) v4 - 铁电材料生成
=============================================
改进策略:
1. 更强的分布匹配 (直方图匹配)
2. 更深的网络
3. 更长的训练
4. 学习率调度
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from feature_engineering import (
    UnifiedFeatureExtractor,
    FEATURE_DIM,
    FEATURE_NAMES,
)


class Config:
    FEATURE_DIM = FEATURE_DIM  # 64
    LATENT_DIM = 128
    HIDDEN_DIM = 768  # 更大
    
    BATCH_SIZE = 64
    EPOCHS = 500
    LR_G = 3e-4
    LR_D = 1e-4
    
    # 损失权重
    DIST_WEIGHT = 30.0    # 分布匹配
    HIST_WEIGHT = 10.0    # 直方图匹配
    ADV_WEIGHT = 1.0
    
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_gan_v4'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_gan_v4'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        
        self.fe_features = self.features[self.labels == 1]
        self.nonfe_features = self.features[self.labels == 0]
        
        # 计算统计量
        self.fe_stats = self._compute_stats(self.fe_features)
        self.nonfe_stats = self._compute_stats(self.nonfe_features)
        
        # 计算直方图 (用于分布匹配)
        self.fe_hists = self._compute_histograms(self.fe_features)
        self.nonfe_hists = self._compute_histograms(self.nonfe_features)
        
        print(f"Loaded {len(self.features)} samples (FE: {len(self.fe_features)}, Non-FE: {len(self.nonfe_features)})")
    
    def _compute_stats(self, data):
        return {
            'mean': torch.tensor(data.mean(0), dtype=torch.float32),
            'std': torch.tensor(data.std(0) + 1e-6, dtype=torch.float32),
            'min': torch.tensor(data.min(0), dtype=torch.float32),
            'max': torch.tensor(data.max(0), dtype=torch.float32),
            'median': torch.tensor(np.median(data, axis=0), dtype=torch.float32),
            'q25': torch.tensor(np.percentile(data, 25, axis=0), dtype=torch.float32),
            'q75': torch.tensor(np.percentile(data, 75, axis=0), dtype=torch.float32),
        }
    
    def _compute_histograms(self, data, n_bins=20):
        """计算每个特征的直方图"""
        hists = []
        for i in range(data.shape[1]):
            hist, _ = np.histogram(data[:, i], bins=n_bins, range=(0, 1), density=True)
            hists.append(hist)
        return torch.tensor(np.array(hists), dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx])


class Generator(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=768, output_dim=64):
        super().__init__()
        
        self.cond_embed = nn.Embedding(2, 64)
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z, labels):
        cond = self.cond_embed(labels)
        x = torch.cat([z, cond], dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=768):
        super().__init__()
        
        self.cond_embed = nn.Embedding(2, 64)
        
        self.features = nn.Sequential(
            nn.Linear(input_dim + 64, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )
        
        self.validity = nn.Linear(hidden_dim // 2, 1)
    
    def forward(self, x, labels):
        cond = self.cond_embed(labels)
        h = torch.cat([x, cond], dim=1)
        feat = self.features(h)
        return self.validity(feat), feat


def soft_histogram(x, n_bins=20, min_val=0, max_val=1):
    """可微分软直方图"""
    bin_width = (max_val - min_val) / n_bins
    bin_centers = torch.linspace(min_val + bin_width/2, max_val - bin_width/2, n_bins, device=x.device)
    
    # 高斯核软化
    sigma = bin_width * 0.5
    diff = x.unsqueeze(-1) - bin_centers
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)
    hist = weights.mean(dim=0)
    hist = hist / (hist.sum() + 1e-8)
    return hist


class Trainer:
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.extractor = UnifiedFeatureExtractor()
        
        self.G = Generator(self.config.LATENT_DIM, self.config.HIDDEN_DIM, self.config.FEATURE_DIM).to(self.device)
        self.D = Discriminator(self.config.FEATURE_DIM, self.config.HIDDEN_DIM).to(self.device)
        
        self.opt_G = optim.AdamW(self.G.parameters(), lr=self.config.LR_G, weight_decay=1e-4)
        self.opt_D = optim.AdamW(self.D.parameters(), lr=self.config.LR_D, weight_decay=1e-4)
        
        self.scheduler_G = optim.lr_scheduler.CosineAnnealingLR(self.opt_G, self.config.EPOCHS)
        self.scheduler_D = optim.lr_scheduler.CosineAnnealingLR(self.opt_D, self.config.EPOCHS)
        
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
        
        # 移动统计量到设备
        self.fe_stats = {k: v.to(self.device) for k, v in self.dataset.fe_stats.items()}
        self.nonfe_stats = {k: v.to(self.device) for k, v in self.dataset.nonfe_stats.items()}
        self.fe_hists = self.dataset.fe_hists.to(self.device)
        self.nonfe_hists = self.dataset.nonfe_hists.to(self.device)
        
        return DataLoader(self.dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    def distribution_loss(self, fake_data, labels):
        """全面分布匹配损失"""
        loss = torch.tensor(0.0, device=self.device)
        
        fe_mask = (labels == 1)
        nonfe_mask = (labels == 0)
        
        for mask, stats, hists in [(fe_mask, self.fe_stats, self.fe_hists), 
                                    (nonfe_mask, self.nonfe_stats, self.nonfe_hists)]:
            if mask.sum() < 2:
                continue
            
            fake = fake_data[mask]
            
            # 1. 均值匹配
            loss = loss + F.mse_loss(fake.mean(0), stats['mean']) * 15
            
            # 2. 标准差匹配
            loss = loss + F.mse_loss(fake.std(0), stats['std']) * 10
            
            # 3. 四分位数匹配
            fake_sorted, _ = torch.sort(fake, dim=0)
            n = fake.size(0)
            q25_idx = int(0.25 * n)
            q75_idx = int(0.75 * n)
            
            loss = loss + F.mse_loss(fake_sorted[q25_idx], stats['q25']) * 5
            loss = loss + F.mse_loss(fake_sorted[q75_idx], stats['q75']) * 5
            
            # 4. 直方图匹配 (前20个特征)
            for i in range(min(20, self.config.FEATURE_DIM)):
                fake_hist = soft_histogram(fake[:, i])
                real_hist = hists[i]
                loss = loss + F.mse_loss(fake_hist, real_hist) * self.config.HIST_WEIGHT
        
        return loss
    
    def train_step(self, real_data, real_labels):
        bs = real_data.size(0)
        real_data = real_data.to(self.device)
        real_labels = real_labels.long().to(self.device)
        
        ones = torch.ones(bs, 1, device=self.device)
        zeros = torch.zeros(bs, 1, device=self.device)
        
        # Train D
        self.opt_D.zero_grad()
        
        real_pred, _ = self.D(real_data, real_labels)
        d_real = self.bce(real_pred, ones * 0.9)
        
        z = torch.randn(bs, self.config.LATENT_DIM, device=self.device)
        fake_data = self.G(z, real_labels)
        fake_pred, _ = self.D(fake_data.detach(), real_labels)
        d_fake = self.bce(fake_pred, zeros + 0.1)
        
        d_loss = d_real + d_fake
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.D.parameters(), 1.0)
        self.opt_D.step()
        
        # Train G
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
        print("GAN v4 Training (Histogram Matching)")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Distribution Weight: {self.config.DIST_WEIGHT}")
        print(f"Histogram Weight: {self.config.HIST_WEIGHT}")
        print(f"{'='*60}\n")
        
        best_corr = 0
        
        for epoch in range(epochs):
            metrics = {'d_loss': [], 'g_loss': [], 'dist_loss': []}
            
            for real_data, real_labels in loader:
                m = self.train_step(real_data, real_labels)
                for k, v in m.items():
                    metrics[k].append(v)
            
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            self.history['epoch'].append(epoch + 1)
            self.history['d_loss'].append(np.mean(metrics['d_loss']))
            self.history['g_loss'].append(np.mean(metrics['g_loss']))
            self.history['dist_loss'].append(np.mean(metrics['dist_loss']))
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
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
        self.final_evaluate()
    
    def save(self, suffix):
        torch.save({
            'generator': self.G.state_dict(),
            'discriminator': self.D.state_dict(),
            'config': {
                'latent_dim': self.config.LATENT_DIM, 
                'hidden_dim': self.config.HIDDEN_DIM,
                'cond_dim': 64
            }
        }, self.config.MODEL_DIR / f'gan_v4_{suffix}.pt')
        torch.save(self.G.state_dict(), self.config.MODEL_DIR / f'generator_v4_{suffix}.pt')
        print(f"✓ Saved gan_v4_{suffix}.pt")
    
    def final_evaluate(self):
        """最终评估"""
        from scipy import stats as scipy_stats
        
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(1000, self.config.LATENT_DIM, device=self.device)
            labels = torch.ones(1000, dtype=torch.long, device=self.device)
            gen_fe = self.G(z, labels).cpu().numpy()
        
        real_fe = self.dataset.fe_features
        
        print(f"\n{'='*60}")
        print("最终评估结果")
        print(f"{'='*60}")
        
        # KS检验
        ks_passes = 0
        for i in range(20):
            stat, pval = scipy_stats.ks_2samp(real_fe[:, i], gen_fe[:, i])
            if pval > 0.05:
                ks_passes += 1
        
        mean_diff = np.abs(real_fe.mean(0) - gen_fe.mean(0)).mean()
        corr = np.corrcoef(real_fe.mean(0), gen_fe.mean(0))[0, 1]
        
        print(f"KS检验通过率: {ks_passes}/20 ({100*ks_passes/20:.1f}%)")
        print(f"均值差异: {mean_diff:.4f}")
        print(f"均值相关系数: {corr:.4f}")


def main():
    print("="*60)
    print("High Quality GAN v4 - Histogram Matching")
    print("="*60)
    trainer = Trainer()
    trainer.train(epochs=500)


if __name__ == '__main__':
    main()
