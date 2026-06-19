"""
条件变分自编码器 (CVAE) - 铁电材料生成
==========================================
VAE能更好地学习数据分布，比GAN更稳定
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
    LATENT_DIM = 32
    HIDDEN_DIM = 512
    
    BATCH_SIZE = 64
    EPOCHS = 500
    LR = 1e-3
    
    # 损失权重
    KL_WEIGHT = 0.001  # KL散度权重 (beta-VAE)
    RECON_WEIGHT = 1.0
    
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_cvae'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_cvae'
    
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
        
        print(f"Loaded {len(self.features)} samples (FE: {len(self.fe_features)}, Non-FE: {len(self.nonfe_features)})")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx]), torch.tensor(self.labels[idx])


class Encoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=512, latent_dim=32):
        super().__init__()
        
        self.cond_embed = nn.Embedding(2, 32)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 32, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
    
    def forward(self, x, labels):
        cond = self.cond_embed(labels)
        h = torch.cat([x, cond], dim=1)
        h = self.encoder(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=512, output_dim=64):
        super().__init__()
        
        self.cond_embed = nn.Embedding(2, 32)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 32, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z, labels):
        cond = self.cond_embed(labels)
        h = torch.cat([z, cond], dim=1)
        return self.decoder(h)


class CVAE(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=512, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x, labels):
        mu, logvar = self.encoder(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z, labels)
        return recon, mu, logvar
    
    def sample(self, n_samples, labels, device):
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decoder(z, labels)


class Trainer:
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE
        
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        self.extractor = UnifiedFeatureExtractor()
        
        self.model = CVAE(
            self.config.FEATURE_DIM, 
            self.config.HIDDEN_DIM, 
            self.config.LATENT_DIM
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.LR)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min', patience=30, factor=0.5
        )
        
        self.history = {'epoch': [], 'loss': [], 'recon': [], 'kl': [], 'mean_diff': [], 'corr': []}
        self.dataset = None
    
    def load_data(self):
        data_files = [
            (str(self.config.DATA_DIR / 'dataset_original_ferroelectric.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_known_FE_rest.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_nonFE.jsonl'), 0),
            (str(self.config.DATA_DIR / 'dataset_polar_non_ferroelectric_final.jsonl'), 0),
        ]
        self.dataset = MaterialDataset(data_files, self.extractor)
        return DataLoader(self.dataset, batch_size=self.config.BATCH_SIZE, shuffle=True, drop_last=True)
    
    def loss_function(self, recon, x, mu, logvar):
        # 重建损失 (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        total = self.config.RECON_WEIGHT * recon_loss + self.config.KL_WEIGHT * kl_loss
        
        return total, recon_loss, kl_loss
    
    def train_step(self, x, labels):
        x = x.to(self.device)
        labels = labels.long().to(self.device)
        
        self.optimizer.zero_grad()
        
        recon, mu, logvar = self.model(x, labels)
        loss, recon_loss, kl_loss = self.loss_function(recon, x, mu, logvar)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return {'loss': loss.item(), 'recon': recon_loss.item(), 'kl': kl_loss.item()}
    
    def evaluate(self, n=500):
        self.model.eval()
        with torch.no_grad():
            labels = torch.ones(n, dtype=torch.long, device=self.device)
            fake = self.model.sample(n, labels, self.device).cpu().numpy()
            real = self.dataset.fe_features[:n]
            
            mean_diff = np.abs(fake.mean(0) - real.mean(0)).mean()
            corr = np.corrcoef(fake.mean(0), real.mean(0))[0, 1]
        self.model.train()
        return {'mean_diff': mean_diff, 'corr': corr}
    
    def train(self, epochs=None):
        epochs = epochs or self.config.EPOCHS
        loader = self.load_data()
        
        print(f"\n{'='*60}")
        print("CVAE Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Latent Dim: {self.config.LATENT_DIM}")
        print(f"KL Weight: {self.config.KL_WEIGHT}")
        print(f"{'='*60}\n")
        
        best_corr = 0
        
        for epoch in range(epochs):
            metrics = {'loss': [], 'recon': [], 'kl': []}
            
            for x, labels in loader:
                m = self.train_step(x, labels)
                for k, v in m.items():
                    metrics[k].append(v)
            
            avg_loss = np.mean(metrics['loss'])
            self.scheduler.step(avg_loss)
            
            self.history['epoch'].append(epoch + 1)
            self.history['loss'].append(avg_loss)
            self.history['recon'].append(np.mean(metrics['recon']))
            self.history['kl'].append(np.mean(metrics['kl']))
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                ev = self.evaluate()
                self.history['mean_diff'].append(ev['mean_diff'])
                self.history['corr'].append(ev['corr'])
                
                print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | "
                      f"Recon: {np.mean(metrics['recon']):.4f} | "
                      f"KL: {np.mean(metrics['kl']):.4f} | "
                      f"Mean_diff: {ev['mean_diff']:.4f} | Corr: {ev['corr']:.4f}")
                
                if ev['corr'] > best_corr:
                    best_corr = ev['corr']
                    self.save('best')
        
        print("\n✓ Training complete!")
        self.save('final')
        self.final_evaluate()
    
    def save(self, suffix):
        torch.save({
            'model': self.model.state_dict(),
            'config': {
                'latent_dim': self.config.LATENT_DIM, 
                'hidden_dim': self.config.HIDDEN_DIM,
                'feature_dim': self.config.FEATURE_DIM
            }
        }, self.config.MODEL_DIR / f'cvae_{suffix}.pt')
        print(f"✓ Saved cvae_{suffix}.pt")
    
    def final_evaluate(self):
        from scipy import stats as scipy_stats
        
        self.model.eval()
        with torch.no_grad():
            labels = torch.ones(1000, dtype=torch.long, device=self.device)
            gen_fe = self.model.sample(1000, labels, self.device).cpu().numpy()
        
        real_fe = self.dataset.fe_features
        
        print(f"\n{'='*60}")
        print("最终评估结果")
        print(f"{'='*60}")
        
        ks_passes = 0
        for i in range(20):
            stat, pval = scipy_stats.ks_2samp(real_fe[:, i], gen_fe[:, i])
            if pval > 0.05:
                ks_passes += 1
        
        mean_diff = np.abs(real_fe.mean(0) - gen_fe.mean(0)).mean()
        std_diff = np.abs(real_fe.std(0) - gen_fe.std(0)).mean()
        corr = np.corrcoef(real_fe.mean(0), gen_fe.mean(0))[0, 1]
        
        print(f"KS检验通过率: {ks_passes}/20 ({100*ks_passes/20:.1f}%)")
        print(f"均值差异: {mean_diff:.4f}")
        print(f"标准差差异: {std_diff:.4f}")
        print(f"均值相关系数: {corr:.4f}")


def main():
    print("="*60)
    print("Conditional VAE for Material Generation")
    print("="*60)
    trainer = Trainer()
    trainer.train(epochs=500)


if __name__ == '__main__':
    main()
