"""
增强型对抗生成网络 (GAN) - 铁电材料预测
=============================================
同步使用统一特征工程模块 (64维特征)

功能:
1. 生成可能的铁电材料特征向量
2. 判别铁电/非铁电材料
3. 条件生成指定类型材料
4. 导出生成的材料特征
"""

import sys
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# 添加共享模块路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from feature_engineering import (
    UnifiedFeatureExtractor,
    FEATURE_DIM,
    FEATURE_NAMES,
    extract_features
)


# ==========================================
# 1. 配置
# ==========================================
class Config:
    # 模型维度 (使用统一特征维度)
    FEATURE_DIM = FEATURE_DIM  # 64
    LATENT_DIM = 128           # 隐空间维度
    HIDDEN_DIM = 256           # 隐藏层维度
    
    # 训练参数
    BATCH_SIZE = 64
    EPOCHS = 200
    LR_G = 2e-4
    LR_D = 1e-4
    BETA1 = 0.5
    BETA2 = 0.999
    
    # 正则化
    LABEL_SMOOTH = 0.1
    NOISE_STD = 0.05
    
    # 路径
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_gan_v2'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_gan_v2'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# 2. 数据集
# ==========================================
class MaterialFeatureDataset(Dataset):
    """材料特征数据集 (使用统一特征提取器)"""
    
    def __init__(self, data_files: List[Tuple[str, int]], extractor: UnifiedFeatureExtractor):
        """
        Args:
            data_files: [(文件路径, 标签), ...]
            extractor: 统一特征提取器
        """
        self.features = []
        self.labels = []
        self.extractor = extractor
        
        for file_path, label in data_files:
            if os.path.exists(file_path):
                self._load_file(file_path, label)
        
        if self.features:
            self.features = np.array(self.features, dtype=np.float32)
            self.labels = np.array(self.labels, dtype=np.float32)
        else:
            self.features = np.zeros((0, Config.FEATURE_DIM), dtype=np.float32)
            self.labels = np.zeros(0, dtype=np.float32)
        
        print(f"Loaded {len(self.features)} samples")
    
    def _load_file(self, file_path: str, label: int):
        """加载单个文件"""
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    struct = item.get('structure')
                    if struct:
                        sg = item.get('spacegroup_number', None)
                        feat = self.extractor.extract_from_structure_dict(struct, sg)
                        if np.sum(np.abs(feat)) > 0:
                            self.features.append(feat)
                            self.labels.append(label)
                except:
                    continue
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx])
        )


# ==========================================
# 3. 网络架构
# ==========================================
class Generator(nn.Module):
    """生成器网络 (64维输出)"""
    
    def __init__(self, latent_dim=Config.LATENT_DIM, hidden_dim=Config.HIDDEN_DIM,
                 output_dim=Config.FEATURE_DIM, num_classes=2):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        
        # 条件嵌入
        self.cond_embed = nn.Embedding(num_classes, 32)
        
        # 主网络
        self.net = nn.Sequential(
            # 输入: latent + condition
            nn.Linear(latent_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.cond_embed(labels)
        x = torch.cat([z, cond], dim=1)
        out = self.net(x)
        # 缩放到 [0, 1] 范围 (特征是归一化的)
        return (out + 1) / 2


class Discriminator(nn.Module):
    """判别器网络 (64维输入)"""
    
    def __init__(self, input_dim=Config.FEATURE_DIM, hidden_dim=Config.HIDDEN_DIM,
                 num_classes=2):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 条件嵌入
        self.cond_embed = nn.Embedding(num_classes, 32)
        
        # 特征提取
        self.features = nn.Sequential(
            nn.Linear(input_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
        )
        
        # 真假判别
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1)
        )
        
        # 分类头
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cond = self.cond_embed(labels)
        h = torch.cat([x, cond], dim=1)
        feat = self.features(h)
        validity = self.adv_head(feat)
        cls_pred = self.cls_head(feat)
        return validity, cls_pred


# ==========================================
# 4. 训练器
# ==========================================
class GANTrainer:
    """GAN训练器"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = self.config.DEVICE
        
        # 创建目录
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 特征提取器
        self.extractor = UnifiedFeatureExtractor()
        
        # 模型
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # 优化器
        self.opt_G = optim.Adam(
            self.generator.parameters(),
            lr=self.config.LR_G,
            betas=(self.config.BETA1, self.config.BETA2)
        )
        self.opt_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.LR_D,
            betas=(self.config.BETA1, self.config.BETA2)
        )
        
        # 损失函数
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.CrossEntropyLoss()
        
        # 训练历史
        self.history = {
            'epoch': [],
            'd_loss': [],
            'g_loss': [],
            'd_real_acc': [],
            'd_fake_acc': [],
            'cls_acc': []
        }
    
    def load_data(self) -> DataLoader:
        """加载训练数据"""
        data_files = [
            (str(self.config.DATA_DIR / 'dataset_original_ferroelectric.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_known_FE_rest.jsonl'), 1),
            (str(self.config.DATA_DIR / 'dataset_nonFE.jsonl'), 0),
            (str(self.config.DATA_DIR / 'dataset_polar_non_ferroelectric_final.jsonl'), 0),
        ]
        
        dataset = MaterialFeatureDataset(data_files, self.extractor)
        return DataLoader(
            dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )
    
    def train_step(self, real_data: torch.Tensor, real_labels: torch.Tensor):
        """单步训练"""
        batch_size = real_data.size(0)
        
        real_data = real_data.to(self.device)
        real_labels = real_labels.long().to(self.device)
        
        # 标签
        real_target = torch.full((batch_size, 1), 1 - self.config.LABEL_SMOOTH, device=self.device)
        fake_target = torch.full((batch_size, 1), self.config.LABEL_SMOOTH, device=self.device)
        
        # ========== 训练判别器 ==========
        self.opt_D.zero_grad()
        
        # 真实数据
        real_noisy = real_data + torch.randn_like(real_data) * self.config.NOISE_STD
        real_validity, real_cls = self.discriminator(real_noisy, real_labels)
        d_real_loss = self.adv_loss(real_validity, real_target)
        d_cls_loss = self.cls_loss(real_cls, real_labels)
        
        # 生成假数据
        z = torch.randn(batch_size, self.config.LATENT_DIM, device=self.device)
        fake_labels = torch.randint(0, 2, (batch_size,), device=self.device)
        fake_data = self.generator(z, fake_labels)
        
        fake_validity, _ = self.discriminator(fake_data.detach(), fake_labels)
        d_fake_loss = self.adv_loss(fake_validity, fake_target)
        
        d_loss = d_real_loss + d_fake_loss + 0.5 * d_cls_loss
        d_loss.backward()
        self.opt_D.step()
        
        # ========== 训练生成器 ==========
        self.opt_G.zero_grad()
        
        z = torch.randn(batch_size, self.config.LATENT_DIM, device=self.device)
        fake_labels = torch.randint(0, 2, (batch_size,), device=self.device)
        fake_data = self.generator(z, fake_labels)
        
        fake_validity, fake_cls = self.discriminator(fake_data, fake_labels)
        g_adv_loss = self.adv_loss(fake_validity, real_target)
        g_cls_loss = self.cls_loss(fake_cls, fake_labels)
        
        # 特征匹配损失
        with torch.no_grad():
            real_feat = self.discriminator.features(torch.cat([real_data, self.discriminator.cond_embed(real_labels)], dim=1))
        fake_feat = self.discriminator.features(torch.cat([fake_data, self.discriminator.cond_embed(fake_labels)], dim=1))
        fm_loss = torch.mean((real_feat.mean(0) - fake_feat.mean(0)) ** 2)
        
        g_loss = g_adv_loss + 0.5 * g_cls_loss + 0.1 * fm_loss
        g_loss.backward()
        self.opt_G.step()
        
        # 计算准确率
        d_real_acc = (torch.sigmoid(real_validity) > 0.5).float().mean().item()
        d_fake_acc = (torch.sigmoid(fake_validity) < 0.5).float().mean().item()
        cls_acc = (real_cls.argmax(1) == real_labels).float().mean().item()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'd_real_acc': d_real_acc,
            'd_fake_acc': d_fake_acc,
            'cls_acc': cls_acc
        }
    
    def train(self, epochs: int = None):
        """完整训练"""
        epochs = epochs or self.config.EPOCHS
        dataloader = self.load_data()
        
        print(f"\n{'='*60}")
        print(f"GAN Training (64-dim features)")
        print(f"Device: {self.device}")
        print(f"Samples: {len(dataloader.dataset)}")
        print(f"Epochs: {epochs}")
        print(f"{'='*60}\n")
        
        best_d_acc = 0
        
        for epoch in range(epochs):
            epoch_metrics = {'d_loss': [], 'g_loss': [], 'd_real_acc': [], 'd_fake_acc': [], 'cls_acc': []}
            
            for batch_data, batch_labels in dataloader:
                metrics = self.train_step(batch_data, batch_labels)
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)
            
            # 记录历史
            self.history['epoch'].append(epoch + 1)
            for k in epoch_metrics:
                self.history[k].append(np.mean(epoch_metrics[k]))
            
            # 输出
            if (epoch + 1) % 10 == 0 or epoch == 0:
                d_acc = (self.history['d_real_acc'][-1] + self.history['d_fake_acc'][-1]) / 2
                print(f"Epoch {epoch+1:3d} | "
                      f"D_loss: {self.history['d_loss'][-1]:.4f} | "
                      f"G_loss: {self.history['g_loss'][-1]:.4f} | "
                      f"D_acc: {d_acc:.1%} | "
                      f"Cls_acc: {self.history['cls_acc'][-1]:.1%}")
                
                if d_acc > best_d_acc:
                    best_d_acc = d_acc
                    self.save_model('best')
        
        print("\n✓ Training complete!")
        self.save_model('final')
        self.generate_report()
    
    def save_model(self, suffix: str = 'final'):
        """保存模型"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'config': {
                'feature_dim': self.config.FEATURE_DIM,
                'latent_dim': self.config.LATENT_DIM,
                'hidden_dim': self.config.HIDDEN_DIM
            }
        }, self.config.MODEL_DIR / f'gan_v2_{suffix}.pt')
        
        # 单独保存生成器
        torch.save(self.generator.state_dict(), self.config.MODEL_DIR / f'generator_v2_{suffix}.pt')
        
        print(f"✓ Model saved: gan_v2_{suffix}.pt")
    
    def generate_report(self):
        """生成训练报告"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 绘图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.history['epoch'], self.history['d_loss'], label='D Loss', alpha=0.8)
        axes[0, 0].plot(self.history['epoch'], self.history['g_loss'], label='G Loss', alpha=0.8)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 判别准确率
        d_acc = [(r + f) / 2 for r, f in zip(self.history['d_real_acc'], self.history['d_fake_acc'])]
        axes[0, 1].plot(self.history['epoch'], d_acc, label='D Accuracy', color='blue')
        axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].set_title('Discriminator Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 分类准确率
        axes[1, 0].plot(self.history['epoch'], self.history['cls_acc'], color='green')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Classification Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 生成样本分析
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(100, self.config.LATENT_DIM, device=self.device)
            labels = torch.ones(100, dtype=torch.long, device=self.device)  # 铁电
            samples = self.generator(z, labels).cpu().numpy()
        
        # 显示前10个特征的分布
        sample_means = samples.mean(axis=0)[:20]
        axes[1, 1].bar(range(20), sample_means, alpha=0.7)
        axes[1, 1].set_xlabel('Feature Index')
        axes[1, 1].set_ylabel('Mean Value')
        axes[1, 1].set_title('Generated Feature Distribution (First 20)')
        axes[1, 1].set_xticks(range(0, 20, 2))
        
        plt.tight_layout()
        plt.savefig(self.config.REPORT_DIR / f'training_report_{timestamp}.png', dpi=150)
        plt.close()
        
        # 文本报告
        report_path = self.config.REPORT_DIR / f'training_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("GAN Training Report (64-dim features)\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Feature Dimension: {self.config.FEATURE_DIM}\n")
            f.write(f"Latent Dimension: {self.config.LATENT_DIM}\n")
            f.write(f"Total Epochs: {len(self.history['epoch'])}\n\n")
            
            f.write("Final Metrics:\n")
            f.write(f"  D Loss: {self.history['d_loss'][-1]:.4f}\n")
            f.write(f"  G Loss: {self.history['g_loss'][-1]:.4f}\n")
            f.write(f"  D Accuracy: {d_acc[-1]:.1%}\n")
            f.write(f"  Classification Accuracy: {self.history['cls_acc'][-1]:.1%}\n")
        
        print(f"✓ Report saved: {report_path}")
    
    def generate_samples(self, n_samples: int = 100, label: int = 1) -> np.ndarray:
        """生成材料特征样本"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.config.LATENT_DIM, device=self.device)
            labels = torch.full((n_samples,), label, dtype=torch.long, device=self.device)
            samples = self.generator(z, labels)
        return samples.cpu().numpy()


# ==========================================
# 5. 主函数
# ==========================================
def main():
    print("="*60)
    print("Ferroelectric Material GAN (64-dim unified features)")
    print("="*60)
    
    trainer = GANTrainer()
    trainer.train(epochs=150)
    
    # 生成样本测试
    print("\n" + "="*60)
    print("Generating test samples...")
    samples = trainer.generate_samples(n_samples=10, label=1)
    print(f"Generated {len(samples)} ferroelectric material features")
    print(f"Sample shape: {samples.shape}")
    print(f"Sample mean: {samples.mean():.4f}")
    print(f"Sample std: {samples.std():.4f}")
    
    # 保存样本
    np.save(trainer.config.MODEL_DIR / 'generated_samples_test.npy', samples)
    print(f"✓ Test samples saved")


if __name__ == '__main__':
    main()
