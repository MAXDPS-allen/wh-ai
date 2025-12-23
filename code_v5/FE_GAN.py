"""
对抗生成网络 (GAN) - 铁电材料生成与预测
============================================
功能:
1. 生成器: 从随机噪声生成材料特征向量
2. 判别器: 区分真实铁电材料和生成的材料
3. 条件GAN: 支持按空间群/晶系等条件生成
4. 完整的模型保存和报表模块

架构:
- Generator: 噪声 -> 材料特征向量 (32维)
- Discriminator: 特征向量 -> 铁电/非铁电分类
- Conditional Generator: 条件 + 噪声 -> 特征向量
"""

import json
import os
import time
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# 设置
sns.set(style="whitegrid")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

# 创建输出目录
os.makedirs('model_gan', exist_ok=True)
os.makedirs('reports_gan', exist_ok=True)
os.makedirs('generated_materials', exist_ok=True)

# ==========================================
# 1. 配置和数据处理
# ==========================================
# 特征维度配置
FEATURE_DIM = 32  # 特征向量维度
LATENT_DIM = 64   # 潜在空间维度
CONDITION_DIM = 10  # 条件向量维度 (空间群、晶系等)

# 元素数据
ELEMENT_DATA = {
    'H': [1, 1.008, 0.37, 2.20], 'Li': [3, 6.94, 1.34, 0.98], 
    'Be': [4, 9.012, 0.90, 1.57], 'B': [5, 10.81, 0.82, 2.04],
    'C': [6, 12.011, 0.77, 2.55], 'N': [7, 14.007, 0.75, 3.04],
    'O': [8, 15.999, 0.73, 3.44], 'F': [9, 18.998, 0.71, 3.98],
    'Na': [11, 22.990, 1.54, 0.93], 'Mg': [12, 24.305, 1.30, 1.31],
    'Al': [13, 26.982, 1.18, 1.61], 'Si': [14, 28.085, 1.11, 1.90],
    'K': [19, 39.098, 1.96, 0.82], 'Ca': [20, 40.078, 1.74, 1.00],
    'Ti': [22, 47.867, 1.36, 1.54], 'V': [23, 50.942, 1.25, 1.63],
    'Mn': [25, 54.938, 1.39, 1.55], 'Fe': [26, 55.845, 1.25, 1.83],
    'Co': [27, 58.933, 1.26, 1.88], 'Ni': [28, 58.693, 1.21, 1.91],
    'Cu': [29, 63.546, 1.38, 1.90], 'Zn': [30, 65.38, 1.31, 1.65],
    'Sr': [38, 87.62, 1.92, 0.95], 'Y': [39, 88.906, 1.62, 1.22],
    'Zr': [40, 91.224, 1.48, 1.33], 'Nb': [41, 92.906, 1.37, 1.60],
    'Ba': [56, 137.33, 1.98, 0.89], 'La': [57, 138.91, 1.69, 1.10],
    'Ce': [58, 140.12, 1.65, 1.12], 'Nd': [60, 144.24, 1.64, 1.14],
    'Pb': [82, 207.2, 1.47, 2.33], 'Bi': [83, 208.98, 1.46, 2.02],
}

TM_SET = {'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
          'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W'}


class MaterialDataset(Dataset):
    """材料特征数据集"""
    
    def __init__(self, features, labels, conditions=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.conditions = torch.FloatTensor(conditions) if conditions is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.conditions is not None:
            return self.features[idx], self.labels[idx], self.conditions[idx]
        return self.features[idx], self.labels[idx]


def extract_features_and_conditions(item):
    """
    从材料数据提取特征向量和条件向量
    
    特征向量 (32维): 晶格、元素统计、配位等
    条件向量 (10维): 空间群、晶系、极性、元素数量等
    """
    if 'structure' not in item:
        return None, None, None
    
    features = np.zeros(FEATURE_DIM, dtype=np.float32)
    conditions = np.zeros(CONDITION_DIM, dtype=np.float32)
    
    try:
        struct = item['structure']
        sites = struct.get('sites', [])
        lattice = struct.get('lattice', {})
        
        if not sites or not lattice:
            return None, None, None
        
        # 晶格参数
        a = lattice.get('a', 0)
        b = lattice.get('b', 0)
        c = lattice.get('c', 0)
        alpha = lattice.get('alpha', 90)
        beta = lattice.get('beta', 90)
        gamma = lattice.get('gamma', 90)
        vol = lattice.get('volume', 0)
        
        if vol <= 0:
            return None, None, None
        
        # 成分统计
        comp = {}
        total_atoms = 0
        for site in sites:
            for species in site['species']:
                el = species['element']
                occu = species['occu']
                comp[el] = comp.get(el, 0) + occu
                total_atoms += occu
        
        if total_atoms == 0:
            return None, None, None
        
        # 元素特征
        masses, radii, ens = [], [], []
        tm_count, o_count = 0, 0
        
        for el, count in comp.items():
            if el in ELEMENT_DATA:
                data = ELEMENT_DATA[el]
                masses.append(data[1])
                radii.append(data[2])
                ens.append(data[3])
                if el in TM_SET:
                    tm_count += count
                if el == 'O':
                    o_count += count
        
        if not masses:
            return None, None, None
        
        # 密度
        total_mass = sum(masses)
        density = total_mass / vol * 10
        
        # 特征向量填充
        # [0-2] 晶格
        features[0] = min(vol, 5000) / 5000.0
        features[1] = min(density, 10) / 10.0
        features[2] = min(vol / total_atoms, 50) / 50.0
        
        # [3-8] 形状
        features[3] = a / max(b, 0.1)
        features[4] = a / max(c, 0.1)
        features[5] = b / max(c, 0.1)
        features[6] = (alpha - 90) / 90.0
        features[7] = (beta - 90) / 90.0
        features[8] = (gamma - 90) / 90.0
        
        # [9-11] 对称性
        sg = item.get('spacegroup_number', 1)
        features[9] = (sg if sg else 1) / 230.0
        features[10] = 0.5  # 晶系
        features[11] = 1.0 if sg and sg < 100 else 0  # 极性估计
        
        # [12-17] 元素统计
        features[12] = np.mean(masses) / 200.0
        features[13] = np.mean(radii) / 2.0
        features[14] = np.mean(ens) / 4.0
        features[15] = 0.5
        features[16] = 0.5
        features[17] = len(comp) / 10.0
        
        # [18-21] 元素范围
        features[18] = (max(radii) - min(radii)) / 2.0 if len(radii) > 1 else 0
        features[19] = (max(ens) - min(ens)) / 4.0 if len(ens) > 1 else 0
        features[20] = 0.5
        features[21] = max(radii) / max(min(radii), 0.1)
        
        # [22-25] 配位
        features[22] = 6.0 / 12.0
        features[23] = 0.2
        features[24] = 2.0 / 4.0
        features[25] = 0.1
        
        # [26-28] 化学
        features[26] = tm_count / total_atoms
        features[27] = 0.5
        features[28] = o_count / total_atoms
        
        # [29-31] 其他
        features[29] = 0.5
        features[30] = 0.5
        features[31] = 0.5
        
        # 条件向量填充
        # [0] 空间群 (归一化)
        conditions[0] = (sg if sg else 1) / 230.0
        
        # [1] 晶系 (one-hot编码简化)
        # 根据空间群号估计晶系
        if sg:
            if sg <= 2: conditions[1] = 0  # triclinic
            elif sg <= 15: conditions[1] = 1/6  # monoclinic
            elif sg <= 74: conditions[1] = 2/6  # orthorhombic
            elif sg <= 142: conditions[1] = 3/6  # tetragonal
            elif sg <= 167: conditions[1] = 4/6  # trigonal
            elif sg <= 194: conditions[1] = 5/6  # hexagonal
            else: conditions[1] = 1  # cubic
        
        # [2] 是否极性
        conditions[2] = features[11]
        
        # [3] 元素数量
        conditions[3] = len(comp) / 10.0
        
        # [4] 过渡金属比例
        conditions[4] = tm_count / total_atoms
        
        # [5] 氧含量
        conditions[5] = o_count / total_atoms
        
        # [6-9] 主要晶格参数
        conditions[6] = min(a, 20) / 20.0
        conditions[7] = min(b, 20) / 20.0
        conditions[8] = min(c, 20) / 20.0
        conditions[9] = min(vol, 5000) / 5000.0
        
        return features, conditions, 1  # 返回特征、条件和标签
        
    except Exception as e:
        return None, None, None


def load_gan_data(pos_files, neg_files):
    """加载GAN训练数据"""
    X_pos, C_pos = [], []
    X_neg, C_neg = [], []
    
    print("Loading data for GAN training...")
    
    # 加载正样本 (铁电材料)
    for f_name in pos_files:
        print(f"  Loading positive: {f_name}")
        try:
            with open(f_name, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        feat, cond, _ = extract_features_and_conditions(item)
                        if feat is not None:
                            X_pos.append(feat)
                            C_pos.append(cond)
                    except:
                        continue
        except FileNotFoundError:
            print(f"    [Warning] File not found: {f_name}")
    
    # 加载负样本 (非铁电材料)
    for f_name in neg_files:
        print(f"  Loading negative: {f_name}")
        try:
            with open(f_name, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        feat, cond, _ = extract_features_and_conditions(item)
                        if feat is not None:
                            X_neg.append(feat)
                            C_neg.append(cond)
                    except:
                        continue
        except FileNotFoundError:
            print(f"    [Warning] File not found: {f_name}")
    
    X_pos = np.array(X_pos)
    C_pos = np.array(C_pos)
    X_neg = np.array(X_neg)
    C_neg = np.array(C_neg)
    
    print(f"Loaded: {len(X_pos)} positive, {len(X_neg)} negative samples")
    
    return X_pos, C_pos, X_neg, C_neg


# ==========================================
# 2. GAN 模型架构
# ==========================================
class Generator(nn.Module):
    """
    生成器网络
    
    输入: 潜在向量 (LATENT_DIM)
    输出: 材料特征向量 (FEATURE_DIM)
    """
    
    def __init__(self, latent_dim=LATENT_DIM, feature_dim=FEATURE_DIM):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, feature_dim),
            nn.Sigmoid()  # 输出归一化到 [0, 1]
        )
    
    def forward(self, z):
        return self.model(z)
    
    def generate(self, num_samples=1):
        """生成材料特征向量"""
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            features = self.forward(z)
        return features.cpu().numpy()


class Discriminator(nn.Module):
    """
    判别器网络
    
    输入: 材料特征向量 (FEATURE_DIM)
    输出: 真实/生成 概率 + 铁电/非铁电 分类
    """
    
    def __init__(self, feature_dim=FEATURE_DIM):
        super(Discriminator, self).__init__()
        
        self.feature_dim = feature_dim
        
        # 共享特征提取层
        self.features = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
        )
        
        # 真假判别头
        self.validity_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 铁电分类头
        self.classification_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.features(x)
        validity = self.validity_head(features)
        classification = self.classification_head(features)
        return validity, classification
    
    def predict_ferroelectric(self, x):
        """预测是否为铁电材料"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            if len(x.shape) == 1:
                x = x.unsqueeze(0)
            x = x.to(device)
            _, classification = self.forward(x)
        return classification.cpu().numpy()


class ConditionalGenerator(nn.Module):
    """
    条件生成器
    
    输入: 潜在向量 + 条件向量
    输出: 满足条件的材料特征向量
    """
    
    def __init__(self, latent_dim=LATENT_DIM, condition_dim=CONDITION_DIM, feature_dim=FEATURE_DIM):
        super(ConditionalGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.feature_dim = feature_dim
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64)
        )
        
        # 生成器主网络
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 64, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, z, condition):
        cond_encoded = self.condition_encoder(condition)
        combined = torch.cat([z, cond_encoded], dim=1)
        return self.model(combined)
    
    def generate(self, condition, num_samples=1):
        """给定条件生成材料特征"""
        self.eval()
        with torch.no_grad():
            if isinstance(condition, np.ndarray):
                condition = torch.FloatTensor(condition)
            if len(condition.shape) == 1:
                condition = condition.unsqueeze(0).repeat(num_samples, 1)
            condition = condition.to(device)
            
            z = torch.randn(num_samples, self.latent_dim).to(device)
            features = self.forward(z, condition)
        return features.cpu().numpy()


class ConditionalDiscriminator(nn.Module):
    """
    条件判别器
    
    输入: 材料特征向量 + 条件向量
    输出: 真实/生成 概率
    """
    
    def __init__(self, feature_dim=FEATURE_DIM, condition_dim=CONDITION_DIM):
        super(ConditionalDiscriminator, self).__init__()
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64)
        )
        
        # 主网络
        self.model = nn.Sequential(
            nn.Linear(feature_dim + 64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, condition):
        cond_encoded = self.condition_encoder(condition)
        combined = torch.cat([x, cond_encoded], dim=1)
        return self.model(combined)


# ==========================================
# 3. GAN 训练器
# ==========================================
class GANTrainer:
    """GAN 训练管理器"""
    
    def __init__(self, generator, discriminator, lr_g=0.0002, lr_d=0.0002):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        self.adversarial_loss = nn.BCELoss()
        self.classification_loss = nn.BCELoss()
        
        self.history = {
            'epoch': [],
            'g_loss': [],
            'd_loss': [],
            'd_accuracy': [],
            'classification_accuracy': []
        }
    
    def train_step(self, real_features, real_labels):
        """单步训练"""
        batch_size = real_features.size(0)
        
        # 标签
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        
        # -----------------
        # 训练判别器
        # -----------------
        self.optimizer_D.zero_grad()
        
        # 真实样本
        validity_real, class_real = self.discriminator(real_features)
        d_real_loss = self.adversarial_loss(validity_real, valid)
        d_class_loss = self.classification_loss(class_real, real_labels.unsqueeze(1))
        
        # 生成样本
        z = torch.randn(batch_size, self.generator.latent_dim).to(device)
        gen_features = self.generator(z)
        validity_fake, _ = self.discriminator(gen_features.detach())
        d_fake_loss = self.adversarial_loss(validity_fake, fake)
        
        # 判别器总损失
        d_loss = (d_real_loss + d_fake_loss) / 2 + d_class_loss * 0.5
        d_loss.backward()
        self.optimizer_D.step()
        
        # -----------------
        # 训练生成器
        # -----------------
        self.optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, self.generator.latent_dim).to(device)
        gen_features = self.generator(z)
        validity, gen_class = self.discriminator(gen_features)
        
        # 生成器损失: 欺骗判别器 + 生成铁电材料
        g_adv_loss = self.adversarial_loss(validity, valid)
        g_class_loss = self.classification_loss(gen_class, torch.ones(batch_size, 1).to(device))
        
        g_loss = g_adv_loss + g_class_loss * 0.3
        g_loss.backward()
        self.optimizer_G.step()
        
        # 计算准确率
        d_accuracy = ((validity_real > 0.5).float().mean() + (validity_fake < 0.5).float().mean()) / 2
        class_accuracy = ((class_real > 0.5).float() == real_labels.unsqueeze(1)).float().mean()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_accuracy': d_accuracy.item(),
            'class_accuracy': class_accuracy.item()
        }
    
    def train(self, dataloader, num_epochs=100, verbose_interval=10):
        """训练GAN"""
        print(f"\n开始GAN训练 (设备: {device})")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            epoch_metrics = {'g_loss': 0, 'd_loss': 0, 'd_accuracy': 0, 'class_accuracy': 0}
            num_batches = 0
            
            for batch in dataloader:
                features = batch[0].to(device)
                labels = batch[1].to(device)
                
                metrics = self.train_step(features, labels)
                
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                num_batches += 1
            
            # 平均
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
            
            # 记录历史
            self.history['epoch'].append(epoch + 1)
            for key in ['g_loss', 'd_loss', 'd_accuracy']:
                self.history[key].append(epoch_metrics[key])
            self.history['classification_accuracy'].append(epoch_metrics['class_accuracy'])
            
            if (epoch + 1) % verbose_interval == 0:
                print(f"Epoch {epoch+1:03d} | G_Loss: {epoch_metrics['g_loss']:.4f} | "
                      f"D_Loss: {epoch_metrics['d_loss']:.4f} | D_Acc: {epoch_metrics['d_accuracy']:.4f} | "
                      f"Class_Acc: {epoch_metrics['class_accuracy']:.4f}")
        
        return self.history


class ConditionalGANTrainer:
    """条件GAN训练管理器"""
    
    def __init__(self, generator, discriminator, lr_g=0.0002, lr_d=0.0002):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        self.adversarial_loss = nn.BCELoss()
        
        self.history = {
            'epoch': [],
            'g_loss': [],
            'd_loss': [],
            'd_accuracy': []
        }
    
    def train_step(self, real_features, conditions):
        """单步训练"""
        batch_size = real_features.size(0)
        
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)
        
        # 训练判别器
        self.optimizer_D.zero_grad()
        
        validity_real = self.discriminator(real_features, conditions)
        d_real_loss = self.adversarial_loss(validity_real, valid)
        
        z = torch.randn(batch_size, self.generator.latent_dim).to(device)
        gen_features = self.generator(z, conditions)
        validity_fake = self.discriminator(gen_features.detach(), conditions)
        d_fake_loss = self.adversarial_loss(validity_fake, fake)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()
        
        # 训练生成器
        self.optimizer_G.zero_grad()
        
        z = torch.randn(batch_size, self.generator.latent_dim).to(device)
        gen_features = self.generator(z, conditions)
        validity = self.discriminator(gen_features, conditions)
        
        g_loss = self.adversarial_loss(validity, valid)
        g_loss.backward()
        self.optimizer_G.step()
        
        d_accuracy = ((validity_real > 0.5).float().mean() + (validity_fake < 0.5).float().mean()) / 2
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'd_accuracy': d_accuracy.item()
        }
    
    def train(self, dataloader, num_epochs=100, verbose_interval=10):
        """训练条件GAN"""
        print(f"\n开始条件GAN训练 (设备: {device})")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            epoch_metrics = {'g_loss': 0, 'd_loss': 0, 'd_accuracy': 0}
            num_batches = 0
            
            for batch in dataloader:
                features = batch[0].to(device)
                conditions = batch[2].to(device)
                
                metrics = self.train_step(features, conditions)
                
                for key in epoch_metrics:
                    epoch_metrics[key] += metrics[key]
                num_batches += 1
            
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
            
            self.history['epoch'].append(epoch + 1)
            for key in ['g_loss', 'd_loss', 'd_accuracy']:
                self.history[key].append(epoch_metrics[key])
            
            if (epoch + 1) % verbose_interval == 0:
                print(f"Epoch {epoch+1:03d} | G_Loss: {epoch_metrics['g_loss']:.4f} | "
                      f"D_Loss: {epoch_metrics['d_loss']:.4f} | D_Acc: {epoch_metrics['d_accuracy']:.4f}")
        
        return self.history


# ==========================================
# 4. 模型保存模块
# ==========================================
class GANModelSaver:
    """GAN模型保存管理器"""
    
    def __init__(self, save_dir='model_gan'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_gan(self, generator, discriminator, trainer, filename=None):
        """保存GAN模型"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'gan_model_{timestamp}.pt'
        
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': trainer.optimizer_G.state_dict(),
            'optimizer_D_state_dict': trainer.optimizer_D.state_dict(),
            'history': trainer.history,
            'generator_config': {
                'latent_dim': generator.latent_dim,
                'feature_dim': generator.feature_dim
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"GAN model saved: {filepath}")
        return filepath
    
    def save_generator_only(self, generator, filename='generator.pt'):
        """只保存生成器"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save(generator.state_dict(), filepath)
        print(f"Generator saved: {filepath}")
        return filepath
    
    def save_discriminator_only(self, discriminator, filename='discriminator.pt'):
        """只保存判别器"""
        filepath = os.path.join(self.save_dir, filename)
        torch.save(discriminator.state_dict(), filepath)
        print(f"Discriminator saved: {filepath}")
        return filepath
    
    def load_gan(self, generator, discriminator, filepath):
        """加载GAN模型"""
        checkpoint = torch.load(filepath, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        return checkpoint['history']
    
    def save_conditional_gan(self, generator, discriminator, trainer, filename='conditional_gan.pt'):
        """保存条件GAN"""
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'history': trainer.history
        }
        
        torch.save(checkpoint, filepath)
        print(f"Conditional GAN saved: {filepath}")
        return filepath


# ==========================================
# 5. 报表生成模块
# ==========================================
class GANReporter:
    """GAN训练报表生成器"""
    
    def __init__(self, report_dir='reports_gan'):
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
    
    def generate_training_report(self, history, save=True):
        """生成训练报告"""
        report = []
        report.append("=" * 70)
        report.append("  GAN 训练报告")
        report.append(f"  生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        report.append(f"\n总训练轮数: {len(history['epoch'])}")
        
        # 最终指标
        report.append("\n【最终训练指标】")
        report.append(f"  生成器损失: {history['g_loss'][-1]:.4f}")
        report.append(f"  判别器损失: {history['d_loss'][-1]:.4f}")
        report.append(f"  判别器准确率: {history['d_accuracy'][-1]:.4f}")
        
        if 'classification_accuracy' in history:
            report.append(f"  分类准确率: {history['classification_accuracy'][-1]:.4f}")
        
        # 训练过程统计
        report.append("\n【训练过程统计】")
        report.append(f"  G_Loss: min={min(history['g_loss']):.4f}, max={max(history['g_loss']):.4f}")
        report.append(f"  D_Loss: min={min(history['d_loss']):.4f}, max={max(history['d_loss']):.4f}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = os.path.join(self.report_dir, f'gan_training_report_{timestamp}.txt')
            with open(report_path, 'w') as f:
                f.write(report_text)
            print(f"\n报告已保存: {report_path}")
        
        return report_text
    
    def plot_training_curves(self, history, save=True):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 损失曲线
        axes[0].plot(history['epoch'], history['g_loss'], label='Generator')
        axes[0].plot(history['epoch'], history['d_loss'], label='Discriminator')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('GAN Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # 判别器准确率
        axes[1].plot(history['epoch'], history['d_accuracy'], label='D Accuracy')
        if 'classification_accuracy' in history:
            axes[1].plot(history['epoch'], history['classification_accuracy'], label='Classification')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Discriminator Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # 损失比率
        g_d_ratio = np.array(history['g_loss']) / (np.array(history['d_loss']) + 1e-6)
        axes[2].plot(history['epoch'], g_d_ratio)
        axes[2].axhline(y=1, color='r', linestyle='--', label='Balance')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('G_Loss / D_Loss')
        axes[2].set_title('Loss Ratio')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.report_dir, f'gan_training_curves_{timestamp}.png')
            plt.savefig(plot_path, dpi=150)
            print(f"训练曲线已保存: {plot_path}")
        
        plt.close()
    
    def plot_generated_distribution(self, real_features, generated_features, save=True):
        """绘制生成样本分布对比"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        feature_names = ['Volume', 'Density', 'a/b', 'α_dev', 
                         'AvgMass', 'AvgRadius', 'TMFrac', 'OFrac']
        feature_indices = [0, 1, 3, 6, 12, 13, 26, 28]
        
        for ax, name, idx in zip(axes, feature_names, feature_indices):
            ax.hist(real_features[:, idx], bins=30, alpha=0.5, label='Real', density=True)
            ax.hist(generated_features[:, idx], bins=30, alpha=0.5, label='Generated', density=True)
            ax.set_xlabel(name)
            ax.set_ylabel('Density')
            ax.legend()
        
        plt.suptitle('Real vs Generated Feature Distributions')
        plt.tight_layout()
        
        if save:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.report_dir, f'distribution_comparison_{timestamp}.png')
            plt.savefig(plot_path, dpi=150)
            print(f"分布对比图已保存: {plot_path}")
        
        plt.close()
    
    def evaluate_generator_quality(self, generator, real_features, num_samples=1000, save=True):
        """评估生成器质量"""
        report = []
        report.append("=" * 70)
        report.append("  生成器质量评估报告")
        report.append("=" * 70)
        
        # 生成样本
        generated = generator.generate(num_samples)
        
        report.append(f"\n生成样本数: {num_samples}")
        report.append(f"真实样本数: {len(real_features)}")
        
        # 统计对比
        report.append("\n【特征统计对比】")
        feature_names = ['Volume', 'Density', 'VolPerAtom', 'a/b', 'a/c', 'b/c',
                         'αDev', 'βDev', 'γDev', 'SpaceGroup']
        
        report.append(f"\n{'Feature':<12} {'Real_Mean':<12} {'Gen_Mean':<12} {'Real_Std':<12} {'Gen_Std':<12}")
        report.append("-" * 60)
        
        for i, name in enumerate(feature_names[:min(10, len(feature_names))]):
            real_mean = real_features[:, i].mean()
            gen_mean = generated[:, i].mean()
            real_std = real_features[:, i].std()
            gen_std = generated[:, i].std()
            report.append(f"{name:<12} {real_mean:>10.4f}   {gen_mean:>10.4f}   {real_std:>10.4f}   {gen_std:>10.4f}")
        
        # 范围检查
        report.append("\n【范围检查】")
        valid_count = 0
        for i in range(num_samples):
            # 检查生成的特征是否在合理范围内 [0, 1]
            if (generated[i] >= 0).all() and (generated[i] <= 1).all():
                valid_count += 1
        
        report.append(f"有效样本比例: {valid_count / num_samples:.1%}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = os.path.join(self.report_dir, f'generator_quality_{timestamp}.txt')
            with open(report_path, 'w') as f:
                f.write(report_text)
        
        # 绘制分布对比图
        self.plot_generated_distribution(real_features, generated)
        
        return generated
    
    def save_generated_materials(self, generated_features, filename=None):
        """保存生成的材料特征"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'generated_materials_{timestamp}.csv'
        
        filepath = os.path.join('generated_materials', filename)
        
        # 创建特征名
        feature_names = [f'feature_{i}' for i in range(generated_features.shape[1])]
        
        df = pd.DataFrame(generated_features, columns=feature_names)
        df.to_csv(filepath, index=False)
        
        print(f"生成材料已保存: {filepath}")
        return filepath


# ==========================================
# 6. 主训练流程
# ==========================================
def train_ferroelectric_gan():
    """训练铁电材料GAN"""
    
    # 数据文件
    pos_files = ['new_data/dataset_original_ferroelectric.jsonl', 'new_data/dataset_known_FE_rest.jsonl']
    neg_files = ['new_data/dataset_nonFE.jsonl', 'new_data/dataset_polar_non_ferroelectric_final.jsonl']
    
    # 加载数据
    X_pos, C_pos, X_neg, C_neg = load_gan_data(pos_files, neg_files)
    
    if len(X_pos) == 0:
        print("Error: No positive data loaded!")
        return None
    
    # 合并数据
    X_all = np.vstack([X_pos, X_neg])
    labels = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    C_all = np.vstack([C_pos, C_neg])
    
    # 创建数据集
    dataset = MaterialDataset(X_all, labels, C_all)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"\n数据集大小: {len(dataset)}")
    print(f"正样本: {len(X_pos)}, 负样本: {len(X_neg)}")
    
    # 初始化模型
    generator = Generator(latent_dim=LATENT_DIM, feature_dim=FEATURE_DIM)
    discriminator = Discriminator(feature_dim=FEATURE_DIM)
    
    # 训练
    trainer = GANTrainer(generator, discriminator, lr_g=0.0002, lr_d=0.0002)
    history = trainer.train(dataloader, num_epochs=150, verbose_interval=15)
    
    # 初始化工具
    model_saver = GANModelSaver()
    reporter = GANReporter()
    
    # 生成报告
    reporter.generate_training_report(history)
    reporter.plot_training_curves(history)
    
    # 评估生成器
    generated = reporter.evaluate_generator_quality(generator, X_pos)
    reporter.save_generated_materials(generated)
    
    # 保存模型
    model_saver.save_gan(generator, discriminator, trainer, 'ferroelectric_gan.pt')
    model_saver.save_generator_only(generator)
    model_saver.save_discriminator_only(discriminator)
    
    return generator, discriminator, history


def train_conditional_gan():
    """训练条件GAN"""
    
    # 数据文件 (只用铁电材料)
    pos_files = ['new_data/dataset_original_ferroelectric.jsonl', 'new_data/dataset_known_FE_rest.jsonl']
    neg_files = []
    
    X_pos, C_pos, _, _ = load_gan_data(pos_files, neg_files)
    
    if len(X_pos) == 0:
        print("Error: No data loaded!")
        return None
    
    # 创建数据集
    labels = np.ones(len(X_pos))
    dataset = MaterialDataset(X_pos, labels, C_pos)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"\n条件GAN数据集: {len(dataset)} 样本")
    
    # 初始化模型
    cond_generator = ConditionalGenerator(
        latent_dim=LATENT_DIM, 
        condition_dim=CONDITION_DIM, 
        feature_dim=FEATURE_DIM
    )
    cond_discriminator = ConditionalDiscriminator(
        feature_dim=FEATURE_DIM, 
        condition_dim=CONDITION_DIM
    )
    
    # 训练
    cond_trainer = ConditionalGANTrainer(cond_generator, cond_discriminator)
    history = cond_trainer.train(dataloader, num_epochs=150, verbose_interval=15)
    
    # 保存
    model_saver = GANModelSaver()
    model_saver.save_conditional_gan(cond_generator, cond_discriminator, cond_trainer)
    
    # 报告
    reporter = GANReporter()
    reporter.generate_training_report(history)
    reporter.plot_training_curves(history)
    
    # 测试条件生成
    print("\n【条件生成测试】")
    
    # 创建一个条件: 空间群=99 (P4mm), 极性=1
    test_condition = np.zeros(CONDITION_DIM, dtype=np.float32)
    test_condition[0] = 99 / 230.0  # 空间群
    test_condition[2] = 1.0  # 极性
    test_condition[4] = 0.3  # TM比例
    test_condition[5] = 0.6  # O比例
    
    generated_conditional = cond_generator.generate(test_condition, num_samples=10)
    print(f"条件生成样本形状: {generated_conditional.shape}")
    print(f"生成样本特征 (前5个特征):")
    print(generated_conditional[:, :5])
    
    return cond_generator, cond_discriminator, history


# ==========================================
# 7. 主程序入口
# ==========================================
if __name__ == '__main__':
    print("=" * 60)
    print("  对抗生成网络 (GAN) - 铁电材料生成与预测")
    print("=" * 60)
    
    # 训练标准GAN
    print("\n[1] 训练标准GAN")
    print("-" * 40)
    generator, discriminator, gan_history = train_ferroelectric_gan()
    
    # 训练条件GAN
    print("\n[2] 训练条件GAN")
    print("-" * 40)
    cond_generator, cond_discriminator, cond_history = train_conditional_gan()
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("\n模型文件:")
    print("  - model_gan/ferroelectric_gan.pt")
    print("  - model_gan/generator.pt")
    print("  - model_gan/discriminator.pt")
    print("  - model_gan/conditional_gan.pt")
    print("\n报告文件:")
    print("  - reports_gan/")
    print("\n生成材料:")
    print("  - generated_materials/")
    print("=" * 60)
    
    # 演示生成功能
    print("\n【生成器演示】")
    print("-" * 40)
    
    # 生成10个候选铁电材料
    candidates = generator.generate(10)
    print(f"生成 {len(candidates)} 个候选材料特征向量")
    print(f"特征向量形状: {candidates.shape}")
    
    # 使用判别器评估
    fe_probs = discriminator.predict_ferroelectric(candidates)
    print(f"\n铁电概率:")
    for i, prob in enumerate(fe_probs):
        print(f"  候选 {i+1}: {prob[0]:.4f}")
