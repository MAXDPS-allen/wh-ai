"""
逆向设计模型 - 从特征向量预测材料条目
==========================================
功能:
1. 从GCNN嵌入向量逆向预测材料的化学成分和晶格参数
2. 使用多任务学习同时预测多个目标
3. 支持条件生成 (给定约束条件生成候选材料)
4. 完整的模型保存和报表模块

输入特征 (来自GCNN或手工特征):
- 32维全局特征向量
- 或256维GCNN嵌入向量 (可选)

输出预测:
- 化学成分: 元素类型 + 比例 (最多4种元素)
- 晶格参数: a, b, c, α, β, γ
- 空间群号
"""

import json
import os
import time
import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# 设置
sns.set(style="whitegrid")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")

# 创建输出目录
os.makedirs('invs_dgn_model', exist_ok=True)
os.makedirs('reports_inverse', exist_ok=True)

# ==========================================
# 1. 元素数据库和特征配置
# ==========================================
ELEMENT_DATA = {
    'H': [1, 1.008, 0.37, 2.20], 'Li': [3, 6.94, 1.34, 0.98], 
    'Be': [4, 9.012, 0.90, 1.57], 'B': [5, 10.81, 0.82, 2.04],
    'C': [6, 12.011, 0.77, 2.55], 'N': [7, 14.007, 0.75, 3.04],
    'O': [8, 15.999, 0.73, 3.44], 'F': [9, 18.998, 0.71, 3.98],
    'Na': [11, 22.990, 1.54, 0.93], 'Mg': [12, 24.305, 1.30, 1.31],
    'Al': [13, 26.982, 1.18, 1.61], 'Si': [14, 28.085, 1.11, 1.90],
    'P': [15, 30.974, 1.06, 2.19], 'S': [16, 32.06, 1.02, 2.58],
    'Cl': [17, 35.45, 0.99, 3.16], 'K': [19, 39.098, 1.96, 0.82],
    'Ca': [20, 40.078, 1.74, 1.00], 'Sc': [21, 44.956, 1.44, 1.36],
    'Ti': [22, 47.867, 1.36, 1.54], 'V': [23, 50.942, 1.25, 1.63],
    'Cr': [24, 51.996, 1.27, 1.66], 'Mn': [25, 54.938, 1.39, 1.55],
    'Fe': [26, 55.845, 1.25, 1.83], 'Co': [27, 58.933, 1.26, 1.88],
    'Ni': [28, 58.693, 1.21, 1.91], 'Cu': [29, 63.546, 1.38, 1.90],
    'Zn': [30, 65.38, 1.31, 1.65], 'Ga': [31, 69.723, 1.26, 1.81],
    'Ge': [32, 72.63, 1.22, 2.01], 'As': [33, 74.922, 1.19, 2.18],
    'Se': [34, 78.96, 1.16, 2.55], 'Br': [35, 79.904, 1.14, 2.96],
    'Rb': [37, 85.468, 2.11, 0.82], 'Sr': [38, 87.62, 1.92, 0.95],
    'Y': [39, 88.906, 1.62, 1.22], 'Zr': [40, 91.224, 1.48, 1.33],
    'Nb': [41, 92.906, 1.37, 1.60], 'Mo': [42, 95.96, 1.45, 2.16],
    'Pd': [46, 106.42, 1.31, 2.20], 'Ag': [47, 107.87, 1.53, 1.93],
    'Cd': [48, 112.41, 1.48, 1.69], 'In': [49, 114.82, 1.44, 1.78],
    'Sn': [50, 118.71, 1.41, 1.96], 'Sb': [51, 121.76, 1.38, 2.05],
    'Te': [52, 127.60, 1.35, 2.10], 'I': [53, 126.90, 1.33, 2.66],
    'Cs': [55, 132.91, 2.25, 0.79], 'Ba': [56, 137.33, 1.98, 0.89],
    'La': [57, 138.91, 1.69, 1.10], 'Ce': [58, 140.12, 1.65, 1.12],
    'Nd': [60, 144.24, 1.64, 1.14], 'Sm': [62, 150.36, 1.62, 1.17],
    'Eu': [63, 151.96, 1.85, 1.20], 'Gd': [64, 157.25, 1.61, 1.20],
    'Tb': [65, 158.93, 1.59, 1.10], 'Dy': [66, 162.50, 1.59, 1.22],
    'Ho': [67, 164.93, 1.58, 1.23], 'Er': [68, 167.26, 1.57, 1.24],
    'Yb': [70, 173.05, 1.74, 1.10], 'Lu': [71, 174.97, 1.56, 1.27],
    'Hf': [72, 178.49, 1.44, 1.30], 'Ta': [73, 180.95, 1.34, 1.50],
    'W': [74, 183.84, 1.30, 2.36], 'Pb': [82, 207.2, 1.47, 2.33],
    'Bi': [83, 208.98, 1.46, 2.02], 'Th': [90, 232.04, 1.79, 1.30],
    'U': [92, 238.03, 1.56, 1.38],
}

# 元素到索引的映射 (用于分类)
ELEMENT_TO_IDX = {el: i for i, el in enumerate(ELEMENT_DATA.keys())}
IDX_TO_ELEMENT = {i: el for el, i in ELEMENT_TO_IDX.items()}
NUM_ELEMENTS = len(ELEMENT_DATA)

# 特征配置
MAX_ELEMENTS_PER_COMPOUND = 5  # 最多5种元素
INPUT_FEATURE_DIM = 32  # 输入特征维度 (来自GCNN全局特征)

# 输出目标配置
# 晶格参数: vol_root, b/a, c/a, alpha, beta, gamma (6维)
# 元素: 每个位置 (元素索引 + 比例) × MAX_ELEMENTS = 5 × 2 = 10维
# 空间群: 1维
# 总计: 17维
LATTICE_DIM = 6
COMPOSITION_DIM = MAX_ELEMENTS_PER_COMPOUND * 2  # 元素索引 + 比例
SPACEGROUP_DIM = 1
OUTPUT_DIM = LATTICE_DIM + COMPOSITION_DIM + SPACEGROUP_DIM


# ==========================================
# 2. 数据集和特征提取
# ==========================================
class InverseDesignDataset(Dataset):
    """逆向设计数据集"""
    
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def extract_input_features(item):
    """
    从材料数据提取输入特征向量 (32维)
    这些特征与GCNN_enhanced.py中的全局特征对应
    """
    if 'structure' not in item:
        return None
    
    features = np.zeros(32, dtype=np.float32)
    
    try:
        struct = item['structure']
        sites = struct.get('sites', [])
        lattice = struct.get('lattice', {})
        
        if not sites or not lattice:
            return None
        
        # 晶格参数
        a = lattice.get('a', 0)
        b = lattice.get('b', 0)
        c = lattice.get('c', 0)
        alpha = lattice.get('alpha', 90)
        beta = lattice.get('beta', 90)
        gamma = lattice.get('gamma', 90)
        vol = lattice.get('volume', 0)
        
        if vol <= 0 or a <= 0:
            return None
        
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
            return None
        
        # 元素特征统计
        masses, radii, ens = [], [], []
        tm_count, o_count = 0, 0
        
        TM_SET = {'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                  'Zr', 'Nb', 'Mo', 'Hf', 'Ta', 'W'}
        
        for el, count in comp.items():
            if el in ELEMENT_DATA:
                data = ELEMENT_DATA[el]
                frac = count / total_atoms
                masses.append(data[1] * frac)
                radii.append(data[2])
                ens.append(data[3])
                if el in TM_SET:
                    tm_count += count
                if el == 'O':
                    o_count += count
        
        if not masses:
            return None
        
        # 密度估算
        total_mass = sum(masses)
        density = total_mass / vol if vol > 0 else 0
        
        # 填充特征向量 [0-31]
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
        
        # [9-11] 对称性 (从数据中获取)
        sg = item.get('spacegroup_number', 1)
        features[9] = sg / 230.0 if sg else 0
        features[10] = 0.5  # 晶系 (简化)
        features[11] = 1.0 if sg in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] else 0  # 极性估计
        
        # [12-17] 元素统计
        features[12] = np.sum(masses) / 200.0
        features[13] = np.mean(radii) / 2.0 if radii else 0
        features[14] = np.mean(ens) / 4.0 if ens else 0
        features[15] = 0.5  # 平均电离能 (简化)
        features[16] = 0.5  # 平均价电子 (简化)
        features[17] = len(comp) / 10.0
        
        # [18-21] 元素范围
        features[18] = (max(radii) - min(radii)) / 2.0 if len(radii) > 1 else 0
        features[19] = (max(ens) - min(ens)) / 4.0 if len(ens) > 1 else 0
        features[20] = 0.5
        features[21] = max(radii) / max(min(radii), 0.1) if radii else 1.0
        
        # [22-25] 配位 (简化估计)
        features[22] = 6.0 / 12.0  # 假设平均配位数6
        features[23] = 0.2
        features[24] = 2.0 / 4.0  # 假设平均键长2Å
        features[25] = 0.1
        
        # [26-28] 化学
        features[26] = tm_count / total_atoms
        features[27] = 0.5
        features[28] = o_count / total_atoms
        
        # [29-31] 热力学 (简化)
        features[29] = 0.5
        features[30] = 0.5
        features[31] = 0.5
        
        return features
        
    except Exception as e:
        return None


def extract_target_vector(item):
    """
    从材料数据提取目标向量
    
    返回: [vol_root, b/a, c/a, alpha, beta, gamma, 
           el1_idx, el1_frac, el2_idx, el2_frac, ..., 
           spacegroup]
    """
    if 'structure' not in item:
        return None
    
    try:
        struct = item['structure']
        lattice = struct.get('lattice', {})
        sites = struct.get('sites', [])
        
        # 晶格参数
        a = lattice.get('a', 0)
        b = lattice.get('b', 0)
        c = lattice.get('c', 0)
        alpha = lattice.get('alpha', 90)
        beta = lattice.get('beta', 90)
        gamma = lattice.get('gamma', 90)
        vol = lattice.get('volume', 0)
        
        if vol <= 0 or a <= 0:
            return None
        
        # 物理变换
        vol_root = vol ** (1/3)
        ratio_ba = b / a
        ratio_ca = c / a
        
        lattice_targets = [vol_root, ratio_ba, ratio_ca, alpha, beta, gamma]
        
        # 成分
        comp = {}
        total_atoms = 0
        for site in sites:
            for species in site['species']:
                el = species['element']
                occu = species['occu']
                comp[el] = comp.get(el, 0) + occu
                total_atoms += occu
        
        # 排序并编码
        el_list = []
        for el, count in comp.items():
            if el in ELEMENT_TO_IDX:
                idx = ELEMENT_TO_IDX[el]
                frac = count / total_atoms
                el_list.append((idx, frac))
        
        # 按原子序数排序
        el_list.sort(key=lambda x: x[0])
        
        # 填充到固定长度
        comp_vector = []
        for i in range(MAX_ELEMENTS_PER_COMPOUND):
            if i < len(el_list):
                comp_vector.extend([el_list[i][0] / NUM_ELEMENTS, el_list[i][1]])
            else:
                comp_vector.extend([0, 0])
        
        # 空间群
        sg = item.get('spacegroup_number', 1)
        sg_normalized = (sg if sg else 1) / 230.0
        
        target = lattice_targets + comp_vector + [sg_normalized]
        
        return np.array(target, dtype=np.float32)
        
    except Exception as e:
        return None


def load_training_data(files_config):
    """加载训练数据"""
    X_all, Y_all = [], []
    
    print("Loading data for inverse design...")
    
    for f_name, weight in files_config.items():
        print(f"  Loading: {f_name}")
        try:
            with open(f_name, 'r') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        feat = extract_input_features(item)
                        target = extract_target_vector(item)
                        
                        if feat is not None and target is not None:
                            # 按权重复制样本
                            for _ in range(int(weight)):
                                X_all.append(feat)
                                Y_all.append(target)
                    except:
                        continue
        except FileNotFoundError:
            print(f"    [Warning] File not found: {f_name}")
    
    X_all = np.array(X_all)
    Y_all = np.array(Y_all)
    
    print(f"Loaded {len(X_all)} samples")
    print(f"Input shape: {X_all.shape}, Target shape: {Y_all.shape}")
    
    return X_all, Y_all


# ==========================================
# 3. 逆向设计神经网络
# ==========================================
class InverseDesignNetwork(nn.Module):
    """
    逆向设计神经网络
    
    从特征向量预测材料的化学成分和晶格参数
    """
    
    def __init__(self, input_dim=32, hidden_dim=256, output_dim=17):
        super(InverseDesignNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
        )
        
        # 多任务头
        # 晶格参数预测 (6维)
        self.lattice_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, LATTICE_DIM)
        )
        
        # 成分预测 (10维: 5元素 × 2)
        self.composition_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, COMPOSITION_DIM),
            nn.Sigmoid()  # 归一化到 [0, 1]
        )
        
        # 空间群预测 (1维)
        self.spacegroup_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, SPACEGROUP_DIM),
            nn.Sigmoid()  # 归一化到 [0, 1]
        )
    
    def forward(self, x):
        # 编码
        hidden = self.encoder(x)
        
        # 多任务预测
        lattice = self.lattice_head(hidden)
        composition = self.composition_head(hidden)
        spacegroup = self.spacegroup_head(hidden)
        
        # 拼接输出
        output = torch.cat([lattice, composition, spacegroup], dim=1)
        
        return output
    
    def predict_material(self, features):
        """
        从特征向量预测材料
        
        返回:
            dict: 包含晶格参数、成分、空间群
        """
        self.eval()
        with torch.no_grad():
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            
            features = features.to(device)
            output = self.forward(features).cpu().numpy()[0]
        
        # 解析输出
        lattice = output[:LATTICE_DIM]
        composition_raw = output[LATTICE_DIM:LATTICE_DIM + COMPOSITION_DIM]
        spacegroup = output[-1]
        
        # 还原晶格参数
        vol_root = lattice[0]
        ratio_ba = max(lattice[1], 0.1)
        ratio_ca = max(lattice[2], 0.1)
        
        a = vol_root / (ratio_ba * ratio_ca) ** (1/3)
        b = a * ratio_ba
        c = a * ratio_ca
        alpha, beta, gamma = lattice[3], lattice[4], lattice[5]
        
        # 解析成分
        elements = []
        for i in range(MAX_ELEMENTS_PER_COMPOUND):
            el_idx = int(composition_raw[i * 2] * NUM_ELEMENTS)
            el_frac = composition_raw[i * 2 + 1]
            if el_frac > 0.05 and el_idx < NUM_ELEMENTS:  # 过滤小于5%的元素
                el_symbol = IDX_TO_ELEMENT.get(el_idx, 'X')
                elements.append({'element': el_symbol, 'fraction': el_frac})
        
        # 归一化成分
        total_frac = sum([e['fraction'] for e in elements])
        if total_frac > 0:
            for e in elements:
                e['fraction'] /= total_frac
        
        return {
            'lattice': {
                'a': float(a), 'b': float(b), 'c': float(c),
                'alpha': float(alpha), 'beta': float(beta), 'gamma': float(gamma),
                'volume': float(vol_root ** 3)
            },
            'composition': elements,
            'spacegroup': int(spacegroup * 230),
            'raw_output': output
        }


# ==========================================
# 4. 条件生成器 (VAE-like)
# ==========================================
class ConditionalGenerator(nn.Module):
    """
    条件材料生成器
    
    给定约束条件 (如空间群、元素类型)，生成满足条件的特征向量
    """
    
    def __init__(self, latent_dim=64, condition_dim=10, output_dim=32):
        super(ConditionalGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(condition_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # 生成器
        self.generator = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, output_dim),
            nn.Sigmoid()  # 归一化输出
        )
    
    def forward(self, z, condition):
        # 编码条件
        cond_encoded = self.condition_encoder(condition)
        
        # 拼接潜在变量和条件
        combined = torch.cat([z, cond_encoded], dim=1)
        
        # 生成特征
        features = self.generator(combined)
        
        return features
    
    def generate(self, condition, num_samples=1):
        """给定条件生成特征向量"""
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


# ==========================================
# 5. 模型保存模块
# ==========================================
class InverseModelSaver:
    """逆向设计模型保存管理器"""
    
    def __init__(self, save_dir='invs_dgn_model'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_pytorch_model(self, model, optimizer, epoch, metrics, filename=None):
        """保存PyTorch模型"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'inverse_design_{timestamp}.pt'
        
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'model_config': {
                'input_dim': model.input_dim,
                'output_dim': model.output_dim
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"PyTorch model saved: {filepath}")
        return filepath
    
    def save_sklearn_model(self, model, filename='inverse_design_sklearn.pkl'):
        """保存sklearn模型"""
        filepath = os.path.join(self.save_dir, filename)
        joblib.dump(model, filepath)
        print(f"Sklearn model saved: {filepath}")
        return filepath
    
    def save_scaler(self, scaler, filename='feature_scaler.pkl'):
        """保存特征缩放器"""
        filepath = os.path.join(self.save_dir, filename)
        joblib.dump(scaler, filepath)
        print(f"Scaler saved: {filepath}")
        return filepath
    
    def load_pytorch_model(self, model, filepath, optimizer=None):
        """加载PyTorch模型"""
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def save_best_model(self, model, optimizer, epoch, metrics):
        """保存最佳模型"""
        return self.save_pytorch_model(model, optimizer, epoch, metrics, 'best_inverse_model.pt')


# ==========================================
# 6. 报表生成模块
# ==========================================
class InverseDesignReporter:
    """逆向设计报表生成器"""
    
    def __init__(self, report_dir='reports_inverse'):
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'lattice_mae': [],
            'composition_mae': [],
            'spacegroup_mae': []
        }
    
    def log_training(self, epoch, train_loss, val_loss=None, metrics=None):
        """记录训练日志"""
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss if val_loss else 0)
        
        # 确保所有数组长度一致
        if metrics:
            self.training_history['lattice_mae'].append(metrics.get('lattice_mae', 0))
            self.training_history['composition_mae'].append(metrics.get('composition_mae', 0))
            self.training_history['spacegroup_mae'].append(metrics.get('spacegroup_mae', 0))
        else:
            self.training_history['lattice_mae'].append(0)
            self.training_history['composition_mae'].append(0)
            self.training_history['spacegroup_mae'].append(0)
    
    def generate_accuracy_report(self, y_true, y_pred, save=True):
        """生成准确率报告"""
        report = []
        report.append("=" * 70)
        report.append("  逆向设计模型准确率报告")
        report.append(f"  生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        # 分解预测结果
        lattice_true = y_true[:, :LATTICE_DIM]
        lattice_pred = y_pred[:, :LATTICE_DIM]
        
        comp_true = y_true[:, LATTICE_DIM:LATTICE_DIM + COMPOSITION_DIM]
        comp_pred = y_pred[:, LATTICE_DIM:LATTICE_DIM + COMPOSITION_DIM]
        
        sg_true = y_true[:, -1]
        sg_pred = y_pred[:, -1]
        
        # 晶格参数评估
        report.append("\n【晶格参数预测】")
        lattice_names = ['vol_root', 'b/a', 'c/a', 'alpha', 'beta', 'gamma']
        
        for i, name in enumerate(lattice_names):
            mae = mean_absolute_error(lattice_true[:, i], lattice_pred[:, i])
            r2 = r2_score(lattice_true[:, i], lattice_pred[:, i])
            report.append(f"  {name:<10}: MAE = {mae:.4f}, R² = {r2:.4f}")
        
        # 还原实际晶格参数
        report.append("\n【还原晶格参数 (a, b, c)】")
        
        def restore_abc(y):
            vol_root = y[:, 0]
            ratio_ba = np.maximum(y[:, 1], 0.1)
            ratio_ca = np.maximum(y[:, 2], 0.1)
            a = vol_root / (ratio_ba * ratio_ca) ** (1/3)
            b = a * ratio_ba
            c = a * ratio_ca
            return np.stack([a, b, c], axis=1)
        
        abc_true = restore_abc(lattice_true)
        abc_pred = restore_abc(lattice_pred)
        
        for i, name in enumerate(['a', 'b', 'c']):
            mae = mean_absolute_error(abc_true[:, i], abc_pred[:, i])
            mape = np.mean(np.abs((abc_true[:, i] - abc_pred[:, i]) / (abc_true[:, i] + 1e-6)))
            r2 = r2_score(abc_true[:, i], abc_pred[:, i])
            report.append(f"  {name:<10}: MAE = {mae:.4f} Å, MAPE = {mape:.1%}, R² = {r2:.4f}")
        
        # 成分预测评估
        report.append("\n【化学成分预测】")
        
        for i in range(MAX_ELEMENTS_PER_COMPOUND):
            idx_true = comp_true[:, i * 2] * NUM_ELEMENTS
            idx_pred = comp_pred[:, i * 2] * NUM_ELEMENTS
            frac_true = comp_true[:, i * 2 + 1]
            frac_pred = comp_pred[:, i * 2 + 1]
            
            # 元素命中率 (预测索引与真实索引差距 < 1)
            idx_hits = np.sum(np.abs(idx_true - idx_pred) < 1)
            idx_acc = idx_hits / len(idx_true)
            
            frac_mae = mean_absolute_error(frac_true, frac_pred)
            
            report.append(f"  Element {i+1}: 索引命中率 = {idx_acc:.1%}, 比例MAE = {frac_mae:.4f}")
        
        # 空间群预测
        report.append("\n【空间群预测】")
        sg_true_real = sg_true * 230
        sg_pred_real = sg_pred * 230
        sg_mae = mean_absolute_error(sg_true_real, sg_pred_real)
        sg_hits = np.sum(np.abs(sg_true_real - sg_pred_real) < 5)
        sg_acc = sg_hits / len(sg_true)
        report.append(f"  MAE = {sg_mae:.2f}, 命中率 (±5) = {sg_acc:.1%}")
        
        # 总体评估
        report.append("\n【总体评估】")
        overall_mae = mean_absolute_error(y_true, y_pred)
        overall_r2 = r2_score(y_true.flatten(), y_pred.flatten())
        report.append(f"  Overall MAE: {overall_mae:.4f}")
        report.append(f"  Overall R²: {overall_r2:.4f}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = os.path.join(self.report_dir, f'accuracy_report_{timestamp}.txt')
            with open(report_path, 'w') as f:
                f.write(report_text)
            print(f"\n报告已保存: {report_path}")
        
        return {
            'lattice_mae': mean_absolute_error(lattice_true, lattice_pred),
            'composition_mae': mean_absolute_error(comp_true, comp_pred),
            'spacegroup_mae': sg_mae,
            'overall_mae': overall_mae,
            'overall_r2': overall_r2
        }
    
    def plot_parity_plots(self, y_true, y_pred, save=True):
        """绘制对比图"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 晶格参数
        lattice_names = ['vol_root', 'b/a', 'c/a', 'alpha', 'beta', 'gamma']
        
        for i, (ax, name) in enumerate(zip(axes.flatten(), lattice_names)):
            ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=10)
            min_val = min(y_true[:, i].min(), y_pred[:, i].min())
            max_val = max(y_true[:, i].max(), y_pred[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax.set_xlabel(f'True {name}')
            ax.set_ylabel(f'Predicted {name}')
            ax.set_title(f'{name} Parity Plot')
            
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                    verticalalignment='top')
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.report_dir, f'parity_plots_{timestamp}.png')
            plt.savefig(plot_path, dpi=150)
            print(f"对比图已保存: {plot_path}")
        
        plt.close()
    
    def save_training_history(self):
        """保存训练历史"""
        df = pd.DataFrame(self.training_history)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(self.report_dir, f'training_history_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"训练历史已保存: {csv_path}")


# ==========================================
# 7. 训练流程
# ==========================================
def train_inverse_design_model():
    """训练逆向设计模型"""
    
    # 数据文件配置
    files_config = {
        'new_data/dataset_known_FE_rest.jsonl': 3,
        'new_data/dataset_original_ferroelectric.jsonl': 3,
        'new_data/dataset_polar_non_ferroelectric_final.jsonl': 1,
        'new_data/dataset_nonFE.jsonl': 1
    }
    
    # 加载数据
    X_all, Y_all = load_training_data(files_config)
    
    if len(X_all) == 0:
        print("Error: No data loaded!")
        return None
    
    # 特征缩放
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, Y_all, test_size=0.15, random_state=42
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 创建数据加载器
    train_dataset = InverseDesignDataset(X_train, y_train)
    test_dataset = InverseDesignDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    model = InverseDesignNetwork(
        input_dim=INPUT_FEATURE_DIM,
        hidden_dim=256,
        output_dim=OUTPUT_DIM
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.MSELoss()
    
    # 初始化工具
    model_saver = InverseModelSaver()
    reporter = InverseDesignReporter()
    
    print(f"\n开始训练逆向设计模型 (设备: {device})")
    print("-" * 60)
    
    best_loss = float('inf')
    num_epochs = 100
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        
        # 每10个epoch评估
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0
            all_pred = []
            all_true = []
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
                    
                    all_pred.extend(output.cpu().numpy())
                    all_true.extend(batch_y.cpu().numpy())
            
            val_loss /= len(test_loader)
            
            all_pred = np.array(all_pred)
            all_true = np.array(all_true)
            
            metrics = {
                'lattice_mae': mean_absolute_error(all_true[:, :LATTICE_DIM], all_pred[:, :LATTICE_DIM]),
                'composition_mae': mean_absolute_error(
                    all_true[:, LATTICE_DIM:LATTICE_DIM+COMPOSITION_DIM],
                    all_pred[:, LATTICE_DIM:LATTICE_DIM+COMPOSITION_DIM]
                ),
                'spacegroup_mae': mean_absolute_error(all_true[:, -1] * 230, all_pred[:, -1] * 230)
            }
            
            reporter.log_training(epoch + 1, avg_loss, val_loss, metrics)
            
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Lattice MAE: {metrics['lattice_mae']:.4f}")
            
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                model_saver.save_best_model(model, optimizer, epoch, metrics)
        else:
            reporter.log_training(epoch + 1, avg_loss)
    
    # 最终评估
    print("\n" + "=" * 60)
    print("最终评估")
    print("=" * 60)
    
    model.eval()
    all_pred = []
    all_true = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            all_pred.extend(output.cpu().numpy())
            all_true.extend(batch_y.numpy())
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    # 生成报告
    final_metrics = reporter.generate_accuracy_report(all_true, all_pred)
    reporter.plot_parity_plots(all_true, all_pred)
    reporter.save_training_history()
    
    # 保存模型和缩放器
    model_saver.save_pytorch_model(model, optimizer, num_epochs, final_metrics, 'final_inverse_model.pt')
    model_saver.save_scaler(scaler)
    
    # 测试预测功能
    print("\n【预测示例】")
    sample_features = X_scaled[0]
    prediction = model.predict_material(sample_features)
    print(f"输入特征形状: {sample_features.shape}")
    print(f"预测晶格: a={prediction['lattice']['a']:.3f}, b={prediction['lattice']['b']:.3f}, c={prediction['lattice']['c']:.3f}")
    print(f"预测成分: {prediction['composition']}")
    print(f"预测空间群: {prediction['spacegroup']}")
    
    return model, scaler, final_metrics


# ==========================================
# 8. 主程序入口
# ==========================================
if __name__ == '__main__':
    print("=" * 60)
    print("  逆向设计模型 - 从特征向量预测材料条目")
    print("=" * 60)
    
    model, scaler, metrics = train_inverse_design_model()
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"晶格参数 MAE: {metrics['lattice_mae']:.4f}")
    print(f"成分 MAE: {metrics['composition_mae']:.4f}")
    print(f"空间群 MAE: {metrics['spacegroup_mae']:.2f}")
    print(f"总体 R²: {metrics['overall_r2']:.4f}")
    print("=" * 60)
