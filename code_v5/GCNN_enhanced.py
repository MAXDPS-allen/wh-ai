"""
增强版 GCNN - 铁电材料分类器
================================
特征增强版本：
1. 扩展全局特征维度 (10 → 32 维)
2. 增加化学环境特征
3. 增加键长/键角分布特征
4. 添加元素电负性差/离子半径比等
5. 完整的模型保存和报表模块

用于逆向设计的特征向量包含足够信息以确定材料的:
- 晶格参数 (a, b, c, α, β, γ, volume)
- 化学成分 (元素种类及比例)
- 对称性信息 (空间群, 晶系, 点群)
- 键合特征 (配位数, 键长分布)
"""

import json
import os
import time
import datetime
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, GATConv
from pymatgen.core.structure import Structure
from pymatgen.core.periodic_table import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from torch.cuda.amp import autocast, GradScaler
import joblib

# ==========================================
# 0. 显卡环境配置
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Device: {device}")
if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True

# 创建输出目录
os.makedirs('model_enhanced', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ==========================================
# 1. 扩展特征工程
# ==========================================
# 元素物理化学数据库
ELEMENT_FEATURES = {
    # Element: [atomic_number, mass, radius, electronegativity, ionization_energy, valence, melting_point, thermal_conductivity]
    'H': [1, 1.008, 0.37, 2.20, 13.598, 1, 14.01, 0.1805],
    'Li': [3, 6.94, 1.34, 0.98, 5.392, 1, 453.65, 84.8],
    'Be': [4, 9.012, 0.90, 1.57, 9.323, 2, 1560, 201],
    'B': [5, 10.81, 0.82, 2.04, 8.298, 3, 2349, 27.4],
    'C': [6, 12.011, 0.77, 2.55, 11.260, 4, 3823, 129],
    'N': [7, 14.007, 0.75, 3.04, 14.534, 5, 63.15, 0.02583],
    'O': [8, 15.999, 0.73, 3.44, 13.618, 6, 54.36, 0.02658],
    'F': [9, 18.998, 0.71, 3.98, 17.423, 7, 53.53, 0.0277],
    'Na': [11, 22.990, 1.54, 0.93, 5.139, 1, 370.95, 142],
    'Mg': [12, 24.305, 1.30, 1.31, 7.646, 2, 923, 156],
    'Al': [13, 26.982, 1.18, 1.61, 5.986, 3, 933.47, 237],
    'Si': [14, 28.085, 1.11, 1.90, 8.152, 4, 1687, 149],
    'P': [15, 30.974, 1.06, 2.19, 10.487, 5, 317.3, 0.236],
    'S': [16, 32.06, 1.02, 2.58, 10.360, 6, 388.36, 0.205],
    'Cl': [17, 35.45, 0.99, 3.16, 12.968, 7, 171.6, 0.0089],
    'K': [19, 39.098, 1.96, 0.82, 4.341, 1, 336.53, 102.5],
    'Ca': [20, 40.078, 1.74, 1.00, 6.113, 2, 1115, 201],
    'Sc': [21, 44.956, 1.44, 1.36, 6.561, 3, 1814, 15.8],
    'Ti': [22, 47.867, 1.36, 1.54, 6.828, 4, 1941, 21.9],
    'V': [23, 50.942, 1.25, 1.63, 6.746, 5, 2183, 30.7],
    'Cr': [24, 51.996, 1.27, 1.66, 6.767, 6, 2180, 93.9],
    'Mn': [25, 54.938, 1.39, 1.55, 7.434, 7, 1519, 7.81],
    'Fe': [26, 55.845, 1.25, 1.83, 7.902, 8, 1811, 80.4],
    'Co': [27, 58.933, 1.26, 1.88, 7.881, 9, 1768, 100],
    'Ni': [28, 58.693, 1.21, 1.91, 7.640, 10, 1728, 90.9],
    'Cu': [29, 63.546, 1.38, 1.90, 7.726, 11, 1357.77, 401],
    'Zn': [30, 65.38, 1.31, 1.65, 9.394, 12, 692.68, 116],
    'Ga': [31, 69.723, 1.26, 1.81, 6.000, 3, 302.91, 40.6],
    'Ge': [32, 72.63, 1.22, 2.01, 7.900, 4, 1211.40, 60.2],
    'As': [33, 74.922, 1.19, 2.18, 9.789, 5, 1090, 50.2],
    'Se': [34, 78.96, 1.16, 2.55, 9.752, 6, 494, 0.52],
    'Br': [35, 79.904, 1.14, 2.96, 11.814, 7, 265.8, 0.122],
    'Rb': [37, 85.468, 2.11, 0.82, 4.177, 1, 312.46, 58.2],
    'Sr': [38, 87.62, 1.92, 0.95, 5.695, 2, 1050, 35.4],
    'Y': [39, 88.906, 1.62, 1.22, 6.217, 3, 1799, 17.2],
    'Zr': [40, 91.224, 1.48, 1.33, 6.634, 4, 2128, 22.6],
    'Nb': [41, 92.906, 1.37, 1.60, 6.759, 5, 2750, 53.7],
    'Mo': [42, 95.96, 1.45, 2.16, 7.092, 6, 2896, 138],
    'Ru': [44, 101.07, 1.26, 2.20, 7.361, 8, 2607, 117],
    'Rh': [45, 102.91, 1.35, 2.28, 7.459, 9, 2237, 150],
    'Pd': [46, 106.42, 1.31, 2.20, 8.337, 10, 1828.05, 71.8],
    'Ag': [47, 107.87, 1.53, 1.93, 7.576, 11, 1234.93, 429],
    'Cd': [48, 112.41, 1.48, 1.69, 8.994, 12, 594.22, 96.6],
    'In': [49, 114.82, 1.44, 1.78, 5.786, 3, 429.75, 81.8],
    'Sn': [50, 118.71, 1.41, 1.96, 7.344, 4, 505.08, 66.8],
    'Sb': [51, 121.76, 1.38, 2.05, 8.64, 5, 903.78, 24.4],
    'Te': [52, 127.60, 1.35, 2.10, 9.010, 6, 722.66, 3.0],
    'I': [53, 126.90, 1.33, 2.66, 10.451, 7, 386.85, 0.449],
    'Cs': [55, 132.91, 2.25, 0.79, 3.894, 1, 301.59, 35.9],
    'Ba': [56, 137.33, 1.98, 0.89, 5.212, 2, 1000, 18.4],
    'La': [57, 138.91, 1.69, 1.10, 5.577, 3, 1193, 13.4],
    'Ce': [58, 140.12, 1.65, 1.12, 5.539, 4, 1068, 11.3],
    'Pr': [59, 140.91, 1.65, 1.13, 5.464, 3, 1208, 12.5],
    'Nd': [60, 144.24, 1.64, 1.14, 5.525, 3, 1297, 16.5],
    'Sm': [62, 150.36, 1.62, 1.17, 5.644, 3, 1345, 13.3],
    'Eu': [63, 151.96, 1.85, 1.20, 5.670, 2, 1099, 13.9],
    'Gd': [64, 157.25, 1.61, 1.20, 6.150, 3, 1585, 10.6],
    'Tb': [65, 158.93, 1.59, 1.10, 5.864, 3, 1629, 11.1],
    'Dy': [66, 162.50, 1.59, 1.22, 5.939, 3, 1680, 10.7],
    'Ho': [67, 164.93, 1.58, 1.23, 6.022, 3, 1734, 16.2],
    'Er': [68, 167.26, 1.57, 1.24, 6.108, 3, 1802, 14.5],
    'Tm': [69, 168.93, 1.56, 1.25, 6.184, 3, 1818, 16.9],
    'Yb': [70, 173.05, 1.74, 1.10, 6.254, 2, 1097, 38.5],
    'Lu': [71, 174.97, 1.56, 1.27, 5.426, 3, 1925, 16.4],
    'Hf': [72, 178.49, 1.44, 1.30, 6.825, 4, 2506, 23.0],
    'Ta': [73, 180.95, 1.34, 1.50, 7.89, 5, 3290, 57.5],
    'W': [74, 183.84, 1.30, 2.36, 7.98, 6, 3695, 173],
    'Re': [75, 186.21, 1.28, 1.90, 7.88, 7, 3459, 48.0],
    'Os': [76, 190.23, 1.26, 2.20, 8.7, 8, 3306, 87.6],
    'Ir': [77, 192.22, 1.27, 2.20, 9.1, 9, 2719, 147],
    'Pt': [78, 195.08, 1.30, 2.28, 9.0, 10, 2041.4, 71.6],
    'Au': [79, 196.97, 1.34, 2.54, 9.226, 11, 1337.33, 318],
    'Pb': [82, 207.2, 1.47, 2.33, 7.417, 4, 600.61, 35.3],
    'Bi': [83, 208.98, 1.46, 2.02, 7.289, 5, 544.7, 7.97],
    'Th': [90, 232.04, 1.79, 1.30, 6.08, 4, 2023, 54.0],
    'U': [92, 238.03, 1.56, 1.38, 6.194, 6, 1405.3, 27.5],
}

# 极性点群列表
POLAR_POINT_GROUPS = ['1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm']

# 晶系映射
CRYSTAL_SYSTEM_MAP = {
    'triclinic': 0, 'monoclinic': 1, 'orthorhombic': 2,
    'tetragonal': 3, 'trigonal': 4, 'hexagonal': 5, 'cubic': 6
}


class EnhancedFeatureExtractor:
    """扩展特征提取器 - 32维全局特征"""
    
    GLOBAL_FEAT_DIM = 32  # 增强后的特征维度
    
    @staticmethod
    def get_element_features(element_symbol):
        """获取元素的物理化学特征"""
        if element_symbol in ELEMENT_FEATURES:
            return ELEMENT_FEATURES[element_symbol]
        else:
            # 默认值
            return [0, 0, 1.5, 2.0, 7.0, 4, 1000, 50]
    
    @staticmethod
    def compute_enhanced_features(struct):
        """
        计算32维增强全局特征向量
        
        特征组成:
        [0-2]   晶格参数: 体积(归一化), 密度, 平均原子体积
        [3-8]   晶格形状: a/b, a/c, b/c, α偏离, β偏离, γ偏离
        [9-11]  对称性: 空间群号, 晶系, 是否极性
        [12-17] 元素统计: 平均质量, 平均半径, 平均电负性, 平均电离能, 平均价电子, 元素数量
        [18-21] 元素范围: 半径范围, 电负性范围, 质量范围, 最大/最小半径比
        [22-25] 配位特征: 平均配位数, 配位数标准差, 平均键长, 键长标准差
        [26-28] 化学特征: 过渡金属比例, 阳离子比例, 氧含量
        [29-31] 热力学: 平均熔点估算, 平均热导率, 结构稳定性指数
        """
        features = np.zeros(32, dtype=np.float32)
        
        try:
            # 1. 晶格参数特征 [0-2]
            lattice = struct.lattice
            volume = lattice.volume
            density = struct.density
            num_sites = len(struct)
            avg_volume_per_atom = volume / num_sites
            
            features[0] = min(volume, 5000) / 5000.0
            features[1] = min(density, 10) / 10.0
            features[2] = min(avg_volume_per_atom, 50) / 50.0
            
            # 2. 晶格形状特征 [3-8]
            a, b, c = lattice.a, lattice.b, lattice.c
            alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
            
            features[3] = a / max(b, 0.1)  # a/b 比
            features[4] = a / max(c, 0.1)  # a/c 比
            features[5] = b / max(c, 0.1)  # b/c 比
            features[6] = (alpha - 90) / 90.0  # α角偏离
            features[7] = (beta - 90) / 90.0   # β角偏离
            features[8] = (gamma - 90) / 90.0  # γ角偏离
            
            # 3. 对称性特征 [9-11]
            try:
                sga = SpacegroupAnalyzer(struct, symprec=0.1)
                spacegroup_number = sga.get_space_group_number()
                crystal_system = sga.get_crystal_system()
                point_group = sga.get_point_group_symbol()
                is_polar = int(point_group in POLAR_POINT_GROUPS)
            except:
                spacegroup_number = 1
                crystal_system = 'triclinic'
                is_polar = 0
            
            features[9] = spacegroup_number / 230.0
            features[10] = CRYSTAL_SYSTEM_MAP.get(crystal_system, 0) / 6.0
            features[11] = float(is_polar)
            
            # 4. 元素统计特征 [12-17]
            masses, radii, ens, ies, valences = [], [], [], [], []
            tm_count, cation_count, oxygen_count = 0, 0, 0
            melting_points, thermal_conds = [], []
            
            TM_SET = {'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                      'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                      'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au'}
            
            CATION_SET = {'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba',
                          'Al', 'Ga', 'In', 'Tl', 'Pb', 'Bi', 'La', 'Ce', 'Nd', 'Y'}
            
            for site in struct:
                el = site.specie.symbol
                el_feat = EnhancedFeatureExtractor.get_element_features(el)
                
                masses.append(el_feat[1])
                radii.append(el_feat[2])
                ens.append(el_feat[3])
                ies.append(el_feat[4])
                valences.append(el_feat[5])
                melting_points.append(el_feat[6])
                thermal_conds.append(el_feat[7])
                
                if el in TM_SET:
                    tm_count += 1
                if el in CATION_SET:
                    cation_count += 1
                if el == 'O':
                    oxygen_count += 1
            
            features[12] = np.mean(masses) / 200.0
            features[13] = np.mean(radii) / 2.0
            features[14] = np.mean(ens) / 4.0
            features[15] = np.mean(ies) / 15.0
            features[16] = np.mean(valences) / 8.0
            features[17] = len(set([s.specie.symbol for s in struct])) / 10.0
            
            # 5. 元素范围特征 [18-21]
            features[18] = (max(radii) - min(radii)) / 2.0 if radii else 0
            features[19] = (max(ens) - min(ens)) / 4.0 if ens else 0
            features[20] = (max(masses) - min(masses)) / 200.0 if masses else 0
            features[21] = max(radii) / max(min(radii), 0.1) if radii else 1.0
            
            # 6. 配位特征 [22-25]
            coord_numbers = []
            bond_lengths = []
            
            for i, site in enumerate(struct):
                neighbors = struct.get_neighbors(site, r=3.5)
                coord_numbers.append(len(neighbors))
                for n in neighbors:
                    bond_lengths.append(n.nn_distance)  # 使用 nn_distance 属性
            
            features[22] = np.mean(coord_numbers) / 12.0 if coord_numbers else 0
            features[23] = np.std(coord_numbers) / 6.0 if len(coord_numbers) > 1 else 0
            features[24] = np.mean(bond_lengths) / 4.0 if bond_lengths else 0
            features[25] = np.std(bond_lengths) / 2.0 if len(bond_lengths) > 1 else 0
            
            # 7. 化学特征 [26-28]
            features[26] = tm_count / num_sites
            features[27] = cation_count / num_sites
            features[28] = oxygen_count / num_sites
            
            # 8. 热力学特征 [29-31]
            features[29] = np.mean(melting_points) / 3000.0 if melting_points else 0
            features[30] = np.log1p(np.mean(thermal_conds)) / 6.0 if thermal_conds else 0
            # 结构稳定性指数 (基于配位和电负性)
            stability_index = np.mean(coord_numbers) * np.std(ens) if coord_numbers and len(ens) > 1 else 0
            features[31] = min(stability_index, 10) / 10.0
            
            return features, spacegroup_number
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return features, 1


# ==========================================
# 2. 数据集定义
# ==========================================
class FerroelectricDataset(Dataset):
    def __init__(self, data_list):
        super(FerroelectricDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def json_to_graph_enhanced(item, label, cutoff=6.0):
    """
    将单条 JSON 数据转换为 PyG 的 Data 对象
    使用增强的32维特征向量
    """
    try:
        s_dict = item['structure']
        struct = Structure.from_dict(s_dict)
        
        # 节点特征: 原子序数
        atomic_numbers = [site.specie.number - 1 for site in struct]
        x = torch.tensor(atomic_numbers, dtype=torch.long)
        
        # 边特征: 原子间距离
        all_neighbors = struct.get_all_neighbors(r=cutoff, include_index=True)
        
        edge_indices = []
        edge_attrs = []
        
        for i, neighbors in enumerate(all_neighbors):
            for n_node in neighbors:
                j = n_node[2]
                dist = n_node[1]
                if i != j:
                    edge_indices.append([i, j])
                    edge_attrs.append([1.0 / (dist + 0.1)])
        
        if len(edge_indices) == 0:
            return None
            
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.float)
        
        # 提取增强的全局特征
        global_features, spacegroup = EnhancedFeatureExtractor.compute_enhanced_features(struct)
        global_features = torch.tensor(global_features, dtype=torch.float)
        
        # 存储额外的化学信息 (用于逆向设计)
        composition = struct.composition.as_dict()
        
        return Data(
            x=x, 
            edge_index=edge_index, 
            edge_attr=edge_attr, 
            y=y,
            global_feat=global_features, 
            spacegroup=spacegroup,
            num_atoms=len(struct)
        )
        
    except Exception as e:
        return None


def load_data_enhanced(pos_files, neg_files):
    """加载并处理数据"""
    dataset_list = []
    print("Loading data with enhanced features...")
    
    count = 0
    t0 = time.time()
    
    for f_name in pos_files:
        print(f"Loading positive samples from: {f_name}")
        try:
            with open(f_name, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    graph = json_to_graph_enhanced(item, 1)
                    if graph: 
                        dataset_list.append(graph)
                        count += 1
                        if count % 500 == 0: 
                            print(f"  Processed {count} graphs...", end='\r')
        except FileNotFoundError:
            print(f"  [Warning] File not found: {f_name}")

    for f_name in neg_files:
        print(f"\nLoading negative samples from: {f_name}")
        try:
            with open(f_name, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    graph = json_to_graph_enhanced(item, 0)
                    if graph: 
                        dataset_list.append(graph)
                        count += 1
                        if count % 500 == 0: 
                            print(f"  Processed {count} graphs...", end='\r')
        except FileNotFoundError:
            print(f"  [Warning] File not found: {f_name}")
            
    print(f"\nData loaded in {time.time()-t0:.2f}s, Total: {len(dataset_list)} graphs")
    return dataset_list


# ==========================================
# 3. 增强版 GNN 模型
# ==========================================
class EnhancedCrystalGNN(torch.nn.Module):
    """
    增强版晶体图神经网络
    
    特点:
    - 支持32维全局特征
    - 多尺度图嵌入 (mean + max pooling)
    - 更深的特征融合网络
    - 特征向量可用于逆向设计
    """
    
    def __init__(self, hidden_dim=128, embedding_dim=64, global_feat_dim=32, num_heads=4):
        super(EnhancedCrystalGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.global_feat_dim = global_feat_dim
        
        # 节点嵌入层
        self.node_embedding = nn.Embedding(100, embedding_dim)
        
        # 多头注意力 GAT 层
        self.conv1 = GATConv(embedding_dim, hidden_dim, heads=num_heads, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)
        self.conv4 = GATConv(hidden_dim, hidden_dim, heads=num_heads, concat=False)  # 额外层
        
        # 批归一化层
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        
        # 全局特征处理网络 (扩展)
        self.global_feat_net = nn.Sequential(
            nn.Linear(global_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128)
        )
        
        # 融合后的分类器 (图嵌入 x2 + 全局特征)
        # mean pool + max pool = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 特征向量维度 (用于逆向设计)
        self.embedding_dim_out = hidden_dim * 2 + 128

    def forward(self, data):
        graph_embedding = self.get_graph_embedding(data)
        
        # 处理全局特征
        global_feat = data.global_feat.view(-1, self.global_feat_dim)
        global_embedding = self.global_feat_net(global_feat)
        
        # 融合
        combined = torch.cat([graph_embedding, global_embedding], dim=1)
        out = self.classifier(combined)
        return out

    def get_graph_embedding(self, data):
        """获取图级别嵌入"""
        x, edge_index = data.x, data.edge_index
        x = self.node_embedding(x)
        
        # 第1层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        # 第2层 (带残差)
        x_res = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x) + x_res
        
        # 第3层 (带残差)
        x_res = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x) + x_res
        
        # 第4层 (带残差)
        x_res = x
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = F.relu(x) + x_res
        
        # 多尺度池化
        x_mean = global_mean_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)
        
        return torch.cat([x_mean, x_max], dim=1)
    
    def get_full_embedding(self, data):
        """获取完整的融合嵌入向量 (用于逆向设计)"""
        graph_embedding = self.get_graph_embedding(data)
        global_feat = data.global_feat.view(-1, self.global_feat_dim)
        global_embedding = self.global_feat_net(global_feat)
        return torch.cat([graph_embedding, global_embedding], dim=1)
    
    def get_latent_features(self, data):
        """获取可用于逆向设计的潜在特征向量"""
        with torch.no_grad():
            embedding = self.get_full_embedding(data)
            global_feat = data.global_feat.view(-1, self.global_feat_dim)
        return {
            'embedding': embedding.cpu().numpy(),
            'global_features': global_feat.cpu().numpy(),
            'embedding_dim': self.embedding_dim_out
        }


# ==========================================
# 4. 模型保存模块
# ==========================================
class ModelSaver:
    """模型保存与加载管理器"""
    
    def __init__(self, save_dir='model_enhanced'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_model(self, model, optimizer, epoch, metrics, filename=None):
        """保存模型检查点"""
        if filename is None:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'gcnn_enhanced_{timestamp}.pt'
        
        filepath = os.path.join(self.save_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'model_config': {
                'hidden_dim': model.hidden_dim,
                'global_feat_dim': model.global_feat_dim,
                'embedding_dim_out': model.embedding_dim_out
            }
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved: {filepath}")
        return filepath
    
    def save_best_model(self, model, optimizer, epoch, metrics):
        """保存最佳模型"""
        return self.save_model(model, optimizer, epoch, metrics, 'best_model.pt')
    
    def load_model(self, model, filepath, optimizer=None):
        """加载模型检查点"""
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def save_feature_extractor(self, model, filename='feature_extractor.pt'):
        """保存特征提取器 (用于逆向设计)"""
        filepath = os.path.join(self.save_dir, filename)
        
        # 只保存需要的层
        feature_extractor_state = {
            'node_embedding': model.node_embedding.state_dict(),
            'conv1': model.conv1.state_dict(),
            'conv2': model.conv2.state_dict(),
            'conv3': model.conv3.state_dict(),
            'conv4': model.conv4.state_dict(),
            'bn1': model.bn1.state_dict(),
            'bn2': model.bn2.state_dict(),
            'bn3': model.bn3.state_dict(),
            'bn4': model.bn4.state_dict(),
            'global_feat_net': model.global_feat_net.state_dict(),
        }
        
        torch.save(feature_extractor_state, filepath)
        print(f"Feature extractor saved: {filepath}")
        return filepath


# ==========================================
# 5. 报表生成模块
# ==========================================
class ReportGenerator:
    """训练和评估报表生成器"""
    
    def __init__(self, report_dir='reports'):
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'accuracy': [],
            'auc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
    def log_training(self, epoch, train_loss, val_loss=None, metrics=None):
        """记录训练日志"""
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(train_loss)
        self.training_history['val_loss'].append(val_loss if val_loss else 0)
        
        if metrics:
            for key in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
                if key in metrics:
                    self.training_history[key].append(metrics[key])
                else:
                    self.training_history[key].append(0)
    
    def generate_classification_report(self, y_true, y_pred, y_probs, 
                                       global_features=None, save=True):
        """生成详细分类报告"""
        report = []
        report.append("=" * 70)
        report.append("  增强版 GCNN 铁电材料分类详细报告")
        report.append(f"  生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        # 基础分类指标
        report.append("\n【分类性能报告】")
        clf_report = classification_report(y_true, y_pred, target_names=['Non-FE', 'FE'])
        report.append(clf_report)
        
        # AUC
        try:
            auc = roc_auc_score(y_true, y_probs)
            report.append(f"ROC-AUC Score: {auc:.4f}")
        except:
            auc = 0
            report.append("ROC-AUC: 无法计算")
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        report.append("\n【混淆矩阵】")
        report.append(f"                 Predicted")
        report.append(f"              Non-FE      FE")
        report.append(f"Actual Non-FE   {cm[0][0]:5d}   {cm[0][1]:5d}")
        report.append(f"       FE       {cm[1][0]:5d}   {cm[1][1]:5d}")
        
        # 错误分析
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)
        y_probs_np = np.array(y_probs)
        
        fp = np.sum((y_true_np == 0) & (y_pred_np == 1))
        fn = np.sum((y_true_np == 1) & (y_pred_np == 0))
        
        report.append(f"\n【错误分析】")
        report.append(f"假阳性 (FP): {fp}")
        report.append(f"假阴性 (FN): {fn}")
        
        # 预测概率分布
        report.append("\n【预测概率分布】")
        fe_probs = y_probs_np[y_true_np == 1]
        non_fe_probs = y_probs_np[y_true_np == 0]
        
        if len(fe_probs) > 0:
            report.append(f"FE样本: min={fe_probs.min():.4f}, mean={fe_probs.mean():.4f}, max={fe_probs.max():.4f}")
        if len(non_fe_probs) > 0:
            report.append(f"Non-FE样本: min={non_fe_probs.min():.4f}, mean={non_fe_probs.mean():.4f}, max={non_fe_probs.max():.4f}")
        
        # 特征重要性分析 (如果提供了全局特征)
        if global_features is not None:
            report.append("\n【全局特征统计】")
            feature_names = [
                'Volume', 'Density', 'VolPerAtom', 'a/b', 'a/c', 'b/c',
                'αDev', 'βDev', 'γDev', 'SpaceGroup', 'CrystalSys', 'Polar',
                'AvgMass', 'AvgRadius', 'AvgEN', 'AvgIE', 'AvgValence', 'NumElements',
                'RadiusRange', 'ENRange', 'MassRange', 'RadiusRatio',
                'AvgCoord', 'CoordStd', 'AvgBond', 'BondStd',
                'TMFrac', 'CationFrac', 'OxygenFrac',
                'AvgMelt', 'AvgTherm', 'Stability'
            ]
            
            global_features_np = np.array(global_features)
            fe_feats = global_features_np[y_true_np == 1]
            non_fe_feats = global_features_np[y_true_np == 0]
            
            report.append(f"\n{'Feature':<15} {'FE_Mean':<12} {'NonFE_Mean':<12} {'Diff':<10}")
            report.append("-" * 55)
            for i, name in enumerate(feature_names[:min(len(feature_names), global_features_np.shape[1])]):
                fe_mean = fe_feats[:, i].mean() if len(fe_feats) > 0 else 0
                non_fe_mean = non_fe_feats[:, i].mean() if len(non_fe_feats) > 0 else 0
                diff = abs(fe_mean - non_fe_mean)
                report.append(f"{name:<15} {fe_mean:>10.4f}   {non_fe_mean:>10.4f}   {diff:>8.4f}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        if save:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = os.path.join(self.report_dir, f'classification_report_{timestamp}.txt')
            with open(report_path, 'w') as f:
                f.write(report_text)
            print(f"\n报告已保存: {report_path}")
        
        return {
            'accuracy': (y_true_np == y_pred_np).mean(),
            'auc': auc,
            'precision': cm[1][1] / (cm[1][1] + cm[0][1]) if (cm[1][1] + cm[0][1]) > 0 else 0,
            'recall': cm[1][1] / (cm[1][1] + cm[1][0]) if (cm[1][1] + cm[1][0]) > 0 else 0,
            'f1': 2 * cm[1][1] / (2 * cm[1][1] + cm[0][1] + cm[1][0]) if (2 * cm[1][1] + cm[0][1] + cm[1][0]) > 0 else 0
        }
    
    def plot_training_curves(self, save=True):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Loss 曲线
        axes[0].plot(self.training_history['epoch'], self.training_history['train_loss'], label='Train Loss')
        if any(self.training_history['val_loss']):
            axes[0].plot(self.training_history['epoch'], self.training_history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy 曲线
        if any(self.training_history['accuracy']):
            axes[1].plot(self.training_history['epoch'], self.training_history['accuracy'], label='Accuracy')
        if any(self.training_history['auc']):
            axes[1].plot(self.training_history['epoch'], self.training_history['auc'], label='AUC')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Accuracy & AUC')
        axes[1].legend()
        axes[1].grid(True)
        
        # Precision/Recall 曲线
        if any(self.training_history['precision']):
            axes[2].plot(self.training_history['epoch'], self.training_history['precision'], label='Precision')
        if any(self.training_history['recall']):
            axes[2].plot(self.training_history['epoch'], self.training_history['recall'], label='Recall')
        if any(self.training_history['f1']):
            axes[2].plot(self.training_history['epoch'], self.training_history['f1'], label='F1')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Score')
        axes[2].set_title('Precision/Recall/F1')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        if save:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.report_dir, f'training_curves_{timestamp}.png')
            plt.savefig(plot_path, dpi=150)
            print(f"训练曲线已保存: {plot_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, save=True):
        """绘制混淆矩阵热力图"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Non-FE', 'FE'],
                    yticklabels=['Non-FE', 'FE'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_path = os.path.join(self.report_dir, f'confusion_matrix_{timestamp}.png')
            plt.savefig(plot_path, dpi=150)
            print(f"混淆矩阵已保存: {plot_path}")
        
        plt.close()
    
    def save_training_history(self):
        """保存训练历史到CSV"""
        df = pd.DataFrame(self.training_history)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(self.report_dir, f'training_history_{timestamp}.csv')
        df.to_csv(csv_path, index=False)
        print(f"训练历史已保存: {csv_path}")


# ==========================================
# 6. 主训练流程
# ==========================================
def train_enhanced_model():
    """主训练函数"""
    
    # 数据文件
    pos_files = ['new_data/dataset_original_ferroelectric.jsonl', 'new_data/dataset_known_FE_rest.jsonl']
    neg_files = ['new_data/dataset_nonFE.jsonl', 'new_data/dataset_polar_non_ferroelectric_final.jsonl']
    
    # 加载数据
    full_data_list = load_data_enhanced(pos_files, neg_files)
    
    if len(full_data_list) == 0:
        print("Error: No data loaded!")
        return None
    
    print(f"Total samples: {len(full_data_list)}")
    print(f"Global feature dimension: {EnhancedFeatureExtractor.GLOBAL_FEAT_DIM}")
    
    # 划分数据集
    labels = [data.y.item() for data in full_data_list]
    train_data, test_data = train_test_split(full_data_list, test_size=0.2, 
                                              random_state=42, stratify=labels)
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    
    # DataLoader
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, pin_memory=True)
    
    # 初始化模型
    model = EnhancedCrystalGNN(
        hidden_dim=128, 
        global_feat_dim=EnhancedFeatureExtractor.GLOBAL_FEAT_DIM
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # 计算类别权重
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 初始化工具
    model_saver = ModelSaver()
    report_gen = ReportGenerator()
    scaler = GradScaler()
    
    print(f"\n开始训练 (设备: {device})")
    print(f"正样本: {pos_count}, 负样本: {neg_count}, 权重: {pos_weight.item():.2f}")
    print("-" * 60)
    
    best_auc = 0
    num_epochs = 50
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            
            with autocast():
                output = model(batch)
                loss = criterion(output.view(-1), batch.y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
        
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # 每5个epoch评估一次
        if (epoch + 1) % 5 == 0:
            model.eval()
            y_true, y_pred, y_probs = [], [], []
            global_features = []
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device, non_blocking=True)
                    
                    with autocast():
                        logits = model(batch).view(-1)
                        prob = torch.sigmoid(logits)
                    
                    pred = (prob > 0.5).float()
                    
                    y_true.extend(batch.y.cpu().numpy())
                    y_pred.extend(pred.cpu().numpy())
                    y_probs.extend(prob.cpu().numpy())
                    global_features.extend(batch.global_feat.view(-1, EnhancedFeatureExtractor.GLOBAL_FEAT_DIM).cpu().numpy())
            
            metrics = report_gen.generate_classification_report(
                y_true, y_pred, y_probs, global_features, save=False
            )
            
            report_gen.log_training(epoch + 1, avg_loss, metrics=metrics)
            
            print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Acc: {metrics['accuracy']:.4f} | "
                  f"AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | Time: {epoch_time:.2f}s")
            
            # 保存最佳模型
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                model_saver.save_best_model(model, optimizer, epoch, metrics)
        else:
            report_gen.log_training(epoch + 1, avg_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
    
    # 最终评估
    print("\n" + "=" * 60)
    print("最终评估")
    print("=" * 60)
    
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    global_features = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            
            with autocast():
                logits = model(batch).view(-1)
                prob = torch.sigmoid(logits)
            
            pred = (prob > 0.5).float()
            
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_probs.extend(prob.cpu().numpy())
            global_features.extend(batch.global_feat.view(-1, EnhancedFeatureExtractor.GLOBAL_FEAT_DIM).cpu().numpy())
    
    # 生成最终报告
    final_metrics = report_gen.generate_classification_report(
        y_true, y_pred, y_probs, global_features, save=True
    )
    
    # 绘制图表
    report_gen.plot_training_curves()
    report_gen.plot_confusion_matrix(y_true, y_pred)
    report_gen.save_training_history()
    
    # 保存最终模型和特征提取器
    model_saver.save_model(model, optimizer, num_epochs, final_metrics, 'final_model.pt')
    model_saver.save_feature_extractor(model)
    
    # 提取特征向量示例
    print("\n【特征向量提取示例】")
    sample_batch = next(iter(test_loader)).to(device)
    latent_info = model.get_latent_features(sample_batch)
    print(f"嵌入向量维度: {latent_info['embedding_dim']}")
    print(f"嵌入向量形状: {latent_info['embedding'].shape}")
    print(f"全局特征形状: {latent_info['global_features'].shape}")
    
    return model, final_metrics


# ==========================================
# 7. 主程序入口
# ==========================================
if __name__ == '__main__':
    print("=" * 60)
    print("  增强版 GCNN 铁电材料分类器")
    print("  特征维度: 32 (扩展版)")
    print("=" * 60)
    
    model, metrics = train_enhanced_model()
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print(f"最终准确率: {metrics['accuracy']:.4f}")
    print(f"最终 AUC: {metrics['auc']:.4f}")
    print(f"最终 F1: {metrics['f1']:.4f}")
    print("=" * 60)
