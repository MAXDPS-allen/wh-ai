import pandas as pd
import numpy as np
import json
import re
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# ==========================================
# 1. 元素属性数据库 (用于化学特征)
# ==========================================
# 包含：原子序数, 原子质量, 原子半径, 电负性, 电离能, 价电子数
element_data = {
    'H': [1, 1.008, 0.37, 2.20, 13.598, 1], 'He': [2, 4.0026, 0.32, 0.0, 24.587, 8], 
    'Li': [3, 6.94, 1.34, 0.98, 5.392, 1], 'Be': [4, 9.0122, 0.90, 1.57, 9.323, 2],
    'B': [5, 10.81, 0.82, 2.04, 8.298, 3], 'C': [6, 12.011, 0.77, 2.55, 11.260, 4], 
    'N': [7, 14.007, 0.75, 3.04, 14.534, 5], 'O': [8, 15.999, 0.73, 3.44, 13.618, 6],
    'F': [9, 18.998, 0.71, 3.98, 17.423, 7], 'Ne': [10, 20.180, 0.69, 0.0, 21.565, 8], 
    'Na': [11, 22.990, 1.54, 0.93, 5.139, 1], 'Mg': [12, 24.305, 1.30, 1.31, 7.646, 2],
    'Al': [13, 26.982, 1.18, 1.61, 5.986, 3], 'Si': [14, 28.085, 1.11, 1.90, 8.152, 4], 
    'P': [15, 30.974, 1.06, 2.19, 10.487, 5], 'S': [16, 32.06, 1.02, 2.58, 10.360, 6],
    'Cl': [17, 35.45, 0.99, 3.16, 12.968, 7], 'K': [19, 39.098, 1.96, 0.82, 4.341, 1], 
    'Ca': [20, 40.078, 1.74, 1.00, 6.113, 2], 'Sc': [21, 44.956, 1.44, 1.36, 6.561, 3],
    'Ti': [22, 47.867, 1.36, 1.54, 6.828, 4], 'V': [23, 50.942, 1.25, 1.63, 6.746, 5], 
    'Cr': [24, 51.996, 1.27, 1.66, 6.767, 6], 'Mn': [25, 54.938, 1.39, 1.55, 7.434, 7],
    'Fe': [26, 55.845, 1.25, 1.83, 7.902, 8], 'Co': [27, 58.933, 1.26, 1.88, 7.881, 9], 
    'Ni': [28, 58.693, 1.21, 1.91, 7.640, 10], 'Cu': [29, 63.546, 1.38, 1.90, 7.726, 11],
    'Zn': [30, 65.38, 1.31, 1.65, 9.394, 12], 'Ga': [31, 69.723, 1.26, 1.81, 6.000, 3], 
    'Ge': [32, 72.63, 1.22, 2.01, 7.900, 4], 'As': [33, 74.922, 1.19, 2.18, 9.789, 5],
    'Se': [34, 78.96, 1.16, 2.55, 9.752, 6], 'Br': [35, 79.904, 1.14, 2.96, 11.814, 7], 
    'Rb': [37, 85.468, 2.11, 0.82, 4.177, 1], 'Sr': [38, 87.62, 1.92, 0.95, 5.695, 2],
    'Y': [39, 88.906, 1.62, 1.22, 6.217, 3], 'Zr': [40, 91.224, 1.48, 1.33, 6.634, 4], 
    'Nb': [41, 92.906, 1.37, 1.60, 6.759, 5], 'Mo': [42, 95.96, 1.45, 2.16, 7.092, 6],
    'Tc': [43, 98.0, 1.56, 1.90, 7.28, 7], 'Ru': [44, 101.07, 1.26, 2.20, 7.361, 8], 
    'Rh': [45, 102.91, 1.35, 2.28, 7.459, 9], 'Pd': [46, 106.42, 1.31, 2.20, 8.337, 10],
    'Ag': [47, 107.87, 1.53, 1.93, 7.576, 11], 'Cd': [48, 112.41, 1.48, 1.69, 8.994, 12], 
    'In': [49, 114.82, 1.44, 1.78, 5.786, 3], 'Sn': [50, 118.71, 1.41, 1.96, 7.344, 4],
    'Sb': [51, 121.76, 1.38, 2.05, 8.64, 5], 'Te': [52, 127.60, 1.35, 2.10, 9.010, 6], 
    'I': [53, 126.90, 1.33, 2.66, 10.451, 7], 'Cs': [55, 132.91, 2.25, 0.79, 3.894, 1],
    'Ba': [56, 137.33, 1.98, 0.89, 5.212, 2], 'La': [57, 138.91, 1.69, 1.10, 5.577, 3], 
    'Ce': [58, 140.12, 1.65, 1.12, 5.539, 3], 'Pr': [59, 140.91, 1.65, 1.13, 5.464, 3],
    'Nd': [60, 144.24, 1.64, 1.14, 5.525, 3], 'Pm': [61, 145.0, 1.63, 1.13, 5.55, 3], 
    'Sm': [62, 150.36, 1.62, 1.17, 5.644, 3], 'Eu': [63, 151.96, 1.85, 1.20, 5.670, 2],
    'Gd': [64, 157.25, 1.61, 1.20, 6.150, 3], 'Tb': [65, 158.93, 1.59, 1.10, 5.864, 3], 
    'Dy': [66, 162.50, 1.59, 1.22, 5.939, 3], 'Ho': [67, 164.93, 1.58, 1.23, 6.022, 3],
    'Er': [68, 167.26, 1.57, 1.24, 6.108, 3], 'Tm': [69, 168.93, 1.56, 1.25, 6.184, 3], 
    'Yb': [70, 173.05, 1.74, 1.10, 6.254, 2], 'Lu': [71, 174.97, 1.56, 1.27, 5.426, 3],
    'Hf': [72, 178.49, 1.44, 1.30, 6.825, 4], 'Ta': [73, 180.95, 1.34, 1.50, 7.89, 5], 
    'W': [74, 183.84, 1.30, 2.36, 7.98, 6], 'Re': [75, 186.21, 1.28, 1.90, 7.88, 7],
    'Os': [76, 190.23, 1.26, 2.20, 8.7, 8], 'Ir': [77, 192.22, 1.27, 2.20, 9.1, 9], 
    'Pt': [78, 195.08, 1.30, 2.28, 9.0, 10], 'Au': [79, 196.97, 1.34, 2.54, 9.226, 11],
    'Hg': [80, 200.59, 1.49, 2.00, 10.438, 12], 'Tl': [81, 204.38, 1.48, 1.62, 6.108, 3], 
    'Pb': [82, 207.2, 1.47, 2.33, 7.417, 4], 'Bi': [83, 208.98, 1.46, 2.02, 7.289, 5],
    'Th': [90, 232.04, 1.79, 1.30, 6.08, 4], 'Pa': [91, 231.04, 1.63, 1.50, 5.89, 5], 
    'U': [92, 238.03, 1.56, 1.38, 6.194, 6]
}

# 常见铁电相关元素
KEY_ELEMENTS = ['O', 'Ti', 'Ba', 'Pb', 'Sr', 'Zr', 'Hf', 'Nb', 'Ta', 'Bi', 'K', 'Li']
TM_SET = set(['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'])

# ==========================================
# 2. 特征工程核心函数 (Structure + Chemistry)
# ==========================================
def compute_features(item):
    """
    输入: 单个 JSON 对象
    输出: 特征列表 (list) 或 None
    """
    if 'structure' not in item:
        return None
        
    struct = item['structure']
    sites = struct.get('sites', [])
    lattice = struct.get('lattice', {})
    
    # --- A. 提取化学成分 ---
    comp = {}
    total_atoms = 0
    for site in sites:
        for species in site['species']:
            el = species['element']
            occu = species['occu']
            comp[el] = comp.get(el, 0) + occu
            total_atoms += occu
            
    if total_atoms == 0: return None
    
    # --- B. 计算化学统计特征 (Mass, Radius, EN, etc.) ---
    # 属性: 1:Mass, 2:Radius, 3:EN, 4:IE, 5:Valence
    props = [[] for _ in range(5)]
    tm_fraction = 0.0
    
    current_mass_sum = 0 # 用于计算密度
    
    for el, count in comp.items():
        if el in element_data:
            data = element_data[el]
            frac = count / total_atoms
            current_mass_sum += data[1] * count # Mass * count
            
            for i in range(5):
                props[i].append((data[i+1], frac))
            
            if el in TM_SET:
                tm_fraction += frac

    if not props[0]: return None
    
    feats = []
    
    # 1. 基础化学属性加权平均 (5维)
    for i in range(5):
        vals = [p[0] for p in props[i]]
        weights = [p[1] for p in props[i]]
        feats.append(np.average(vals, weights=weights))
        
    # 2. 关键属性的极差与比值 (6维)
    # Radius
    radii = [p[0] for p in props[1]]
    max_r, min_r = max(radii), min(radii)
    feats.extend([max_r, min_r, max_r - min_r])
    feats.append(max_r / min_r if min_r > 0 else 0) # 尺寸失配度
    
    # Electronegativity
    ens = [p[0] for p in props[2]]
    max_en, min_en = max(ens), min(ens)
    feats.extend([max_en - min_en]) # 键的离子性
    
    # --- C. 计算结构物理特征 (关键升级) ---
    
    # 3. 晶格参数与畸变 (Structure Info)
    # lattice keys: a, b, c, alpha, beta, gamma, volume
    a, b, c = lattice.get('a', 0), lattice.get('b', 0), lattice.get('c', 0)
    alpha, beta, gamma = lattice.get('alpha', 90), lattice.get('beta', 90), lattice.get('gamma', 90)
    vol = lattice.get('volume', 0)
    
    # (1) 晶格角度畸变度 (偏离90度的程度) - 铁电体常为三方/单斜/四方畸变
    angle_distortion = abs(alpha - 90) + abs(beta - 90) + abs(gamma - 90)
    feats.append(angle_distortion)
    
    # (2) 晶格各向异性 (长轴/短轴)
    lengths = [x for x in [a, b, c] if x > 0]
    if lengths:
        feats.append(max(lengths) / min(lengths))
    else:
        feats.append(1.0)
        
    # (3) 密度与空间占位
    # 密度 = 质量 / 体积
    density = current_mass_sum / vol if vol > 0 else 0
    feats.append(density)
    
    # 原子平均体积 (反映原子堆积紧密程度)
    vol_per_atom = vol / total_atoms if total_atoms > 0 else 0
    feats.append(vol_per_atom)
    
    # (4) 空间群 (Space Group)
    # 优先从顶层获取，如果没有则设为 0 (缺失值)
    sg = item.get('spacegroup_number', 0)
    if sg is None: sg = 0
    feats.append(sg)

    # --- D. 其他特征 ---
    feats.append(tm_fraction) # 过渡金属含量
    feats.append(len(comp))   # 元素种类数
    
    # Bandgap
    bg = item.get('band_gap')
    if bg is None: bg = 0
    feats.append(bg)
    
    # One-Hot 关键元素 (Key Elements Presence)
    for k_el in KEY_ELEMENTS:
        feats.append(1.0 if k_el in comp else 0.0)
        
    return feats

# ==========================================
# 3. 数据加载与处理
# ==========================================
print("正在加载并解析数据...")

# 文件列表 (请根据实际情况修改文件名)
pos_files = ['new_data/dataset_original_ferroelectric.jsonl', 'new_data/dataset_known_FE_rest.jsonl']
neg_file = 'new_data/dataset_nonFE.jsonl' # 或者 'dataset_nonFE.jsonl'

X_pos, y_pos = [], []
X_neg, y_neg = [], []

# 加载正样本
for f_name in pos_files:
    try:
        with open(f_name, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    feats = compute_features(item)
                    if feats:
                        X_pos.append(feats)
                        y_pos.append(1)
                except: continue
    except FileNotFoundError:
        print(f"Warning: 文件 {f_name} 未找到。")

# 加载负样本
try:
    with open(neg_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                feats = compute_features(item)
                if feats:
                    X_neg.append(feats)
                    y_neg.append(0)
            except: continue
except FileNotFoundError:
    print(f"Warning: 文件 {neg_file} 未找到。")

X = np.array(X_pos + X_neg)
y = np.array(y_pos + y_neg)

print(f"数据加载完成: 正样本 {len(X_pos)}, 负样本 {len(X_neg)}, 总特征数 {X.shape[1]}")

# ==========================================
# 4. 训练神经网络 (MLP) + Balanced Bagging
# ==========================================
# 切分测试集 (保留 20% 纯净数据用于验证)
X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 数据标准化 (神经网络对尺度非常敏感)
scaler = StandardScaler()
X_train_scaled_all = scaler.fit_transform(X_train_all)
X_test_scaled = scaler.transform(X_test)

# Bagging 配置
n_estimators = 10  # 训练 10 个神经网络子模型
models = []

pos_idx = np.where(y_train_all == 1)[0]
neg_idx = np.where(y_train_all == 0)[0]
n_pos = len(pos_idx)

print(f"开始训练 {n_estimators} 个 Bagging MLP 模型...")

for i in range(n_estimators):
    # 1. 下采样负样本 (1:3 比例，即每个正样本配3个负样本)
    # 这样既保证了正样本权重，又让网络看更多的负样本
    sample_neg_idx = np.random.choice(neg_idx, int(n_pos * 3.0), replace=False) # 如果负样本不够会报错，但这里5000>>600*3，安全
    sample_idx = np.concatenate([pos_idx, sample_neg_idx])
    np.random.shuffle(sample_idx)
    
    X_sub = X_train_scaled_all[sample_idx]
    y_sub = y_train_all[sample_idx]
    
    # 2. 构建神经网络 (MLP)
    # 结构: 隐藏层 (100, 50), ReLU 激活, L2 正则化 (alpha=0.01) 防止过拟合
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.01,
        max_iter=500,
        early_stopping=True, # 早停法
        random_state=42 + i
    )
    mlp.fit(X_sub, y_sub)
    models.append(mlp)

# ==========================================
# 5. 集成预测与评估
# ==========================================
# 对测试集进行平均概率投票
y_prob_sum = np.zeros(len(y_test))
for clf in models:
    y_prob_sum += clf.predict_proba(X_test_scaled)[:, 1]

y_prob_avg = y_prob_sum / n_estimators

# 使用 0.5 作为阈值 (或者可以调整为更高以提高 Precision)
y_pred = (y_prob_avg >= 0.42).astype(int)

print("\n====== 最终模型评估 (Neural Network Ensemble) ======")
print(classification_report(y_test, y_pred, target_names=['Non-Ferroelectric', 'Ferroelectric']))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob_avg):.4f}")

# 保存模型 (可选)
# import joblib
# joblib.dump(models, 'ensemble_mlp_models.pkl')
# joblib.dump(scaler, 'scaler.pkl')