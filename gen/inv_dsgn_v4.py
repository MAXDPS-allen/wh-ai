import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 基础配置与氧化态库
# ==========================================
# [AtomicNumber, Mass, Radius, Electronegativity, IonizationEnergy, Valence_Electron]
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
TM_SET = set(['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'])

# [新增] 常见氧化态库 (用于电荷平衡检查)
COMMON_OXIDATION_STATES = {
    'O': -2, 'S': -2, 'F': -1, 'Cl': -1, 'Br': -1, 'I': -1,
    'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
    'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
    'Ti': 4, 'Zr': 4, 'Hf': 4,
    'V': 5, 'Nb': 5, 'Ta': 5,
    'Cr': 3, 'Mo': 6, 'W': 6,
    'Mn': 4, 'Fe': 3, 'Co': 3, 'Ni': 2,
    'Cu': 2, 'Zn': 2, 'Ag': 1, 'Cd': 2,
    'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 1,
    'Si': 4, 'Ge': 4, 'Sn': 4, 'Pb': 2,
    'Bi': 3, 'La': 3, 'Y': 3
}

FEATURE_NAMES = [
    'Avg_Mass', 'Avg_Radius', 'Avg_EN', 'Avg_IE', 'Avg_Valence',
    'Max_Radius', 'Min_Radius', 'Range_Radius', 'Ratio_Radius',
    'Range_EN',
    'Lattice_Distortion_Angle', 'Lattice_Anisotropy', 'Density', 'Vol_Per_Atom',
    'SpaceGroup_Number', 'TM_Fraction', 'Num_Elements', 'Bandgap'
]

# ==========================================
# 2. 特征计算函数 (复用)
# ==========================================
def compute_features(item):
    if 'structure' not in item: return None
    struct = item['structure']
    sites = struct.get('sites', [])
    lattice = struct.get('lattice', {})
    comp = {}
    total_atoms = 0
    for site in sites:
        for species in site['species']:
            el = species['element']; occu = species['occu']
            comp[el] = comp.get(el, 0) + occu; total_atoms += occu
    if total_atoms == 0: return None
    props = [[] for _ in range(5)]; tm_fraction = 0.0; current_mass_sum = 0
    for el, count in comp.items():
        if el in element_data:
            data = element_data[el]; frac = count / total_atoms
            current_mass_sum += data[1] * count
            for i in range(5): props[i].append((data[i+1], frac))
            if el in TM_SET: tm_fraction += frac
    if not props[0]: return None
    feats = []
    for i in range(5):
        vals = [p[0] for p in props[i]]; weights = [p[1] for p in props[i]]
        feats.append(np.average(vals, weights=weights))
    radii = [p[0] for p in props[1]]
    max_r, min_r = max(radii), min(radii)
    feats.extend([max_r, min_r, max_r - min_r]); feats.append(max_r / min_r if min_r > 0 else 0)
    ens = [p[0] for p in props[2]]
    max_en, min_en = max(ens), min(ens); feats.extend([max_en - min_en])
    a, b, c = lattice.get('a', 0), lattice.get('b', 0), lattice.get('c', 0)
    alpha, beta, gamma = lattice.get('alpha', 90), lattice.get('beta', 90), lattice.get('gamma', 90)
    vol = lattice.get('volume', 0)
    angle_distortion = abs(alpha - 90) + abs(beta - 90) + abs(gamma - 90); feats.append(angle_distortion)
    lengths = [x for x in [a, b, c] if x > 0]
    feats.append(max(lengths) / min(lengths) if lengths else 1.0)
    density = current_mass_sum / vol if vol > 0 else 0; feats.append(density)
    vol_per_atom = vol / total_atoms if total_atoms > 0 else 0; feats.append(vol_per_atom)
    sg = item.get('spacegroup_number', 0); feats.append(sg if sg is not None else 0)
    feats.append(tm_fraction); feats.append(len(comp))
    bg = item.get('band_gap'); feats.append(bg if bg is not None else 0)
    return feats

# ==========================================
# 3. 严格的物理过滤器 (Strict Check)
# ==========================================
def is_physically_valid_strict(item):
    """
    严苛的物理过滤器，用于确保训练数据的高质量
    """
    if 'structure' not in item: return False
    struct = item['structure']
    sites = struct.get('sites', [])
    
    total_atoms = 0
    total_charge = 0
    is_integer = True
    
    # 1. 严格的基元限制 (Primary Cell Only)
    # 我们只想要最基本的单元，不要超胞。超胞会增加学习难度。
    # 对于 ABO3，原子数是 5。限制设为 10 以包含双钙钛矿等。
    atom_limit = 12 
    
    for site in sites:
        for species in site['species']:
            el = species['element']
            occu = species['occu']
            
            # 2. 绝对整数占位 (No Partial Occupancy)
            # 容差设为极小 (0.001)
            if abs(occu - round(occu)) > 0.001:
                is_integer = False
            
            total_atoms += occu
            
            # 3. 电荷平衡估算
            if el in COMMON_OXIDATION_STATES:
                total_charge += COMMON_OXIDATION_STATES[el] * occu
            else:
                # 如果遇到未知元素，用简单价电子数估算 (可能会有偏差，但这只是辅助过滤)
                val = element_data[el][5] if el in element_data else 0
                if val > 4: val -= 8 # 假设阴离子
                total_charge += val * occu

    if not is_integer: return False
    if total_atoms > atom_limit: return False
    
    # 电荷不平衡容忍度 (有些材料可能有缺陷或非标准价态)
    # 设为 ±2 以包容一定偏差，但过滤掉极端的错误
    if abs(total_charge) > 3.0: return False
    
    return True

# ==========================================
# 4. 目标提取 (Target Extraction)
# ==========================================
MAX_ELEMENTS = 4
def extract_target_vector_physics(item):
    if 'structure' not in item: return None
    struct = item['structure']; lattice = struct.get('lattice', {})
    a, b, c = lattice.get('a', 0), lattice.get('b', 0), lattice.get('c', 0)
    alpha, beta, gamma = lattice.get('alpha', 90), lattice.get('beta', 90), lattice.get('gamma', 90)
    vol = lattice.get('volume', 0)
    if vol <= 0 or a <= 0: return None
    
    # 物理变换: 体积立方根 + 形状比
    vol_root = vol ** (1/3)
    ratio_b_a = b / a
    ratio_c_a = c / a
    lattice_targets = [vol_root, ratio_b_a, ratio_c_a, alpha, beta, gamma]
    
    comp = {}; total_atoms = 0; sites = struct.get('sites', [])
    for site in sites:
        for species in site['species']:
            el = species['element']; occu = species['occu']
            comp[el] = comp.get(el, 0) + occu; total_atoms += occu
    el_list = []
    for el, count in comp.items():
        if el in element_data:
            z = element_data[el][0]; frac = count / total_atoms
            el_list.append((z, frac))
    el_list.sort(key=lambda x: x[0])
    comp_vector = []
    for i in range(MAX_ELEMENTS):
        if i < len(el_list): comp_vector.extend(el_list[i])
        else: comp_vector.extend([0, 0])
    return lattice_targets + comp_vector

# ==========================================
# 5. 加载数据并应用权重 (Weighted Loading)
# ==========================================
# 你的文件列表
files_config = {
    'new_data/dataset_known_FE_rest.jsonl': 10.0,           # [权重] 已知铁电 (最重要)
    'new_data/dataset_original_ferroelectric.jsonl': 10.0,  # [权重] 原始铁电 (最重要)
    'new_data/dataset_polar_non_ferroelectric_final.jsonl': 2.0, # [权重] 极性非铁电 (次重要，用于学习极性结构)
    'new_data/dataset_nonFE.jsonl': 1.0                     # [权重] 非铁电 (基准)
}

X_all, Y_all, weights_all = [], [], []

print("Loading data with Physics Filters and Importance Weighting...")
stats = {'valid': 0, 'dropped': 0}

for f_name, weight in files_config.items():
    print(f"  - Processing {f_name} (Weight={weight})...")
    try:
        with open(f_name, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    
                    # 严苛的物理过滤
                    if not is_physically_valid_strict(item):
                        stats['dropped'] += 1
                        continue
                    
                    feat = compute_features(item)
                    target = extract_target_vector_physics(item)
                    
                    if feat and target:
                        X_all.append(feat)
                        Y_all.append(target)
                        weights_all.append(weight) # 记录该样本的权重
                        stats['valid'] += 1
                except: continue
    except FileNotFoundError:
        print(f"    [Warning] File not found: {f_name}")

print(f"\nData Summary: {stats['valid']} samples loaded, {stats['dropped']} samples dropped (strict filter).")

X_all = np.array(X_all)
Y_all = np.array(Y_all)
weights_all = np.array(weights_all)

# 划分训练集/测试集 (同时划分权重)
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X_all, Y_all, weights_all, test_size=0.1, random_state=42
)

# ==========================================
# 6. 训练加权模型 (Weighted Training)
# ==========================================
categorical_features = [14] # SpaceGroup
numerical_features = [i for i in range(len(FEATURE_NAMES)) if i not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), categorical_features)
    ])

# XGBoost 支持 sample_weight
xgb_estimator = xgb.XGBRegressor(
    n_estimators=800,       # 更多树，拟合更精细
    learning_rate=0.03,     # 更慢的学习率，更稳健
    max_depth=8,            # 允许捕捉更复杂的相互作用
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MultiOutputRegressor(xgb_estimator))
])

print("\nTraining Weighted XGBoost Model...")
# 注意：MultiOutputRegressor 的 fit 方法支持 fit_params
# 我们需要按照 sklearn 的规范传递 sample_weight
# 格式通常是: model.fit(X, y, regressor__sample_weight=w)
# 但 MultiOutputRegressor 内部也是一个个 estimator，比较复杂。
# 最简单且稳健的方法是直接对内部的每个 regressor 传递。
# 然而 Pipeline 封装了一层。
# 这里的标准做法是使用 fit_params 字典传递给最后一步。

# 修正：直接传递给 Pipeline 的 fit 方法
# 语法: stepname__parametername
model.fit(X_train, y_train, **{'regressor__sample_weight': w_train})
print("Training Completed.")

# ==========================================
# 7. 全面分析与保存模块
# ==========================================
class ComprehensiveAnalyzer:
    def __init__(self, model, X_test, y_test, feature_names):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.y_pred = model.predict(X_test)
        
    def restore_lattice_params(self, y_vec):
        vol_root = y_vec[:, 0]
        ratio_ba = np.maximum(y_vec[:, 1], 1e-4)
        ratio_ca = np.maximum(y_vec[:, 2], 1e-4)
        a = vol_root / (ratio_ba * ratio_ca)**(1/3)
        b = a * ratio_ba
        c = a * ratio_ca
        return np.stack([a, b, c], axis=1)

    def generate_accuracy_report(self):
        print("\n====== 最终准确率报表 (Weighted Model) ======")
        z_indices = [6, 8, 10, 12] # Z1-Z4
        total = len(self.y_test)
        
        print("[化学成分]")
        for i, idx in enumerate(z_indices):
            real_z = self.y_test[:, idx]
            pred_z = self.y_pred[:, idx]
            # 计算绝对命中 (偏差 < 0.8)
            hits = np.sum(np.abs(real_z - pred_z) < 0.8)
            mae = mean_absolute_error(real_z, pred_z)
            valid_mask = real_z > 0
            if np.sum(valid_mask) > 0:
                acc = hits / total
                print(f"  Element {i+1}: MAE={mae:.2f}, Hit Rate={acc:.1%}")

        print("[晶格参数]")
        real_abc = self.restore_lattice_params(self.y_test)
        pred_abc = self.restore_lattice_params(self.y_pred)
        for i, lab in enumerate(['a', 'b', 'c']):
            mape = np.mean(np.abs((real_abc[:, i] - pred_abc[:, i]) / real_abc[:, i]))
            r2 = r2_score(real_abc[:, i], pred_abc[:, i])
            print(f"  Lattice {lab}: MAPE={mape:.1%}, R2={r2:.3f}")
            
    def save_analysis_plots(self):
        # 简单绘制 Parity Plot
        real_abc = self.restore_lattice_params(self.y_test)
        pred_abc = self.restore_lattice_params(self.y_pred)
        plt.figure(figsize=(15, 5))
        for i, lab in enumerate(['a', 'b', 'c']):
            plt.subplot(1, 3, i+1)
            plt.scatter(real_abc[:, i], pred_abc[:, i], alpha=0.5, s=10)
            plt.plot([0, 15], [0, 15], 'r--')
            plt.title(f'Lattice {lab}')
        plt.tight_layout()
        plt.savefig('final_analysis_parity.png')
        print(">> Plots saved to final_analysis_parity.png")

# 执行分析
analyzer = ComprehensiveAnalyzer(model, X_test, y_test, FEATURE_NAMES)
analyzer.generate_accuracy_report()
analyzer.save_analysis_plots()

# 保存模型
print("\n====== Saving Weighted Model ======")
joblib.dump(model, 'invs_dgn_model/inverse_design_weighted_v5.pkl')
print("Model saved: invs_dgn_model/inverse_design_weighted_v5.pkl")