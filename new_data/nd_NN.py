import pandas as pd
import numpy as np
import json
import re
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# ==========================================
# 1. 基础配置与属性库
# ==========================================
# 关键铁电相关元素 + 数据集中常见元素
KEY_ELEMENTS = ['O', 'Ti', 'Ba', 'Pb', 'Sr', 'Zr', 'Hf', 'Nb', 'Ta', 'Bi', 
                'K', 'Na', 'Li', 'H', 'C', 'F', 'Cl']

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
tm_set = set(['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'])

# ==========================================
# 2. 增强型特征计算 (物理14维 + 关键元素17维 = 31维)
# ==========================================
def compute_features_v2(composition_dict, bandgap):
    if not composition_dict: return None
    total_atoms = sum(composition_dict.values())
    if total_atoms == 0: return None
    
    # --- Part A: 物理统计特征 (14维) ---
    props = [[] for _ in range(5)]
    tm_fraction = 0.0
    for el, count in composition_dict.items():
        if el in element_data:
            data = element_data[el]
            frac = count / total_atoms
            for i in range(5):
                props[i].append((data[i+1], frac))
            if el in tm_set: tm_fraction += frac
            
    if not props[0]: return None
    
    feats = []
    # 1-5: 平均值
    for i in range(5):
        vals = [p[0] for p in props[i]]
        weights = [p[1] for p in props[i]]
        feats.append(np.average(vals, weights=weights))
    # 6-11: 极差统计
    radii = [p[0] for p in props[1]]
    ens = [p[0] for p in props[2]]
    feats.extend([max(radii), min(radii), max(radii)-min(radii)])
    feats.extend([max(ens), min(ens), max(ens)-min(ens)])
    # 12-14: 其他
    feats.extend([tm_fraction, len(composition_dict), bandgap])
    
    # --- Part B: 关键元素特征 (One-Hot like) ---
    # 检查材料中是否包含关键元素 (0 或 1)
    element_presence = []
    for key_el in KEY_ELEMENTS:
        element_presence.append(1.0 if key_el in composition_dict else 0.0)
    
    feats.extend(element_presence)
    
    return feats

# 辅助解析函数
def parse_formula(formula):
    def get_composition(formula_str):
        formula_str = formula_str.strip()
        while '(' in formula_str:
            m = re.search(r'\(([^()]+)\)([\d\.]*)', formula_str)
            if m:
                content = m.group(1)
                mult = float(m.group(2)) if m.group(2) else 1.0
                sub_comp = get_composition(content)
                expanded = ""
                for el, count in sub_comp.items():
                    expanded += f"{el}{count * mult}"
                formula_str = formula_str.replace(m.group(0), expanded)
            else: break
        matches = re.findall(r'([A-Z][a-z]*)([\d\.]*)', formula_str)
        comp = {}
        for el, count_str in matches:
            count = float(count_str) if count_str else 1.0
            comp[el] = comp.get(el, 0) + count
        return comp
    return get_composition(formula)

# ==========================================
# 3. 数据处理
# ==========================================
print("正在处理数据...")
X_list = []
y_list = []

# CSV
df_csv = pd.read_csv('ferroelectric_database_labeled.csv')
for _, row in df_csv.iterrows():
    try:
        comp = parse_formula(row['Material_Name'])
        bg = row['Bandgap (eV)']
        if pd.isna(bg): bg = 0
        feats = compute_features_v2(comp, bg)
        label = 1 if row['Label'] == 'ferroelectric' else 0
        if feats:
            X_list.append(feats)
            y_list.append(label)
    except: pass

# JSONL
with open('new_data/dataset_nonFE.jsonl', 'r') as f:
    for line in f:
        try:
            item = json.loads(line)
            sites = item['structure']['sites']
            comp = {}
            for site in sites:
                for species in site['species']:
                    el = species['element']
                    occu = species['occu']
                    comp[el] = comp.get(el, 0) + occu
            bg = item['band_gap']
            if bg is None: bg = 0
            feats = compute_features_v2(comp, bg)
            label = 0 
            if feats:
                X_list.append(feats)
                y_list.append(label)
        except: pass

X = np.array(X_list)
y = np.array(y_list)
print(f"完整数据集形状: {X.shape}, 正样本: {sum(y==1)}, 负样本: {sum(y==0)}")

# ==========================================
# 4. 集成学习：Balanced Bagging + Gradient Boosting
# ==========================================
# 切分训练/测试集 (保留20%作为最终验证，不参与任何训练)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 自定义 Balanced Bagging 逻辑
n_estimators = 20  # 训练20个模型
models = []

pos_idx = np.where(y_train == 1)[0]
neg_idx = np.where(y_train == 0)[0]
n_pos = len(pos_idx)

print(f"\n开始训练集成模型 (共 {n_estimators} 个子模型)...")
for i in range(n_estimators):
    # 1. 采样: 取所有正样本 + 随机取 1.5倍 数量的负样本
    # 注意：每次循环负样本都是重新随机抽取的，确保覆盖面
    sample_neg_idx = np.random.choice(neg_idx, int(n_pos * 1.5), replace=False)
    sample_idx = np.concatenate([pos_idx, sample_neg_idx])
    np.random.shuffle(sample_idx)
    
    X_subset = X_train_scaled[sample_idx]
    y_subset = y_train[sample_idx]
    
    # 2. 训练子模型: 使用 Gradient Boosting (比 MLP 更适合此类特征)
    clf = GradientBoostingClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=5, 
        random_state=42+i
    )
    clf.fit(X_subset, y_subset)
    models.append(clf)

print("训练完成。")

# ==========================================
# 5. 预测与评估 (软投票 Soft Voting)
# ==========================================
# 对测试集，让这20个模型分别预测概率，然后取平均
y_pred_proba_sum = np.zeros(len(y_test))

for clf in models:
    y_pred_proba_sum += clf.predict_proba(X_test_scaled)[:, 1]

y_pred_proba_avg = y_pred_proba_sum / n_estimators
y_pred_final = (y_pred_proba_avg > 0.5).astype(int)

print("\n====== 最终评估报告 (Ensemble GBDT) ======")
print(classification_report(y_test, y_pred_final, target_names=['非铁电', '铁电']))
print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, y_pred_proba_avg):.4f}")