import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 准备环境 & 加载模型
# ==========================================
# 加载之前保存的 XGBoost Pipeline 和 工具
try:
    model = joblib.load('invs_dgn_model/inverse_design_physics_v5.pkl')
    # 注意：XGBoost Pipeline 已经包含了 Preprocessor (One-Hot等)，所以不需要手动处理 X
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'invs_dgn_model/inverse_design_physics_v5.pkl' not found. Please ensure training was successful.")
    exit()

# 加载候选特征数据
try:
    df_gen = pd.read_csv('generated_candidates_v3.csv')
    print(f"Loaded {len(df_gen)} generated candidates.")
except FileNotFoundError:
    print("Error: 'generated_candidates_v3.csv' not found.")
    exit()

# ==========================================
# 2. 预测
# ==========================================
# 确保列名与训练时一致 (排除 Predicted_FE_Prob 列)
# 训练时用的 FEATURE_NAMES (v3_XGB_save.py 定义的)
FEATURE_NAMES = [
    'Avg_Mass', 'Avg_Radius', 'Avg_EN', 'Avg_IE', 'Avg_Valence',
    'Max_Radius', 'Min_Radius', 'Range_Radius', 'Ratio_Radius',
    'Range_EN',
    'Lattice_Distortion_Angle', 'Lattice_Anisotropy', 'Density', 'Vol_Per_Atom',
    'SpaceGroup_Number', 'TM_Fraction', 'Num_Elements', 'Bandgap'
]

# 提取特征矩阵
X_gen = df_gen[FEATURE_NAMES]

# 核心预测
print("Running inverse design prediction...")
y_pred = model.predict(X_gen)

# ==========================================
# 3. 结果解析与化学式重构
# ==========================================
# 元素周期表映射 (Atomic Z -> Symbol)
element_data = {
    'H': [1], 'He': [2], 'Li': [3], 'Be': [4], 'B': [5], 'C': [6], 'N': [7], 'O': [8],
    'F': [9], 'Ne': [10], 'Na': [11], 'Mg': [12], 'Al': [13], 'Si': [14], 'P': [15], 
    'S': [16], 'Cl': [17], 'K': [19], 'Ca': [20], 'Sc': [21], 'Ti': [22], 'V': [23], 
    'Cr': [24], 'Mn': [25], 'Fe': [26], 'Co': [27], 'Ni': [28], 'Cu': [29], 'Zn': [30], 
    'Ga': [31], 'Ge': [32], 'As': [33], 'Se': [34], 'Br': [35], 'Rb': [37], 'Sr': [38], 
    'Y': [39], 'Zr': [40], 'Nb': [41], 'Mo': [42], 'Tc': [43], 'Ru': [44], 'Rh': [45], 
    'Pd': [46], 'Ag': [47], 'Cd': [48], 'In': [49], 'Sn': [50], 'Sb': [51], 'Te': [52], 
    'I': [53], 'Cs': [55], 'Ba': [56], 'La': [57], 'Ce': [58], 'Pr': [59], 'Nd': [60], 
    'Pm': [61], 'Sm': [62], 'Eu': [63], 'Gd': [64], 'Tb': [65], 'Dy': [66], 'Ho': [67], 
    'Er': [68], 'Tm': [69], 'Yb': [70], 'Lu': [71], 'Hf': [72], 'Ta': [73], 'W': [74], 
    'Re': [75], 'Os': [76], 'Ir': [77], 'Pt': [78], 'Au': [79], 'Hg': [80], 'Tl': [81], 
    'Pb': [82], 'Bi': [83], 'Th': [90], 'Pa': [91], 'U': [92]
}
z_to_el = {v[0]: k for k, v in element_data.items()}

results = []
for i in range(len(y_pred)):
    row = y_pred[i]
    # y 结构: [a, b, c, alpha, beta, gamma, Z1, frac1, Z2, frac2, Z3, frac3, Z4, frac4]
    
    # 1. 晶格参数
    lattice_str = f"a={row[0]:.2f}, b={row[1]:.2f}, c={row[2]:.2f}, α={row[3]:.1f}, β={row[4]:.1f}, γ={row[5]:.1f}"
    
    # 2. 化学式解析
    elements_found = []
    # 遍历 4 个可能的元素槽位 (索引 6 到 13)
    for j in range(6, 14, 2):
        z_pred = row[j]
        frac_pred = row[j+1]
        
        # 过滤无效元素
        if frac_pred < 0.05 or z_pred < 0.5: continue
        
        # 寻找最近的真实元素
        z_int = int(round(z_pred))
        
        # 简单的价态修正逻辑 (可选): 如果 Z=21.8 (Ti/Sc), 倾向于更常见的那个
        # 这里暂时直接取整
        
        if z_int in z_to_el:
            sym = z_to_el[z_int]
            # 格式化分数: 如果是 0.98 -> 1, 0.33 -> 0.33
            if abs(frac_pred - 1.0) < 0.1:
                elements_found.append(f"{sym}")
            else:
                elements_found.append(f"{sym}{frac_pred:.2f}")
    
    formula = "".join(elements_found)
    
    results.append({
        "Predicted_Formula": formula,
        "Lattice_Params": lattice_str,
        "Confidence_Score": df_gen.iloc[i]['Predicted_FE_Prob'],
        "Bandgap_Target": df_gen.iloc[i]['Bandgap'],
        "SpaceGroup_Target": int(df_gen.iloc[i]['SpaceGroup_Number'])
    })

# ==========================================
# 4. 保存最终结果
# ==========================================
final_df = pd.DataFrame(results)

# 简单的后处理去重 (可能会生成重复的化学式)
final_df.drop_duplicates(subset=['Predicted_Formula'], keep='first', inplace=True)

# 按置信度排序
final_df.sort_values(by='Confidence_Score', ascending=False, inplace=True)

print("\n====== Final Inverse Design Results (Top 10) ======")
print(final_df.head(10).to_string(index=False))

output_file = 'final_designed_materials.csv'
final_df.to_csv(output_file, index=False)
print(f"\nSaved full results to '{output_file}'. Ready for DFT validation!")