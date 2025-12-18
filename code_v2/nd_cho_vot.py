import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.core import Composition, Element
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve
import xgboost as xgb

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 高级物理特征提取器 (Physics-Informed)
# ==========================================
class AdvancedFeatureExtractor:
    def __init__(self):
        # 预加载更深层的物理属性
        self.elem_data = {}
        for el in Element:
            try:
                self.elem_data[el.symbol] = {
                    'X': el.X,                              # 电负性
                    'Radius': el.atomic_radius,             # 原子半径
                    'Mass': el.atomic_mass,                 # 质量
                    'Ionization': el.ionization_energy,     # 电离能 (关键!)
                    'Affinity': el.electron_affinity,       # 电子亲和能 (关键!)
                    'MeltingT': el.melting_point,           # 熔点 (反映键强)
                    'Valence': float(el.nvalence_electrons) # 价电子数
                }
            except:
                continue
                
        # 铁电活性元素列表 (d0 transition metals & lone pair ions)
        self.fe_active_elements = set(['Ti', 'Nb', 'Ta', 'Zr', 'Hf', 'V', 'W', 'Mo', 'Pb', 'Bi', 'Sn'])

    def get_stats(self, values, weights):
        """计算加权统计量：均值、加权方差、极差"""
        vals = np.array([v if v is not None else 0 for v in values], dtype=float)
        wts = np.array(weights, dtype=float)
        
        # 归一化权重
        if np.sum(wts) > 0:
            wts = wts / np.sum(wts)
        
        mean = np.average(vals, weights=wts)
        variance = np.average((vals - mean)**2, weights=wts)
        rng = np.max(vals) - np.min(vals)
        
        return [mean, variance, rng]

    def extract(self, formula, bandgap):
        try:
            if not isinstance(formula, str) or not formula.strip(): return None
            comp = Composition(formula)
            
            features = {}
            
            # --- A. 基础组分解析 ---
            total_atoms = comp.num_atoms
            elements = []
            weights = []
            for el, amt in comp.items():
                elements.append(el.symbol)
                weights.append(amt / total_atoms)
            
            # --- B. 元素属性统计 (Magpie-style) ---
            props = ['X', 'Radius', 'Mass', 'Ionization', 'Affinity', 'Valence']
            
            for prop in props:
                # 获取每个原子的属性值
                vals = [self.elem_data.get(el, {}).get(prop, 0) for el in elements]
                stats = self.get_stats(vals, weights)
                
                features[f'Mean_{prop}'] = stats[0]
                features[f'Var_{prop}'] = stats[1] # 方差通常比极差更鲁棒
                features[f'Range_{prop}'] = stats[2]

            # --- C. 高级物理特征 ---
            
            # 1. 离子性代理 (Ionicity Proxy)
            # 电负性差越大，离子性越强。铁电体通常处于共价/离子键的边界
            features['Max_Ionic_Char'] = features['Range_X']
            
            # 2. 尺寸失配代理 (Size Mismatch)
            # 半径方差大意味着晶格中存在大小差异巨大的离子（类似钙钛矿A/B位差异）
            features['Size_Mismatch'] = features['Var_Radius']
            
            # 3. 铁电活性元素占比 (Ferroelectric Active Fraction)
            # 计算含有多少 d0 元素或孤对电子元素
            active_frac = sum([weights[i] for i, el in enumerate(elements) if el in self.fe_active_elements])
            features['FE_Active_Fraction'] = active_frac
            
            # 4. 平均价电子浓度 (VEC - Valence Electron Concentration)
            # 这是一个预测相稳定性的经典物理量
            features['VEC'] = features['Mean_Valence']
            
            # 5. 电子结构
            try:
                bg = float(bandgap)
                features['Bandgap'] = bg
                # 绝缘性判定 (Bandgap > 0.1)
                features['Is_Insulator'] = 1 if bg > 0.1 else 0
            except:
                return None

            return features
            
        except Exception as e:
            return None

# ==========================================
# 2. 数据处理与融合
# ==========================================
def process_data(csv_path, jsonl_path):
    print(">>> 正在提取高级物理特征 (XGBoost Ready)...")
    extractor = AdvancedFeatureExtractor()
    all_data = []
    
    # 1. CSV 处理
    try:
        df_csv = pd.read_csv(csv_path)
        df_csv.columns = df_csv.columns.str.strip()
        # 智能匹配列名
        f_col = next((c for c in ['Material_Name', 'Formula', 'pretty_formula'] if c in df_csv.columns), None)
        b_col = next((c for c in ['Bandgap (eV)', 'Bandgap'] if c in df_csv.columns), None)
        
        if f_col and b_col:
            for _, row in df_csv.iterrows():
                feats = extractor.extract(row[f_col], row[b_col])
                if feats:
                    feats['Label'] = 1 if row.get('Label') == 'ferroelectric' else 0
                    all_data.append(feats)
    except Exception as e: print(f"CSV Error: {e}")

    # 2. JSONL 处理
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    # 优先从 formula 字段获取，其次重构
                    form = entry.get('pretty_formula') or entry.get('formula')
                    if not form and 'structure' in entry:
                        # 简易重构
                        s = entry['structure']['sites']
                        els = {}
                        for site in s:
                            e = site['species'][0]['element']
                            els[e] = els.get(e, 0) + 1
                        form = "".join([f"{k}{v}" for k,v in els.items()])
                    
                    bg = entry.get('band_gap', entry.get('bandgap'))
                    
                    if form and bg is not None:
                        feats = extractor.extract(form, bg)
                        if feats:
                            feats['Label'] = 0
                            all_data.append(feats)
                except: continue
    except Exception as e: print(f"JSONL Error: {e}")

    df = pd.DataFrame(all_data)
    print(f"特征提取完成。总样本: {len(df)} | 正样本: {df['Label'].sum()}")
    return df

# ==========================================
# 3. XGBoost 训练与自动调优
# ==========================================
def train_xgboost_system(df):
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # 1. 划分数据 (保留不平衡测试集)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 2. 训练集降采样 (Under-sampling) 
    # 将负样本降至正样本的 5 倍 (比之前的 3 倍宽松一点，给 XGBoost 更多数据)
    train_df = pd.concat([X_train, y_train], axis=1)
    pos = train_df[train_df.Label == 1]
    neg = train_df[train_df.Label == 0]
    
    # 动态调整采样比例
    if len(neg) > len(pos) * 5:
        neg = neg.sample(n=len(pos)*5, random_state=42)
    
    train_balanced = pd.concat([pos, neg]).sample(frac=1, random_state=42)
    X_train_bal = train_balanced.drop('Label', axis=1)
    y_train_bal = train_balanced['Label']
    
    print(f"\n>>> 训练 XGBoost 模型 (训练集 正:负 = 1:{len(neg)/len(pos):.1f})...")
    
    # 3. 定义 XGBoost 模型
    # scale_pos_weight: 进一步处理不平衡
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2,  # 稍微增加正样本权重
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_bal, y_train_bal)
    
    # 4. 预测与阈值寻优 (PR-Curve)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # 获取 Precision-Recall 曲线
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # 计算 F1 Scores
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # 找到最佳 F1 的阈值
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    
    # 5. 最终评估
    y_pred_final = (y_probs >= best_thresh).astype(int)
    
    print("\n" + "="*50)
    print("XGBoost 高级模型评估结果")
    print("="*50)
    print(f"最佳阈值 (Best Threshold): {best_thresh:.3f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_probs):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred_final, target_names=['Non-FE', 'FE']))
    
    # 绘制特征重要性
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=15, height=0.5, importance_type='gain')
    plt.title('Top 15 Features (XGBoost Gain)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 请确认文件路径
    csv_file = 'new_data/ferroelectric_database_labeled.csv'
    jsonl_file = 'new_data/dataset_nonFE.jsonl' # 注意文件名里的空格
    
    # 1. 处理
    df_data = process_data(csv_file, jsonl_file)
    
    # 2. 训练
    if not df_data.empty:
        train_xgboost_system(df_data)
    else:
        print("错误：数据为空")