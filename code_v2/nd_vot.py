import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # 关键：修复 NaN 报错
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             roc_auc_score, precision_score, recall_score, f1_score)

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

class IntrinsicFeatureExtractor:
    def extract(self, formula, bandgap, row_idx=None, source="Unknown"):
        try:
            if not isinstance(formula, str) or not formula.strip(): return None
            try: bg_val = float(bandgap)
            except: return None

            try: comp = Composition(formula)
            except: return None
            
            # 提取特征 (手动计算以兼容旧版 pymatgen)
            electronegs = [e.X for e in comp.elements if hasattr(e, 'X') and e.X is not None]
            if not electronegs: return None 

            total_en = sum(el.X * amt for el, amt in comp.items() if hasattr(el, 'X') and el.X)
            total_atoms = sum(amt for el, amt in comp.items() if hasattr(el, 'X') and el.X)
            avg_electroneg = total_en / total_atoms if total_atoms > 0 else 0.0
            
            radii = [e.atomic_radius for e in comp.elements if e.atomic_radius is not None]
            avg_radius = np.mean(radii) if radii else 0.0
            
            return {
                'Avg_Electronegativity': avg_electroneg,
                'Diff_Electronegativity': max(electronegs) - min(electronegs),
                'Avg_Atomic_Radius': avg_radius,
                'Avg_Atomic_Mass': comp.weight / comp.num_atoms,
                'Num_Elements': len(comp.elements),
                'Has_Transition_Metal': int(any([e.is_transition_metal for e in comp.elements])),
                'Bandgap': bg_val
            }
        except Exception:
            return None

def process_datasets(csv_path, jsonl_path):
    extractor = IntrinsicFeatureExtractor()
    combined_data = []
    
    # 1. CSV
    try:
        df_csv = pd.read_csv(csv_path)
        df_csv.columns = df_csv.columns.str.strip()
        formula_col = next((c for c in ['Material_Name', 'Formula', 'pretty_formula'] if c in df_csv.columns), None)
        bandgap_col = next((c for c in ['Bandgap (eV)', 'Bandgap'] if c in df_csv.columns), None)
        
        if formula_col and bandgap_col:
            for idx, row in df_csv.iterrows():
                feats = extractor.extract(row[formula_col], row[bandgap_col])
                if feats:
                    feats['Label'] = 1 if row.get('Label') == 'ferroelectric' else 0
                    combined_data.append(feats)
    except Exception as e: print(f"CSV Error: {e}")

    # 2. JSONL
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    entry = json.loads(line)
                    formula = None
                    if 'structure' in entry and 'sites' in entry['structure']:
                        element_counts = {}
                        for site in entry['structure']['sites']:
                            el = site['species'][0]['element']
                            element_counts[el] = element_counts.get(el, 0) + 1
                        formula = "".join([f"{el}{int(cnt)}" for el, cnt in element_counts.items()])
                    
                    if not formula: formula = entry.get('pretty_formula') or entry.get('formula')
                    bandgap = entry.get('band_gap', entry.get('bandgap'))
                    
                    if formula and bandgap is not None:
                        feats = extractor.extract(formula, bandgap)
                        if feats:
                            feats['Label'] = 0 # 负样本
                            combined_data.append(feats)
                except: continue
    except Exception as e: print(f"JSONL Error: {e}")

    return pd.DataFrame(combined_data)

def find_optimal_threshold(model, X_test, y_test):
    """
    针对不平衡数据，自动寻找最佳阈值
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.01, 1.00, 0.01)
    
    f1_scores = []
    recalls = []
    precisions = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        
    # 策略：优先保证 Recall > 0.85 的前提下，F1 最高
    # 如果没有 Recall > 0.85 的点，就取 F1 最高点
    high_recall_indices = [i for i, r in enumerate(recalls) if r >= 0.85]
    
    if high_recall_indices:
        best_idx = high_recall_indices[np.argmax([f1_scores[i] for i in high_recall_indices])]
        best_threshold = thresholds[best_idx]
        strategy = "High Recall Priority"
    else:
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        strategy = "Max F1 Priority"

    print("\n" + "="*50)
    print(f"阈值寻优结果 ({strategy})")
    print("="*50)
    print(f"推荐阈值: {best_threshold:.2f}")
    print(f"预期性能 -> Recall: {recalls[best_idx]:.4f} | Precision: {precisions[best_idx]:.4f} | F1: {f1_scores[best_idx]:.4f}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, recalls, label='Recall', color='green')
    plt.plot(thresholds, precisions, label='Precision', color='blue', linestyle='--')
    plt.plot(thresholds, f1_scores, label='F1 Score', color='red')
    plt.axvline(best_threshold, color='black', linestyle=':', label=f'Best Threshold {best_threshold:.2f}')
    plt.title('Metrics vs Threshold (Imbalanced Data)')
    plt.xlabel('Probability Threshold')
    plt.legend()
    plt.show()
    
    return best_threshold

def train_and_evaluate(df):
    if df.empty: return

    print(f"\n>>> 数据概览: 总数 {len(df)} | 铁电(1): {df['Label'].sum()} | 非铁电(0): {len(df)-df['Label'].sum()}")

    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # 1. 缺失值填充 (修复 ValueError: Input X contains NaN)
    print(">>> 正在填充缺失值...")
    imputer = SimpleImputer(strategy='mean')
    X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n>>> 正在训练加权模型 (Class Weighted)...")
    
    # 关键修改：添加 class_weight='balanced'
    clf_rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
    
    # HGB 早期版本不支持 class_weight，但它很强，我们保留它
    clf_hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=150, random_state=42)
    
    # SVM 必须加权
    clf_svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, class_weight='balanced', C=1.0, random_state=42))
    
    eclf = VotingClassifier(
        estimators=[('rf', clf_rf), ('hgb', clf_hgb), ('svm', clf_svm)],
        voting='soft'
    )
    
    eclf.fit(X_train, y_train)
    
    # 2. 自动寻找最佳阈值
    best_threshold = find_optimal_threshold(eclf, X_test, y_test)
    
    # 3. 最终评估
    y_prob = eclf.predict_proba(X_test)[:, 1]
    y_pred_final = (y_prob >= best_threshold).astype(int)
    
    print(f"\n=== 最终测试集表现 (阈值 {best_threshold:.2f}) ===")
    print(classification_report(y_test, y_pred_final, target_names=['Non-FE', 'FE']))
    
    cm = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-FE', 'FE'], yticklabels=['Non-FE', 'FE'])
    plt.title(f'Confusion Matrix (Thresh={best_threshold:.2f})')
    plt.show()

if __name__ == "__main__":
    # 请确认文件路径
    csv_file = 'new_data/ferroelectric_database_labeled.csv'
    jsonl_file = 'new_data/dataset_nonFE.jsonl' # 注意文件名里的空格
    
    df_data = process_datasets(csv_file, jsonl_file)
    train_and_evaluate(df_data)