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
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.impute import SimpleImputer

# 设置绘图风格
sns.set(style="whitegrid")
# 尝试解决中文乱码 (如果不支持可忽略)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

class IntrinsicFeatureExtractor:
    """
    内禀特征提取器：修复了 pymatgen 版本兼容性问题
    """
    def extract(self, formula, bandgap, row_idx=None, source="Unknown"):
        try:
            # 1. 数据校验
            if not isinstance(formula, str) or not formula.strip():
                return None
            
            # 处理 bandgap 可能为字符串的情况
            try:
                bg_val = float(bandgap)
            except (ValueError, TypeError):
                return None

            # 2. 使用 pymatgen 解析化学式
            try:
                comp = Composition(formula)
            except Exception:
                # 某些特殊格式（如带括号的基团）如果解析失败，直接跳过
                return None
            
            # 3. 提取特征 (手动计算以确保兼容性)
            
            # A. 电负性特征 (Manual Calculation)
            # 获取所有元素的电负性列表
            # 注意: 某些稀有元素可能没有电负性数据，设为 0 或跳过
            electronegs = [e.X for e in comp.elements if hasattr(e, 'X') and e.X is not None]
            
            if not electronegs:
                return None # 无法计算电负性，跳过此样本

            # 计算加权平均电负性
            total_en = 0.0
            total_atoms = 0.0
            for el, amt in comp.items():
                if hasattr(el, 'X') and el.X is not None:
                    total_en += el.X * amt
                    total_atoms += amt
            
            avg_electroneg = total_en / total_atoms if total_atoms > 0 else 0.0
            
            # 最大电负性差
            diff_electroneg = max(electronegs) - min(electronegs)
            
            # B. 原子半径特征
            radii = [e.atomic_radius for e in comp.elements if e.atomic_radius is not None]
            avg_radius = np.mean(radii) if radii else 0.0
            
            # C. 质量与组成
            avg_mass = comp.weight / comp.num_atoms
            num_elements = len(comp.elements)
            
            # D. 是否含过渡金属
            has_transition_metal = int(any([e.is_transition_metal for e in comp.elements]))
            
            return {
                'Avg_Electronegativity': avg_electroneg,
                'Diff_Electronegativity': diff_electroneg,
                'Avg_Atomic_Radius': avg_radius,
                'Avg_Atomic_Mass': avg_mass,
                'Num_Elements': num_elements,
                'Has_Transition_Metal': has_transition_metal,
                'Bandgap': bg_val
            }
        except Exception as e:
            # 仅在调试时打印前几个错误
            if row_idx is not None and row_idx < 3:
                print(f"[解析警告] {source} 第 {row_idx} 行: 化学式='{formula}' 解析失败. 原因: {e}")
            return None

def process_datasets(csv_path, jsonl_path):
    extractor = IntrinsicFeatureExtractor()
    combined_data = []
    
    # ==========================
    # 1. 处理 CSV (旧数据)
    # ==========================
    print(f"\n>>> 正在处理 CSV 文件: {csv_path}")
    try:
        df_csv = pd.read_csv(csv_path)
        # 自动清洗列名
        df_csv.columns = df_csv.columns.str.strip()
        
        # 寻找化学式列 (兼容多种命名)
        formula_col = None
        for col in ['Material_Name', 'Formula', 'pretty_formula', 'formula', 'Composition']:
            if col in df_csv.columns:
                formula_col = col
                break
        
        # 寻找带隙列
        bandgap_col = None
        for col in ['Bandgap (eV)', 'Bandgap', 'band_gap', 'bandgap']:
            if col in df_csv.columns:
                bandgap_col = col
                break
        
        if not formula_col or not bandgap_col:
            print(f"[错误] CSV 列名匹配失败。找到的列: {list(df_csv.columns)}")
        else:
            success_count = 0
            for idx, row in df_csv.iterrows():
                feats = extractor.extract(row[formula_col], row[bandgap_col], row_idx=idx, source="CSV")
                if feats:
                    label = 1 if row.get('Label') == 'ferroelectric' else 0
                    feats['Label'] = label
                    feats['Source'] = 'CSV'
                    combined_data.append(feats)
                    success_count += 1
            print(f"    CSV 处理完毕: 成功提取 {success_count} / {len(df_csv)} 行")

    except Exception as e:
        print(f"[CSV 读取失败] {e}")

    # ==========================
    # 2. 处理 JSONL (新数据)
    # ==========================
    print(f"\n>>> 正在处理 JSONL 文件: {jsonl_path}")
    try:
        json_success = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                try:
                    entry = json.loads(line)
                    
                    # 提取化学式
                    formula = None
                    if 'structure' in entry and 'sites' in entry['structure']:
                        sites = entry['structure']['sites']
                        element_counts = {}
                        for site in sites:
                            species_list = site.get('species', [])
                            if species_list:
                                el = species_list[0].get('element')
                                if el:
                                    element_counts[el] = element_counts.get(el, 0) + 1
                        if element_counts:
                            formula = "".join([f"{el}{int(cnt)}" for el, cnt in element_counts.items()])

                    if not formula:
                        formula = entry.get('pretty_formula') or entry.get('formula') or entry.get('material_name')

                    # 提取带隙
                    bandgap = entry.get('band_gap')
                    if bandgap is None:
                        bandgap = entry.get('bandgap')

                    if formula and bandgap is not None:
                        feats = extractor.extract(formula, bandgap, row_idx=idx, source="JSONL")
                        if feats:
                            feats['Label'] = 0 # 负样本
                            feats['Source'] = 'JSONL'
                            combined_data.append(feats)
                            json_success += 1
                    
                except Exception:
                    continue 

        print(f"    JSONL 处理完毕: 成功提取 {json_success} 行")

    except Exception as e:
        print(f"[JSONL 读取失败] {e}")

    # ==========================
    # 3. 返回
    # ==========================
    if not combined_data:
        print("\n[严重错误] 融合后数据为空！")
        return pd.DataFrame()
    
    return pd.DataFrame(combined_data)

def train_and_evaluate(df):
    if df.empty:
        return

    print(f"\n>>> 数据融合成功。总样本数: {len(df)}")
    print(f"    正样本 (铁电): {df['Label'].sum()}")
    print(f"    负样本 (非铁电): {len(df) - df['Label'].sum()}")

    # 准备特征
    feature_cols = [c for c in df.columns if c not in ['Label', 'Source']]
    X = df[feature_cols]
    y = df['Label']
    
    # --- 关键修正：全局清洗数据，填补 NaN ---
    # 使用列的平均值来填补缺失值 (例如 Ar 的电负性缺失，就用其他元素的平均值代替)
    print(">>> 正在清洗数据 (填补缺失值 NaN)...")
    imputer = SimpleImputer(strategy='mean')
    X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # 再次检查是否还有 NaN
    if X_clean.isnull().sum().sum() > 0:
        print("[警告] 数据清洗后仍有空值，改用 0 填充。")
        X_clean = X_clean.fillna(0)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n>>> 正在训练内禀特征模型 (RF + HistGB + SVM)...")
    
    # 1. 随机森林 (基准)
    clf_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    
    # 2. HistGradientBoosting (原生支持 NaN，但我们已经清洗过了，也没问题)
    clf_hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=150, random_state=42)
    
    # 3. SVM (Pipeline中已经有Scaler，现在数据纯净了，可以直接跑)
    clf_svm = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, C=1.0, random_state=42))
    
    # 集成
    eclf = VotingClassifier(
        estimators=[('rf', clf_rf), ('hgb', clf_hgb), ('svm', clf_svm)],
        voting='soft'
    )
    
    # 训练
    try:
        eclf.fit(X_train, y_train)
    except Exception as e:
        print(f"[训练失败] {e}")
        return

    # 预测与评估
    y_pred = eclf.predict(X_test)
    y_prob = eclf.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("模型评估报告 (Intrinsic Features)")
    print("="*50)
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['非铁电', '铁电']))
    
    # 混淆矩阵
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-FE', 'FE'], yticklabels=['Non-FE', 'FE'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # 特征重要性 (用随机森林近似)
    clf_rf.fit(X_train, y_train)
    importances = clf_rf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
    plt.title('Feature Importance (Chemistry + Bandgap)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 请确保路径正确 (根据你之前的报错，路径似乎是当前目录或 new_data 目录)
    # 如果文件在当前目录下：
    csv_file = 'new_data/ferroelectric_database_labeled.csv'
    jsonl_file = 'new_data/dataset_nonFE.jsonl' # 注意文件名里的空格
    
    # 如果文件在 new_data 目录下，请取消注释下面两行并注释上面两行：
    # csv_file = 'new_data/ferroelectric_database_labeled.csv'
    # jsonl_file = 'new_data/dataset_nonFE.jsonl'
    
    df_data = process_datasets(csv_file, jsonl_file)
    train_and_evaluate(df_data)