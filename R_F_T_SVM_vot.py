import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(filepath):
    """
    加载数据并构建物理交互特征 (与之前一致)
    """
    df = pd.read_csv(filepath)
    df_clean = df.dropna(subset=['Bandgap (eV)']).copy()
    
    # --- 特征工程 ---
    # 1. 畸变密度: 畸变/体积变化
    df_clean['Distortion_Density'] = df_clean['Max_Distortion (Å)'] / (df_clean['Volume_Change_Ratio'] + 1e-6)
    # 2. 带隙-应变耦合
    df_clean['Bandgap_Strain_Interaction'] = df_clean['Bandgap (eV)'] * df_clean['Spontaneous_Strain (unitless)']
    # 3. 对称性破缺程度 * 平均畸变
    df_clean['Symmetry_Distortion_Product'] = df_clean['Symmetry_Change (Delta_SG)'] * df_clean['Avg_Distortion (Å)']

    feature_cols = [
        'Max_Distortion (Å)', 'Spontaneous_Strain (unitless)', 'Symmetry_Change (Delta_SG)', 
        'Bandgap (eV)', 'Avg_Distortion (Å)', 'Structural_Delta', 'Volume_Change_Ratio',
        'Distortion_Density', 'Bandgap_Strain_Interaction', 'Symmetry_Distortion_Product'
    ]
    
    X = df_clean[feature_cols]
    y = df_clean['Label'].map({'ferroelectric': 1, 'none_ferroelectric': 0})
    
    return X, y, feature_cols

def train_voting_model(X, y):
    """
    构建并训练投票模型
    """
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n正在构建基模型...")
    
    # --- 模型 1: 随机森林 (Random Forest) ---
    # 使用之前调优过的最佳参数 (或者一套鲁棒的参数)
    clf_rf = RandomForestClassifier(
        n_estimators=200, 
        min_samples_split=2,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    
    # --- 模型 2: 梯度提升树 (Gradient Boosting) ---
    # 树模型中的"特种兵"，精度高
    clf_gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # --- 模型 3: 支持向量机 (SVM) ---
    # 注意：SVM 是基于距离的，必须先进行标准化 (StandardScaler)!
    # probability=True 是为了让 SVM 输出概率，从而能够参与 Soft Voting
    clf_svm = make_pipeline(
        StandardScaler(), 
        SVC(kernel='rbf', probability=True, C=1.0, random_state=42)
    )
    
    # --- 构建投票器 (Voting Classifier) ---
    print("正在集成模型 (Voting='soft')...")
    # weights: 可以给表现好的模型更高的权重，这里我们先假设它们同等重要 [1, 1, 1]
    eclf = VotingClassifier(
        estimators=[('rf', clf_rf), ('gb', clf_gb), ('svm', clf_svm)],
        voting='soft', 
        weights=[1, 1, 1] 
    )
    
    # 训练集成模型
    eclf.fit(X_train, y_train)
    
    return eclf, X_test, y_test

def evaluate_ensemble(model, X_test, y_test, X_full, y_full):
    """
    评估集成模型性能
    """
    # 1. 获取概率预测 (用于 AUC 和 阈值调整)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    
    # 3. 调整阈值预测 (例如 0.35) - 优先保证不漏掉铁电体
    custom_threshold = 0.24
    y_pred_adjusted = (y_prob >= custom_threshold).astype(int)
    
    print("\n" + "="*50)
    print("集成模型评估 (RF + GB + SVM)")
    print("="*50)
    
    # 打印 AUC
    print(f"AUC Score (排序能力): {roc_auc_score(y_test, y_prob):.4f}")
    

    
    print(f"\n--- 调整阈值 ({custom_threshold}) 以提升召回率 (Recall) ---")
    print(classification_report(y_test, y_pred_adjusted, target_names=['非铁电体', '铁电体']))
    
    # 4. 稳健性检查
    print("\n正在进行 10折交叉验证 (全数据集)...")
    cv_scores = cross_val_score(model, X_full, y_full, cv=10, scoring='accuracy')
    print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # 5. 可视化混淆矩阵 (使用调整后的阈值)
    cm = confusion_matrix(y_test, y_pred_adjusted)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Non-FE', 'FE'], yticklabels=['Non-FE', 'FE'])
    plt.title(f'Confusion Matrix (Voting Ensemble, Threshold={custom_threshold})', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    data_path = 'ferroelectric_database_labeled.csv' # 确保文件名正确
    
    try:
        X, y, _ = load_and_preprocess_data(data_path)
        ensemble_model, X_test, y_test = train_voting_model(X, y)
        evaluate_ensemble(ensemble_model, X_test, y_test, X, y)
        
    except Exception as e:
        print(f"发生错误: {e}")