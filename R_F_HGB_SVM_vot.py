import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
# 引入 HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             roc_auc_score, precision_score, recall_score, f1_score)

# 设置绘图风格
sns.set(style="whitegrid")
# 尝试解决中文乱码问题 (如果您的环境支持 SimHei)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(filepath):
    """
    加载数据并构建物理交互特征
    """
    df = pd.read_csv(filepath)
    df_clean = df.dropna(subset=['Bandgap (eV)']).copy()
    
    # --- 特征工程 (Feature Engineering) ---
    # 1. 畸变密度: 畸变/体积变化 (衡量单位体积的形变能)
    df_clean['Distortion_Density'] = df_clean['Max_Distortion (Å)'] / (df_clean['Volume_Change_Ratio'] + 1e-6)
    # 2. 带隙-应变耦合 (绝缘体中的强应变往往对应极化)
    df_clean['Bandgap_Strain_Interaction'] = df_clean['Bandgap (eV)'] * df_clean['Spontaneous_Strain (unitless)']
    # 3. 对称性破缺程度 * 平均畸变 (综合描述结构的非中心对称性)
    df_clean['Symmetry_Distortion_Product'] = df_clean['Symmetry_Change (Delta_SG)'] * df_clean['Avg_Distortion (Å)']

    feature_cols = [
        'Max_Distortion (Å)', 'Spontaneous_Strain (unitless)', 'Symmetry_Change (Delta_SG)', 
        'Bandgap (eV)', 'Avg_Distortion (Å)', 'Structural_Delta', 'Volume_Change_Ratio',
        'Distortion_Density', 'Bandgap_Strain_Interaction', 'Symmetry_Distortion_Product'
    ]
    
    X = df_clean[feature_cols]
    y = df_clean['Label'].map({'ferroelectric': 1, 'none_ferroelectric': 0})
    
    print(f"数据加载完成。样本数: {len(X)}")
    return X, y, feature_cols

def train_advanced_voting_model(X, y):
    """
    构建并训练新一代投票模型 (RF + HistGB + SVM)
    """
    # 划分数据集 (80% 训练, 20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n正在构建新一代集成模型 (RF + HistGB + SVM)...")
    
    # --- 模型 1: 随机森林 (Random Forest) ---
    # 稳健的基石，抗过拟合
    clf_rf = RandomForestClassifier(
        n_estimators=200, min_samples_split=2, max_depth=None,
        random_state=42, n_jobs=-1
    )
    
    # --- 模型 2: 直方图梯度提升树 (HistGradientBoosting) ---
    # 升级点！速度更快，精度更高，擅长处理大量样本和非线性特征
    clf_hgb = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        random_state=42
    )
    
    # --- 模型 3: 支持向量机 (SVM) ---
    # 提供基于几何距离的分类边界，与树模型互补
    clf_svm = make_pipeline(
        StandardScaler(), 
        SVC(kernel='rbf', probability=True, C=1.0, random_state=42)
    )
    
    # --- 构建软投票集成 (Soft Voting) ---
    # Soft Voting 利用了每个模型输出的概率值，通常比 Hard Voting 效果更好
    eclf = VotingClassifier(
        estimators=[('rf', clf_rf), ('hgb', clf_hgb), ('svm', clf_svm)],
        voting='soft', 
        weights=[1, 1, 1] 
    )
    
    eclf.fit(X_train, y_train)
    
    return eclf, X_test, y_test

def find_optimal_threshold(model, X_test, y_test):
    """
    自动寻找并可视化最佳分类阈值
    """
    # 1. 获取正样本预测概率
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 2. 扫描阈值 (0.01 到 0.99)
    thresholds = np.arange(0.01, 1.00, 0.01)
    precisions, recalls, f1_scores = [], [], []
    
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred_t, zero_division=0))
        
    # 3. 寻找关键点
    # 策略 A: F1 Score 最高点 (综合最优)
    best_f1_idx = np.argmax(f1_scores)
    best_thresh_f1 = thresholds[best_f1_idx]
    
    # 策略 B: 发现模式 (Recall >= 0.90 且 Precision 最高)
    high_recall_indices = [i for i, r in enumerate(recalls) if r >= 0.90]
    if high_recall_indices:
        best_discovery_idx = high_recall_indices[np.argmax([precisions[i] for i in high_recall_indices])]
        best_thresh_discovery = thresholds[best_discovery_idx]
    else:
        best_thresh_discovery = thresholds[np.argmax(recalls)]

    # 4. 打印报告
    print("\n" + "="*50)
    print("自动化阈值寻优报告 (Threshold Optimization)")
    print("="*50)
    print(f"AUC Score (排序能力): {roc_auc_score(y_test, y_prob):.4f}")
    
    print(f"\n[平衡模式] 推荐阈值: {best_thresh_f1:.2f}")
    print(f"  Recall: {recalls[best_f1_idx]:.4f}, Precision: {precisions[best_f1_idx]:.4f}, F1: {f1_scores[best_f1_idx]:.4f}")
    
    print(f"\n[发现模式] 推荐阈值: {best_thresh_discovery:.2f} (优先保证 Recall >= 90%)")
    print(f"  Recall: {recalls[best_discovery_idx]:.4f}, Precision: {precisions[best_discovery_idx]:.4f}, F1: {f1_scores[best_discovery_idx]:.4f}")
    
    # 5. 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', linestyle='--', color='blue')
    plt.plot(thresholds, recalls, label='Recall', color='green', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1 Score', color='red', linewidth=2)
    
    plt.axvline(best_thresh_f1, color='red', linestyle=':', label=f'Best F1 ({best_thresh_f1:.2f})')
    plt.axvline(best_thresh_discovery, color='green', linestyle=':', label=f'Discovery ({best_thresh_discovery:.2f})')
    
    plt.title('Performance Metrics vs. Threshold (RF+HGB+SVM)')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
    
    return best_thresh_discovery  # 返回发现模式的阈值

def evaluate_final_performance(model, X_test, y_test, threshold, X_full, y_full):
    """
    使用选定的最佳阈值进行最终评估
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_final = (y_prob >= threshold).astype(int)
    
    print("\n" + "="*50)
    print(f"最终模型评估 (使用阈值: {threshold:.2f})")
    print("="*50)
    print(classification_report(y_test, y_pred_final, target_names=['非铁电体', '铁电体']))
    
    # 混淆矩阵可视化
    cm = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['Non-FE', 'FE'], yticklabels=['Non-FE', 'FE'])
    plt.title(f'Confusion Matrix (Threshold={threshold:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 稳健性检查
    print("\n正在进行 10折交叉验证 (全数据集)...")
    cv_scores = cross_val_score(model, X_full, y_full, cv=10, scoring='accuracy')
    print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

if __name__ == "__main__":
    # 请确保csv文件在当前目录下
    data_path = 'ferroelectric_database_labeled.csv' 
    
    try:
        # 1. 加载数据
        X, y, _ = load_and_preprocess_data(data_path)
        
        # 2. 训练集成模型 (RF + HGB + SVM)
        ensemble_model, X_test, y_test = train_advanced_voting_model(X, y)
        
        # 3. 寻找最佳阈值 (返回的是“发现模式”的阈值)
        best_threshold = find_optimal_threshold(ensemble_model, X_test, y_test)
        
        # 4. 使用最佳阈值进行最终评估
        evaluate_final_performance(ensemble_model, X_test, y_test, best_threshold, X, y)
        
    except FileNotFoundError:
        print("错误: 找不到文件，请确保 'ferroelectric_database_labeled.csv' 在当前目录下。")
    except Exception as e:
        print(f"发生错误: {e}")