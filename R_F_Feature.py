import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 防止乱码，根据环境可调整
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(filepath):
    """
    加载数据并进行预处理和特征工程
    """
    # 1. 加载数据
    df = pd.read_csv(filepath)
    
    # 2. 数据清洗
    # 删除关键特征缺失的行 (例如 Bandgap)
    df_clean = df.dropna(subset=['Bandgap (eV)']).copy()
    
    # 3. 特征工程 (Feature Engineering) - 核心提升点
    # 物理直觉：畸变越大且体积变化越小，铁电性可能越强（应变能密度高）
    df_clean['Distortion_Density'] = df_clean['Max_Distortion (Å)'] / (df_clean['Volume_Change_Ratio'] + 1e-6)
    
    # 物理直觉：带隙与应变的耦合，绝缘体中的强应变往往对应极化
    df_clean['Bandgap_Strain_Interaction'] = df_clean['Bandgap (eV)'] * df_clean['Spontaneous_Strain (unitless)']
    
    # 物理直觉：对称性破缺程度与平均畸变的乘积
    df_clean['Symmetry_Distortion_Product'] = df_clean['Symmetry_Change (Delta_SG)'] * df_clean['Avg_Distortion (Å)']

    # 4. 定义特征列表
    # 原始特征 + 新构造的特征
    feature_cols = [
        'Max_Distortion (Å)', 
        'Spontaneous_Strain (unitless)', 
        'Symmetry_Change (Delta_SG)', 
        'Bandgap (eV)', 
        'Avg_Distortion (Å)', 
        'Structural_Delta', 
        'Volume_Change_Ratio',
        'Distortion_Density',           # 新特征
        'Bandgap_Strain_Interaction',   # 新特征
        'Symmetry_Distortion_Product'   # 新特征
    ]
    
    # 5. 准备 X 和 y
    X = df_clean[feature_cols]
    # 将标签映射为数字: ferroelectric -> 1, none_ferroelectric -> 0
    y = df_clean['Label'].map({'ferroelectric': 1, 'none_ferroelectric': 0})
    
    print(f"数据加载完成。样本数: {len(df_clean)}, 特征数: {len(feature_cols)}")
    print(f"正样本(铁电体): {y.sum()}, 负样本: {len(y) - y.sum()}")
    
    return X, y, feature_cols

def train_optimized_model(X, y):
    """
    使用网格搜索训练优化后的随机森林模型
    """
    # 划分训练集和测试集 (80% / 20%)
    # stratify=y 保证训练集和测试集中正负样本比例一致
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 定义超参数搜索空间
    param_grid = {
        'n_estimators': [100, 200, 300],        # 决策树数量
        'max_depth': [None, 10, 20, 30],        # 树的最大深度，防止过拟合
        'min_samples_split': [2, 5, 10],        # 节点分裂所需的最小样本数
        'min_samples_leaf': [1, 2, 4],          # 叶子节点最小样本数
        'class_weight': ['balanced', None]      # 是否自动平衡样本权重
    }
    
    # 初始化随机森林
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # 使用网格搜索 (GridSearchCV) 进行 5 折交叉验证
    print("\n开始超参数调优 (Grid Search)... 这可能需要几秒钟...")
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n最佳参数组合: {grid_search.best_params_}")
    print(f"最佳验证集准确率: {grid_search.best_score_:.4f}")
    
    # 返回最佳模型和测试数据
    return grid_search.best_estimator_, X_test, y_test

def evaluate_model(model, X_test, y_test, X_full, y_full, feature_cols):
    """
    评估模型性能并可视化
    """
    # 1. 基础预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # 获取正样本概率
    
    # 2. 打印分类报告
    print("\n" + "="*40)
    print("最终测试集评估报告")
    print("="*40)
    print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=['非铁电体', '铁电体']))
    
    # 3. 稳健性检查 (全数据集交叉验证)
    cv_scores = cross_val_score(model, X_full, y_full, cv=10, scoring='accuracy')
    print(f"10折交叉验证平均准确率 (稳健性): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # --- 可视化部分 ---
    plt.figure(figsize=(16, 6))

    # 图1: 混淆矩阵热力图
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-FE', 'FE'], yticklabels=['Non-FE', 'FE'])
    plt.title('Confusion Matrix (Test Set)', fontsize=14)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 图2: 特征重要性排序
    plt.subplot(1, 2, 2)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # 创建DataFrame方便绘图
    feat_imp_df = pd.DataFrame({
        'Feature': [feature_cols[i] for i in indices],
        'Importance': importances[indices]
    })
    
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, palette='viridis')
    plt.title('Feature Importance (with Engineered Features)', fontsize=14)
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    # 保存图片（可选）
    # plt.savefig('model_evaluation_report.png', dpi=300)
    plt.show()

# --- 主程序执行入口 ---
if __name__ == "__main__":
    # 假设文件在当前目录下
    data_path = 'ferroelectric_database_labeled.csv'
    
    try:
        # 1. 数据处理
        X, y, features = load_and_preprocess_data(data_path)
        
        # 2. 训练与优化
        best_rf_model, X_test, y_test = train_optimized_model(X, y)
        
        # 3. 评估与可视化
        evaluate_model(best_rf_model, X_test, y_test, X, y, features)
        
    except FileNotFoundError:
        print("错误: 找不到文件，请确保 'ferroelectric_database_labeled.csv' 在当前目录下。")
    except Exception as e:
        print(f"发生错误: {e}")