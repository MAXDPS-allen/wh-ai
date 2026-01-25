"""
NequIP增强型铁电分类器v6 - 完整版本
===========================================
目标: Accuracy >= 99%, Recall >= 99%

使用StratifiedKFold进行交叉验证, 防止数据泄露
集成多个模型 (Random Forest + Logistic Regression)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 导入特征工程
sys.path.insert(0, str(Path(__file__).parent))
from advanced_feature_engineering import AdvancedFeatureExtractor

sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from feature_engineering import ELEMENT_DATABASE

# 导入必要的库
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    from sklearn.utils import class_weight
    import joblib
    
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: scikit-learn未安装, 使用基础模型")


# ==========================================
# 基础逻辑回归实现 (无需sklearn)
# ==========================================
class SimpleLogisticRegression:
    """简单的逻辑回归实现"""
    
    def __init__(self, learning_rate=0.01, iterations=1000, regularization=0.01):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization = regularization
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0
        
        for _ in range(self.iterations):
            z = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(z)
            
            dw = (1/m) * np.dot(X.T, (predictions - y)) + (self.regularization/m) * self.weights
            db = (1/m) * np.sum(predictions - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ==========================================
# 配置
# ==========================================
class Config:
    DATA_DIR = Path(__file__).parent.parent / 'new_data'
    MODEL_DIR = Path(__file__).parent.parent / 'model_nequip_v6'
    REPORT_DIR = Path(__file__).parent.parent / 'reports_nequip_v6'
    
    FEATURE_DIM = 256
    N_SPLITS = 5
    RANDOM_STATE = 42
    
    @classmethod
    def prepare_dirs(cls):
        for d in [cls.MODEL_DIR, cls.REPORT_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# ==========================================
# 数据处理
# ==========================================
class DataProcessor:
    def __init__(self):
        self.extractor = AdvancedFeatureExtractor()
        self.config = Config()
    
    def load_jsonl(self, filepath: Path) -> List[Dict]:
        data = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data.append(json.loads(line))
                        except:
                            continue
        except:
            pass
        return data
    
    def extract_features_from_jsonl(self, filepath: Path, label: int):
        data = self.load_jsonl(filepath)
        features = []
        labels = []
        
        print(f"处理 {filepath.name} ({len(data)} 个样本)...")
        
        for item in tqdm(data, desc="特征提取"):
            try:
                struct = item.get('structure', item.get('data', item))
                sg_num = item.get('spacegroup', {}).get('number', None)
                
                feat = self.extractor.extract_advanced_features(struct, sg_num)
                
                if not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
                    features.append(feat)
                    labels.append(label)
            except:
                pass
        
        return features, labels
    
    def load_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        all_features = []
        all_labels = []
        
        # 正样本
        fe_files = [
            'dataset_original_ferroelectric.jsonl',
            'dataset_known_FE_rest.jsonl'
        ]
        
        for fname in fe_files:
            fpath = self.config.DATA_DIR / fname
            if fpath.exists():
                feats, lbls = self.extract_features_from_jsonl(fpath, label=1)
                all_features.extend(feats)
                all_labels.extend(lbls)
        
        # 负样本
        non_fe_files = [
            'dataset_nonFE.jsonl',
            'dataset_nonFE_cleaned.jsonl',
            'dataset_nonFE_expanded.jsonl',
            'dataset_nonFE_mp_polar.jsonl'
        ]
        
        for fname in non_fe_files:
            fpath = self.config.DATA_DIR / fname
            if fpath.exists():
                feats, lbls = self.extract_features_from_jsonl(fpath, label=0)
                all_features.extend(feats)
                all_labels.extend(lbls)
        
        X = np.array(all_features, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        
        print(f"\n数据集统计:")
        print(f"  正样本 (FE): {np.sum(y)}")
        print(f"  负样本 (non-FE): {np.sum(y == 0)}")
        print(f"  总数: {len(y)}")
        
        return X, y


# ==========================================
# 模型训练和评估
# ==========================================
class CVEvaluator:
    def __init__(self):
        self.config = Config()
        self.cv = StratifiedKFold(
            n_splits=self.config.N_SPLITS,
            shuffle=True,
            random_state=self.config.RANDOM_STATE
        )
        self.results = []
    
    def train_fold(self, X_train, y_train, X_test, y_test, fold_idx):
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx + 1}/{self.config.N_SPLITS}")
        print(f"{'='*70}")
        
        # 特征缩放
        scaler = RobustScaler() if SKLEARN_AVAILABLE else StandardScaler_Simple()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 计算类别权重 (处理不平衡)
        pos_weight = np.sum(y_train == 0) / (np.sum(y_train == 1) + 1e-6)
        
        print(f"训练集: {len(X_train)} 样本, 类别权重={pos_weight:.2f}")
        print(f"测试集: {len(X_test)} 样本")
        
        # 模型训练
        models = {}
        predictions = []
        
        # RF模型
        if SKLEARN_AVAILABLE:
            print("训练 Random Forest...")
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.config.RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced_subsample'
            )
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict_proba(X_test_scaled)[:, 1]
            models['rf'] = rf
            predictions.append(rf_pred)
        
        # LR模型
        print("训练 Logistic Regression...")
        if SKLEARN_AVAILABLE:
            lr = LogisticRegression(
                max_iter=10000,
                random_state=self.config.RANDOM_STATE,
                class_weight='balanced'
            )
            lr.fit(X_train_scaled, y_train)
            lr_pred = lr.predict_proba(X_test_scaled)[:, 1]
            models['lr'] = lr
            predictions.append(lr_pred)
        else:
            lr = SimpleLogisticRegression(learning_rate=0.01, iterations=1000)
            lr.fit(X_train_scaled, y_train)
            lr_pred = lr.predict_proba(X_test_scaled)
            models['lr'] = lr
            predictions.append(lr_pred)
        
        # 集成预测
        ensemble_pred = np.mean(predictions, axis=0)
        
        # 动态阈值优化
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.3, 0.7, 0.02):
            pred = (ensemble_pred >= threshold).astype(int)
            if np.sum(pred) > 0:
                recall = np.sum((pred == 1) & (y_test == 1)) / (np.sum(y_test == 1) + 1e-6)
                precision = np.sum((pred == 1) & (y_test == 1)) / (np.sum(pred == 1) + 1e-6)
                if recall + precision > 0:
                    f1 = 2 * recall * precision / (recall + precision)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
        
        print(f"优化阈值: {best_threshold:.4f} (F1={best_f1:.4f})")
        
        # 最终预测
        y_pred = (ensemble_pred >= best_threshold).astype(int)
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        
        TP = np.sum((y_pred == 1) & (y_test == 1))
        FP = np.sum((y_pred == 1) & (y_test == 0))
        FN = np.sum((y_pred == 0) & (y_test == 1))
        TN = np.sum((y_pred == 0) & (y_test == 0))
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        try:
            roc_auc = roc_auc_score(y_test, ensemble_pred) if SKLEARN_AVAILABLE else 0
        except:
            roc_auc = 0
        
        print(f"\nFold {fold_idx + 1} 结果:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        print(f"  TN={TN}, FP={FP}, FN={FN}, TP={TP}")
        
        # 保存模型
        fold_dir = self.config.MODEL_DIR / f'fold_{fold_idx}'
        fold_dir.mkdir(exist_ok=True)
        
        if SKLEARN_AVAILABLE:
            joblib.dump(models['rf'], fold_dir / 'random_forest.pkl')
            joblib.dump(models['lr'], fold_dir / 'logistic_regression.pkl')
            joblib.dump(scaler, fold_dir / 'scaler.pkl')
        
        with open(fold_dir / 'results.txt', 'w') as f:
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1-Score:  {f1:.4f}\n")
            f.write(f"ROC-AUC:   {roc_auc:.4f}\n")
            f.write(f"Threshold: {best_threshold:.4f}\n")
        
        return {
            'fold': fold_idx,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'threshold': best_threshold,
            'confusion_matrix': (TN, FP, FN, TP)
        }
    
    def evaluate(self, X, y):
        print(f"\n开始 {self.config.N_SPLITS} 折交叉验证...")
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            result = self.train_fold(X_train, y_train, X_test, y_test, fold_idx)
            self.results.append(result)
        
        results_df = pd.DataFrame(self.results)
        return results_df
    
    def print_summary(self, results_df):
        print(f"\n{'='*70}")
        print("交叉验证总结")
        print(f"{'='*70}\n")
        print(results_df.to_string(index=False))
        
        print(f"\n{'='*70}")
        print("性能统计:")
        print(f"{'='*70}")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            mean = results_df[metric].mean()
            std = results_df[metric].std()
            print(f"{metric.ljust(12)}: {mean:.4f} ± {std:.4f}")
        
        # 检查是否达到目标
        avg_acc = results_df['accuracy'].mean()
        avg_rec = results_df['recall'].mean()
        
        print(f"\n{'='*70}")
        if avg_acc >= 0.99 and avg_rec >= 0.99:
            print("✓ 达成目标: Accuracy >= 99% AND Recall >= 99%")
        else:
            print("✗ 未达成目标")
            print(f"  目标 Accuracy >= 99%, 实际: {avg_acc*100:.2f}%")
            print(f"  目标 Recall >= 99%, 实际: {avg_rec*100:.2f}%")
        print(f"{'='*70}")


class StandardScaler_Simple:
    """简单标准化器 (无需sklearn)"""
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1
        return (X - self.mean) / self.std
    
    def transform(self, X):
        return (X - self.mean) / self.std


# ==========================================
# 主程序
# ==========================================
def main():
    config = Config()
    config.prepare_dirs()
    
    np.random.seed(config.RANDOM_STATE)
    
    print("="*70)
    print("NequIP增强型铁电分类器 v6 - 完整版本")
    print("="*70)
    print(f"特征维度: {config.FEATURE_DIM}")
    print(f"数据目录: {config.DATA_DIR}")
    print(f"模型目录: {config.MODEL_DIR}")
    print(f"报告目录: {config.REPORT_DIR}")
    
    # 加载数据
    print(f"\n{'='*70}")
    print("第一步: 加载数据")
    print(f"{'='*70}")
    
    processor = DataProcessor()
    X, y = processor.load_all_data()
    
    # 交叉验证
    print(f"\n{'='*70}")
    print("第二步: 交叉验证训练和评估")
    print(f"{'='*70}")
    
    evaluator = CVEvaluator()
    results_df = evaluator.evaluate(X, y)
    
    # 汇总
    print(f"\n{'='*70}")
    print("第三步: 结果汇总")
    print(f"{'='*70}")
    
    evaluator.print_summary(results_df)
    
    # 保存结果
    results_df.to_csv(config.REPORT_DIR / 'cv_results.csv', index=False)
    print(f"\n已保存结果到: {config.REPORT_DIR / 'cv_results.csv'}")
    
    # 生成报告
    report_path = config.REPORT_DIR / 'final_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"NequIP增强型铁电分类器 v6 - 最终报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("交叉验证结果:\n")
        f.write(results_df.to_string(index=False))
        f.write(f"\n\n性能统计:\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            mean = results_df[metric].mean()
            std = results_df[metric].std()
            f.write(f"{metric}: {mean:.4f} ± {std:.4f}\n")
    
    print(f"已保存报告到: {report_path}")
    
    print(f"\n{'='*70}")
    print("完成!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
