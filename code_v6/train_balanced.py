#!/usr/bin/env python3
"""
NequIP增强型铁电分类器 v6 - 平衡版本
目标：在准确率和召回率之间找到最佳平衡点
策略：多阈值测试，报告不同权衡点的性能
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from feature_engineering import UnifiedFeatureExtractor
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import (
    RandomForestClassifier, 
    ExtraTreesClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


@dataclass
class Config:
    base_dir: Path = Path("/home/ubuntu/ai_wh/wh-ai")
    data_dir: Path = None
    report_dir: Path = None
    n_splits: int = 5
    random_state: int = 42
    
    def __post_init__(self):
        self.data_dir = self.base_dir / "new_data"
        self.report_dir = self.base_dir / "reports_nequip_v6"
        self.report_dir.mkdir(exist_ok=True)


class DataProcessor:
    """数据处理器 - 使用基础64维特征"""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_extractor = UnifiedFeatureExtractor()
        
    def load_jsonl(self, filepath: Path) -> List[Dict]:
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def process_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        fe_files = [
            ("dataset_original_ferroelectric.jsonl", 1),
            ("dataset_known_FE_rest.jsonl", 1),
        ]
        
        non_fe_files = [
            ("dataset_nonFE.jsonl", 0),
            ("dataset_nonFE_cleaned.jsonl", 0),
            ("dataset_nonFE_expanded.jsonl", 0),
        ]
        
        all_features = []
        all_labels = []
        
        print("处理正样本 (铁电材料)...")
        for filename, label in fe_files:
            filepath = self.config.data_dir / filename
            if filepath.exists():
                data = self.load_jsonl(filepath)
                print(f"  处理 {filename} ({len(data)} 个样本)...")
                features = self._extract_features_batch(data)
                all_features.extend(features)
                all_labels.extend([label] * len(features))
        
        print("\n处理负样本 (非铁电材料)...")
        seen_formulas = set()
        for filename, label in non_fe_files:
            filepath = self.config.data_dir / filename
            if filepath.exists():
                data = self.load_jsonl(filepath)
                unique_data = []
                for item in data:
                    formula = item.get('formula', str(item.get('structure', '')))
                    if formula not in seen_formulas:
                        seen_formulas.add(formula)
                        unique_data.append(item)
                
                print(f"  处理 {filename} ({len(unique_data)}/{len(data)} 个唯一样本)...")
                features = self._extract_features_batch(unique_data)
                all_features.extend(features)
                all_labels.extend([label] * len(features))
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        print(f"\n数据集统计:")
        print(f"  正样本 (FE): {sum(y == 1)}")
        print(f"  负样本 (non-FE): {sum(y == 0)}")
        print(f"  特征维度: {X.shape[1]}")
        
        return X, y
    
    def _extract_features_batch(self, data: List[Dict]) -> List[np.ndarray]:
        features = []
        for item in tqdm(data, desc=f"特征提取"):
            try:
                struct = item.get('structure', item.get('data', item))
                sg_num = item.get('spacegroup', {}).get('number', None)
                feat = self.feature_extractor.extract_from_structure_dict(struct, sg_num)
                if not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
                    features.append(feat)
                else:
                    features.append(np.zeros(64))
            except Exception:
                features.append(np.zeros(64))
        return features


class BalancedEvaluator:
    """平衡评估器 - 测试不同的准确率-召回率权衡"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def _apply_smote(self, X: np.ndarray, y: np.ndarray, target_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE过采样 - 可调整目标比例"""
        minority_mask = y == 1
        X_minority = X[minority_mask]
        n_minority = len(X_minority)
        n_majority = len(X) - n_minority
        
        # 目标: minority / majority = target_ratio
        n_to_generate = int(n_majority * target_ratio) - n_minority
        
        if n_to_generate <= 0:
            return X, y
        
        synthetic = []
        np.random.seed(self.config.random_state)
        
        for _ in range(n_to_generate):
            idx = np.random.randint(0, n_minority)
            idx2 = np.random.randint(0, n_minority)
            alpha = np.random.random()
            new_sample = X_minority[idx] + alpha * (X_minority[idx2] - X_minority[idx])
            synthetic.append(new_sample)
        
        synthetic = np.array(synthetic)
        X_resampled = np.vstack([X, synthetic])
        y_resampled = np.concatenate([y, np.ones(len(synthetic))])
        
        return X_resampled, y_resampled
    
    def train_fold(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   fold: int) -> Dict:
        """训练单个折，返回不同阈值的结果"""
        
        n_pos = sum(y_train == 1)
        n_neg = sum(y_train == 0)
        ratio = n_neg / n_pos
        print(f"训练集: {len(y_train)} 样本 (正:{n_pos}, 负:{n_neg}, 比例 1:{ratio:.1f})")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 使用 SMOTE + 多模型集成
        print("训练集成模型...")
        X_smote, y_smote = self._apply_smote(X_train_scaled, y_train, target_ratio=0.3)
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=500,
            class_weight={0: 1, 1: ratio * 2},
            random_state=self.config.random_state,
            n_jobs=-1
        )
        rf.fit(X_smote, y_smote)
        rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
        
        # Extra Trees
        et = ExtraTreesClassifier(
            n_estimators=500,
            class_weight={0: 1, 1: ratio * 2},
            random_state=self.config.random_state,
            n_jobs=-1
        )
        et.fit(X_smote, y_smote)
        et_proba = et.predict_proba(X_test_scaled)[:, 1]
        
        # Gradient Boosting
        gb = GradientBoostingClassifier(
            n_estimators=200,
            random_state=self.config.random_state
        )
        gb.fit(X_smote, y_smote)
        gb_proba = gb.predict_proba(X_test_scaled)[:, 1]
        
        # 集成概率 (加权平均)
        ensemble_proba = (rf_proba + et_proba + gb_proba) / 3
        
        # 测试不同阈值
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results_by_threshold = {}
        
        for thresh in thresholds:
            y_pred = (ensemble_proba >= thresh).astype(int)
            results_by_threshold[thresh] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
            }
        
        # 寻找最佳平衡点 (F1最大)
        best_f1_thresh = max(thresholds, key=lambda t: results_by_threshold[t]['f1'])
        
        # 寻找满足recall >= 95%的最高精度阈值
        recall_95_thresh = None
        for thresh in sorted(thresholds):
            if results_by_threshold[thresh]['recall'] >= 0.95:
                if recall_95_thresh is None or results_by_threshold[thresh]['accuracy'] > results_by_threshold[recall_95_thresh]['accuracy']:
                    recall_95_thresh = thresh
        
        # 寻找满足recall >= 90%的最高精度阈值
        recall_90_thresh = None
        for thresh in sorted(thresholds):
            if results_by_threshold[thresh]['recall'] >= 0.90:
                if recall_90_thresh is None or results_by_threshold[thresh]['accuracy'] > results_by_threshold[recall_90_thresh]['accuracy']:
                    recall_90_thresh = thresh
        
        results = {
            'fold': fold,
            'roc_auc': roc_auc_score(y_test, ensemble_proba),
            'results_by_threshold': results_by_threshold,
            'best_f1_threshold': best_f1_thresh,
            'best_f1_results': results_by_threshold[best_f1_thresh],
            'recall_95_threshold': recall_95_thresh,
            'recall_95_results': results_by_threshold[recall_95_thresh] if recall_95_thresh else None,
            'recall_90_threshold': recall_90_thresh,
            'recall_90_results': results_by_threshold[recall_90_thresh] if recall_90_thresh else None,
        }
        
        print(f"\nFold {fold + 1} 结果:")
        print(f"  ROC-AUC: {results['roc_auc']:.4f}")
        print(f"  最佳F1阈值 {best_f1_thresh}: Acc={results_by_threshold[best_f1_thresh]['accuracy']:.4f}, Recall={results_by_threshold[best_f1_thresh]['recall']:.4f}")
        if recall_95_thresh:
            print(f"  Recall≥95%阈值 {recall_95_thresh}: Acc={results_by_threshold[recall_95_thresh]['accuracy']:.4f}")
        if recall_90_thresh:
            print(f"  Recall≥90%阈值 {recall_90_thresh}: Acc={results_by_threshold[recall_90_thresh]['accuracy']:.4f}")
        
        return results
    
    def run_cv(self, X: np.ndarray, y: np.ndarray) -> List[Dict]:
        print(f"\n开始 {self.config.n_splits} 折交叉验证...\n")
        
        skf = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state
        )
        
        results = []
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\n{'='*70}")
            print(f"Fold {fold + 1}/{self.config.n_splits}")
            print(f"{'='*70}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            fold_results = self.train_fold(X_train, y_train, X_test, y_test, fold)
            results.append(fold_results)
        
        return results


def main():
    print("=" * 70)
    print("NequIP增强型铁电分类器 v6 - 平衡版本")
    print("目标：测试不同的准确率-召回率权衡点")
    print("=" * 70)
    
    config = Config()
    
    print(f"\n{'='*70}")
    print("第一步: 加载数据")
    print(f"{'='*70}")
    
    processor = DataProcessor(config)
    X, y = processor.process_all_data()
    
    print(f"\n{'='*70}")
    print("第二步: 交叉验证训练和评估")
    print(f"{'='*70}")
    
    evaluator = BalancedEvaluator(config)
    results = evaluator.run_cv(X, y)
    
    # 汇总结果
    print(f"\n{'='*70}")
    print("交叉验证总结 - 不同阈值的平均性能")
    print(f"{'='*70}\n")
    
    # 收集所有阈值的结果
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    summary = []
    
    for thresh in thresholds:
        acc_list = [r['results_by_threshold'][thresh]['accuracy'] for r in results]
        prec_list = [r['results_by_threshold'][thresh]['precision'] for r in results]
        recall_list = [r['results_by_threshold'][thresh]['recall'] for r in results]
        f1_list = [r['results_by_threshold'][thresh]['f1'] for r in results]
        
        summary.append({
            'threshold': thresh,
            'accuracy': np.mean(acc_list),
            'precision': np.mean(prec_list),
            'recall': np.mean(recall_list),
            'f1': np.mean(f1_list),
        })
    
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("ROC-AUC统计:")
    print(f"{'='*70}")
    roc_auc_list = [r['roc_auc'] for r in results]
    print(f"  平均: {np.mean(roc_auc_list):.4f} ± {np.std(roc_auc_list):.4f}")
    
    print(f"\n{'='*70}")
    print("关键发现:")
    print(f"{'='*70}")
    
    # 找到最佳F1阈值
    best_f1_idx = summary_df['f1'].idxmax()
    best_f1_row = summary_df.iloc[best_f1_idx]
    print(f"\n最佳F1平衡点 (阈值={best_f1_row['threshold']}):")
    print(f"  Accuracy:  {best_f1_row['accuracy']*100:.2f}%")
    print(f"  Recall:    {best_f1_row['recall']*100:.2f}%")
    print(f"  Precision: {best_f1_row['precision']*100:.2f}%")
    print(f"  F1-Score:  {best_f1_row['f1']*100:.2f}%")
    
    # 找到recall >= 95%的最佳准确率
    high_recall = summary_df[summary_df['recall'] >= 0.95]
    if len(high_recall) > 0:
        best_hr = high_recall.loc[high_recall['accuracy'].idxmax()]
        print(f"\nRecall ≥ 95%时的最佳准确率 (阈值={best_hr['threshold']}):")
        print(f"  Accuracy:  {best_hr['accuracy']*100:.2f}%")
        print(f"  Recall:    {best_hr['recall']*100:.2f}%")
    
    # 找到recall >= 90%的最佳准确率
    recall_90 = summary_df[summary_df['recall'] >= 0.90]
    if len(recall_90) > 0:
        best_90 = recall_90.loc[recall_90['accuracy'].idxmax()]
        print(f"\nRecall ≥ 90%时的最佳准确率 (阈值={best_90['threshold']}):")
        print(f"  Accuracy:  {best_90['accuracy']*100:.2f}%")
        print(f"  Recall:    {best_90['recall']*100:.2f}%")
    
    print(f"\n{'='*70}")
    
    # 保存结果
    summary_df.to_csv(config.report_dir / "cv_results_balanced.csv", index=False)
    print(f"\n已保存结果到: {config.report_dir / 'cv_results_balanced.csv'}")
    
    print(f"\n{'='*70}")
    print("完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
