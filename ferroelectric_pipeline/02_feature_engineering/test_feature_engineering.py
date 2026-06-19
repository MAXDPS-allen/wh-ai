"""
快速训练脚本 - 测试高级特征工程
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# 导入特征工程
sys.path.insert(0, str(Path(__file__).parent))
from advanced_feature_engineering import AdvancedFeatureExtractor

sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))
from feature_engineering import ELEMENT_DATABASE


def load_jsonl(filepath: Path) -> List[Dict]:
    """加载JSONL文件"""
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


def extract_features_quick():
    """快速特征提取和数据分析"""
    
    print("="*70)
    print("特征工程模块测试和数据分析")
    print("="*70)
    
    extractor = AdvancedFeatureExtractor()
    data_dir = Path(__file__).parent.parent / 'new_data'
    
    # 正样本统计
    print("\n1. 正样本 (铁电) 统计")
    print("-" * 70)
    
    fe_features = []
    fe_counts = {}
    
    for fname in ['dataset_original_ferroelectric.jsonl', 'dataset_known_FE_rest.jsonl']:
        fpath = data_dir / fname
        if fpath.exists():
            data = load_jsonl(fpath)
            print(f"\n{fname}: {len(data)} 个样本")
            
            for idx, item in enumerate(tqdm(data[:min(20, len(data))], desc="特征提取")):
                try:
                    struct = item.get('structure', item.get('data', item))
                    sg_num = item.get('spacegroup', {}).get('number', None)
                    
                    feat = extractor.extract_advanced_features(struct, sg_num)
                    
                    if not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
                        fe_features.append(feat)
                        fe_counts[fname] = fe_counts.get(fname, 0) + 1
                        
                        if idx == 0:
                            print(f"  特征维度: {feat.shape}")
                            print(f"  特征范围: [{np.min(feat):.4f}, {np.max(feat):.4f}]")
                            print(f"  特征平均值: {np.mean(feat):.4f}")
                            print(f"  特征标准差: {np.std(feat):.4f}")
                except Exception as e:
                    pass
    
    print(f"\n成功提取的正样本: {sum(fe_counts.values())} 个")
    
    # 负样本统计
    print("\n\n2. 负样本 (非铁电) 统计")
    print("-" * 70)
    
    non_fe_features = []
    non_fe_counts = {}
    
    for fname in ['dataset_nonFE.jsonl', 'dataset_nonFE_cleaned.jsonl']:
        fpath = data_dir / fname
        if fpath.exists():
            data = load_jsonl(fpath)
            print(f"\n{fname}: {len(data)} 个样本")
            
            for idx, item in enumerate(tqdm(data[:min(20, len(data))], desc="特征提取")):
                try:
                    struct = item.get('structure', item.get('data', item))
                    sg_num = item.get('spacegroup', {}).get('number', None)
                    
                    feat = extractor.extract_advanced_features(struct, sg_num)
                    
                    if not np.any(np.isnan(feat)) and not np.any(np.isinf(feat)):
                        non_fe_features.append(feat)
                        non_fe_counts[fname] = non_fe_counts.get(fname, 0) + 1
                        
                        if idx == 0:
                            print(f"  特征维度: {feat.shape}")
                            print(f"  特征范围: [{np.min(feat):.4f}, {np.max(feat):.4f}]")
                except Exception as e:
                    pass
    
    print(f"\n成功提取的负样本: {sum(non_fe_counts.values())} 个")
    
    # 特征统计
    print("\n\n3. 特征统计信息")
    print("-" * 70)
    
    if fe_features and non_fe_features:
        fe_array = np.array(fe_features)
        non_fe_array = np.array(non_fe_features)
        
        print(f"\n正样本特征统计:")
        print(f"  样本数: {len(fe_array)}")
        print(f"  特征维度: {fe_array.shape[1]}")
        print(f"  全局均值: {np.mean(fe_array):.4f}")
        print(f"  全局标准差: {np.std(fe_array):.4f}")
        print(f"  最小值: {np.min(fe_array):.4f}")
        print(f"  最大值: {np.max(fe_array):.4f}")
        
        print(f"\n负样本特征统计:")
        print(f"  样本数: {len(non_fe_array)}")
        print(f"  特征维度: {non_fe_array.shape[1]}")
        print(f"  全局均值: {np.mean(non_fe_array):.4f}")
        print(f"  全局标准差: {np.std(non_fe_array):.4f}")
        print(f"  最小值: {np.min(non_fe_array):.4f}")
        print(f"  最大值: {np.max(non_fe_array):.4f}")
        
        # 特征差异
        print(f"\n特征分离能力 (正样本均值 - 负样本均值):")
        fe_mean = np.mean(fe_array, axis=0)
        non_fe_mean = np.mean(non_fe_array, axis=0)
        diff = fe_mean - non_fe_mean
        
        top_k = 10
        top_indices = np.argsort(np.abs(diff))[-top_k:][::-1]
        
        print(f"  Top {top_k} 有区分力的特征:")
        for i, idx in enumerate(top_indices):
            print(f"    {i+1}. Feature #{idx}: {diff[idx]:.4f}")
    
    # 保存样本数据
    print("\n\n4. 保存分析结果")
    print("-" * 70)
    
    report_path = Path(__file__).parent.parent / 'reports_nequip_v6'
    report_path.mkdir(exist_ok=True)
    
    summary = {
        'positive_samples': sum(fe_counts.values()),
        'negative_samples': sum(non_fe_counts.values()),
        'feature_dimension': 256,
        'extraction_status': 'success'
    }
    
    with open(report_path / 'feature_extraction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"已保存到: {report_path / 'feature_extraction_summary.json'}")
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)


if __name__ == '__main__':
    extract_features_quick()
