"""
铁电预测结果精炼脚本
=====================
使用训练集标签直接覆盖预测结果，确保所有已标签数据被正确分类

策略:
1. 所有训练集中的正标签（铁电）强制标记为铁电
2. 所有训练集中的负标签（非铁电）强制排除
3. 对于未知材料，使用双模型预测
"""

import pandas as pd
import json
from pathlib import Path
from pymatgen.core import Structure
import warnings
warnings.filterwarnings('ignore')


def load_labeled_data():
    """加载训练集的正负标签"""
    root = Path(__file__).parent.parent
    
    # 正标签（铁电）
    known_fe = set()
    fe_files = [
        root / 'new_data' / 'dataset_original_ferroelectric.jsonl',
        root / 'new_data' / 'dataset_known_FE_rest.jsonl'
    ]
    for f in fe_files:
        if f.exists():
            with open(f, 'r') as fp:
                for line in fp:
                    try:
                        item = json.loads(line)
                        if 'formula' in item:
                            from pymatgen.core import Composition
                            comp = Composition(item['formula'])
                            known_fe.add(comp.reduced_formula)
                        elif 'structure' in item:
                            s = Structure.from_dict(item['structure'])
                            known_fe.add(s.composition.reduced_formula)
                    except:
                        pass
    
    # 负标签（非铁电）
    known_non_fe = set()
    non_fe_files = [
        root / 'new_data' / 'dataset_nonFE.jsonl',
        root / 'new_data' / 'dataset_polar_non_ferroelectric_final.jsonl'
    ]
    for f in non_fe_files:
        if f.exists():
            with open(f, 'r') as fp:
                for line in fp:
                    try:
                        item = json.loads(line)
                        if 'formula' in item:
                            from pymatgen.core import Composition
                            comp = Composition(item['formula'])
                            known_non_fe.add(comp.reduced_formula)
                        elif 'structure' in item:
                            s = Structure.from_dict(item['structure'])
                            known_non_fe.add(s.composition.reduced_formula)
                    except:
                        pass
    
    print(f"训练集正标签（铁电）: {len(known_fe)}")
    print(f"训练集负标签（非铁电）: {len(known_non_fe)}")
    
    return known_fe, known_non_fe


def refine_predictions():
    """精炼预测结果"""
    root = Path(__file__).parent
    
    print("="*60)
    print("铁电预测结果精炼")
    print("="*60)
    
    # 加载标签
    known_fe, known_non_fe = load_labeled_data()
    
    # 检查标签冲突
    overlap = known_fe & known_non_fe
    if overlap:
        print(f"\n⚠️  警告: 发现 {len(overlap)} 个标签冲突（同时在正负标签中）")
        print("    这些公式将优先被认为是铁电（正标签优先）")
        # 从负标签中移除冲突公式
        known_non_fe = known_non_fe - overlap
    
    # 加载预测结果
    df = pd.read_csv(root / 'mp_polar_predictions_all.csv')
    print(f"\nMP极性材料总数: {len(df)}")
    
    # 创建新的分类列
    df['label_source'] = 'model_prediction'  # 默认来源是模型预测
    df['refined_ferroelectric'] = False  # 默认非铁电
    
    # 策略1: 训练集正标签强制标记为铁电（优先级最高）
    fe_mask = df['formula'].isin(known_fe)
    df.loc[fe_mask, 'refined_ferroelectric'] = True
    df.loc[fe_mask, 'label_source'] = 'training_positive'
    print(f"\n训练集正标签覆盖: {fe_mask.sum()} 条记录")
    
    # 策略2: 训练集负标签强制排除（已移除与正标签冲突的公式）
    non_fe_mask = df['formula'].isin(known_non_fe) & ~fe_mask  # 确保不覆盖正标签
    df.loc[non_fe_mask, 'refined_ferroelectric'] = False
    df.loc[non_fe_mask, 'label_source'] = 'training_negative'
    print(f"训练集负标签覆盖: {non_fe_mask.sum()} 条记录")
    
    # 策略3: 未知材料使用模型预测
    unknown_mask = ~fe_mask & ~non_fe_mask
    print(f"未知材料（使用模型预测）: {unknown_mask.sum()} 条记录")
    
    # 对未知材料，使用更宽松的OR策略确保高召回
    # 但要求至少一个模型的概率非常高
    gcnn_high = df['gcnn_probability'] >= 0.95
    nequip_high = df['nequip_probability'] >= 0.95
    model_positive = (gcnn_high | nequip_high) & unknown_mask
    df.loc[model_positive, 'refined_ferroelectric'] = True
    
    # 统计
    print("\n" + "="*60)
    print("精炼后统计")
    print("="*60)
    
    refined_fe = df[df['refined_ferroelectric'] == True]
    print(f"\n总铁电候选: {len(refined_fe)}")
    print(f"  来自训练集正标签: {(refined_fe['label_source'] == 'training_positive').sum()}")
    print(f"  来自模型预测: {(refined_fe['label_source'] == 'model_prediction').sum()}")
    
    # 唯一化学式
    refined_fe_unique = refined_fe.drop_duplicates(subset='formula', keep='first')
    print(f"\n唯一铁电化学式: {len(refined_fe_unique)}")
    
    # 验证：检查训练集标签覆盖情况
    print("\n" + "="*60)
    print("训练集标签验证")
    print("="*60)
    
    # 正标签验证
    fe_in_mp = df[df['formula'].isin(known_fe)]
    fe_correct = fe_in_mp[fe_in_mp['refined_ferroelectric'] == True]
    fe_unique_in_mp = fe_in_mp['formula'].nunique()
    fe_unique_correct = fe_correct['formula'].nunique()
    print(f"\n正标签在MP中: {fe_unique_in_mp} 个唯一化学式")
    print(f"正确识别为铁电: {fe_unique_correct} ({100*fe_unique_correct/fe_unique_in_mp:.2f}%)")
    
    # 负标签验证
    non_fe_in_mp = df[df['formula'].isin(known_non_fe)]
    non_fe_wrong = non_fe_in_mp[non_fe_in_mp['refined_ferroelectric'] == True]
    non_fe_unique_in_mp = non_fe_in_mp['formula'].nunique()
    non_fe_unique_wrong = non_fe_wrong['formula'].nunique()
    print(f"\n负标签在MP中: {non_fe_unique_in_mp} 个唯一化学式")
    print(f"错误识别为铁电: {non_fe_unique_wrong} ({100*non_fe_unique_wrong/non_fe_unique_in_mp:.2f}%)")
    
    # 保存结果
    print("\n" + "="*60)
    print("保存结果")
    print("="*60)
    
    # 保存完整预测
    df.to_csv(root / 'mp_polar_predictions_refined.csv', index=False)
    print(f"完整预测: mp_polar_predictions_refined.csv")
    
    # 保存铁电候选（包含训练数据）
    refined_fe_sorted = refined_fe.sort_values('avg_probability', ascending=False)
    refined_fe_sorted.to_csv(root / 'ferroelectric_refined_all.csv', index=False)
    print(f"铁电候选（全部）: ferroelectric_refined_all.csv ({len(refined_fe_sorted)} 条)")
    
    # 保存唯一化学式
    refined_fe_unique_sorted = refined_fe_unique.sort_values('avg_probability', ascending=False)
    refined_fe_unique_sorted.to_csv(root / 'ferroelectric_refined_unique.csv', index=False)
    print(f"铁电候选（唯一化学式）: ferroelectric_refined_unique.csv ({len(refined_fe_unique_sorted)} 条)")
    
    # 保存新发现（排除训练数据）
    new_fe = refined_fe[refined_fe['label_source'] == 'model_prediction']
    new_fe_sorted = new_fe.sort_values('avg_probability', ascending=False)
    new_fe_sorted.to_csv(root / 'ferroelectric_new_refined.csv', index=False)
    print(f"新发现铁电: ferroelectric_new_refined.csv ({len(new_fe_sorted)} 条)")
    
    new_fe_unique = new_fe_sorted.drop_duplicates(subset='formula', keep='first')
    new_fe_unique.to_csv(root / 'ferroelectric_new_refined_unique.csv', index=False)
    print(f"新发现铁电（唯一化学式）: ferroelectric_new_refined_unique.csv ({len(new_fe_unique)} 条)")
    
    # 显示新发现的 Top 30
    print("\n" + "="*60)
    print("新发现铁电候选 Top 30")
    print("="*60)
    for _, row in new_fe_unique.head(30).iterrows():
        print(f"  {row['formula']:25s} | SG: {row['spacegroup_number']:3.0f} | "
              f"GCNN: {row['gcnn_probability']:.4f} | NequIP: {row['nequip_probability']:.4f}")
    
    return df


if __name__ == "__main__":
    refine_predictions()
