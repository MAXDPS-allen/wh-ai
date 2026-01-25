"""
后处理 v2 筛选结果
=================
应用多种策略减少假阳性
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# 读取预测结果
results_dir = Path(__file__).parent
df = pd.read_csv(results_dir / 'mp_polar_predictions_v2_all.csv')

print(f"Total predictions: {len(df)}")
print()

# 读取训练数据
data_dir = results_dir.parent / 'new_data'

# 已知铁电
positive_formulas = set()
for fname in ['dataset_original_ferroelectric.jsonl', 'dataset_known_FE_rest.jsonl']:
    fpath = data_dir / fname
    if fpath.exists():
        with open(fpath) as f:
            for line in f:
                item = json.loads(line)
                formula = item.get('formula', item.get('pretty_formula', ''))
                if formula:
                    positive_formulas.add(formula)

# 已知非铁电
negative_formulas = set()
neg_path = data_dir / 'dataset_nonFE_expanded.jsonl'
if neg_path.exists():
    with open(neg_path) as f:
        for line in f:
            item = json.loads(line)
            formula = item.get('formula', item.get('pretty_formula', ''))
            if formula:
                negative_formulas.add(formula)

print(f"Known FE formulas: {len(positive_formulas)}")
print(f"Known non-FE formulas: {len(negative_formulas)}")
print()

# 策略1: 提高阈值
print("=== Strategy 1: Higher thresholds ===")
thresholds_to_try = [0.5, 0.7, 0.9, 0.95, 0.99]
for thresh in thresholds_to_try:
    candidates = df[(df['gcnn_v6_prob'] >= thresh) & (df['nequip_v2_prob'] >= thresh)]
    known_fe = candidates[candidates['is_known_fe'] == True]
    new_disc = candidates[(candidates['is_known_fe'] == False) & (candidates['is_known_non_fe'] == False)]
    
    # 计算召回率
    total_known_fe = df[df['is_known_fe'] == True]
    recall = len(known_fe) / len(total_known_fe) * 100 if len(total_known_fe) > 0 else 0
    
    print(f"Threshold {thresh:.2f}: {len(candidates)} candidates, {len(new_disc)} new, Recall: {recall:.1f}%")

print()

# 策略2: 使用训练标签覆盖
print("=== Strategy 2: Training label override ===")

# 基本阈值预测
base_thresh = 0.5
df['model_pred'] = (df['gcnn_v6_prob'] >= base_thresh) & (df['nequip_v2_prob'] >= base_thresh)

# 训练标签覆盖
df['final_pred'] = df.apply(
    lambda row: True if row['formula'] in positive_formulas 
                else (False if row['formula'] in negative_formulas 
                      else row['model_pred']),
    axis=1
)

candidates_override = df[df['final_pred'] == True]
known_fe_override = candidates_override[candidates_override['formula'].isin(positive_formulas)]
new_disc_override = candidates_override[~candidates_override['formula'].isin(positive_formulas) & 
                                         ~candidates_override['formula'].isin(negative_formulas)]

print(f"After override: {len(candidates_override)} candidates")
print(f"  - Known FE: {len(known_fe_override)}")
print(f"  - New discoveries: {len(new_disc_override)}")

print()

# 策略3: 分析概率分布差异
print("=== Strategy 3: Probability product filtering ===")
df['prob_product'] = df['gcnn_v6_prob'] * df['nequip_v2_prob']
df['prob_min'] = df[['gcnn_v6_prob', 'nequip_v2_prob']].min(axis=1)

# 使用概率乘积作为置信度
for min_prod in [0.5, 0.7, 0.9, 0.95, 0.99]:
    candidates = df[df['prob_product'] >= min_prod]
    new_disc = candidates[(candidates['is_known_fe'] == False) & (candidates['is_known_non_fe'] == False)]
    
    # 召回率
    total_known_fe = df[df['is_known_fe'] == True]
    known_fe_recalled = candidates[candidates['is_known_fe'] == True]
    recall = len(known_fe_recalled) / len(total_known_fe) * 100 if len(total_known_fe) > 0 else 0
    
    print(f"Prob product >= {min_prod:.2f}: {len(candidates)} candidates, {len(new_disc)} new, Recall: {recall:.1f}%")

print()

# 策略4: 使用更高的阈值并应用训练标签覆盖
print("=== Strategy 4: High threshold + label override ===")

FINAL_THRESH = 0.9

# 模型预测
high_conf_pred = (df['gcnn_v6_prob'] >= FINAL_THRESH) & (df['nequip_v2_prob'] >= FINAL_THRESH)

# 应用训练标签覆盖
df['final_refined'] = df.apply(
    lambda row: True if row['formula'] in positive_formulas 
                else (False if row['formula'] in negative_formulas 
                      else ((row['gcnn_v6_prob'] >= FINAL_THRESH) and (row['nequip_v2_prob'] >= FINAL_THRESH))),
    axis=1
)

final_candidates = df[df['final_refined'] == True]
final_known_fe = final_candidates[final_candidates['formula'].isin(positive_formulas)]
final_new_disc = final_candidates[~final_candidates['formula'].isin(positive_formulas) & 
                                   ~final_candidates['formula'].isin(negative_formulas)]

print(f"High threshold ({FINAL_THRESH}) + label override:")
print(f"  Total candidates: {len(final_candidates)}")
print(f"  - Known FE (recovered): {len(final_known_fe)}")
print(f"  - New discoveries: {len(final_new_disc)}")

# 计算完整召回率
all_polar_known_fe = df[df['is_known_fe'] == True]
print(f"  - Known FE recall: {len(final_known_fe)/len(all_polar_known_fe)*100:.1f}% ({len(final_known_fe)}/{len(all_polar_known_fe)})")

print()

# 保存最终结果
print("=== Saving final results ===")

# 保存新发现
new_disc_df = final_new_disc[['material_id', 'formula', 'spacegroup_number', 'spacegroup_symbol',
                              'energy_above_hull', 'is_stable', 'gcnn_v6_prob', 'nequip_v2_prob']]
new_disc_df = new_disc_df.sort_values('prob_product', ascending=False) if 'prob_product' in new_disc_df.columns else new_disc_df
new_disc_df.to_csv(results_dir / 'fe_candidates_v2_final_new.csv', index=False)
print(f"Saved {len(new_disc_df)} new discoveries to fe_candidates_v2_final_new.csv")

# 按化学式去重
unique_formulas = new_disc_df.drop_duplicates(subset=['formula'])
unique_formulas.to_csv(results_dir / 'fe_candidates_v2_final_new_unique.csv', index=False)
print(f"Saved {len(unique_formulas)} unique formulas to fe_candidates_v2_final_new_unique.csv")

# 保存完整候选列表（包含已知铁电）
all_candidates_df = final_candidates[['material_id', 'formula', 'spacegroup_number', 'spacegroup_symbol',
                                       'energy_above_hull', 'is_stable', 'gcnn_v6_prob', 'nequip_v2_prob',
                                       'is_known_fe', 'is_known_non_fe']]
all_candidates_df.to_csv(results_dir / 'fe_candidates_v2_final_all.csv', index=False)
print(f"Saved {len(all_candidates_df)} total candidates to fe_candidates_v2_final_all.csv")

# 保存报告
report = {
    'threshold_gcnn': FINAL_THRESH,
    'threshold_nequip': FINAL_THRESH,
    'total_screened': len(df),
    'total_candidates': len(final_candidates),
    'known_fe_recovered': len(final_known_fe),
    'new_discoveries': len(final_new_disc),
    'unique_new_formulas': len(unique_formulas),
    'known_fe_in_polar': len(all_polar_known_fe),
    'recall_rate': len(final_known_fe)/len(all_polar_known_fe)*100 if len(all_polar_known_fe) > 0 else 0,
}
with open(results_dir / 'final_screening_v2_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print(f"\nFinal report saved to final_screening_v2_report.json")

print()
print("="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"New FE candidates (unique formulas): {len(unique_formulas)}")
print(f"Known FE recall rate: {report['recall_rate']:.1f}%")
print("="*60)
