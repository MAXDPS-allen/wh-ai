"""
Final MP Polar Screening - Combined v1 + v2 Analysis
====================================================
结合 v1 和 v2 模型的预测结果，生成不同置信度级别的候选列表
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# 读取预测结果
results_dir = Path('/home/ubuntu/ai_wh/wh-ai/mp_polar_screening')
data_dir = Path('/home/ubuntu/ai_wh/wh-ai/new_data')

v1 = pd.read_csv(results_dir / 'mp_polar_predictions_all.csv')
v2 = pd.read_csv(results_dir / 'mp_polar_predictions_v2_all.csv')

# 合并
merged = v1.merge(v2[['material_id', 'gcnn_v6_prob', 'nequip_v2_prob']], 
                  on='material_id', how='inner')

print(f"Total materials: {len(merged)}")

# 加载训练集
positive_formulas = set()
negative_formulas = set()

for fname in ['dataset_original_ferroelectric.jsonl', 'dataset_known_FE_rest.jsonl']:
    fpath = data_dir / fname
    if fpath.exists():
        with open(fpath) as f:
            for line in f:
                item = json.loads(line)
                formula = item.get('formula', item.get('pretty_formula', ''))
                if formula:
                    positive_formulas.add(formula)

neg_path = data_dir / 'dataset_nonFE_expanded.jsonl'
if neg_path.exists():
    with open(neg_path) as f:
        for line in f:
            item = json.loads(line)
            formula = item.get('formula', item.get('pretty_formula', ''))
            if formula:
                negative_formulas.add(formula)

print(f"Training positives: {len(positive_formulas)}")
print(f"Training negatives: {len(negative_formulas)}")

# 计算综合得分
merged['avg_prob'] = (merged['gcnn_probability'] + merged['nequip_probability'] + 
                       merged['gcnn_v6_prob'] + merged['nequip_v2_prob']) / 4
merged['min_prob'] = merged[['gcnn_probability', 'nequip_probability', 
                             'gcnn_v6_prob', 'nequip_v2_prob']].min(axis=1)
merged['max_prob'] = merged[['gcnn_probability', 'nequip_probability', 
                             'gcnn_v6_prob', 'nequip_v2_prob']].max(axis=1)

# v1 和 v2 的共识
merged['v1_consensus'] = merged['consensus_ferroelectric']
merged['v2_consensus'] = (merged['gcnn_v6_prob'] >= 0.87) & (merged['nequip_v2_prob'] >= 0.94)
merged['both_consensus'] = merged['v1_consensus'] & merged['v2_consensus']

# 标记训练样本
merged['is_training_positive'] = merged['formula'].isin(positive_formulas)
merged['is_training_negative'] = merged['formula'].isin(negative_formulas)

# 已知铁电
known_fe = merged[merged['is_known_fe'] == True]
print(f"Known FE in polar materials: {len(known_fe)}")

# 定义置信度级别
confidence_levels = {
    'high': {
        'description': 'Both v1 and v2 agree (strictest)',
        'condition': merged['both_consensus'] == True,
    },
    'medium_high': {
        'description': 'Min probability >= 0.5 across all 4 models',
        'condition': merged['min_prob'] >= 0.5,
    },
    'medium': {
        'description': 'Average probability >= 0.7',
        'condition': merged['avg_prob'] >= 0.7,
    },
    'low': {
        'description': 'Either v1 or v2 consensus',
        'condition': merged['v1_consensus'] | merged['v2_consensus'],
    },
}

print("\n" + "="*70)
print("CONFIDENCE LEVEL ANALYSIS")
print("="*70)

all_results = {}
for level, config in confidence_levels.items():
    candidates = merged[config['condition']]
    
    # 排除训练负样本，应用标签覆盖
    # 训练正样本保留，训练负样本排除
    final_candidates = candidates[~candidates['is_training_negative']]
    
    # 添加被遗漏的训练正样本
    missed_positives = merged[merged['is_training_positive'] & ~config['condition']]
    if len(missed_positives) > 0:
        # 将遗漏的正样本也加入
        final_candidates = pd.concat([final_candidates, missed_positives]).drop_duplicates('material_id')
    
    # 统计
    new_discoveries = final_candidates[~final_candidates['is_training_positive'] & ~final_candidates['is_training_negative']]
    unique_new = new_discoveries['formula'].nunique()
    
    # 召回率 (训练正样本中有多少在候选中)
    training_pos_recalled = final_candidates[final_candidates['is_training_positive']]
    recall = len(training_pos_recalled) / len(positive_formulas) * 100 if len(positive_formulas) > 0 else 0
    
    # 保存结果
    all_results[level] = {
        'candidates': final_candidates,
        'new_discoveries': new_discoveries,
        'unique_new': unique_new,
        'recall': recall,
    }
    
    print(f"\n{level.upper()}: {config['description']}")
    print(f"  Total candidates: {len(final_candidates)}")
    print(f"  New discoveries: {len(new_discoveries)} ({unique_new} unique formulas)")
    print(f"  Training recall: {recall:.1f}%")

# 保存不同置信度级别的结果
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

for level, results in all_results.items():
    new_disc = results['new_discoveries']
    
    # 按平均概率排序
    new_disc_sorted = new_disc.sort_values('avg_prob', ascending=False)
    
    # 选择输出列
    output_cols = ['material_id', 'formula', 'spacegroup_number', 'spacegroup_symbol',
                   'energy_above_hull', 'is_stable', 
                   'gcnn_probability', 'nequip_probability', 'gcnn_v6_prob', 'nequip_v2_prob',
                   'avg_prob', 'min_prob']
    
    # 完整列表
    output_df = new_disc_sorted[[c for c in output_cols if c in new_disc_sorted.columns]]
    output_path = results_dir / f'fe_candidates_final_{level}.csv'
    output_df.to_csv(output_path, index=False)
    print(f"Saved {level}: {len(output_df)} entries -> {output_path.name}")
    
    # 唯一化学式
    unique_df = output_df.drop_duplicates(subset=['formula'])
    unique_path = results_dir / f'fe_candidates_final_{level}_unique.csv'
    unique_df.to_csv(unique_path, index=False)
    print(f"  Unique formulas: {len(unique_df)} -> {unique_path.name}")

# 保存综合报告
report = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_materials_screened': len(merged),
    'training_positives': len(positive_formulas),
    'training_negatives': len(negative_formulas),
    'known_fe_in_polar': len(known_fe),
    'confidence_levels': {},
}

for level, results in all_results.items():
    report['confidence_levels'][level] = {
        'total_candidates': len(results['candidates']),
        'new_discoveries': len(results['new_discoveries']),
        'unique_new_formulas': results['unique_new'],
        'training_recall': results['recall'],
    }

report_path = results_dir / 'final_combined_screening_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f"\nReport saved: {report_path.name}")

# 最终总结
print("\n" + "="*70)
print("FINAL SUMMARY - RECOMMENDED CANDIDATES")
print("="*70)
print(f"HIGH confidence (most reliable): {all_results['high']['unique_new']} unique formulas")
print(f"MEDIUM-HIGH confidence (conservative): {all_results['medium_high']['unique_new']} unique formulas")
print(f"MEDIUM confidence (balanced): {all_results['medium']['unique_new']} unique formulas")
print(f"LOW confidence (exploratory): {all_results['low']['unique_new']} unique formulas")
print("="*70)
