"""
完整材料发现流水线
=============================================
一站式铁电材料发现工作流

流程:
1. GAN生成候选材料特征
2. 逆向设计还原材料描述
3. 合理性验证过滤
4. Materials Project数据库对比
5. 生成最终候选材料报告

使用方法:
    python full_discovery_pipeline.py --n_samples 500 --output_dir discovery_results
"""

import sys
import os
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))


def run_full_pipeline(n_samples: int = 200,
                      output_dir: str = None,
                      min_validity_level: str = 'GOOD',
                      compare_mp: bool = True) -> dict:
    """
    运行完整的材料发现流水线
    
    Args:
        n_samples: 生成的材料数量
        output_dir: 输出目录
        min_validity_level: 最低合理性等级
        compare_mp: 是否与MP数据库对比
    
    Returns:
        包含各阶段结果的字典
    """
    from generate_materials_pipeline import MaterialGenerationPipeline
    from material_validator import MaterialValidator, ValidityLevel
    from mp_comparison import compare_with_mp
    
    # 设置输出目录
    base_dir = Path(__file__).parent.parent
    if output_dir:
        out_path = Path(output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = base_dir / f'discovery_{timestamp}'
    
    out_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'output_dir': str(out_path),
        'timestamp': datetime.now().isoformat(),
        'n_samples': n_samples,
    }
    
    print("="*70)
    print("FERROELECTRIC MATERIAL DISCOVERY PIPELINE")
    print("="*70)
    print(f"Output directory: {out_path}")
    print(f"Samples to generate: {n_samples}")
    print()
    
    # ==========================================
    # 阶段1: 生成材料
    # ==========================================
    print("\n" + "="*70)
    print("STAGE 1: Generating Candidate Materials")
    print("="*70)
    
    pipeline = MaterialGenerationPipeline()
    pipeline.load_models()
    
    raw_df = pipeline.generate_and_save(
        n_samples, 
        output_path=out_path / 'stage1_raw_generated.csv'
    )
    
    results['stage1_generated'] = len(raw_df)
    print(f"\n✓ Generated {len(raw_df)} raw materials")
    
    # ==========================================
    # 阶段2: 合理性验证
    # ==========================================
    print("\n" + "="*70)
    print("STAGE 2: Validity Validation")
    print("="*70)
    
    validated_df = pipeline.validate_materials(raw_df, min_validity_level)
    validated_df.to_csv(out_path / 'stage2_validated.csv', index=False)
    
    results['stage2_validated'] = len(validated_df)
    
    # 按等级统计
    level_counts = validated_df['validity_level'].value_counts().to_dict()
    results['validity_distribution'] = level_counts
    
    print(f"\n✓ Validated {len(validated_df)} materials (>= {min_validity_level})")
    
    # ==========================================
    # 阶段3: Materials Project对比
    # ==========================================
    if compare_mp and len(validated_df) > 0:
        print("\n" + "="*70)
        print("STAGE 3: Materials Project Database Comparison")
        print("="*70)
        
        mp_results = compare_with_mp(
            str(out_path / 'stage2_validated.csv'),
            str(out_path / 'stage3_mp_matched.csv'),
            top_n=5
        )
        
        results['stage3_mp_matches'] = len(mp_results)
        
        if len(mp_results) > 0:
            # 提取铁电候选
            polar_stable = mp_results[
                (mp_results['mp_is_polar'] == True) & 
                (mp_results['mp_is_stable'] == True)
            ]
            
            results['ferroelectric_candidates'] = len(polar_stable)
            
            # 保存铁电候选
            if len(polar_stable) > 0:
                polar_stable.to_csv(
                    out_path / 'ferroelectric_candidates.csv', 
                    index=False
                )
    else:
        results['stage3_mp_matches'] = 0
        results['ferroelectric_candidates'] = 0
    
    # ==========================================
    # 生成最终报告
    # ==========================================
    print("\n" + "="*70)
    print("GENERATING FINAL REPORT")
    print("="*70)
    
    report_path = out_path / 'discovery_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FERROELECTRIC MATERIAL DISCOVERY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Date: {results['timestamp']}\n")
        f.write(f"Output Directory: {results['output_dir']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("PIPELINE SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Stage 1 - Generated:        {results['stage1_generated']} materials\n")
        f.write(f"Stage 2 - Validated:        {results['stage2_validated']} materials\n")
        f.write(f"Stage 3 - MP Matches:       {results.get('stage3_mp_matches', 0)} materials\n")
        f.write(f"Ferroelectric Candidates:   {results.get('ferroelectric_candidates', 0)} materials\n\n")
        
        if 'validity_distribution' in results:
            f.write("-"*70 + "\n")
            f.write("VALIDITY DISTRIBUTION\n")
            f.write("-"*70 + "\n")
            for level, count in results['validity_distribution'].items():
                f.write(f"  {level}: {count}\n")
            f.write("\n")
        
        # 铁电候选详情
        if results.get('ferroelectric_candidates', 0) > 0:
            f.write("-"*70 + "\n")
            f.write("DISCOVERED FERROELECTRIC CANDIDATES\n")
            f.write("-"*70 + "\n\n")
            
            fe_df = pd.read_csv(out_path / 'ferroelectric_candidates.csv')
            for _, row in fe_df.iterrows():
                f.write(f"Material: {row['mp_formula']}\n")
                f.write(f"  MP ID: {row['mp_id']}\n")
                f.write(f"  Point Group: {row['mp_point_group']}\n")
                f.write(f"  Space Group: {row['mp_spacegroup']}\n")
                f.write(f"  Band Gap: {row['mp_band_gap']:.2f} eV\n")
                f.write(f"  Lattice: a={row['mp_a']:.3f}, b={row['mp_b']:.3f}, c={row['mp_c']:.3f} Å\n")
                f.write(f"  Match Score: {row['total_score']:.1f}\n")
                f.write("\n")
        
        f.write("-"*70 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("-"*70 + "\n")
        f.write(f"  stage1_raw_generated.csv     - Raw generated materials\n")
        f.write(f"  stage2_validated.csv         - Validated materials\n")
        f.write(f"  stage3_mp_matched.csv        - MP database matches\n")
        f.write(f"  ferroelectric_candidates.csv - Final FE candidates\n")
        f.write(f"  discovery_report.txt         - This report\n")
    
    print(f"\n✓ Report saved: {report_path}")
    
    # ==========================================
    # 最终总结
    # ==========================================
    print("\n" + "="*70)
    print("DISCOVERY COMPLETE")
    print("="*70)
    print(f"\n📊 Results Summary:")
    print(f"   Generated:              {results['stage1_generated']}")
    print(f"   Validated:              {results['stage2_validated']}")
    print(f"   MP Database Matches:    {results.get('stage3_mp_matches', 0)}")
    print(f"   Ferroelectric Candidates: {results.get('ferroelectric_candidates', 0)}")
    print(f"\n📁 Output: {out_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Full Material Discovery Pipeline')
    parser.add_argument('--n_samples', type=int, default=200, 
                        help='Number of materials to generate')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--min_level', type=str, default='GOOD',
                        choices=['EXCELLENT', 'GOOD', 'ACCEPTABLE'],
                        help='Minimum validity level')
    parser.add_argument('--no-mp', dest='compare_mp', action='store_false',
                        help='Skip MP database comparison')
    args = parser.parse_args()
    
    run_full_pipeline(
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        min_validity_level=args.min_level,
        compare_mp=args.compare_mp
    )


if __name__ == '__main__':
    main()
