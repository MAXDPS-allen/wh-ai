"""
Materials Project 数据库对比模块
=============================================
将生成的候选材料与MP数据库对比，找出真实存在的材料

功能:
1. 按化学系统搜索MP数据库
2. 对比晶格参数相似度
3. 对比空间群
4. 计算综合匹配分数
5. 生成对比报告

使用方法:
    python mp_comparison.py --input validated_materials.csv --output mp_matched.csv
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

# Materials Project API
try:
    from mp_api.client import MPRester
    HAS_MP_API = True
except ImportError:
    HAS_MP_API = False
    print("Warning: mp_api not installed. Install with: pip install mp-api")

from pymatgen.core import Composition


# ==========================================
# 1. 配置
# ==========================================
class MPConfig:
    # API配置
    API_KEY = "1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq"
    
    # 匹配阈值
    VOLUME_TOLERANCE = 0.15      # 体积差异容忍度 (15%)
    LATTICE_TOLERANCE = 0.10    # 晶格参数容忍度 (10%)
    ANGLE_TOLERANCE = 5.0       # 角度容忍度 (度)
    
    # 路径
    BASE_DIR = Path(__file__).parent.parent
    OUTPUT_DIR = BASE_DIR / 'mp_comparison'
    
    # 请求限制
    REQUEST_DELAY = 0.5  # 请求间隔 (秒)
    MAX_RESULTS_PER_SYSTEM = 50  # 每个化学系统最多返回结果数


# ==========================================
# 2. 匹配结果类
# ==========================================
@dataclass
class MatchResult:
    """匹配结果"""
    generated_id: int
    generated_formula: str
    mp_id: str
    mp_formula: str
    mp_spacegroup: int
    
    # 匹配分数
    total_score: float
    composition_match: float
    lattice_score: float
    volume_score: float
    spacegroup_match: bool
    
    # MP材料属性
    mp_volume: float
    mp_a: float
    mp_b: float
    mp_c: float
    mp_alpha: float
    mp_beta: float
    mp_gamma: float
    mp_energy_above_hull: float
    mp_band_gap: float
    mp_is_stable: bool
    
    # 铁电相关
    mp_is_polar: bool
    mp_point_group: str


# ==========================================
# 3. MP数据库比较器
# ==========================================
class MPComparator:
    """Materials Project 数据库比较器"""
    
    def __init__(self, config: MPConfig = None):
        self.config = config or MPConfig()
        self.config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        if not HAS_MP_API:
            raise ImportError("mp_api is required. Install with: pip install mp-api")
    
    def _get_chemical_system(self, elements: List[str]) -> str:
        """获取化学系统字符串 (元素按字母排序，用-连接)"""
        unique_elements = sorted(set(elements))
        return '-'.join(unique_elements)
    
    def _parse_elements(self, elements_str: str) -> List[str]:
        """解析元素字符串"""
        if isinstance(elements_str, str):
            return [el.strip() for el in elements_str.split(',') if el.strip()]
        return []
    
    def _calculate_lattice_similarity(self, gen: Dict, mp: Dict) -> float:
        """计算晶格参数相似度 (0-100)"""
        try:
            # 归一化差异
            diffs = []
            
            for param in ['a', 'b', 'c']:
                gen_val = gen.get(param, 0)
                mp_val = mp.get(param, 0)
                if mp_val > 0:
                    diff = abs(gen_val - mp_val) / mp_val
                    diffs.append(min(diff, 1.0))
            
            for param in ['alpha', 'beta', 'gamma']:
                gen_val = gen.get(param, 90)
                mp_val = mp.get(param, 90)
                diff = abs(gen_val - mp_val) / 180.0
                diffs.append(min(diff, 1.0))
            
            if not diffs:
                return 0
            
            avg_diff = np.mean(diffs)
            score = (1 - avg_diff) * 100
            return max(0, score)
            
        except Exception as e:
            return 0
    
    def _calculate_volume_similarity(self, gen_vol: float, mp_vol: float) -> float:
        """计算体积相似度 (0-100)"""
        if mp_vol <= 0:
            return 0
        diff = abs(gen_vol - mp_vol) / mp_vol
        score = (1 - min(diff, 1.0)) * 100
        return max(0, score)
    
    def _search_mp_database(self, elements: List[str]) -> List[Dict]:
        """在MP数据库中搜索指定化学系统的材料"""
        results = []
        
        try:
            chemsys = self._get_chemical_system(elements)
            
            with MPRester(api_key=self.config.API_KEY) as mpr:
                # 搜索该化学系统的所有材料
                docs = mpr.materials.summary.search(
                    chemsys=chemsys,
                    fields=[
                        "material_id", "formula_pretty", "structure",
                        "symmetry", "volume", "density",
                        "energy_above_hull", "is_stable", "band_gap",
                        "formation_energy_per_atom"
                    ],
                    num_chunks=1
                )
                
                for doc in docs[:self.config.MAX_RESULTS_PER_SYSTEM]:
                    try:
                        struct = doc.structure
                        lattice = struct.lattice
                        symmetry = doc.symmetry
                        
                        result = {
                            'material_id': str(doc.material_id),
                            'formula': doc.formula_pretty,
                            'volume': doc.volume,
                            'a': lattice.a,
                            'b': lattice.b,
                            'c': lattice.c,
                            'alpha': lattice.alpha,
                            'beta': lattice.beta,
                            'gamma': lattice.gamma,
                            'spacegroup': symmetry.number if symmetry else 1,
                            'point_group': symmetry.point_group if symmetry else '',
                            'crystal_system': symmetry.crystal_system if symmetry else '',
                            'energy_above_hull': doc.energy_above_hull or 0,
                            'is_stable': doc.is_stable or False,
                            'band_gap': doc.band_gap or 0,
                            'density': doc.density or 0,
                        }
                        
                        # 判断是否极性
                        polar_point_groups = ['1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm']
                        result['is_polar'] = result['point_group'] in polar_point_groups
                        
                        results.append(result)
                        
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"  Warning: MP search failed for {elements}: {e}")
        
        return results
    
    def compare_material(self, generated: Dict) -> List[MatchResult]:
        """将单个生成的材料与MP数据库对比"""
        matches = []
        
        elements = self._parse_elements(generated.get('elements', ''))
        if not elements or len(elements) < 2:
            return matches
        
        # 搜索MP数据库
        mp_materials = self._search_mp_database(elements)
        
        if not mp_materials:
            return matches
        
        gen_lattice = {
            'a': generated.get('a', 0),
            'b': generated.get('b', 0),
            'c': generated.get('c', 0),
            'alpha': generated.get('alpha', 90),
            'beta': generated.get('beta', 90),
            'gamma': generated.get('gamma', 90),
        }
        gen_volume = generated.get('volume', 0)
        gen_spacegroup = generated.get('spacegroup', 1)
        
        for mp_mat in mp_materials:
            mp_lattice = {
                'a': mp_mat['a'],
                'b': mp_mat['b'],
                'c': mp_mat['c'],
                'alpha': mp_mat['alpha'],
                'beta': mp_mat['beta'],
                'gamma': mp_mat['gamma'],
            }
            
            # 计算各项分数
            lattice_score = self._calculate_lattice_similarity(gen_lattice, mp_lattice)
            volume_score = self._calculate_volume_similarity(gen_volume, mp_mat['volume'])
            spacegroup_match = (gen_spacegroup == mp_mat['spacegroup'])
            
            # 成分匹配 (化学系统相同即为100%)
            composition_match = 100.0
            
            # 综合分数
            total_score = (
                composition_match * 0.30 +
                lattice_score * 0.35 +
                volume_score * 0.20 +
                (100 if spacegroup_match else 50) * 0.15
            )
            
            match = MatchResult(
                generated_id=generated.get('id', 0),
                generated_formula=generated.get('formula', ''),
                mp_id=mp_mat['material_id'],
                mp_formula=mp_mat['formula'],
                mp_spacegroup=mp_mat['spacegroup'],
                total_score=total_score,
                composition_match=composition_match,
                lattice_score=lattice_score,
                volume_score=volume_score,
                spacegroup_match=spacegroup_match,
                mp_volume=mp_mat['volume'],
                mp_a=mp_mat['a'],
                mp_b=mp_mat['b'],
                mp_c=mp_mat['c'],
                mp_alpha=mp_mat['alpha'],
                mp_beta=mp_mat['beta'],
                mp_gamma=mp_mat['gamma'],
                mp_energy_above_hull=mp_mat['energy_above_hull'],
                mp_band_gap=mp_mat['band_gap'],
                mp_is_stable=mp_mat['is_stable'],
                mp_is_polar=mp_mat['is_polar'],
                mp_point_group=mp_mat['point_group'],
            )
            
            matches.append(match)
        
        # 按总分排序
        matches.sort(key=lambda x: x.total_score, reverse=True)
        
        return matches
    
    def compare_dataframe(self, df: pd.DataFrame, 
                          top_n_per_material: int = 3) -> pd.DataFrame:
        """对比DataFrame中的所有材料"""
        all_matches = []
        processed_systems = set()
        
        print(f"\n{'='*60}")
        print("Comparing with Materials Project Database")
        print(f"{'='*60}\n")
        
        total = len(df)
        
        for idx, (_, row) in enumerate(df.iterrows()):
            elements = self._parse_elements(row.get('elements', ''))
            chemsys = self._get_chemical_system(elements)
            
            # 避免重复搜索相同的化学系统
            if chemsys in processed_systems:
                continue
            processed_systems.add(chemsys)
            
            print(f"[{idx+1}/{total}] Searching: {chemsys}...", end=" ")
            
            generated = row.to_dict()
            matches = self.compare_material(generated)
            
            if matches:
                print(f"Found {len(matches)} matches")
                # 只保留top N
                for match in matches[:top_n_per_material]:
                    all_matches.append(match)
            else:
                print("No matches")
            
            # 请求延迟
            time.sleep(self.config.REQUEST_DELAY)
        
        # 转换为DataFrame
        if all_matches:
            results_df = pd.DataFrame([vars(m) for m in all_matches])
            results_df = results_df.sort_values('total_score', ascending=False)
        else:
            results_df = pd.DataFrame()
        
        return results_df
    
    def generate_report(self, matches_df: pd.DataFrame, output_path: Path = None):
        """生成对比报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if output_path is None:
            output_path = self.config.OUTPUT_DIR / f'mp_comparison_report_{timestamp}.txt'
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("Materials Project Comparison Report\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total matches found: {len(matches_df)}\n\n")
            
            if len(matches_df) == 0:
                f.write("No matches found.\n")
                return
            
            # 统计
            f.write("-"*70 + "\n")
            f.write("Summary Statistics\n")
            f.write("-"*70 + "\n")
            
            high_matches = matches_df[matches_df['total_score'] >= 80]
            good_matches = matches_df[(matches_df['total_score'] >= 60) & (matches_df['total_score'] < 80)]
            
            f.write(f"High matches (>=80): {len(high_matches)}\n")
            f.write(f"Good matches (60-80): {len(good_matches)}\n")
            f.write(f"Stable materials: {matches_df['mp_is_stable'].sum()}\n")
            f.write(f"Polar materials: {matches_df['mp_is_polar'].sum()}\n")
            
            # 最佳匹配
            f.write("\n" + "-"*70 + "\n")
            f.write("Top 10 Best Matches\n")
            f.write("-"*70 + "\n\n")
            
            for _, row in matches_df.head(10).iterrows():
                f.write(f"Generated: {row['generated_formula']}\n")
                f.write(f"  MP Match: {row['mp_id']} ({row['mp_formula']})\n")
                f.write(f"  Score: {row['total_score']:.1f}/100\n")
                f.write(f"    - Lattice: {row['lattice_score']:.1f}\n")
                f.write(f"    - Volume: {row['volume_score']:.1f}\n")
                f.write(f"    - Spacegroup: {'✓' if row['spacegroup_match'] else '✗'} (MP: {row['mp_spacegroup']})\n")
                f.write(f"  Properties:\n")
                f.write(f"    - Stable: {'Yes' if row['mp_is_stable'] else 'No'}\n")
                f.write(f"    - Polar: {'Yes' if row['mp_is_polar'] else 'No'}\n")
                f.write(f"    - Band gap: {row['mp_band_gap']:.2f} eV\n")
                f.write(f"    - E above hull: {row['mp_energy_above_hull']:.4f} eV/atom\n")
                f.write("\n")
            
            # 铁电候选
            polar_stable = matches_df[(matches_df['mp_is_polar']) & (matches_df['mp_is_stable'])]
            if len(polar_stable) > 0:
                f.write("\n" + "-"*70 + "\n")
                f.write("Potential Ferroelectric Candidates (Polar & Stable)\n")
                f.write("-"*70 + "\n\n")
                
                for _, row in polar_stable.head(10).iterrows():
                    f.write(f"{row['mp_id']}: {row['mp_formula']}\n")
                    f.write(f"  Point group: {row['mp_point_group']}, Spacegroup: {row['mp_spacegroup']}\n")
                    f.write(f"  Band gap: {row['mp_band_gap']:.2f} eV\n")
                    f.write(f"  Match score: {row['total_score']:.1f}\n\n")
        
        print(f"\n✓ Report saved: {output_path}")


# ==========================================
# 4. 主函数
# ==========================================
def compare_with_mp(input_csv: str, output_csv: str = None, 
                    top_n: int = 3) -> pd.DataFrame:
    """
    将生成的材料与MP数据库对比
    
    Args:
        input_csv: 输入CSV文件 (验证后的材料)
        output_csv: 输出CSV文件
        top_n: 每个材料保留的最佳匹配数
    
    Returns:
        匹配结果DataFrame
    """
    # 加载数据
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} materials from {input_csv}")
    
    # 过滤高分材料
    if 'score' in df.columns:
        df = df[df['score'] >= 65]  # 只对比GOOD及以上的材料
        print(f"Filtered to {len(df)} high-quality materials")
    
    # 对比
    comparator = MPComparator()
    matches_df = comparator.compare_dataframe(df, top_n_per_material=top_n)
    
    # 保存
    if output_csv is None:
        output_csv = comparator.config.OUTPUT_DIR / 'mp_matched_materials.csv'
    
    if len(matches_df) > 0:
        matches_df.to_csv(output_csv, index=False)
        print(f"\n✓ Saved {len(matches_df)} matches to: {output_csv}")
        
        # 生成报告
        comparator.generate_report(matches_df)
        
        # 显示摘要
        print(f"\n{'='*60}")
        print("Match Summary")
        print(f"{'='*60}")
        
        print(f"\nTotal matches: {len(matches_df)}")
        print(f"High matches (>=80): {(matches_df['total_score'] >= 80).sum()}")
        print(f"Stable materials: {matches_df['mp_is_stable'].sum()}")
        print(f"Polar materials: {matches_df['mp_is_polar'].sum()}")
        
        print(f"\nTop 5 Best Matches:")
        for _, row in matches_df.head(5).iterrows():
            flag = "🔥" if row['mp_is_polar'] and row['mp_is_stable'] else ""
            print(f"  {row['mp_id']}: {row['mp_formula']} "
                  f"(Score: {row['total_score']:.1f}, "
                  f"Polar: {'Y' if row['mp_is_polar'] else 'N'}, "
                  f"Stable: {'Y' if row['mp_is_stable'] else 'N'}) {flag}")
    else:
        print("\nNo matches found in MP database.")
    
    return matches_df


def main():
    parser = argparse.ArgumentParser(description='Compare generated materials with MP database')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    parser.add_argument('--top-n', type=int, default=3, help='Top N matches per material')
    args = parser.parse_args()
    
    compare_with_mp(args.input, args.output, args.top_n)


if __name__ == '__main__':
    main()
