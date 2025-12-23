"""
铁电候选材料深度分析模块
=============================================
对从MP数据库匹配到的铁电候选材料进行深入分析

功能:
1. 获取详细的材料属性（介电、压电等）
2. 分析极性空间群特征
3. 评估铁电可能性
4. 与已知铁电材料对比
5. 生成深度分析报告
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

# MP API
try:
    from mp_api.client import MPRester
    HAS_MP_API = True
except ImportError:
    HAS_MP_API = False


# 已知铁电材料及其关键参数
KNOWN_FERROELECTRICS = {
    'BaTiO3': {
        'Tc': 393,  # Curie temperature (K)
        'Ps': 26.0,  # Spontaneous polarization (μC/cm²)
        'crystal_system': 'tetragonal',
        'space_groups': [99, 160, 38],  # P4mm, R3m, Amm2
    },
    'PbTiO3': {
        'Tc': 763,
        'Ps': 75.0,
        'crystal_system': 'tetragonal',
        'space_groups': [99],
    },
    'LiNbO3': {
        'Tc': 1483,
        'Ps': 71.0,
        'crystal_system': 'trigonal',
        'space_groups': [161],  # R3c
    },
    'KNbO3': {
        'Tc': 708,
        'Ps': 30.0,
        'crystal_system': 'orthorhombic',
        'space_groups': [38],  # Amm2
    },
    'BiFeO3': {
        'Tc': 1103,
        'Ps': 90.0,
        'crystal_system': 'rhombohedral',
        'space_groups': [161],
    },
}

# 极性点群（可能产生铁电性的点群）
POLAR_POINT_GROUPS = [
    '1', '2', 'm', 'mm2', 
    '4', '4mm', 
    '3', '3m', 
    '6', '6mm'
]

# 铁电相关空间群
FERROELECTRIC_SPACE_GROUPS = {
    # 四方系
    99: 'P4mm',    # BaTiO3 tetragonal
    # 三方系
    160: 'R3m',    # BaTiO3 rhombohedral
    161: 'R3c',    # LiNbO3, BiFeO3
    # 正交系
    38: 'Amm2',    # BaTiO3 orthorhombic
    33: 'Pna2_1',  # KNbO3
    36: 'Cmc2_1',
    # 单斜系
    6: 'Pm',
    8: 'Cm',
    9: 'Cc',
    # 三斜系
    1: 'P1',
}


@dataclass
class FerroelectricScore:
    """铁电可能性评分"""
    material_id: str
    formula: str
    
    # 结构相关得分 (0-100)
    polar_symmetry_score: float  # 极性对称性
    lattice_distortion_score: float  # 晶格畸变
    
    # 化学相关得分 (0-100)
    composition_score: float  # 化学成分适合性
    
    # 物理性质得分 (0-100)
    band_gap_score: float  # 带隙（绝缘性）
    stability_score: float  # 稳定性
    
    # 总分
    total_score: float
    
    # 评价
    assessment: str
    details: List[str]


class FerroelectricAnalyzer:
    """铁电材料分析器"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('MP_API_KEY', '1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq')
        
        # 常见铁电化学元素
        self.fe_elements = {
            'A_site': ['Ba', 'Pb', 'Sr', 'Ca', 'K', 'Na', 'Li', 'Bi', 'La', 'Nd'],
            'B_site': ['Ti', 'Zr', 'Nb', 'Ta', 'Fe', 'Mn', 'W', 'Hf', 'Sn'],
            'anion': ['O', 'S', 'Se', 'F', 'Cl'],
        }
    
    def analyze_candidate(self, mp_id: str) -> FerroelectricScore:
        """
        深度分析一个MP材料的铁电可能性
        
        Args:
            mp_id: Materials Project ID (e.g., 'mp-2998')
        
        Returns:
            FerroelectricScore 评估结果
        """
        if not HAS_MP_API:
            raise ImportError("mp_api not installed")
        
        with MPRester(self.api_key) as mpr:
            # 获取详细信息
            docs = mpr.materials.summary.search(
                material_ids=[mp_id],
                fields=[
                    'material_id', 'formula_pretty', 
                    'symmetry', 'structure',
                    'band_gap', 'is_stable', 'energy_above_hull',
                    'elements', 'nelements',
                    'volume', 'density',
                ]
            )
            
            if not docs:
                raise ValueError(f"Material {mp_id} not found")
            
            doc = docs[0]
        
        details = []
        
        # 1. 极性对称性评分
        polar_score = self._evaluate_polar_symmetry(doc, details)
        
        # 2. 晶格畸变评分
        distortion_score = self._evaluate_lattice_distortion(doc, details)
        
        # 3. 化学成分评分
        composition_score = self._evaluate_composition(doc, details)
        
        # 4. 带隙评分
        bandgap_score = self._evaluate_band_gap(doc, details)
        
        # 5. 稳定性评分
        stability_score = self._evaluate_stability(doc, details)
        
        # 计算总分（加权平均）
        total_score = (
            polar_score * 0.30 +       # 极性对称性最重要
            distortion_score * 0.15 +
            composition_score * 0.20 +
            bandgap_score * 0.20 +
            stability_score * 0.15
        )
        
        # 评估
        if total_score >= 80:
            assessment = "EXCELLENT - Highly likely ferroelectric"
        elif total_score >= 65:
            assessment = "GOOD - Strong ferroelectric candidate"
        elif total_score >= 50:
            assessment = "MODERATE - Possible ferroelectric"
        else:
            assessment = "LOW - Unlikely to be ferroelectric"
        
        return FerroelectricScore(
            material_id=str(doc.material_id),
            formula=doc.formula_pretty,
            polar_symmetry_score=polar_score,
            lattice_distortion_score=distortion_score,
            composition_score=composition_score,
            band_gap_score=bandgap_score,
            stability_score=stability_score,
            total_score=total_score,
            assessment=assessment,
            details=details
        )
    
    def _evaluate_polar_symmetry(self, doc, details: List) -> float:
        """评估极性对称性"""
        score = 50.0  # 基础分
        
        symmetry = doc.symmetry
        if symmetry:
            point_group = symmetry.point_group
            space_group_num = symmetry.number
            
            # 极性点群加分
            if point_group in POLAR_POINT_GROUPS:
                score += 30
                details.append(f"✓ Polar point group: {point_group}")
            else:
                details.append(f"✗ Non-polar point group: {point_group}")
            
            # 铁电空间群加分
            if space_group_num in FERROELECTRIC_SPACE_GROUPS:
                score += 20
                details.append(
                    f"✓ Known FE space group: {FERROELECTRIC_SPACE_GROUPS[space_group_num]} (#{space_group_num})"
                )
        
        return min(100, max(0, score))
    
    def _evaluate_lattice_distortion(self, doc, details: List) -> float:
        """评估晶格畸变（通常与铁电相变相关）"""
        score = 60.0
        
        structure = doc.structure
        if structure:
            lattice = structure.lattice
            a, b, c = lattice.a, lattice.b, lattice.c
            
            # 计算轴向各向异性
            avg = (a + b + c) / 3
            anisotropy = max(abs(a-avg), abs(b-avg), abs(c-avg)) / avg
            
            # 适度的畸变是好的（钙钛矿通常有轻微畸变）
            if 0.01 < anisotropy < 0.15:
                score += 20
                details.append(f"✓ Moderate lattice distortion: {anisotropy:.3f}")
            elif anisotropy <= 0.01:
                score -= 10
                details.append(f"○ Very low distortion: {anisotropy:.3f}")
            
            # 检查角度畸变
            alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
            angle_dev = sum([
                abs(alpha - 90), 
                abs(beta - 90), 
                abs(gamma - 90)
            ]) / 3
            
            if 0.1 < angle_dev < 10:
                score += 10
                details.append(f"✓ Non-cubic angles detected")
        
        return min(100, max(0, score))
    
    def _evaluate_composition(self, doc, details: List) -> float:
        """评估化学成分"""
        score = 50.0
        
        elements = [str(e) for e in doc.elements]
        nelements = doc.nelements
        
        # 检查是否包含典型铁电元素
        has_a_site = any(e in elements for e in self.fe_elements['A_site'])
        has_b_site = any(e in elements for e in self.fe_elements['B_site'])
        has_anion = any(e in elements for e in self.fe_elements['anion'])
        
        if has_a_site:
            score += 15
            a_found = [e for e in elements if e in self.fe_elements['A_site']]
            details.append(f"✓ Contains A-site elements: {a_found}")
        
        if has_b_site:
            score += 15
            b_found = [e for e in elements if e in self.fe_elements['B_site']]
            details.append(f"✓ Contains B-site elements: {b_found}")
        
        if has_anion:
            score += 10
        
        # 元素数量（2-5个元素的化合物最常见）
        if 2 <= nelements <= 5:
            score += 10
        
        return min(100, max(0, score))
    
    def _evaluate_band_gap(self, doc, details: List) -> float:
        """评估带隙（铁电体通常是绝缘体/半导体）"""
        score = 60.0
        
        band_gap = doc.band_gap
        
        if band_gap is not None:
            if band_gap >= 2.5:
                # 理想绝缘体
                score = 90
                details.append(f"✓ Large band gap: {band_gap:.2f} eV (insulator)")
            elif band_gap >= 1.0:
                score = 75
                details.append(f"✓ Moderate band gap: {band_gap:.2f} eV (semiconductor)")
            elif band_gap >= 0.1:
                score = 50
                details.append(f"○ Small band gap: {band_gap:.2f} eV")
            else:
                score = 20
                details.append(f"✗ Metallic/near-zero gap: {band_gap:.2f} eV")
        else:
            details.append("○ Band gap not available")
        
        return score
    
    def _evaluate_stability(self, doc, details: List) -> float:
        """评估稳定性"""
        score = 50.0
        
        if doc.is_stable:
            score += 40
            details.append("✓ Thermodynamically stable")
        else:
            ehull = doc.energy_above_hull
            if ehull is not None:
                if ehull < 0.025:
                    score += 30
                    details.append(f"✓ Nearly stable (E_hull = {ehull*1000:.1f} meV/atom)")
                elif ehull < 0.050:
                    score += 15
                    details.append(f"○ Metastable (E_hull = {ehull*1000:.1f} meV/atom)")
                else:
                    details.append(f"✗ Unstable (E_hull = {ehull*1000:.1f} meV/atom)")
        
        return min(100, max(0, score))
    
    def compare_to_known(self, formula: str) -> Optional[Dict]:
        """与已知铁电材料对比"""
        # 简化公式比较
        for known, props in KNOWN_FERROELECTRICS.items():
            if known.lower() in formula.lower().replace(' ', ''):
                return {
                    'known_ferroelectric': known,
                    'properties': props
                }
        return None
    
    def batch_analyze(self, mp_ids: List[str]) -> pd.DataFrame:
        """批量分析多个材料"""
        results = []
        
        for mp_id in mp_ids:
            try:
                score = self.analyze_candidate(mp_id)
                results.append({
                    'mp_id': score.material_id,
                    'formula': score.formula,
                    'polar_symmetry': score.polar_symmetry_score,
                    'distortion': score.lattice_distortion_score,
                    'composition': score.composition_score,
                    'band_gap': score.band_gap_score,
                    'stability': score.stability_score,
                    'total_score': score.total_score,
                    'assessment': score.assessment,
                })
            except Exception as e:
                print(f"Error analyzing {mp_id}: {e}")
        
        df = pd.DataFrame(results)
        return df.sort_values('total_score', ascending=False)


def analyze_matched_candidates(
    mp_matched_csv: str,
    output_path: str = None
) -> pd.DataFrame:
    """
    分析MP匹配结果中的铁电候选
    
    Args:
        mp_matched_csv: MP匹配结果CSV文件路径
        output_path: 分析结果输出路径
    
    Returns:
        包含详细铁电评分的DataFrame
    """
    df = pd.read_csv(mp_matched_csv)
    
    # 筛选极性材料
    polar_materials = df[df['mp_is_polar'] == True] if 'mp_is_polar' in df.columns else df
    
    if len(polar_materials) == 0:
        print("No polar materials found in the dataset")
        return pd.DataFrame()
    
    print(f"\nAnalyzing {len(polar_materials)} polar materials...")
    
    analyzer = FerroelectricAnalyzer()
    
    results = []
    for _, row in polar_materials.iterrows():
        mp_id = row['mp_id']
        print(f"  Analyzing {mp_id}: {row['mp_formula']}...")
        
        try:
            score = analyzer.analyze_candidate(mp_id)
            results.append({
                'mp_id': score.material_id,
                'formula': score.formula,
                'generated_formula': row.get('generated_formula', ''),
                'polar_symmetry_score': score.polar_symmetry_score,
                'distortion_score': score.lattice_distortion_score,
                'composition_score': score.composition_score,
                'band_gap_score': score.band_gap_score,
                'stability_score': score.stability_score,
                'fe_total_score': score.total_score,
                'assessment': score.assessment,
                'mp_match_score': row.get('total_score', 0),
                'details': '; '.join(score.details)
            })
        except Exception as e:
            print(f"    Error: {e}")
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('fe_total_score', ascending=False)
    
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"\n✓ Analysis results saved to: {output_path}")
    
    return result_df


def print_analysis_report(df: pd.DataFrame):
    """打印分析报告"""
    print("\n" + "="*70)
    print("FERROELECTRIC CANDIDATE ANALYSIS REPORT")
    print("="*70)
    
    if len(df) == 0:
        print("No candidates to analyze")
        return
    
    # 按评级分组
    excellent = df[df['fe_total_score'] >= 80]
    good = df[(df['fe_total_score'] >= 65) & (df['fe_total_score'] < 80)]
    moderate = df[(df['fe_total_score'] >= 50) & (df['fe_total_score'] < 65)]
    
    print(f"\n📊 Summary:")
    print(f"   EXCELLENT (≥80):  {len(excellent)} candidates")
    print(f"   GOOD (65-80):     {len(good)} candidates")
    print(f"   MODERATE (50-65): {len(moderate)} candidates")
    
    if len(excellent) > 0:
        print("\n" + "-"*70)
        print("🏆 EXCELLENT FERROELECTRIC CANDIDATES")
        print("-"*70)
        for _, row in excellent.iterrows():
            print(f"\n{row['mp_id']}: {row['formula']}")
            print(f"  FE Score: {row['fe_total_score']:.1f}")
            print(f"  Assessment: {row['assessment']}")
            print(f"  Details: {row['details']}")
    
    if len(good) > 0:
        print("\n" + "-"*70)
        print("⭐ GOOD FERROELECTRIC CANDIDATES")
        print("-"*70)
        for _, row in good.head(5).iterrows():
            print(f"\n{row['mp_id']}: {row['formula']}")
            print(f"  FE Score: {row['fe_total_score']:.1f}")
            print(f"  Assessment: {row['assessment']}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze ferroelectric candidates')
    parser.add_argument('input_csv', help='MP matched materials CSV')
    parser.add_argument('--output', '-o', help='Output CSV path')
    args = parser.parse_args()
    
    result = analyze_matched_candidates(args.input_csv, args.output)
    print_analysis_report(result)
