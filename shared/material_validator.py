"""
材料合理性验证模块
=============================================
验证逆向设计生成的材料是否合理

检查项目:
1. 化学合理性 - 电荷平衡、氧化态
2. 晶格合理性 - 参数范围、密度
3. 铁电材料特征 - 成分组合、空间群
4. GCNN模型验证 - 使用判别器打分
"""

import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# 添加共享模块路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from feature_engineering import (
    ELEMENT_DATABASE,
    COMMON_OXIDATION_STATES,
    TRANSITION_METALS,
    LANTHANIDES,
    D0_METALS,
    ALKALI_METALS,
    ALKALINE_EARTH,
    HALOGENS,
    CHALCOGENS,
    ELEMENT_TO_IDX
)


# ==========================================
# 1. 验证结果类
# ==========================================
class ValidityLevel(Enum):
    """合理性等级"""
    EXCELLENT = 5    # 非常合理，高度可能存在
    GOOD = 4         # 合理，可能存在
    ACCEPTABLE = 3   # 勉强合理，需验证
    POOR = 2         # 不太合理，存疑
    INVALID = 1      # 不合理，应排除


@dataclass
class ValidationResult:
    """验证结果"""
    level: ValidityLevel
    score: float           # 0-100 综合得分
    is_valid: bool         # 是否通过基本验证
    
    # 分项得分
    chemistry_score: float
    lattice_score: float
    ferroelectric_score: float
    model_score: float
    
    # 问题列表
    issues: List[str]
    warnings: List[str]
    
    def __str__(self):
        return (f"Validity: {self.level.name} (Score: {self.score:.1f}/100)\n"
                f"  Chemistry: {self.chemistry_score:.1f}\n"
                f"  Lattice: {self.lattice_score:.1f}\n"
                f"  Ferroelectric: {self.ferroelectric_score:.1f}\n"
                f"  Model: {self.model_score:.1f}\n"
                f"  Issues: {self.issues}\n"
                f"  Warnings: {self.warnings}")


# ==========================================
# 2. 化学合理性验证
# ==========================================
class ChemistryValidator:
    """化学合理性验证器"""
    
    # 扩展的氧化态
    OXIDATION_STATES = {
        'H': [1, -1], 'Li': [1], 'Na': [1], 'K': [1], 'Rb': [1], 'Cs': [1],
        'Be': [2], 'Mg': [2], 'Ca': [2], 'Sr': [2], 'Ba': [2],
        'B': [3], 'Al': [3], 'Ga': [3], 'In': [3], 'Tl': [1, 3],
        'C': [-4, 4], 'Si': [4], 'Ge': [4], 'Sn': [2, 4], 'Pb': [2, 4],
        'N': [-3, 3, 5], 'P': [-3, 3, 5], 'As': [3, 5], 'Sb': [3, 5], 'Bi': [3],
        'O': [-2], 'S': [-2, 4, 6], 'Se': [-2, 4, 6], 'Te': [-2, 4, 6],
        'F': [-1], 'Cl': [-1, 1, 3, 5, 7], 'Br': [-1, 1, 3, 5], 'I': [-1, 1, 3, 5, 7],
        'Sc': [3], 'Y': [3], 'La': [3],
        'Ti': [2, 3, 4], 'Zr': [4], 'Hf': [4],
        'V': [2, 3, 4, 5], 'Nb': [3, 5], 'Ta': [5],
        'Cr': [2, 3, 6], 'Mo': [4, 6], 'W': [4, 6],
        'Mn': [2, 3, 4, 7], 'Fe': [2, 3], 'Co': [2, 3], 'Ni': [2, 3],
        'Cu': [1, 2], 'Ag': [1], 'Au': [1, 3],
        'Zn': [2], 'Cd': [2], 'Hg': [1, 2],
        'Ce': [3, 4], 'Pr': [3], 'Nd': [3], 'Sm': [2, 3], 'Eu': [2, 3],
        'Gd': [3], 'Tb': [3, 4], 'Dy': [3], 'Ho': [3], 'Er': [3],
        'Tm': [3], 'Yb': [2, 3], 'Lu': [3],
        'Th': [4], 'U': [3, 4, 5, 6],
    }
    
    # 不可能的元素组合
    IMPOSSIBLE_COMBINATIONS = [
        {'H', 'F'},      # 只有H和F难以形成稳定化合物
        {'O', 'F'},      # O和F很少共存
    ]
    
    # 常见的铁电材料元素组合模式
    FERROELECTRIC_PATTERNS = [
        # ABO3 钙钛矿
        (ALKALI_METALS | ALKALINE_EARTH | LANTHANIDES, D0_METALS | TRANSITION_METALS, {'O'}),
        # 层状钙钛矿
        ({'Bi', 'Pb'}, D0_METALS, {'O'}),
        # 钨青铜
        (ALKALI_METALS | ALKALINE_EARTH, {'W', 'Nb', 'Ta'}, {'O'}),
        # 铌酸锂/钽酸锂类
        ({'Li', 'Na', 'K'}, {'Nb', 'Ta'}, {'O'}),
        # 硫酸盐/磷酸盐
        (ALKALI_METALS | ALKALINE_EARTH, {'S', 'P'}, {'O'}),
    ]
    
    def validate(self, elements: List[str], fractions: List[float]) -> Tuple[float, List[str], List[str]]:
        """
        验证化学合理性
        
        Returns:
            (score, issues, warnings)
        """
        score = 100.0
        issues = []
        warnings = []
        
        if not elements:
            return 0, ["No elements provided"], []
        
        # 1. 检查元素是否有效
        valid_elements = []
        for el in elements:
            if el in ELEMENT_DATABASE:
                valid_elements.append(el)
            else:
                issues.append(f"Unknown element: {el}")
                score -= 20
        
        if not valid_elements:
            return 0, issues, warnings
        
        elements = valid_elements
        element_set = set(elements)
        
        # 2. 检查元素多样性
        unique_elements = set(elements)
        if len(unique_elements) < len(elements):
            issues.append("Duplicate elements in composition")
            score -= 30
        
        # 3. 检查是否只有阴离子或只有阳离子
        cations = element_set & (ALKALI_METALS | ALKALINE_EARTH | TRANSITION_METALS | LANTHANIDES)
        anions = element_set & (HALOGENS | CHALCOGENS | {'N', 'P', 'C'})
        
        if len(element_set) > 1:
            if not cations and not anions:
                warnings.append("Unclear oxidation states")
                score -= 10
            elif not cations:
                issues.append("No cations in composition")
                score -= 40
            elif not anions:
                warnings.append("No typical anions (may be metallic)")
                score -= 15
        
        # 4. 检查电荷平衡 (简化版)
        charge_balance = self._check_charge_balance(elements, fractions)
        if charge_balance < 0.5:
            issues.append("Severe charge imbalance")
            score -= 30
        elif charge_balance < 0.8:
            warnings.append("Possible charge imbalance")
            score -= 10
        
        # 5. 检查不可能的组合
        for combo in self.IMPOSSIBLE_COMBINATIONS:
            if combo.issubset(element_set) and len(element_set) == len(combo):
                issues.append(f"Unlikely element combination: {combo}")
                score -= 25
        
        # 6. 检查是否符合铁电材料模式
        matches_pattern = False
        for pattern in self.FERROELECTRIC_PATTERNS:
            a_site, b_site, anion = pattern
            has_a = bool(element_set & a_site)
            has_b = bool(element_set & b_site)
            has_anion = bool(element_set & anion)
            if has_a and has_b and has_anion:
                matches_pattern = True
                break
        
        if not matches_pattern and 'O' not in element_set:
            warnings.append("Does not match typical ferroelectric patterns")
            score -= 15
        
        # 7. 检查氢含量
        h_idx = [i for i, el in enumerate(elements) if el == 'H']
        if h_idx and fractions[h_idx[0]] > 0.3:
            warnings.append("High hydrogen content unusual for ferroelectrics")
            score -= 10
        
        return max(0, score), issues, warnings
    
    def _check_charge_balance(self, elements: List[str], fractions: List[float]) -> float:
        """检查电荷平衡 (返回0-1的平衡度)"""
        try:
            # 简化检查：假设最常见氧化态
            total_positive = 0
            total_negative = 0
            
            for el, frac in zip(elements, fractions):
                if el in self.OXIDATION_STATES:
                    states = self.OXIDATION_STATES[el]
                    # 选择最常见的氧化态
                    if el in COMMON_OXIDATION_STATES:
                        state = COMMON_OXIDATION_STATES[el]
                    else:
                        state = states[0]
                    
                    charge = state * frac
                    if charge > 0:
                        total_positive += charge
                    else:
                        total_negative += abs(charge)
            
            if total_positive == 0 and total_negative == 0:
                return 0.5  # 无法判断
            
            # 计算平衡度
            total = total_positive + total_negative
            if total == 0:
                return 0.5
            
            balance = 1 - abs(total_positive - total_negative) / total
            return balance
            
        except:
            return 0.5


# ==========================================
# 3. 晶格合理性验证
# ==========================================
class LatticeValidator:
    """晶格参数验证器"""
    
    # 合理的晶格参数范围 (Å)
    LATTICE_RANGE = {
        'a': (2.0, 30.0),
        'b': (2.0, 30.0),
        'c': (2.0, 50.0),
        'alpha': (30.0, 150.0),
        'beta': (30.0, 150.0),
        'gamma': (30.0, 150.0),
        'volume': (10.0, 5000.0),
    }
    
    # 密度范围 (g/cm³)
    DENSITY_RANGE = (1.0, 15.0)
    
    def validate(self, lattice: Dict[str, float], 
                 elements: List[str] = None, 
                 fractions: List[float] = None) -> Tuple[float, List[str], List[str]]:
        """验证晶格参数合理性"""
        score = 100.0
        issues = []
        warnings = []
        
        # 1. 检查参数范围
        for param, (min_val, max_val) in self.LATTICE_RANGE.items():
            val = lattice.get(param, 0)
            if val < min_val:
                issues.append(f"{param} too small: {val:.2f} < {min_val}")
                score -= 15
            elif val > max_val:
                issues.append(f"{param} too large: {val:.2f} > {max_val}")
                score -= 15
        
        # 2. 检查角度是否合理
        angles = [lattice.get('alpha', 90), lattice.get('beta', 90), lattice.get('gamma', 90)]
        if all(80 < a < 100 for a in angles):
            pass  # 接近正交，合理
        elif any(a < 50 or a > 130 for a in angles):
            warnings.append("Unusual lattice angles")
            score -= 10
        
        # 3. 检查轴长比例
        a = lattice.get('a', 1)
        b = lattice.get('b', 1)
        c = lattice.get('c', 1)
        
        max_ratio = max(a/b, b/a, a/c, c/a, b/c, c/b)
        if max_ratio > 5:
            warnings.append(f"Highly anisotropic lattice (ratio: {max_ratio:.1f})")
            score -= 10
        
        # 4. 估算密度
        if elements and fractions:
            vol = lattice.get('volume', 1)
            total_mass = sum(
                ELEMENT_DATABASE.get(el, [0, 100])[1] * frac 
                for el, frac in zip(elements, fractions)
            )
            # 假设每个分数单元大约10个原子
            estimated_mass = total_mass * 10
            density = estimated_mass / vol * 1.66  # 转换为 g/cm³
            
            if density < self.DENSITY_RANGE[0]:
                warnings.append(f"Low estimated density: {density:.2f} g/cm³")
                score -= 5
            elif density > self.DENSITY_RANGE[1]:
                warnings.append(f"High estimated density: {density:.2f} g/cm³")
                score -= 5
        
        return max(0, score), issues, warnings


# ==========================================
# 4. 铁电特征验证
# ==========================================
class FerroelectricValidator:
    """铁电材料特征验证器"""
    
    # 常见铁电空间群
    FERROELECTRIC_SPACEGROUPS = {
        # 极性空间群
        1, 3, 4, 5, 6, 7, 8, 9,  # 三斜/单斜
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,  # 正交
        75, 76, 77, 78, 79, 80, 81, 82,  # 四方
        99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
        143, 144, 145, 146,  # 三方
        156, 157, 158, 159, 160, 161,
        168, 169, 170, 171, 172, 173,  # 六方
        183, 184, 185, 186,
    }
    
    # 已知铁电材料中常见的元素
    COMMON_FE_ELEMENTS = {
        'Ba', 'Ti', 'O',  # BaTiO3
        'Pb', 'Zr',       # PZT
        'K', 'Na', 'Nb',  # KNN
        'Li', 'Ta',       # LiTaO3
        'Bi', 'Fe',       # BiFeO3
        'Sr', 'La', 'Ca', # 掺杂元素
        'Mn', 'Co', 'Ni', # 过渡金属
    }
    
    def validate(self, elements: List[str], fractions: List[float],
                 spacegroup: int = None) -> Tuple[float, List[str], List[str]]:
        """验证是否符合铁电材料特征"""
        score = 100.0
        issues = []
        warnings = []
        
        element_set = set(elements)
        
        # 1. 检查是否含氧
        if 'O' not in element_set:
            if not (element_set & HALOGENS):  # 非氧化物也非卤化物
                issues.append("Most ferroelectrics are oxides or halides")
                score -= 30
        
        # 2. 检查是否含有典型的铁电元素
        fe_elements = element_set & self.COMMON_FE_ELEMENTS
        if len(fe_elements) >= 2:
            pass  # 好
        elif len(fe_elements) == 1:
            warnings.append("Only one typical ferroelectric element")
            score -= 10
        else:
            warnings.append("No typical ferroelectric elements")
            score -= 20
        
        # 3. 检查是否含有 d0 过渡金属
        d0_present = element_set & D0_METALS
        if not d0_present and 'O' in element_set:
            warnings.append("No d0 transition metal (common in oxide ferroelectrics)")
            score -= 10
        
        # 4. 检查空间群
        if spacegroup:
            if spacegroup in self.FERROELECTRIC_SPACEGROUPS:
                pass  # 好
            elif spacegroup > 194:  # 立方
                issues.append(f"Cubic spacegroup {spacegroup} rarely ferroelectric")
                score -= 25
            else:
                warnings.append(f"Spacegroup {spacegroup} not typical for ferroelectrics")
                score -= 10
        
        # 5. 检查元素数量
        if len(element_set) < 2:
            issues.append("Single element cannot be ferroelectric")
            score -= 50
        elif len(element_set) > 6:
            warnings.append("Complex composition (>6 elements)")
            score -= 5
        
        # 6. 检查阳离子/阴离子比例
        cations = element_set & (ALKALI_METALS | ALKALINE_EARTH | TRANSITION_METALS | LANTHANIDES)
        if 'O' in element_set:
            o_idx = elements.index('O') if 'O' in elements else -1
            if o_idx >= 0 and o_idx < len(fractions):
                o_frac = fractions[o_idx]
                if o_frac < 0.3:
                    warnings.append("Low oxygen content for oxide")
                    score -= 10
                elif o_frac > 0.8:
                    warnings.append("Very high oxygen content")
                    score -= 15
        
        return max(0, score), issues, warnings


# ==========================================
# 5. 模型验证器
# ==========================================
class ModelValidator:
    """使用训练好的模型进行验证"""
    
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or Path(__file__).parent.parent / 'model_gcnn_v2'
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """加载GCNN模型"""
        try:
            # 尝试加载判别器
            model_path = self.model_dir / 'gcnn_v2_best.pt'
            if not model_path.exists():
                model_path = self.model_dir / 'gcnn_v2_final.pt'
            
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                # 模型加载需要更复杂的处理，这里简化
                self.model = None  # TODO: 实际加载模型
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            self.model = None
    
    def validate(self, features: np.ndarray = None) -> Tuple[float, List[str]]:
        """使用模型验证"""
        if self.model is None:
            return 50.0, ["Model validation not available"]
        
        try:
            # TODO: 实际模型推理
            return 75.0, []
        except Exception as e:
            return 50.0, [f"Model validation error: {e}"]


# ==========================================
# 6. 综合验证器
# ==========================================
class MaterialValidator:
    """综合材料验证器"""
    
    def __init__(self):
        self.chemistry_validator = ChemistryValidator()
        self.lattice_validator = LatticeValidator()
        self.ferroelectric_validator = FerroelectricValidator()
        self.model_validator = ModelValidator()
    
    def validate(self, material: Dict[str, Any]) -> ValidationResult:
        """
        验证单个材料
        
        Args:
            material: 材料字典，包含:
                - elements: 元素列表
                - fractions: 比例列表
                - lattice: 晶格参数字典
                - spacegroup: 空间群号
        
        Returns:
            ValidationResult
        """
        all_issues = []
        all_warnings = []
        
        # 提取信息
        elements = material.get('elements', [])
        if isinstance(elements, str):
            elements = elements.split(',')
        
        fractions = material.get('fractions', [])
        if not fractions:
            # 从 fraction_1, fraction_2 等提取
            fractions = []
            for i in range(1, 6):
                frac = material.get(f'fraction_{i}', 0)
                if frac > 0:
                    fractions.append(frac)
        
        lattice = material.get('lattice', {})
        if not lattice:
            lattice = {
                'a': material.get('a', 0),
                'b': material.get('b', 0),
                'c': material.get('c', 0),
                'alpha': material.get('alpha', 90),
                'beta': material.get('beta', 90),
                'gamma': material.get('gamma', 90),
                'volume': material.get('volume', 0),
            }
        
        spacegroup = material.get('spacegroup', 1)
        
        # 1. 化学验证
        chem_score, chem_issues, chem_warnings = self.chemistry_validator.validate(
            elements, fractions
        )
        all_issues.extend(chem_issues)
        all_warnings.extend(chem_warnings)
        
        # 2. 晶格验证
        lat_score, lat_issues, lat_warnings = self.lattice_validator.validate(
            lattice, elements, fractions
        )
        all_issues.extend(lat_issues)
        all_warnings.extend(lat_warnings)
        
        # 3. 铁电特征验证
        fe_score, fe_issues, fe_warnings = self.ferroelectric_validator.validate(
            elements, fractions, spacegroup
        )
        all_issues.extend(fe_issues)
        all_warnings.extend(fe_warnings)
        
        # 4. 模型验证 (简化)
        model_score = 50.0  # 默认中性分数
        
        # 计算综合得分
        total_score = (
            chem_score * 0.35 +
            lat_score * 0.20 +
            fe_score * 0.30 +
            model_score * 0.15
        )
        
        # 确定等级
        if total_score >= 80:
            level = ValidityLevel.EXCELLENT
        elif total_score >= 65:
            level = ValidityLevel.GOOD
        elif total_score >= 50:
            level = ValidityLevel.ACCEPTABLE
        elif total_score >= 35:
            level = ValidityLevel.POOR
        else:
            level = ValidityLevel.INVALID
        
        # 是否通过基本验证
        is_valid = len(all_issues) == 0 or total_score >= 50
        
        return ValidationResult(
            level=level,
            score=total_score,
            is_valid=is_valid,
            chemistry_score=chem_score,
            lattice_score=lat_score,
            ferroelectric_score=fe_score,
            model_score=model_score,
            issues=all_issues,
            warnings=all_warnings
        )
    
    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """验证DataFrame中的所有材料"""
        results = []
        
        for idx, row in df.iterrows():
            material = row.to_dict()
            result = self.validate(material)
            
            results.append({
                'id': material.get('id', idx),
                'formula': material.get('formula', 'Unknown'),
                'validity_level': result.level.name,
                'score': result.score,
                'is_valid': result.is_valid,
                'chemistry_score': result.chemistry_score,
                'lattice_score': result.lattice_score,
                'ferroelectric_score': result.ferroelectric_score,
                'issues': '; '.join(result.issues) if result.issues else '',
                'warnings': '; '.join(result.warnings) if result.warnings else '',
            })
        
        return pd.DataFrame(results)
    
    def filter_valid_materials(self, df: pd.DataFrame, 
                                min_level: ValidityLevel = ValidityLevel.ACCEPTABLE) -> pd.DataFrame:
        """过滤出合理的材料"""
        valid_indices = []
        
        for idx, row in df.iterrows():
            material = row.to_dict()
            result = self.validate(material)
            
            if result.level.value >= min_level.value:
                valid_indices.append(idx)
        
        return df.loc[valid_indices].copy()


# ==========================================
# 7. 流水线集成
# ==========================================
def validate_generated_materials(input_csv: str, output_csv: str = None,
                                  filter_level: str = 'ACCEPTABLE') -> pd.DataFrame:
    """
    验证生成的材料CSV
    
    Args:
        input_csv: 输入CSV路径
        output_csv: 输出CSV路径 (可选)
        filter_level: 过滤等级 (EXCELLENT, GOOD, ACCEPTABLE, POOR, INVALID)
    
    Returns:
        验证后的DataFrame
    """
    print(f"\n{'='*60}")
    print("Material Validity Validation")
    print(f"{'='*60}\n")
    
    # 加载数据
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} materials from {input_csv}")
    
    # 验证
    validator = MaterialValidator()
    results_df = validator.validate_dataframe(df)
    
    # 统计
    print(f"\nValidation Results:")
    for level in ValidityLevel:
        count = (results_df['validity_level'] == level.name).sum()
        pct = count / len(results_df) * 100
        print(f"  {level.name}: {count} ({pct:.1f}%)")
    
    # 合并结果
    merged_df = pd.concat([df, results_df.drop(['id', 'formula'], axis=1)], axis=1)
    
    # 过滤
    level_map = {l.name: l for l in ValidityLevel}
    min_level = level_map.get(filter_level, ValidityLevel.ACCEPTABLE)
    
    valid_df = merged_df[merged_df['score'] >= min_level.value * 15].copy()  # 近似阈值
    
    print(f"\nFiltered to {len(valid_df)} materials (>= {filter_level})")
    
    # 保存
    if output_csv:
        valid_df.to_csv(output_csv, index=False)
        print(f"Saved to: {output_csv}")
    
    # 显示一些示例
    print(f"\n{'='*60}")
    print("Sample Valid Materials:")
    print(f"{'='*60}")
    
    top_materials = valid_df.nlargest(5, 'score')
    for _, row in top_materials.iterrows():
        print(f"\n{row['formula']}:")
        print(f"  Score: {row['score']:.1f} ({row['validity_level']})")
        print(f"  Elements: {row['elements']}")
        print(f"  Lattice: a={row['a']:.2f}, b={row['b']:.2f}, c={row['c']:.2f}")
        if row.get('warnings'):
            print(f"  Warnings: {row['warnings']}")
    
    return valid_df


# ==========================================
# 8. 主函数
# ==========================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate generated materials')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, default=None, help='Output CSV file')
    parser.add_argument('--level', type=str, default='ACCEPTABLE',
                        choices=['EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR', 'INVALID'],
                        help='Minimum validity level')
    args = parser.parse_args()
    
    validate_generated_materials(args.input, args.output, args.level)


if __name__ == '__main__':
    # 测试验证器
    print("Testing Material Validator...")
    
    validator = MaterialValidator()
    
    # 测试案例
    test_cases = [
        {
            'formula': 'BaTiO3',
            'elements': ['Ba', 'Ti', 'O'],
            'fractions': [0.2, 0.2, 0.6],
            'a': 4.0, 'b': 4.0, 'c': 4.0,
            'alpha': 90, 'beta': 90, 'gamma': 90,
            'volume': 64,
            'spacegroup': 99,
        },
        {
            'formula': 'O-O-O',
            'elements': ['O', 'O', 'O'],
            'fractions': [0.33, 0.33, 0.34],
            'a': 4.0, 'b': 5.0, 'c': 6.0,
            'alpha': 90, 'beta': 90, 'gamma': 90,
            'volume': 120,
            'spacegroup': 27,
        },
        {
            'formula': 'PbZrTiO3',
            'elements': ['Pb', 'Zr', 'Ti', 'O'],
            'fractions': [0.1, 0.1, 0.1, 0.7],
            'a': 4.1, 'b': 4.1, 'c': 4.1,
            'alpha': 90, 'beta': 90, 'gamma': 90,
            'volume': 68,
            'spacegroup': 99,
        },
        {
            'formula': 'H-O-O',
            'elements': ['H', 'O', 'O'],
            'fractions': [0.48, 0.38, 0.14],
            'a': 4.7, 'b': 5.0, 'c': 6.0,
            'alpha': 96, 'beta': 94, 'gamma': 106,
            'volume': 141,
            'spacegroup': 1,
        },
    ]
    
    print("\n" + "="*60)
    for case in test_cases:
        result = validator.validate(case)
        print(f"\n{case['formula']}:")
        print(f"  Level: {result.level.name} (Score: {result.score:.1f})")
        print(f"  Chemistry: {result.chemistry_score:.1f}")
        print(f"  Lattice: {result.lattice_score:.1f}")
        print(f"  Ferroelectric: {result.ferroelectric_score:.1f}")
        if result.issues:
            print(f"  Issues: {result.issues}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
