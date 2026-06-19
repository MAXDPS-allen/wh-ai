"""
条件过滤模块
用于根据用户约束筛选材料候选
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pymatgen.core import Composition

from constraint_parser import (
    ConstraintSet,
    ElementConstraints,
    BandGapConstraints,
    SpacegroupConstraints,
    StabilityConstraints,
    LatticeConstraints,
    CompositionConstraints,
    POLAR_POINT_GROUPS
)


@dataclass
class MaterialCandidate:
    """材料候选数据结构"""
    # 基本信息
    id: int
    formula: str
    elements: List[str]
    
    # 组成
    composition: Dict[str, float]  # 元素 -> 原子分数
    
    # 晶格参数
    volume: float  # Å³
    a: float  # Å
    b: float  # Å
    c: float  # Å
    alpha: float  # 度
    beta: float  # 度
    gamma: float  # 度
    
    # 空间群信息
    spacegroup: Optional[int] = None
    point_group: Optional[str] = None
    crystal_system: Optional[str] = None
    
    # 电子结构
    band_gap: Optional[float] = None  # eV
    
    # 稳定性
    energy_above_hull: Optional[float] = None  # eV/atom
    is_stable: Optional[bool] = None
    
    # MP匹配信息 (如果匹配到)
    mp_id: Optional[str] = None
    mp_formula: Optional[str] = None
    is_polar: Optional[bool] = None


@dataclass
class FilterResult:
    """过滤结果"""
    passed: bool
    reasons: List[str]  # 通过或失败的原因
    
    def __str__(self):
        status = "通过" if self.passed else "失败"
        return f"{status}: {', '.join(self.reasons)}"


class ConstraintFilter:
    """约束过滤器"""
    
    def __init__(self, constraints: ConstraintSet):
        """
        初始化过滤器
        
        Args:
            constraints: 约束集合
        """
        self.constraints = constraints
        
    def filter_material(self, material: MaterialCandidate) -> FilterResult:
        """
        对单个材料应用所有约束
        
        Args:
            material: 材料候选
            
        Returns:
            FilterResult: 过滤结果
        """
        reasons = []
        passed = True
        
        # 1. 元素约束
        elem_result = self._check_element_constraints(material)
        if not elem_result.passed:
            passed = False
        reasons.extend(elem_result.reasons)
        
        # 2. 带隙约束
        if material.band_gap is not None:
            bg_result = self._check_band_gap_constraints(material)
            if not bg_result.passed:
                passed = False
            reasons.extend(bg_result.reasons)
        
        # 3. 空间群约束
        if material.spacegroup is not None or material.is_polar is not None:
            sg_result = self._check_spacegroup_constraints(material)
            if not sg_result.passed:
                passed = False
            reasons.extend(sg_result.reasons)
        
        # 4. 稳定性约束
        if material.energy_above_hull is not None:
            stab_result = self._check_stability_constraints(material)
            if not stab_result.passed:
                passed = False
            reasons.extend(stab_result.reasons)
        
        # 5. 晶格约束
        lat_result = self._check_lattice_constraints(material)
        if not lat_result.passed:
            passed = False
        reasons.extend(lat_result.reasons)
        
        # 6. 组成约束
        comp_result = self._check_composition_constraints(material)
        if not comp_result.passed:
            passed = False
        reasons.extend(comp_result.reasons)
        
        # 如果全部通过，添加通过原因
        if passed and not reasons:
            reasons.append("满足所有约束条件")
        
        return FilterResult(passed=passed, reasons=reasons)
    
    def _check_element_constraints(self, material: MaterialCandidate) -> FilterResult:
        """检查元素约束"""
        constraints = self.constraints.elements
        reasons = []
        passed = True
        
        material_elements = set(material.elements)
        
        # 检查 must_include
        if constraints.must_include:
            missing = set(constraints.must_include) - material_elements
            if missing:
                passed = False
                reasons.append(f"缺少必需元素: {', '.join(missing)}")
            else:
                reasons.append(f"包含必需元素: {', '.join(constraints.must_include)}")
        
        # 检查 must_exclude
        if constraints.must_exclude:
            forbidden = material_elements & set(constraints.must_exclude)
            if forbidden:
                passed = False
                reasons.append(f"包含禁止元素: {', '.join(forbidden)}")
            else:
                reasons.append(f"不含禁止元素")
        
        # 检查 allow_only
        if constraints.allow_only:
            disallowed = material_elements - set(constraints.allow_only)
            if disallowed:
                passed = False
                reasons.append(f"包含不允许的元素: {', '.join(disallowed)}")
            else:
                reasons.append(f"仅包含允许元素")
        
        # 检查元素数量
        n_elements = len(material_elements)
        if n_elements < constraints.min_elements:
            passed = False
            reasons.append(f"元素数量 {n_elements} 少于最小值 {constraints.min_elements}")
        elif n_elements > constraints.max_elements:
            passed = False
            reasons.append(f"元素数量 {n_elements} 多于最大值 {constraints.max_elements}")
        else:
            reasons.append(f"元素数量 {n_elements} 符合要求")
        
        return FilterResult(passed=passed, reasons=reasons)
    
    def _check_band_gap_constraints(self, material: MaterialCandidate) -> FilterResult:
        """检查带隙约束"""
        constraints = self.constraints.band_gap
        reasons = []
        passed = True
        
        bg = material.band_gap
        
        if constraints.min_value is not None:
            if bg < constraints.min_value:
                passed = False
                reasons.append(f"带隙 {bg:.2f} {constraints.unit} 小于下限 {constraints.min_value}")
            else:
                reasons.append(f"带隙 {bg:.2f} {constraints.unit} 满足下限")
        
        if constraints.max_value is not None:
            if bg > constraints.max_value:
                passed = False
                reasons.append(f"带隙 {bg:.2f} {constraints.unit} 大于上限 {constraints.max_value}")
            else:
                reasons.append(f"带隙 {bg:.2f} {constraints.unit} 满足上限")
        
        return FilterResult(passed=passed, reasons=reasons)
    
    def _check_spacegroup_constraints(self, material: MaterialCandidate) -> FilterResult:
        """检查空间群约束"""
        constraints = self.constraints.spacegroup
        reasons = []
        passed = True
        
        # 检查 polar_only
        if constraints.polar_only:
            if material.is_polar is not None:
                if not material.is_polar:
                    passed = False
                    reasons.append("非极性空间群")
                else:
                    reasons.append("极性空间群")
            elif material.point_group is not None:
                if material.point_group not in POLAR_POINT_GROUPS:
                    passed = False
                    reasons.append(f"点群 {material.point_group} 非极性")
                else:
                    reasons.append(f"点群 {material.point_group} 极性")
        
        # 检查 crystal_system
        if constraints.crystal_system and material.crystal_system:
            if material.crystal_system.lower() not in [s.lower() for s in constraints.crystal_system]:
                passed = False
                reasons.append(f"晶系 {material.crystal_system} 不在允许列表中")
            else:
                reasons.append(f"晶系 {material.crystal_system} 符合要求")
        
        # 检查 spacegroup_numbers
        if constraints.spacegroup_numbers and material.spacegroup:
            if material.spacegroup not in constraints.spacegroup_numbers:
                passed = False
                reasons.append(f"空间群 {material.spacegroup} 不在允许列表中")
            else:
                reasons.append(f"空间群 {material.spacegroup} 符合要求")
        
        return FilterResult(passed=passed, reasons=reasons)
    
    def _check_stability_constraints(self, material: MaterialCandidate) -> FilterResult:
        """检查稳定性约束"""
        constraints = self.constraints.stability
        reasons = []
        passed = True
        
        e_hull = material.energy_above_hull
        
        if constraints.require_stable:
            if e_hull > constraints.max_energy_above_hull:
                passed = False
                reasons.append(
                    f"能量高于凸包 {e_hull:.3f} eV/atom > "
                    f"阈值 {constraints.max_energy_above_hull} eV/atom"
                )
            else:
                reasons.append(
                    f"稳定性良好: 能量高于凸包 {e_hull:.3f} eV/atom ≤ "
                    f"阈值 {constraints.max_energy_above_hull} eV/atom"
                )
        
        return FilterResult(passed=passed, reasons=reasons)
    
    def _check_lattice_constraints(self, material: MaterialCandidate) -> FilterResult:
        """检查晶格约束"""
        constraints = self.constraints.lattice
        reasons = []
        passed = True
        
        # 检查体积
        if constraints.volume_range:
            min_v, max_v = constraints.volume_range
            if material.volume < min_v or material.volume > max_v:
                passed = False
                reasons.append(f"体积 {material.volume:.2f} Å³ 超出范围 [{min_v:.2f}, {max_v:.2f}]")
            else:
                reasons.append(f"体积 {material.volume:.2f} Å³ 符合要求")
        
        # 检查 a
        if constraints.a_range:
            min_a, max_a = constraints.a_range
            if material.a < min_a or material.a > max_a:
                passed = False
                reasons.append(f"晶格参数 a = {material.a:.2f} Å 超出范围 [{min_a:.2f}, {max_a:.2f}]")
            else:
                reasons.append(f"晶格参数 a = {material.a:.2f} Å 符合要求")
        
        # 检查 b
        if constraints.b_range:
            min_b, max_b = constraints.b_range
            if material.b < min_b or material.b > max_b:
                passed = False
                reasons.append(f"晶格参数 b = {material.b:.2f} Å 超出范围 [{min_b:.2f}, {max_b:.2f}]")
            else:
                reasons.append(f"晶格参数 b = {material.b:.2f} Å 符合要求")
        
        # 检查 c
        if constraints.c_range:
            min_c, max_c = constraints.c_range
            if material.c < min_c or material.c > max_c:
                passed = False
                reasons.append(f"晶格参数 c = {material.c:.2f} Å 超出范围 [{min_c:.2f}, {max_c:.2f}]")
            else:
                reasons.append(f"晶格参数 c = {material.c:.2f} Å 符合要求")
        
        return FilterResult(passed=passed, reasons=reasons)
    
    def _check_composition_constraints(self, material: MaterialCandidate) -> FilterResult:
        """检查组成约束"""
        constraints = self.constraints.composition
        reasons = []
        passed = True
        
        # 检查 min_fraction
        for element, min_frac in constraints.min_fraction.items():
            actual_frac = material.composition.get(element, 0.0)
            if actual_frac < min_frac:
                passed = False
                reasons.append(
                    f"元素 {element} 分数 {actual_frac:.3f} 小于下限 {min_frac}"
                )
            else:
                reasons.append(
                    f"元素 {element} 分数 {actual_frac:.3f} 满足下限 {min_frac}"
                )
        
        # 检查 max_fraction
        for element, max_frac in constraints.max_fraction.items():
            actual_frac = material.composition.get(element, 0.0)
            if actual_frac > max_frac:
                passed = False
                reasons.append(
                    f"元素 {element} 分数 {actual_frac:.3f} 大于上限 {max_frac}"
                )
            else:
                reasons.append(
                    f"元素 {element} 分数 {actual_frac:.3f} 满足上限 {max_frac}"
                )
        
        return FilterResult(passed=passed, reasons=reasons)
    
    def filter_batch(self, materials: List[MaterialCandidate]) -> Tuple[List[MaterialCandidate], List[FilterResult]]:
        """
        批量过滤材料
        
        Args:
            materials: 材料候选列表
            
        Returns:
            (通过的材料列表, 所有过滤结果列表)
        """
        passed_materials = []
        all_results = []
        
        for material in materials:
            result = self.filter_material(material)
            all_results.append(result)
            
            if result.passed:
                passed_materials.append(material)
        
        return passed_materials, all_results
    
    def get_statistics(self, results: List[FilterResult]) -> Dict[str, Any]:
        """
        获取过滤统计信息
        
        Args:
            results: 过滤结果列表
            
        Returns:
            统计信息字典
        """
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        
        # 统计失败原因
        failure_reasons = {}
        for result in results:
            if not result.passed:
                for reason in result.reasons:
                    # 提取关键原因
                    if "缺少必需元素" in reason:
                        key = "缺少必需元素"
                    elif "包含禁止元素" in reason:
                        key = "包含禁止元素"
                    elif "包含不允许的元素" in reason:
                        key = "包含不允许元素"
                    elif "元素数量" in reason and "少于" in reason:
                        key = "元素数量过少"
                    elif "元素数量" in reason and "多于" in reason:
                        key = "元素数量过多"
                    elif "带隙" in reason and "小于" in reason:
                        key = "带隙过小"
                    elif "带隙" in reason and "大于" in reason:
                        key = "带隙过大"
                    elif "非极性" in reason:
                        key = "非极性空间群"
                    elif "晶系" in reason:
                        key = "晶系不符"
                    elif "空间群" in reason:
                        key = "空间群不符"
                    elif "能量高于凸包" in reason:
                        key = "稳定性不足"
                    elif "体积" in reason:
                        key = "体积超出范围"
                    elif "晶格参数" in reason:
                        key = "晶格参数超出范围"
                    elif "分数" in reason and "小于" in reason:
                        key = "元素分数过低"
                    elif "分数" in reason and "大于" in reason:
                        key = "元素分数过高"
                    else:
                        key = "其他"
                    
                    failure_reasons[key] = failure_reasons.get(key, 0) + 1
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total if total > 0 else 0,
            'failure_reasons': failure_reasons
        }


# 便捷函数
def create_material_from_dict(data: Dict[str, Any], material_id: int = 0) -> MaterialCandidate:
    """
    从字典创建 MaterialCandidate
    
    Args:
        data: 包含材料信息的字典（支持嵌套的pymatgen结构格式）
        material_id: 材料ID
        
    Returns:
        MaterialCandidate
    """
    # 处理嵌套的 structure 格式
    structure_data = data.get('structure', {})
    lattice_data = structure_data.get('lattice', {})
    sites_data = structure_data.get('sites', [])
    
    # 解析元素 - 优先从 sites 提取
    elements = []
    if sites_data:
        for site in sites_data:
            species = site.get('species', [])
            for sp in species:
                el = sp.get('element', '')
                if el and el not in elements:
                    elements.append(el)
    elif 'elements' in data:
        elements = data['elements']
        if isinstance(elements, str):
            elements = [el.strip() for el in elements.split(',')]
    
    # 解析组成 - 从 sites 计算
    composition = {}
    if sites_data:
        element_counts = {}
        for site in sites_data:
            species = site.get('species', [])
            for sp in species:
                el = sp.get('element', '')
                occu = sp.get('occu', 1.0)
                if el:
                    element_counts[el] = element_counts.get(el, 0) + occu
        total = sum(element_counts.values())
        if total > 0:
            composition = {el: count/total for el, count in element_counts.items()}
    elif 'formula' in data:
        try:
            comp = Composition(data['formula'])
            composition = {str(el): comp.get_atomic_fraction(el) for el in comp.elements}
        except:
            pass
    
    # 获取晶格参数 - 优先从 lattice_data
    volume = lattice_data.get('volume', data.get('volume', 0.0))
    a = lattice_data.get('a', data.get('a', 0.0))
    b = lattice_data.get('b', data.get('b', 0.0))
    c = lattice_data.get('c', data.get('c', 0.0))
    alpha = lattice_data.get('alpha', data.get('alpha', 90.0))
    beta = lattice_data.get('beta', data.get('beta', 90.0))
    gamma = lattice_data.get('gamma', data.get('gamma', 90.0))
    
    # 空间群信息
    spacegroup = data.get('spacegroup_number', data.get('spacegroup'))
    point_group = data.get('point_group_symbol', data.get('point_group'))
    
    # 判断是否极性 - 基于点群
    is_polar = None
    if point_group:
        is_polar = point_group in POLAR_POINT_GROUPS
    
    # 生成公式（如果没有）
    formula = data.get('formula', '')
    if not formula and composition:
        formula = ''.join([f"{el}{composition[el]:.2f}" for el in sorted(composition.keys())])
    
    return MaterialCandidate(
        id=material_id,
        formula=formula,
        elements=elements,
        composition=composition,
        volume=volume if volume else 0.0,
        a=a if a else 0.0,
        b=b if b else 0.0,
        c=c if c else 0.0,
        alpha=alpha if alpha else 90.0,
        beta=beta if beta else 90.0,
        gamma=gamma if gamma else 90.0,
        spacegroup=spacegroup,
        point_group=point_group,
        crystal_system=data.get('crystal_system'),
        band_gap=data.get('band_gap'),
        energy_above_hull=data.get('energy_above_hull'),
        is_stable=data.get('is_stable'),
        mp_id=data.get('material_id', data.get('mp_id')),
        mp_formula=data.get('mp_formula'),
        is_polar=is_polar
    )


# 测试代码
if __name__ == "__main__":
    from pathlib import Path
    from constraint_parser import load_constraints
    
    # 获取当前脚本目录
    script_dir = Path(__file__).parent
    config_dir = script_dir / "config"
    
    # 加载示例约束
    constraint_file = config_dir / "example_titanate.json"
    
    if constraint_file.exists():
        print("加载约束文件...")
        constraints = load_constraints(str(constraint_file))
        print(constraints.summary)
        print()
        
        # 创建过滤器
        filter_obj = ConstraintFilter(constraints)
        
        # 测试材料
        test_materials = [
            # 符合条件: TiO2
            MaterialCandidate(
                id=1,
                formula="TiO2",
                elements=["Ti", "O"],
                composition={"Ti": 0.333, "O": 0.667},
                volume=62.0,
                a=4.59, b=4.59, c=2.96,
                alpha=90, beta=90, gamma=90,
                band_gap=3.2,
                crystal_system="tetragonal",
                is_polar=False
            ),
            # 不符合: 包含Pb
            MaterialCandidate(
                id=2,
                formula="PbTiO3",
                elements=["Pb", "Ti", "O"],
                composition={"Pb": 0.2, "Ti": 0.2, "O": 0.6},
                volume=68.0,
                a=3.90, b=3.90, c=4.15,
                alpha=90, beta=90, gamma=90,
                band_gap=3.5,
                crystal_system="tetragonal",
                is_polar=True
            ),
            # 不符合: 带隙过小
            MaterialCandidate(
                id=3,
                formula="Ti2O3",
                elements=["Ti", "O"],
                composition={"Ti": 0.4, "O": 0.6},
                volume=75.0,
                a=5.0, b=5.0, c=13.6,
                alpha=90, beta=90, gamma=120,
                band_gap=1.5,
                crystal_system="trigonal",
                is_polar=False
            )
        ]
        
        print("\n" + "="*60)
        print("测试材料筛选")
        print("="*60)
        
        passed_materials, results = filter_obj.filter_batch(test_materials)
        
        for material, result in zip(test_materials, results):
            print(f"\n材料 {material.id}: {material.formula}")
            print(f"  {result}")
        
        # 统计
        stats = filter_obj.get_statistics(results)
        print(f"\n统计信息:")
        print(f"  总数: {stats['total']}")
        print(f"  通过: {stats['passed']}")
        print(f"  失败: {stats['failed']}")
        print(f"  通过率: {stats['pass_rate']*100:.1f}%")
        
        if stats['failure_reasons']:
            print(f"\n失败原因:")
            for reason, count in stats['failure_reasons'].items():
                print(f"  {reason}: {count}")
    else:
        print(f"约束文件不存在: {constraint_file}")
