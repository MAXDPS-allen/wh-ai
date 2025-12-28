"""
约束解析器模块
用于解析和验证用户定义的材料生成约束条件
"""

import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path


# 有效元素列表（与 feature_engineering 模块一致）
VALID_ELEMENTS = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
    'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
    'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
    'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
    'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr'
]

# 有效晶系
VALID_CRYSTAL_SYSTEMS = [
    'triclinic', 'monoclinic', 'orthorhombic', 
    'tetragonal', 'trigonal', 'hexagonal', 'cubic'
]

# 极性点群（用于铁电材料筛选）
POLAR_POINT_GROUPS = [
    '1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm'
]


@dataclass
class ElementConstraints:
    """元素约束"""
    must_include: List[str] = field(default_factory=list)
    must_exclude: List[str] = field(default_factory=list)
    allow_only: Optional[List[str]] = None
    min_elements: int = 2
    max_elements: int = 5


@dataclass
class BandGapConstraints:
    """带隙约束"""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    unit: str = "eV"


@dataclass
class SpacegroupConstraints:
    """空间群约束"""
    polar_only: bool = False
    crystal_system: Optional[List[str]] = None
    spacegroup_numbers: Optional[List[int]] = None


@dataclass
class StabilityConstraints:
    """稳定性约束"""
    require_stable: bool = False
    max_energy_above_hull: float = 0.1  # eV/atom


@dataclass
class LatticeConstraints:
    """晶格约束"""
    volume_range: Optional[Tuple[float, float]] = None  # Å³
    a_range: Optional[Tuple[float, float]] = None  # Å
    b_range: Optional[Tuple[float, float]] = None  # Å
    c_range: Optional[Tuple[float, float]] = None  # Å


@dataclass
class CompositionConstraints:
    """组成约束"""
    min_fraction: Dict[str, float] = field(default_factory=dict)
    max_fraction: Dict[str, float] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """生成配置"""
    n_candidates: int = 100
    n_iterations: int = 5
    temperature: float = 1.0
    seed: Optional[int] = None


@dataclass
class OutputConfig:
    """输出配置"""
    output_dir: str = "results"
    save_all_candidates: bool = False
    generate_report: bool = True
    export_format: str = "json"


@dataclass
class ConstraintSet:
    """完整的约束集合"""
    name: str
    description: str
    elements: ElementConstraints
    band_gap: BandGapConstraints
    spacegroup: SpacegroupConstraints
    stability: StabilityConstraints
    lattice: LatticeConstraints
    composition: CompositionConstraints
    generation: GenerationConfig
    output: OutputConfig
    
    @property
    def summary(self) -> str:
        """生成约束摘要"""
        parts = [f"约束集: {self.name}"]
        parts.append(f"描述: {self.description}")
        
        if self.elements.must_include:
            parts.append(f"必须包含元素: {', '.join(self.elements.must_include)}")
        if self.elements.must_exclude:
            parts.append(f"排除元素: {', '.join(self.elements.must_exclude)}")
        if self.elements.allow_only:
            parts.append(f"仅允许元素: {', '.join(self.elements.allow_only)}")
            
        if self.band_gap.min_value is not None:
            parts.append(f"带隙下限: {self.band_gap.min_value} {self.band_gap.unit}")
        if self.band_gap.max_value is not None:
            parts.append(f"带隙上限: {self.band_gap.max_value} {self.band_gap.unit}")
            
        if self.spacegroup.polar_only:
            parts.append("仅极性空间群")
        if self.spacegroup.crystal_system:
            parts.append(f"晶系: {', '.join(self.spacegroup.crystal_system)}")
            
        if self.stability.require_stable:
            parts.append(f"稳定性: e_above_hull ≤ {self.stability.max_energy_above_hull} eV/atom")
            
        return "\n".join(parts)


class ConstraintValidationError(Exception):
    """约束验证错误"""
    pass


class ConstraintParser:
    """约束文件解析器"""
    
    def __init__(self, schema_path: Optional[str] = None):
        """
        初始化解析器
        
        Args:
            schema_path: JSON Schema文件路径（可选，用于验证）
        """
        self.schema_path = schema_path
        self.schema = None
        
        if schema_path and os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
    
    def parse(self, constraint_file: str) -> ConstraintSet:
        """
        解析约束文件
        
        Args:
            constraint_file: 约束JSON文件路径
            
        Returns:
            ConstraintSet: 解析后的约束集合
            
        Raises:
            ConstraintValidationError: 约束验证失败
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON格式错误
        """
        # 读取文件
        with open(constraint_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证基本结构
        self._validate_structure(data)
        
        # 解析各部分约束
        elements = self._parse_element_constraints(data.get('constraints', {}).get('elements', {}))
        band_gap = self._parse_band_gap_constraints(data.get('constraints', {}).get('band_gap', {}))
        spacegroup = self._parse_spacegroup_constraints(data.get('constraints', {}).get('spacegroup', {}))
        stability = self._parse_stability_constraints(data.get('constraints', {}).get('stability', {}))
        lattice = self._parse_lattice_constraints(data.get('constraints', {}).get('lattice', {}))
        composition = self._parse_composition_constraints(data.get('constraints', {}).get('composition', {}))
        generation = self._parse_generation_config(data.get('generation', {}))
        output = self._parse_output_config(data.get('output', {}))
        
        # 交叉验证
        self._cross_validate(elements, band_gap, spacegroup, stability)
        
        return ConstraintSet(
            name=data.get('name', 'Unnamed'),
            description=data.get('description', ''),
            elements=elements,
            band_gap=band_gap,
            spacegroup=spacegroup,
            stability=stability,
            lattice=lattice,
            composition=composition,
            generation=generation,
            output=output
        )
    
    def _validate_structure(self, data: Dict[str, Any]) -> None:
        """验证JSON基本结构"""
        if not isinstance(data, dict):
            raise ConstraintValidationError("约束文件必须是JSON对象")
        
        if 'name' not in data:
            raise ConstraintValidationError("约束文件必须包含 'name' 字段")
    
    def _parse_element_constraints(self, data: Dict[str, Any]) -> ElementConstraints:
        """解析元素约束"""
        constraints = ElementConstraints()
        
        if 'must_include' in data:
            elements = data['must_include']
            self._validate_elements(elements, "must_include")
            constraints.must_include = elements
            
        if 'must_exclude' in data:
            elements = data['must_exclude']
            self._validate_elements(elements, "must_exclude")
            constraints.must_exclude = elements
            
        if 'allow_only' in data:
            elements = data['allow_only']
            self._validate_elements(elements, "allow_only")
            constraints.allow_only = elements
            
        if 'min_elements' in data:
            min_el = data['min_elements']
            if not isinstance(min_el, int) or min_el < 1:
                raise ConstraintValidationError("min_elements 必须是大于0的整数")
            constraints.min_elements = min_el
            
        if 'max_elements' in data:
            max_el = data['max_elements']
            if not isinstance(max_el, int) or max_el < 1:
                raise ConstraintValidationError("max_elements 必须是大于0的整数")
            constraints.max_elements = max_el
            
        # 验证 min <= max
        if constraints.min_elements > constraints.max_elements:
            raise ConstraintValidationError(
                f"min_elements ({constraints.min_elements}) 不能大于 max_elements ({constraints.max_elements})"
            )
        
        # 验证 must_include 和 must_exclude 不冲突
        if constraints.must_include and constraints.must_exclude:
            overlap = set(constraints.must_include) & set(constraints.must_exclude)
            if overlap:
                raise ConstraintValidationError(
                    f"元素 {overlap} 同时出现在 must_include 和 must_exclude 中"
                )
        
        # 验证 must_include 元素都在 allow_only 中（如果设置了allow_only）
        if constraints.allow_only and constraints.must_include:
            missing = set(constraints.must_include) - set(constraints.allow_only)
            if missing:
                raise ConstraintValidationError(
                    f"must_include 中的元素 {missing} 不在 allow_only 列表中"
                )
        
        return constraints
    
    def _validate_elements(self, elements: List[str], field_name: str) -> None:
        """验证元素列表"""
        if not isinstance(elements, list):
            raise ConstraintValidationError(f"{field_name} 必须是数组")
        
        for el in elements:
            if el not in VALID_ELEMENTS:
                raise ConstraintValidationError(f"无效元素 '{el}' 在 {field_name} 中")
    
    def _parse_band_gap_constraints(self, data: Dict[str, Any]) -> BandGapConstraints:
        """解析带隙约束"""
        constraints = BandGapConstraints()
        
        if 'min' in data:
            min_val = data['min']
            if not isinstance(min_val, (int, float)):
                raise ConstraintValidationError("band_gap.min 必须是数值")
            constraints.min_value = float(min_val)
            
        if 'max' in data:
            max_val = data['max']
            if not isinstance(max_val, (int, float)):
                raise ConstraintValidationError("band_gap.max 必须是数值")
            constraints.max_value = float(max_val)
            
        if 'unit' in data:
            constraints.unit = data['unit']
            
        # 验证 min <= max
        if constraints.min_value is not None and constraints.max_value is not None:
            if constraints.min_value > constraints.max_value:
                raise ConstraintValidationError(
                    f"band_gap.min ({constraints.min_value}) 不能大于 band_gap.max ({constraints.max_value})"
                )
        
        return constraints
    
    def _parse_spacegroup_constraints(self, data: Dict[str, Any]) -> SpacegroupConstraints:
        """解析空间群约束"""
        constraints = SpacegroupConstraints()
        
        if 'polar_only' in data:
            constraints.polar_only = bool(data['polar_only'])
            
        if 'crystal_system' in data:
            systems = data['crystal_system']
            if not isinstance(systems, list):
                raise ConstraintValidationError("crystal_system 必须是数组")
            for sys in systems:
                if sys.lower() not in VALID_CRYSTAL_SYSTEMS:
                    raise ConstraintValidationError(f"无效晶系 '{sys}'")
            constraints.crystal_system = [s.lower() for s in systems]
            
        if 'spacegroup_numbers' in data:
            numbers = data['spacegroup_numbers']
            if not isinstance(numbers, list):
                raise ConstraintValidationError("spacegroup_numbers 必须是数组")
            for num in numbers:
                if not isinstance(num, int) or num < 1 or num > 230:
                    raise ConstraintValidationError(f"无效空间群编号 {num}（必须在1-230之间）")
            constraints.spacegroup_numbers = numbers
        
        return constraints
    
    def _parse_stability_constraints(self, data: Dict[str, Any]) -> StabilityConstraints:
        """解析稳定性约束"""
        constraints = StabilityConstraints()
        
        if 'require_stable' in data:
            constraints.require_stable = bool(data['require_stable'])
            
        if 'max_energy_above_hull' in data:
            max_e = data['max_energy_above_hull']
            if not isinstance(max_e, (int, float)):
                raise ConstraintValidationError("max_energy_above_hull 必须是数值")
            if max_e < 0:
                raise ConstraintValidationError("max_energy_above_hull 不能为负数")
            constraints.max_energy_above_hull = float(max_e)
        
        return constraints
    
    def _parse_lattice_constraints(self, data: Dict[str, Any]) -> LatticeConstraints:
        """解析晶格约束"""
        constraints = LatticeConstraints()
        
        def parse_range(key: str) -> Optional[Tuple[float, float]]:
            if key not in data:
                return None
            range_data = data[key]
            if not isinstance(range_data, dict):
                raise ConstraintValidationError(f"{key} 必须是对象")
            min_val = range_data.get('min')
            max_val = range_data.get('max')
            if min_val is not None and max_val is not None:
                if min_val > max_val:
                    raise ConstraintValidationError(f"{key}.min 不能大于 {key}.max")
                return (float(min_val), float(max_val))
            elif min_val is not None:
                return (float(min_val), float('inf'))
            elif max_val is not None:
                return (0.0, float(max_val))
            return None
        
        constraints.volume_range = parse_range('volume')
        constraints.a_range = parse_range('a')
        constraints.b_range = parse_range('b')
        constraints.c_range = parse_range('c')
        
        return constraints
    
    def _parse_composition_constraints(self, data: Dict[str, Any]) -> CompositionConstraints:
        """解析组成约束"""
        constraints = CompositionConstraints()
        
        if 'min_fraction' in data:
            min_frac = data['min_fraction']
            if not isinstance(min_frac, dict):
                raise ConstraintValidationError("min_fraction 必须是对象")
            for el, frac in min_frac.items():
                if el not in VALID_ELEMENTS:
                    raise ConstraintValidationError(f"无效元素 '{el}' 在 min_fraction 中")
                if not 0 <= frac <= 1:
                    raise ConstraintValidationError(f"分数 {frac} 必须在 0-1 之间")
            constraints.min_fraction = min_frac
            
        if 'max_fraction' in data:
            max_frac = data['max_fraction']
            if not isinstance(max_frac, dict):
                raise ConstraintValidationError("max_fraction 必须是对象")
            for el, frac in max_frac.items():
                if el not in VALID_ELEMENTS:
                    raise ConstraintValidationError(f"无效元素 '{el}' 在 max_fraction 中")
                if not 0 <= frac <= 1:
                    raise ConstraintValidationError(f"分数 {frac} 必须在 0-1 之间")
            constraints.max_fraction = max_frac
        
        return constraints
    
    def _parse_generation_config(self, data: Dict[str, Any]) -> GenerationConfig:
        """解析生成配置"""
        config = GenerationConfig()
        
        if 'n_candidates' in data:
            n = data['n_candidates']
            if not isinstance(n, int) or n < 1:
                raise ConstraintValidationError("n_candidates 必须是正整数")
            config.n_candidates = n
            
        if 'n_iterations' in data:
            n = data['n_iterations']
            if not isinstance(n, int) or n < 1:
                raise ConstraintValidationError("n_iterations 必须是正整数")
            config.n_iterations = n
            
        if 'temperature' in data:
            t = data['temperature']
            if not isinstance(t, (int, float)) or t <= 0:
                raise ConstraintValidationError("temperature 必须是正数")
            config.temperature = float(t)
            
        if 'seed' in data:
            config.seed = data['seed']
        
        return config
    
    def _parse_output_config(self, data: Dict[str, Any]) -> OutputConfig:
        """解析输出配置"""
        config = OutputConfig()
        
        if 'output_dir' in data:
            config.output_dir = data['output_dir']
            
        if 'save_all_candidates' in data:
            config.save_all_candidates = bool(data['save_all_candidates'])
            
        if 'generate_report' in data:
            config.generate_report = bool(data['generate_report'])
            
        if 'export_format' in data:
            fmt = data['export_format']
            if fmt not in ['json', 'csv', 'both']:
                raise ConstraintValidationError(f"无效的 export_format '{fmt}'，必须是 json、csv 或 both")
            config.export_format = fmt
        
        return config
    
    def _cross_validate(self, 
                       elements: ElementConstraints,
                       band_gap: BandGapConstraints,
                       spacegroup: SpacegroupConstraints,
                       stability: StabilityConstraints) -> None:
        """交叉验证约束"""
        # 检查元素数量约束是否与 must_include 兼容
        if len(elements.must_include) > elements.max_elements:
            raise ConstraintValidationError(
                f"must_include 包含 {len(elements.must_include)} 个元素，"
                f"但 max_elements 仅为 {elements.max_elements}"
            )
        
        # 检查 allow_only 元素数量是否足够
        if elements.allow_only:
            if len(elements.allow_only) < elements.min_elements:
                raise ConstraintValidationError(
                    f"allow_only 仅包含 {len(elements.allow_only)} 个元素，"
                    f"但 min_elements 要求至少 {elements.min_elements} 个"
                )


def load_constraints(constraint_file: str, 
                    schema_file: Optional[str] = None) -> ConstraintSet:
    """
    便捷函数：加载约束文件
    
    Args:
        constraint_file: 约束JSON文件路径
        schema_file: 可选的JSON Schema文件路径
        
    Returns:
        ConstraintSet: 解析后的约束集合
    """
    parser = ConstraintParser(schema_file)
    return parser.parse(constraint_file)


# 测试代码
if __name__ == "__main__":
    import sys
    
    # 获取当前脚本目录
    script_dir = Path(__file__).parent
    config_dir = script_dir / "config"
    
    # 测试解析示例文件
    example_files = [
        config_dir / "example_titanate.json",
        config_dir / "example_leadfree.json"
    ]
    
    schema_file = config_dir / "constraints_schema.json"
    
    for example_file in example_files:
        if example_file.exists():
            print(f"\n{'='*60}")
            print(f"解析文件: {example_file.name}")
            print('='*60)
            
            try:
                constraints = load_constraints(str(example_file), str(schema_file))
                print(constraints.summary)
                print(f"\n生成配置: {constraints.generation.n_candidates} 候选，"
                      f"{constraints.generation.n_iterations} 轮迭代")
            except Exception as e:
                print(f"错误: {e}")
        else:
            print(f"文件不存在: {example_file}")
