# 约束条件驱动的铁电材料生成模块

## 简介

本模块提供基于用户自定义约束条件的铁电材料发现和生成功能。用户可以通过JSON配置文件指定所需材料的各种约束（如元素组成、带隙范围、空间群类型等），系统将自动筛选已知材料数据库并生成满足约束的新候选材料。

## 功能特性

- ✅ **灵活的约束配置**: 通过JSON文件定义元素、带隙、空间群、稳定性、晶格参数等约束
- ✅ **已知材料搜索**: 自动搜索内置铁电材料数据库中满足约束的材料
- ✅ **智能材料生成**: 基于CVAE+逆向设计网络生成新候选材料
- ✅ **多层筛选**: 对生成的材料应用用户定义的约束条件筛选
- ✅ **MP数据库对比**: 自动与Materials Project数据库对比验证
- ✅ **详细报告**: 生成完整的结果报告和统计信息

## 目录结构

```
constrained_generation/
├── config/                          # 配置文件目录
│   ├── constraints_schema.json      # JSON Schema 定义
│   ├── example_titanate.json        # 示例：钛酸盐铁电材料
│   └── example_leadfree.json        # 示例：无铅压电材料
├── constraint_parser.py             # 约束解析器
├── constrained_filter.py            # 条件筛选模块
├── main.py                          # 主程序入口
└── README.md                        # 本文档
```

## 使用方法

### 1. 运行示例

```bash
# 搜索钛酸盐铁电材料
python main.py --constraint config/example_titanate.json

# 搜索无铅压电材料
python main.py --constraint config/example_leadfree.json

# 自定义输出目录和迭代次数
python main.py --constraint config/example_titanate.json --output results/titanate --iterations 10
```

### 2. 命令行参数

| 参数 | 说明 |
|------|------|
| `--constraint` | **必需**。约束配置文件路径（JSON格式）|
| `--output` | 可选。输出目录路径（覆盖配置文件设置）|
| `--iterations` | 可选。生成迭代次数（覆盖配置文件设置）|
| `--candidates` | 可选。每次迭代生成的候选数量（覆盖配置文件设置）|

### 3. 创建自定义约束文件

创建一个JSON文件，按以下格式定义约束：

```json
{
  "name": "我的材料搜索",
  "description": "搜索描述",
  
  "constraints": {
    "elements": {
      "must_include": ["Ti", "O"],    // 必须包含的元素
      "must_exclude": ["Pb", "Cd"],   // 必须排除的元素
      "allow_only": null,             // 仅允许的元素列表（可选）
      "min_elements": 2,              // 最少元素数量
      "max_elements": 5               // 最多元素数量
    },
    
    "band_gap": {
      "min": 2.0,                     // 带隙下限 (eV)
      "max": 5.0,                     // 带隙上限 (eV)
      "unit": "eV"
    },
    
    "spacegroup": {
      "polar_only": true,             // 仅极性空间群
      "crystal_system": ["tetragonal", "orthorhombic"]  // 晶系限制
    },
    
    "stability": {
      "require_stable": true,         // 要求热力学稳定
      "max_energy_above_hull": 0.1    // 最大凸包上方能量 (eV/atom)
    },
    
    "lattice": {
      "volume": {"min": 30, "max": 500}  // 体积范围 (Å³)
    },
    
    "composition": {
      "min_fraction": {"O": 0.3},     // 最小原子分数
      "max_fraction": {"Ti": 0.5}     // 最大原子分数
    }
  },
  
  "generation": {
    "n_candidates": 100,              // 每次迭代生成候选数
    "n_iterations": 5,                // 迭代次数
    "temperature": 1.0,               // 生成温度
    "seed": null                      // 随机种子（可选）
  },
  
  "output": {
    "output_dir": "results/my_search",
    "save_all_candidates": false,
    "generate_report": true,
    "export_format": "json"
  }
}
```

## 约束类型详解

### 元素约束 (elements)

| 字段 | 类型 | 说明 |
|------|------|------|
| `must_include` | 数组 | 必须包含的元素列表 |
| `must_exclude` | 数组 | 必须排除的元素列表 |
| `allow_only` | 数组 | 仅允许使用的元素白名单 |
| `min_elements` | 整数 | 材料中最少元素种类数 (默认2) |
| `max_elements` | 整数 | 材料中最多元素种类数 (默认5) |

### 带隙约束 (band_gap)

| 字段 | 类型 | 说明 |
|------|------|------|
| `min` | 数值 | 带隙下限 |
| `max` | 数值 | 带隙上限 |
| `unit` | 字符串 | 单位 (默认 "eV") |

### 空间群约束 (spacegroup)

| 字段 | 类型 | 说明 |
|------|------|------|
| `polar_only` | 布尔 | 是否仅筛选极性空间群 |
| `crystal_system` | 数组 | 允许的晶系列表 |
| `spacegroup_numbers` | 数组 | 允许的空间群编号列表 (1-230) |

可用晶系: `triclinic`, `monoclinic`, `orthorhombic`, `tetragonal`, `trigonal`, `hexagonal`, `cubic`

### 稳定性约束 (stability)

| 字段 | 类型 | 说明 |
|------|------|------|
| `require_stable` | 布尔 | 是否要求热力学稳定 |
| `max_energy_above_hull` | 数值 | 最大凸包上方能量 (eV/atom，默认0.1) |

### 晶格约束 (lattice)

| 字段 | 类型 | 说明 |
|------|------|------|
| `volume` | 对象 | 体积范围 {min, max} (Å³) |
| `a`, `b`, `c` | 对象 | 晶格参数范围 {min, max} (Å) |

### 组成约束 (composition)

| 字段 | 类型 | 说明 |
|------|------|------|
| `min_fraction` | 对象 | 元素最小原子分数 {元素: 分数} |
| `max_fraction` | 对象 | 元素最大原子分数 {元素: 分数} |

## 输出文件

运行后会在输出目录生成以下文件：

| 文件名 | 说明 |
|--------|------|
| `known_materials_matched.csv` | 已知数据库中满足约束的材料 |
| `generated_materials_all.csv` | 所有生成并通过筛选的候选材料 |
| `mp_matched_filtered.csv` | 与MP匹配且通过约束筛选的材料 |
| `report.txt` | 详细文本报告 |
| `statistics.json` | 统计信息JSON |

## 示例用例

### 用例1：钛酸盐宽带隙铁电

搜索含钛和氧、带隙大于2eV、不含有毒元素的铁电材料：

```json
{
  "name": "Wide bandgap titanates",
  "constraints": {
    "elements": {
      "must_include": ["Ti", "O"],
      "must_exclude": ["Pb", "Cd", "Hg", "As"]
    },
    "band_gap": {"min": 2.0},
    "spacegroup": {"polar_only": true}
  }
}
```

### 用例2：无铅钙钛矿压电

搜索基于环保元素的钙钛矿压电材料：

```json
{
  "name": "Lead-free perovskites",
  "constraints": {
    "elements": {
      "allow_only": ["K", "Na", "Li", "Ba", "Sr", "Ca", "Bi", "Nb", "Ta", "Ti", "Zr", "O", "F"]
    },
    "spacegroup": {
      "crystal_system": ["tetragonal", "orthorhombic", "trigonal"]
    },
    "stability": {
      "require_stable": true,
      "max_energy_above_hull": 0.025
    }
  }
}
```

### 用例3：高氧含量材料

搜索氧原子分数高于60%的氧化物：

```json
{
  "name": "High-oxygen materials",
  "constraints": {
    "elements": {
      "must_include": ["O"]
    },
    "composition": {
      "min_fraction": {"O": 0.6}
    }
  }
}
```

## 依赖项

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Pandas
- Pymatgen
- mp-api (可选，用于MP对比)

## 注意事项

1. 首次运行需要预训练模型文件：
   - `model_v3/cvae_model_v3.pth`
   - `invs_dgn_model/inverse_design_network_v7.pth`

2. MP数据库对比需要有效的API密钥

3. 约束条件越严格，生成通过率越低，可能需要增加迭代次数

4. 生成温度参数影响多样性：较高温度产生更多样的结果，但可能降低质量

## 问题反馈

如有问题或建议，请联系开发团队。
