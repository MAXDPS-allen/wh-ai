# Ferroelectric Material Prediction & Generation Pipeline

基于机器学习的铁电材料预测与生成完整管线。从 Materials Project 数据库获取极性无机材料，训练级联判别器、变分生成器和逆向设计模型，实现新型铁电材料的自动发现。

## Pipeline Architecture

```
                        ┌─────────────────────────────┐
                        │  01 Data Acquisition        │
                        │  MP极性材料爬取 + 铁电标注   │
                        └─────────────┬───────────────┘
                                      │
                        ┌─────────────▼───────────────┐
                        │  02 Feature Engineering     │
                        │  E(3)-等变特征提取 (64/256D) │
                        └─────────────┬───────────────┘
                                      │
                  ┌───────────────────┼───────────────────┐
                  │                   │                   │
    ┌─────────────▼──────┐  ┌────────▼────────┐  ┌──────▼──────────────┐
    │  03 Cascade        │  │  04 Variational │  │  05 Inverse Design  │
    │  Discriminator     │  │  Generator      │  │  逆向工程模型       │
    │  GCNN+NequIP级联   │  │  CVAE变分生成   │  │  特征→组分+晶格     │
    └─────────────┬──────┘  └────────┬────────┘  └──────┬──────────────┘
                  │                   │                   │
                  └───────────────────┼───────────────────┘
                                      │
                        ┌─────────────▼───────────────┐
                        │  06 Rationality Filter      │
                        │  合理性判别 + 约束过滤       │
                        └─────────────┬───────────────┘
                                      │
                        ┌─────────────▼───────────────┐
                        │  07 MP Comparison           │
                        │  Materials Project数据库比对 │
                        └─────────────┬───────────────┘
                                      │
                        ┌─────────────▼───────────────┐
                        │  08 Full Pipeline           │
                        │  端到端发现流水线            │
                        └─────────────────────────────┘
```

## Directory Structure

```
ferroelectric_pipeline/
├── 01_data_acquisition/          # MP数据爬取与筛选
│   ├── screen_mp_polar_v2.py     # 从MP API获取68个极性空间群的材料，使用GCNN v6 + NequIP v2双模型筛选
│   ├── screen_mp_original.py     # 原始MP数据筛选（GCNN v5 + NequIP共识预测）
│   ├── refine_predictions.py     # 训练集标签校正，确保已知铁电材料被正确分类
│   ├── postprocess_v2.py         # 多阈值假阳性消除
│   └── final_combined_screening.py  # v1+v2预测融合，生成4级置信度候选列表
│
├── 02_feature_engineering/       # 特征工程
│   ├── advanced_feature_engineering.py  # 256维E(3)-等变特征提取器（球谐函数+径向基+图卷积+化学特征）
│   └── test_feature_engineering.py      # 特征提取验证脚本
│
├── 03_cascade_discriminator/     # 级联判别器
│   ├── GCNN_v5.py                # 图卷积神经网络v5（3模型集成+SMOTE，code_v5最佳GCNN）
│   ├── NequIP_Classifier_v2.py   # E(3)-等变NequIP分类器v2
│   ├── train_nequip_v9.py        # NequIP v9: 6层SE(3)等变消息传递 + 注意力机制
│   ├── train_nequip_v9_full.py   # NequIP v9完整训练管线（含模型持久化）
│   ├── train_graph_nn.py         # 消息传递神经网络MPNN（原子→节点，化学键→边）
│   ├── train_cascade.py          # 二阶段级联：高召回率筛选 → 高精度确认
│   ├── train_composite_v2.py     # 4模型组合级联（End-to-End + GCNN + NequIP_v6 + Transformer）
│   └── train_final.py            # 最终版集成分类器（Transformer架构 + 阈值校准）
│
├── 04_variational_generator/     # 变分生成器
│   ├── FE_CVAE.py                # 条件变分自编码器（64D输入→32D隐空间→重构，β-VAE）
│   └── evaluate_cvae.py          # CVAE质量评估（KS检验、t-SNE可视化、分布匹配）
│
├── 05_inverse_design/            # 逆向工程模型
│   ├── inverse_design_v7.py      # 最新版：注意力机制+残差块，sqrt变换预测晶格参数比
│   └── inverse_design_v6.py      # v6版本：分离式晶格/组分网络，元素分类预测
│
├── 06_rationality_filter/        # 合理性判别器
│   ├── constraint_parser.py      # 约束定义与解析（元素、带隙、空间群、稳定性、晶格、组分）
│   ├── constrained_filter.py     # 多级约束过滤引擎
│   ├── main.py                   # 约束驱动的材料生成主控模块（含MP比对）
│   └── config/                   # 约束配置示例
│       ├── constraints_schema.json   # JSON Schema验证
│       ├── example_titanate.json     # 钛酸盐铁电体约束示例
│       ├── example_leadfree.json     # 无铅压电体约束示例
│       └── simple_test.json          # 简易测试配置
│
├── 07_mp_comparison/             # MP数据库比对模块
│   ├── screen_database.py        # GCNN模型数据库筛选
│   └── screen_database_nequip.py # NequIP模型数据库筛选
│
└── 08_full_pipeline/             # 端到端流水线
    ├── full_discovery_pipeline.py     # 完整发现流程（生成→验证→MP比对→报告）
    └── generate_materials_pipeline.py # 材料生成流水线（GAN生成→逆向设计→合理性验证）
```

## Pipeline Stages

### Stage 1: Data Acquisition (数据获取)

从 Materials Project API 爬取所有属于68个极性空间群的无机材料，结合已标注的铁电材料数据库构建训练集。

- **输入**: Materials Project API Key
- **输出**: `mp_polar_predictions_v2_all.csv`, `fe_candidates_final_*.csv`
- **关键参数**: 68个极性空间群（三斜/单斜/正交/四方/三方/六方晶系）
- **双模型共识**: GCNN (threshold=0.87) + NequIP (threshold=0.94)

```bash
python 01_data_acquisition/screen_mp_polar_v2.py
python 01_data_acquisition/final_combined_screening.py
```

### Stage 2: Feature Engineering (特征工程)

从晶体结构中提取高维特征向量，支持64维（基础）和256维（高级E(3)-等变）两种模式。

- **64维特征**: 平均质量、半径、电负性、电离能、价电子数、晶格畸变、密度等
- **256维特征**: 球谐函数编码、径向基函数、图卷积特征、对称性特征

### Stage 3: Cascade Discriminator (级联判别器)

多阶段级联分类器，逐步精炼预测结果：

| 模型 | 架构 | 特点 | 目标 |
|------|------|------|------|
| GCNN v5 | GAT/GCN + 3模型集成 | SMOTE过采样，极端正样本权重(20x) | 高召回率 |
| NequIP v9 | 6层SE(3)等变消息传递 | 注意力增强，Focal Loss | Acc & Recall ≥ 99% |
| 级联分类器 | 二阶段 | 高召回率筛选 → 高精度确认 | 最小化假阴性 |
| 组合v2 | 4模型加权投票 | 极性空间群预筛 + 专家规则验证 | 综合最优 |

```bash
python 03_cascade_discriminator/train_nequip_v9_full.py
python 03_cascade_discriminator/train_composite_v2.py
```

### Stage 4: Variational Generator (变分生成器)

条件变分自编码器 (CVAE) 学习铁电材料的隐空间分布：

- **编码器**: 64D特征 + 条件标签 → (μ, log σ²) → 32D隐空间
- **解码器**: 32D隐变量 + 条件标签 → 重构64D特征
- **损失函数**: 重构损失 + β·KL散度 (β=0.001)
- **训练**: 500 epochs, ReduceLROnPlateau

```bash
python 04_variational_generator/FE_CVAE.py
python 04_variational_generator/evaluate_cvae.py
```

### Stage 5: Inverse Design (逆向工程)

将隐空间特征向量反向映射为具体的材料组分和晶格参数：

- **输入**: 64D特征向量
- **输出**: 元素种类（5位×87元素分类）、原子分数（softmax）、晶格参数（a, b, c, α, β, γ）、空间群
- **v7改进**: sqrt变换预测晶格参数比 (sqrt(b/a), sqrt(c/a))，注意力+残差网络

```bash
python 05_inverse_design/inverse_design_v7.py
```

### Stage 6: Rationality Filter (合理性判别)

基于物理化学规则的多层约束过滤：

- **元素约束**: 必须包含/排除元素，元素数量范围
- **带隙约束**: 最小/最大带隙 (eV)
- **空间群约束**: 仅极性空间群，晶系过滤
- **稳定性约束**: 热力学稳定性（hull上方能量）
- **晶格约束**: 体积和晶格参数范围
- **组分约束**: 元素分数边界

```bash
python 06_rationality_filter/main.py --config config/example_titanate.json
```

### Stage 7: MP Comparison (数据库比对)

将生成的候选材料与 Materials Project 数据库进行结构相似性匹配：

- 匹配评分基于晶格、体积和空间群兼容性 (0-100分)
- 识别已知稳定的极性材料
- 标记新发现的铁电候选

### Stage 8: Full Pipeline (完整流水线)

端到端自动化运行全部阶段：

```bash
python 08_full_pipeline/full_discovery_pipeline.py --n_samples 500 --output_dir results/
```

## Dependencies

```
numpy
pandas
scipy
scikit-learn
torch
torch_geometric
pymatgen
matminer
xgboost
tqdm
matplotlib
mp_api          # Materials Project API
```

## Training Data

| 数据集 | 文件 | 描述 |
|--------|------|------|
| 铁电正样本 | `data_files/dataset_original_ferroelectric.jsonl` | 已知铁电材料结构 |
| 已知铁电补充 | `data_files/dataset_known_FE_rest.jsonl` | 额外标注的铁电材料 |
| 非铁电负样本 | `data_files/dataset_nonFE.jsonl` | 极性但非铁电的材料 |

## Model Checkpoints

训练产生的模型保存在项目根目录下的对应文件夹：

| 模型 | 路径 |
|------|------|
| GCNN v5/v6 | `model_gcnn_v5/`, `model_gcnn_v6/` |
| NequIP | `model_nequip/`, `model_nequip_v2/` |
| 级联模型 | `model_cascade/` (5-fold) |
| CVAE | `model_cvae/cvae_best.pt` |
| 逆向设计 | `invs_dgn_model_v2/inverse_design_v7_best.pt` |
| 组合模型 | `model_composite_v2/` |

## Key Design Decisions

1. **双模型共识**: GCNN + NequIP 必须同时预测为铁电才确认，降低假阳性率
2. **68极性空间群预筛**: 利用对称性物理先验知识缩小搜索空间
3. **CVAE替代GAN**: 更稳定的分布学习（避免模式崩塌）
4. **级联策略**: 第一阶段追求99.5%+召回率（宁可多选），第二阶段追求99%+精度（精确确认）
5. **sqrt晶格变换**: 逆向设计中用 sqrt(b/a), sqrt(c/a) 替代直接比例预测，改善数值稳定性
