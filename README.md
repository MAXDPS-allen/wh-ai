# WH-AI: Ferroelectric Material Prediction & Generation

基于机器学习的铁电材料自动发现系统。通过级联判别器、变分生成器和逆向工程模型，实现从 Materials Project 数据库中筛选已知材料并生成新型铁电候选材料。

## Quick Start

完整管线已整理至 [`ferroelectric_pipeline/`](ferroelectric_pipeline/) 目录，按执行顺序分为8个阶段。详见 [Pipeline README](ferroelectric_pipeline/README.md)。

## Pipeline Overview

```
MP极性材料爬取 → 特征工程(64/256D) → 级联判别器(GCNN+NequIP) → CVAE生成 → 逆向设计 → 合理性过滤 → MP比对
```

| 阶段 | 模块 | 核心方法 |
|------|------|----------|
| 数据获取 | `01_data_acquisition` | MP API + 68极性空间群筛选 |
| 特征工程 | `02_feature_engineering` | E(3)-等变球谐函数 + 径向基 |
| 级联判别 | `03_cascade_discriminator` | GCNN v5 + NequIP v9 + 组合级联 |
| 变分生成 | `04_variational_generator` | 条件VAE (β-VAE, 32D latent) |
| 逆向设计 | `05_inverse_design` | 注意力+残差网络, 特征→组分+晶格 |
| 合理性判别 | `06_rationality_filter` | 物理化学约束过滤 |
| MP比对 | `07_mp_comparison` | 结构相似性匹配 |
| 完整流水线 | `08_full_pipeline` | 端到端自动发现 |

## Repository Structure

```
wh-ai/
├── ferroelectric_pipeline/   # ★ 整理后的完整管线（推荐入口）
├── code_v1/ ~ code_v6/       # 历史开发版本
├── data_files/               # 训练数据（JSONL格式）
├── model_*/                  # 训练好的模型检查点
├── gen/                      # 逆向设计模型源码
├── constrained_generation/   # 约束生成系统
├── mp_screening/             # MP原始筛选
├── mp_polar_screening/       # MP极性材料筛选
├── mp_comparison/            # MP比对结果
├── discovery_results/        # 发现结果
└── new_data/                 # 新增数据源（CIF结构文件）
```

## License

See [LICENSE](LICENSE).
