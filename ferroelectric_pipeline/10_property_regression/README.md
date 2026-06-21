# 10 · 铁电关键物性回归模型

针对原管线"只做是/否分类、不预测物性"的问题，新增**从结构直接回归铁电关键
指标**的模型，训练数据来自 Smidt et al. 2020 的 DFT 数据库（`workflow_data.json`，
255 条 COMPLETED 记录）。

## 回归目标

| 目标 | 含义 | 单位 | 来源字段 |
|------|------|------|----------|
| `Ps` | 自发极化 | μC/cm² | `same_branch_polarization` 端点差 |
| `dw_depth` | 双势阱深度（极性相相对非极性的稳定性） | meV/atom | `energies_per_atom` |
| `path_barrier` | 沿畸变路径的能垒 | meV/atom | `energies_per_atom` |
| `gap_polar` | 极性相带隙 | eV | `bandgaps` |
| `is_switchable` | 高质量可切换铁电（分类） | {0,1} | 平滑度判据 |

## 文件

| 文件 | 功能 |
|------|------|
| `build_dataset.py` | 从 `workflow_data.json.gz` 解析 → `dataset/regression_dataset.csv` + 结构 |
| `featurize.py` | 结构 → 图（GNN）/ 向量（基线）特征 |
| `train.py` | 基线：梯度提升多目标回归（CPU，分钟级，5-fold CV） |
| `model_gnn.py` | 生产：CGCNN 风格几何感知 GNN（多任务 + 异方差不确定性） |
| `train_gnn.py` | GNN 训练（CPU 可跑，GPU 更快；图缓存） |

## 用法

```bash
conda activate fe_dft
python build_dataset.py                       # 构建数据集 (255 条)
python train.py                               # 基线 CV 指标
python train_gnn.py --epochs 300 --device cuda  # GNN (GPU 节点)
```

## 实测指标（验证集 / 5-fold CV）

| 目标 | 基线 GBT (R²) | 几何 GNN (R²) | 说明 |
|------|--------------|--------------|------|
| `gap_polar` | 0.70 | 0.60 | 带隙偏组分性，两者都可学 |
| `dw_depth` | **−1.7** | **0.18** | 几何感知把负 R² 拉正 |
| `path_barrier` | **−0.03** | **0.10** | 同上，能垒需要几何 |
| `Ps` | 0.09 | −0.08 | **数据量瓶颈**（见下） |
| `is_switchable`(AUC) | 0.76 | **0.80** | 几何感知提升 |

## 关键结论（诚实评估）

1. **几何感知确有帮助**：能垒/势阱深度的 R² 从负转正、可切换性 AUC 提升，证明
   组分/晶格统计不足，必须用结构几何（印证 `IMPROVEMENT_PLAN.md` §1.2）。
2. **自发极化 \(P_s\) 仍难预测**：根因是**数据稀缺**（仅 255 条 DFT 标注）。这正是
   `09_dft_validation/` 闭环主动学习要解决的——用 DFT 验证不断扩充标注。
3. **架构洞见（后续改进）**：\(P_s\) 本质是极性↔非极性**位移场**的属性，而非极性
   结构单独的属性。当前模型仅输入极性结构，信息不足。改进方向：以
   **位移 (polar − nonpolar)** 或双结构对作为输入；并引入 E(3) 等变向量输出头直接
   回归极化矢量（团队 `bn_e3nn` 经验可复用）。

> 不确定性输出（异方差 log-variance）用于主动学习：优先把模型最不确定的候选送
> `09_dft_validation/` 做第一性原理验证，单位算力收益最大。
