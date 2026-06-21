# 11 · 质量与物理感知筛选

解决用户指出的核心问题：**极性是铁电的必要不充分条件**。原 `01_data_acquisition`
仅用"极性空间群 + 分类器"筛选，会放过物理上不可能（金属）或性质很差的候选。
本模块在极性基础上叠加物理必要条件 + 物性质量评分。

## 判据

**硬过滤（必要条件，不满足直接淘汰）**
1. 极性点群（10 个之一）；
2. 非金属：带隙 ≥ 0.01 eV（金属不可能是铁电）；
3. 可切换：存在高对称非极性母相（伪对称检测）。

**质量评分（0–100，排序"优质"铁电）**
- \(P_s\) 自发极化（权重 0.40，越大越好）；
- 切换能垒在可切换窗口 5–300 meV/atom（权重 0.20）；
- 极性相为基态 `dw_depth>0`（权重 0.20）；
- 带隙适中 0.1–6 eV（权重 0.20）。

物性来自 `10_property_regression` 的 GNN 预测（无 DFT 时）或 `09_dft_validation`
的 DFT 结果。物性缺失项按中性 0.5 计。

## 用法

```bash
conda activate fe_dft
python quality_filter.py candidates.json --top 50 --out screened.json
```

输入 `candidates.json`：
```json
[{"cid": "x1", "structure": {...pymatgen dict...},
  "properties": {"band_gap": 3.2, "Ps": 25.0, "barrier_meV": 120, "dw_depth_meV": 45}}]
```

## 在管线中的位置

```
01 筛选 / 04-05 生成  →  11 质量筛选 (硬过滤+评分)  →  top-K  →  09 DFT 验证  →  回流 10 重训
```

## 验证状态

已在真实结构上测试：输入 3 个极性铁电 + 3 个非极性参考相，正确淘汰全部非极性参考
（non-polar point group），仅保留满足必要条件者。

## 对生成阶段的建议（见 `IMPROVEMENT_PLAN.md` §2.3）

把 \(P_s\)/带隙作为 CVAE 的**条件变量**，用 `10` 的物性回归作为生成的**奖励/引导**，
使生成偏向"优质铁电"而非"任意极性体"。
