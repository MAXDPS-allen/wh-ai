# 09 · 第一性原理 (DFT) 铁电验证管线

复刻 **Smidt, Mack, Reyes-Lillo, Jain, Neaton, _Scientific Data_ 7, 72 (2020)**
([An automatically curated first-principles database of ferroelectrics](https://www.nature.com/articles/s41597-020-0407-9))
的工作流，用本集群的 **VASP 6.4.2 (CUDA)** 对 ML 预测/生成的候选做第一性原理验证。

> 本仓库 `data_files/workflow_data.json.gz` 即该论文发布的数据集；本管线让我们能
> 用**相同的 DFT 工作流**验证新候选，形成"ML 预测 → DFT 验证 → 回流训练"的闭环。

## 物理判据

铁电的判定（极性是**必要不充分**条件）：

1. **极性**：极性点群（10 个之一）；
2. **绝缘**：DFT-PBE 带隙 ≥ 10 meV（金属不可能是铁电）；
3. **可切换**：存在高对称非极性母相，且二者间有连续极化路径；
4. **优质**（附加）：自发极化 \(P_s\) 大、切换能垒在可切换窗口、极性相为基态。

## 工作流阶段

| 阶段 | 脚本 | 功能 |
|------|------|------|
| S1 | `s1_find_nonpolar_parent.py` | 找/校验高对称非极性母相（群-子群；配对模式或伪对称自动模式） |
| S2 | `s2_interpolate.py` | 非极性↔极性间 8 个线性插值（共 10 个路径结构） |
| S3 | `s3_make_vasp_inputs.py` | 各阶段 VASP 输入（relax / static / polarization），复刻论文参数 |
| S4 | `s4_run_cluster.py` | SSH 派发到 GPU 节点 g1-g9 并行跑 VASP（无 SLURM） |
| S5 | `s5_postprocess.py` | Berry 相极化分支跟踪 → \(P_s\)/能垒/带隙 + 平滑度质量判据 |
| — | `validate.py` | 端到端编排（按论文顺序：弛豫→插值→金属性早停→极化→后处理） |
| — | `config.py` | 集群路径与 DFT 参数集中配置 |

## DFT 参数（复刻论文）

PBE-GGA + PAW；`ENCUT=520 eV`；`EDIFF=5e-5 eV/atom`；`EDIFFG=-5e-4 eV/Å`；
`ISPIN=2`（初始铁磁）；PBE+U（Materials Project 规则，由 pymatgen MP 输入集自动套用）；
k 点密度 弛豫 50、静态/极化 100 /(1/Å)³；非极性↔极性 8 个插值；带隙 < 10 meV 判金属。

## 环境准备

```bash
# 1) 创建 pymatgen 环境 (一次性)
conda create -y -n fe_dft python=3.11
conda activate fe_dft
pip install pymatgen ase scipy scikit-learn

# 2) 软链接 VASP PBE 赝势库成 pymatgen 期望布局 (一次性)
bash environment/setup_psp.sh

# 3) 校验路径
python config.py
```

集群约定（见 `config.py`）：
- VASP GPU：`/share/apps/vasp.6.4.2_nvhpc2311_hpcx_cuda11.8/bin/vasp_std`
- POTCAR(PBE)：`/share/apps/vasp/potentials/potpaw_PBE`（经 `psp_mirror/` 软链接）
- GPU 节点：g1–g9（免密 SSH + `CUDA_VISIBLE_DEVICES`）

## 用法

### 端到端（自动跑 VASP）

```bash
conda activate fe_dft
python validate.py candidate_polar.json --nonpolar candidate_nonpolar.json --cid mycand
```

### 仅准备输入（手动控制集群）

```bash
python validate.py candidate_polar.json --prepare-only --cid mycand
# 然后分阶段跑：
python s4_run_cluster.py runs/mycand/vasp --pattern 'relax_*'
python s4_run_cluster.py runs/mycand/vasp --pattern 'static_*' 'polar_*'
python s5_postprocess.py runs/mycand
```

### 探测可用 GPU（不跑计算）

```bash
python s4_run_cluster.py runs/mycand/vasp --dry-run
```

## 输出

`runs/<cid>/dft_validation_result.json`：

```json
{
  "cid": "...", "Ps_norm_uC_cm2": 23.1, "gap_min_eV": 0.95, "gap_polar_eV": 3.2,
  "dw_depth_meV": 45.0, "path_barrier_meV": 120.0,
  "is_ferroelectric": true, "is_high_quality": true,
  "quality_flags": {"polar_ground_state": true, "large_polarization": true}
}
```

## 验证状态

- S1–S5 + 编排器均已实现并在 LiNbO₃（极性 R3c #161 → 非极性 R-3c #167）上**端到端冒烟测试通过**（输入生成、POTCAR 解析、k 网格、LCALCPOL、GPU 探测）。
- S5 的 Berry 相分支跟踪用论文作者贡献的 `pymatgen.analysis.ferroelectricity`，并已与论文发布的极化值**对拍**（有意义的 \(P_s\) 量级一致；亚 μC/cm² 量级为分支噪声 ≈ 零极化）。
- 完整 VASP 计算需在 GPU 节点提交（单候选约 22 个计算，数小时量级），支持断点续跑。

## 已知限制

- 自动伪对称找母相（无配对时）不保证成功；**推荐配对模式**（提供非极性参考），与论文一致。
- 离子极化约定（`calc_ionic` 点电荷 vs VASP 原生）会带来小幅数值差异；生产路径直接解析 VASP OUTCAR。
- 极化分支跟踪在相邻像极化差超过极化量子时会失败，需加密插值（论文亦有此限制）。
