#!/usr/bin/env python3
"""
质量与物理感知的铁电筛选 (改进筛选/生成阶段)
=====================================================================
解决核心问题: **极性是铁电的必要不充分条件**。原管线只用"极性空间群 +
分类器"筛选, 会放过大量物理上不可能或性质很差的候选。本模块在极性基础上
叠加物理必要条件与物性质量评分:

必要条件 (硬过滤, 不满足直接淘汰):
  1. 极性点群 (10 个极性点群之一)
  2. 非金属: 带隙 ≥ 0.01 eV (金属不可能是铁电; 复刻论文 10 meV 判据)
  3. 可切换: 存在高对称非极性母相 (群-子群关系)

质量评分 (满足必要条件后排序, 我们要"好"的铁电而非"任意极性体"):
  - Ps 自发极化 (越大越好, 决定存储密度/压电响应)
  - 切换能垒在可切换窗口 (太高难切换, 太低不稳定)
  - 带隙适中 (绝缘性好但非过宽, 影响击穿/漏电)
  - 极性相为基态 (dw_depth > 0)

物性来自 10_property_regression 的 GNN 预测 (无 DFT 时) 或 DFT 验证结果。
最终 top-K 候选送 09_dft_validation 做第一性原理确认 (闭环主动学习)。

依赖: pymatgen (对称性); 可选 torch (载入 GNN 预测物性)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

try:
    from pymatgen.core import Structure
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    _HAS_PMG = True
except Exception:
    _HAS_PMG = False

POLAR_POINT_GROUPS = {"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"}

# 物性质量窗口 (可调; 默认基于 DFT 数据库分布与器件经验)
GAP_MIN_EV = 0.1            # 绝缘下限 (高于金属判据, 留裕量)
GAP_MAX_EV = 6.0            # 过宽带隙器件意义有限
PS_GOOD = 5.0              # μC/cm², 数据集中位数; 以上视为有价值
BARRIER_LOW = 5.0          # meV/atom, 低于此可能不稳定 (室温易退极化)
BARRIER_HIGH = 300.0       # meV/atom, 高于此难以电场切换


@dataclass
class Candidate:
    cid: str
    structure: dict = field(default=None, repr=False)
    point_group: str = ""
    space_group: int = 0
    is_polar: bool = False
    band_gap: float = None         # eV (GNN 或 DFT)
    Ps: float = None               # μC/cm²
    barrier_meV: float = None      # meV/atom
    dw_depth_meV: float = None     # meV/atom
    has_nonpolar_parent: bool = None


def assess_symmetry(struct, symprec=0.1):
    sga = SpacegroupAnalyzer(struct, symprec=symprec)
    pg = sga.get_point_group_symbol()
    return pg, sga.get_space_group_number(), pg in POLAR_POINT_GROUPS


def has_nonpolar_parent(struct, symprecs=(0.2, 0.3, 0.5, 0.8, 1.0)):
    """伪对称检测是否存在更高对称的非极性母相 (可切换性必要条件)。"""
    base = SpacegroupAnalyzer(struct, symprec=0.1).get_space_group_number()
    for sp in symprecs:
        sga = SpacegroupAnalyzer(struct, symprec=sp)
        if sga.get_space_group_number() > base and \
           sga.get_point_group_symbol() not in POLAR_POINT_GROUPS:
            return True
    return False


def hard_filter(c: Candidate) -> tuple[bool, list[str]]:
    """必要条件硬过滤。返回 (是否通过, 失败原因列表)。"""
    reasons = []
    if not c.is_polar:
        reasons.append("non-polar point group")
    if c.band_gap is not None and c.band_gap < 0.01:
        reasons.append(f"metallic (gap={c.band_gap:.3f} eV < 0.01)")
    if c.has_nonpolar_parent is False:
        reasons.append("no high-symmetry nonpolar parent (not switchable)")
    return (len(reasons) == 0), reasons


def quality_score(c: Candidate) -> float:
    """0-100 综合质量评分 (满足必要条件后)。物性缺失时该项按中性 0.5 计。"""
    def band(x, lo, hi):
        if x is None:
            return 0.5
        return 1.0 if lo <= x <= hi else max(0.0, 1 - min(abs(x-lo), abs(x-hi)) / (hi-lo+1e-9))

    s_ps = 0.5 if c.Ps is None else float(1 / (1 + np.exp(-(c.Ps - PS_GOOD) / 5.0)))  # sigmoid, Ps↑→分↑
    s_gap = band(c.band_gap, GAP_MIN_EV, GAP_MAX_EV)
    s_bar = band(c.barrier_meV, BARRIER_LOW, BARRIER_HIGH)
    s_dw = 0.5 if c.dw_depth_meV is None else (1.0 if c.dw_depth_meV > 0 else 0.2)
    # 权重: 极化最重要, 其次可切换性(能垒/基态), 再带隙
    score = 100 * (0.40 * s_ps + 0.20 * s_bar + 0.20 * s_dw + 0.20 * s_gap)
    return round(score, 1)


def screen(candidates: list[Candidate]):
    passed, rejected = [], []
    for c in candidates:
        ok, reasons = hard_filter(c)
        if ok:
            passed.append((quality_score(c), c))
        else:
            rejected.append((c, reasons))
    passed.sort(key=lambda x: -x[0])
    return passed, rejected


def build_candidate(cid, structure_dict, properties: dict | None = None, symprec=0.1):
    """从结构 + 可选物性 (GNN/DFT) 构建候选并完成对称性分析。"""
    st = Structure.from_dict(structure_dict)
    pg, sg, polar = assess_symmetry(st, symprec)
    props = properties or {}
    return Candidate(
        cid=cid, structure=structure_dict, point_group=pg, space_group=sg, is_polar=polar,
        band_gap=props.get("band_gap"), Ps=props.get("Ps"),
        barrier_meV=props.get("barrier_meV"), dw_depth_meV=props.get("dw_depth_meV"),
        has_nonpolar_parent=has_nonpolar_parent(st) if polar else False,
    )


def main():
    ap = argparse.ArgumentParser(description="Quality + physics-aware ferroelectric screening")
    ap.add_argument("input", type=Path, help="JSON: [{cid, structure, properties?}, ...]")
    ap.add_argument("--out", type=Path, default=Path("screened_candidates.json"))
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()
    if not _HAS_PMG:
        raise SystemExit("需要 pymatgen: conda activate fe_dft")

    data = json.load(open(args.input))
    cands = [build_candidate(d["cid"], d["structure"], d.get("properties")) for d in data]
    passed, rejected = screen(cands)

    print(f"输入 {len(cands)} | 通过必要条件 {len(passed)} | 淘汰 {len(rejected)}")
    print(f"\nTop {min(args.top, len(passed))} 优质铁电候选:")
    print(f"{'rank':<5}{'cid':<28}{'score':>7}{'pg':>6}{'Ps':>8}{'gap':>7}")
    out = []
    for i, (score, c) in enumerate(passed[:args.top]):
        print(f"{i+1:<5}{c.cid:<28}{score:>7}{c.point_group:>6}"
              f"{(c.Ps or 0):>8.1f}{(c.band_gap or 0):>7.2f}")
        out.append({"rank": i+1, "score": score, **{k: v for k, v in asdict(c).items() if k != 'structure'}})
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n-> {args.out} (top-K 送 09_dft_validation 做第一性原理确认)")


if __name__ == "__main__":
    main()
