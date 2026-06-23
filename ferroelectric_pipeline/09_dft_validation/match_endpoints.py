#!/usr/bin/env python3
"""
端点匹配: 在极性胞内生成"去畸变"的高对称非极性参考
=====================================================================
DFT 畸变工作流要求极性/非极性两端有一一对应的原子 (同数、同序)。MP 的独立
非极性条目通常是不同晶胞设定 (原子数不同), 无法直接插值。

本模块改为在**极性结构自身的晶胞内**, 把原子位置对称化到更高对称 (非极性)
母群, 得到去畸变参考 —— 原子数与顺序天然匹配, 极化路径插值可直接进行。

方法 (伪对称去畸变):
  1. 逐步放宽 symprec 探测极性结构的高对称母群
  2. 取母群的对称操作, 对每个原子按其轨道平均分数坐标 -> 去畸变位置
  3. 校验得到的结构确为非极性点群

依赖: pymatgen, spglib, numpy
"""
from __future__ import annotations

import numpy as np
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

POLAR_PG = {"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"}


def symmetrize_to_parent(polar: Structure, symprecs=(0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)):
    """在极性胞内生成去畸变非极性参考。返回 (nonpolar_struct, info) 或 (None, info)。"""
    base_sga = SpacegroupAnalyzer(polar, symprec=0.05)
    base_sg = base_sga.get_space_group_number()
    base_pg = base_sga.get_point_group_symbol()

    frac = np.array(polar.frac_coords)
    for sp in symprecs:
        sga = SpacegroupAnalyzer(polar, symprec=sp)
        sg = sga.get_space_group_number()
        pg = sga.get_point_group_symbol()
        if sg <= base_sg or pg in POLAR_PG:
            continue
        ops = sga.get_symmetry_operations()           # 母群对称操作 (在该 symprec 下)
        # 对每个原子: 在其对称轨道上平均位置 -> 去畸变
        new_frac = frac.copy()
        for i, fc in enumerate(frac):
            orbit = []
            for op in ops:
                p = op.operate(fc) % 1.0
                # 折叠到与原子最近的镜像, 避免跨边界平均出错
                d = p - fc
                d -= np.round(d)
                orbit.append(fc + d)
            new_frac[i] = np.mean(orbit, axis=0) % 1.0
        nonpolar = Structure(polar.lattice, polar.species, new_frac,
                             coords_are_cartesian=False)
        try:
            np_sga = SpacegroupAnalyzer(nonpolar, symprec=0.05)
            np_pg = np_sga.get_point_group_symbol()
            np_sg = np_sga.get_space_group_number()
        except Exception:
            continue
        max_disp = float(np.abs((new_frac - frac) - np.round(new_frac - frac)).max())
        if np_pg not in POLAR_PG and len(nonpolar) == len(polar):
            return nonpolar, {
                "method": "in-cell symmetrization (undistortion)",
                "symprec_used": sp,
                "polar_sg": base_sg, "polar_pg": base_pg,
                "nonpolar_sg": np_sg, "nonpolar_pg": np_pg,
                "max_frac_displacement": max_disp,
                "natoms": len(polar),
                "valid": True,
            }
    return None, {"polar_sg": base_sg, "polar_pg": base_pg, "valid": False,
                  "reason": "no higher-symmetry nonpolar parent found in-cell"}


if __name__ == "__main__":
    import sys, json
    polar = Structure.from_dict(json.load(open(sys.argv[1])))
    nonpolar, info = symmetrize_to_parent(polar)
    print(json.dumps(info, indent=2, default=str))
    if nonpolar is not None:
        print("nonpolar natoms:", len(nonpolar), "== polar", len(polar))
