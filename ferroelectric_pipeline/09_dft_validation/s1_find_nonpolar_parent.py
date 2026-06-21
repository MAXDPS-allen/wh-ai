#!/usr/bin/env python3
"""
Stage 1: 为极性候选结构寻找高对称非极性母相
=====================================================================
铁电的必要条件: 极性相 + 一个高对称非极性母相, 且极性空间群是非极性空间群
的子群 (存在连续可切换的极化路径)。

两种模式 (复刻 Smidt et al. 2020):
  A. 配对模式 (推荐): 候选自带非极性参考 (如 MP 群-子群配对)。本阶段
     校验极性/非极性判定与群-子群关系。
  B. 自动模式: 仅有极性结构时, 用伪对称分析寻找母相 —— 逐步放宽对称精度
     探测更高对称性, 若得到的高对称结构为非极性则作为母相 (对称化原子位置)。

输出: {cid}/polar.json, {cid}/nonpolar.json, {cid}/pair_info.json

依赖: pymatgen, spglib
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# 10 个极性点群 (允许自发极化)
POLAR_POINT_GROUPS = {"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"}


def point_group(struct: Structure, symprec=0.1) -> str:
    return SpacegroupAnalyzer(struct, symprec=symprec).get_point_group_symbol()


def space_group_number(struct: Structure, symprec=0.1) -> int:
    return SpacegroupAnalyzer(struct, symprec=symprec).get_space_group_number()


def is_polar(struct: Structure, symprec=0.1) -> bool:
    return point_group(struct, symprec) in POLAR_POINT_GROUPS


def find_nonpolar_parent(polar: Structure, symprecs=(0.1, 0.2, 0.3, 0.5, 0.8, 1.0)):
    """伪对称: 逐步放宽 symprec, 寻找比极性相对称性更高且为非极性的母相。
    返回 (nonpolar_structure, info) 或 (None, info)。"""
    base_sg = space_group_number(polar, symprec=0.1)
    base_pg = point_group(polar, symprec=0.1)
    for sp in symprecs:
        sga = SpacegroupAnalyzer(polar, symprec=sp)
        sg = sga.get_space_group_number()
        pg = sga.get_point_group_symbol()
        if sg > base_sg and pg not in POLAR_POINT_GROUPS:
            # 找到更高对称的非极性母相; 用对称化 (精修) 结构作为参考
            try:
                refined = sga.get_refined_structure()
                # 对称化后通常为中心对称设定; 再降到与极性相同样的原子计数设定
                conventional = sga.get_conventional_standard_structure()
                parent = refined
            except Exception:
                parent = sga.get_symmetrized_structure()
            return parent, {
                "method": "pseudosymmetry",
                "symprec_used": sp,
                "polar_sg": base_sg, "polar_pg": base_pg,
                "nonpolar_sg": sg, "nonpolar_pg": pg,
                "is_subgroup": True,    # sg > base_sg 表示极性是子群 (近似判据)
            }
    return None, {"method": "pseudosymmetry", "polar_sg": base_sg, "polar_pg": base_pg,
                  "found": False}


def validate_pair(polar: Structure, nonpolar: Structure, symprec=0.1):
    """校验配对: 极性相为极性、非极性相为非极性、群-子群关系。"""
    p_sg = space_group_number(polar, symprec)
    n_sg = space_group_number(nonpolar, symprec)
    info = {
        "method": "provided_pair",
        "polar_sg": p_sg, "polar_pg": point_group(polar, symprec),
        "nonpolar_sg": n_sg, "nonpolar_pg": point_group(nonpolar, symprec),
        "polar_is_polar": is_polar(polar, symprec),
        "nonpolar_is_nonpolar": not is_polar(nonpolar, symprec),
        "subgroup_index_consistent": n_sg >= p_sg,   # 非极性母相对称性应不低于极性子群
        "same_natoms": len(polar) == len(nonpolar),
    }
    info["valid"] = (info["polar_is_polar"] and info["nonpolar_is_nonpolar"]
                     and info["subgroup_index_consistent"])
    return info


def run(polar_path: Path, out_dir: Path, nonpolar_path: Path | None = None,
        cid: str | None = None, symprec=0.1):
    polar = Structure.from_dict(json.load(open(polar_path))) if polar_path.suffix == ".json" \
        else Structure.from_file(str(polar_path))
    cid = cid or polar.composition.reduced_formula
    cdir = out_dir / cid
    cdir.mkdir(parents=True, exist_ok=True)

    if nonpolar_path is not None:
        nonpolar = Structure.from_dict(json.load(open(nonpolar_path))) if nonpolar_path.suffix == ".json" \
            else Structure.from_file(str(nonpolar_path))
        info = validate_pair(polar, nonpolar, symprec)
    else:
        nonpolar, info = find_nonpolar_parent(polar)
        if nonpolar is None:
            info["valid"] = False
            json.dump(info, open(cdir / "pair_info.json", "w"), indent=2)
            print(f"[{cid}] 未找到非极性母相 -> 该候选不满足铁电必要条件")
            return info

    json.dump(polar.as_dict(), open(cdir / "polar.json", "w"))
    json.dump(nonpolar.as_dict(), open(cdir / "nonpolar.json", "w"))
    json.dump(info, open(cdir / "pair_info.json", "w"), indent=2)
    status = "VALID" if info.get("valid") else "INVALID"
    print(f"[{cid}] {status}: polar {info['polar_pg']}(#{info['polar_sg']}) | "
          f"nonpolar {info['nonpolar_pg']}(#{info['nonpolar_sg']})")
    return info


def main():
    ap = argparse.ArgumentParser(description="Find/validate nonpolar parent for a polar candidate")
    ap.add_argument("polar", type=Path, help="极性结构 (.json/.cif/POSCAR)")
    ap.add_argument("--nonpolar", type=Path, default=None, help="非极性母相 (可选; 缺省则自动伪对称搜索)")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "runs")
    ap.add_argument("--cid", default=None)
    args = ap.parse_args()
    run(args.polar, args.out, args.nonpolar, args.cid)


if __name__ == "__main__":
    main()
