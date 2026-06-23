#!/usr/bin/env python3
"""
端到端测试: 从 MP 找"未标注为铁电的热稳定材料"作为铁电候选
=====================================================================
1. 从 Materials Project 查询: 极性空间群 + 热力学稳定 + 绝缘 + 小胞
2. 排除已被标注为铁电的材料 (Smidt et al. DFT 数据库中的 polar_id)
3. 用工具包筛选: 伪对称找非极性母相 (s1) + 质量过滤 (11)
4. 输出 3-5 个最有希望的候选, 供 DFT 验证 (09)
"""
import json, gzip, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "09_dft_validation"))
sys.path.insert(0, str(HERE / "11_quality_screening"))

from mp_api.client import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

API_KEY = "1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq"
POLAR_PG = {"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"}


def polar_spacegroup_numbers():
    """枚举 68 个极性空间群编号 (其点群属极性点群)。"""
    from pymatgen.symmetry.groups import SpaceGroup
    nums = []
    for n in range(1, 231):
        try:
            pg = SpaceGroup.from_int_number(n).point_group
            if pg in POLAR_PG:
                nums.append(n)
        except Exception:
            pass
    return nums


def labeled_ferroelectric_ids():
    """已标注铁电的 MP id (来自本仓库的 Smidt DFT 数据库)。"""
    wf = HERE.parent / "data_files" / "workflow_data.json.gz"
    ids = set()
    for r in json.load(gzip.open(wf)):
        for k in ("polar_id", "nonpolar_id"):
            if r.get(k):
                ids.add(r[k])
    return ids


def main():
    polar_sgs = polar_spacegroup_numbers()
    fe_ids = labeled_ferroelectric_ids()
    print(f"极性空间群数: {len(polar_sgs)} | 已标注铁电 MP-id: {len(fe_ids)}")

    with MPRester(API_KEY) as m:
        docs = m.materials.summary.search(
            spacegroup_number=polar_sgs,
            energy_above_hull=(0, 0.02),         # 热力学稳定
            band_gap=(0.5, 5.0),                 # 绝缘体 (非金属), 适中带隙
            num_sites=(2, 12),                   # 小胞, DFT 可行
            fields=["material_id", "formula_pretty", "energy_above_hull",
                    "band_gap", "symmetry", "structure", "nsites", "theoretical"],
        )
    print(f"MP 命中 (稳定+极性+绝缘+小胞): {len(docs)}")

    cands = []
    for d in docs:
        mid = str(d.material_id)
        if mid in fe_ids:
            continue                              # 排除已标注铁电
        if d.theoretical:
            continue                              # 优先实验已知 (ICSD) 结构
        cands.append(d)
    print(f"排除已标注铁电 + 仅理论结构后: {len(cands)}")

    # 用工具包检查"可切换性": 是否存在更高对称非极性母相
    from s1_find_nonpolar_parent import find_nonpolar_parent
    selected = []
    # 按带隙适中、胞小排序优先
    cands.sort(key=lambda d: (d.nsites, abs(d.band_gap - 2.0)))
    for d in cands:
        st = d.structure
        try:
            parent, info = find_nonpolar_parent(st)
        except Exception:
            parent = None
        if parent is None:
            continue
        selected.append((d, parent, info))
        print(f"  [{len(selected)}] {d.formula_pretty:12} {d.material_id:12} "
              f"gap={d.band_gap:.2f} nsites={d.nsites} "
              f"polar#{info['polar_sg']}->nonpolar#{info['nonpolar_sg']}")
        if len(selected) >= 5:
            break

    out_dir = HERE / "test_run"
    out_dir.mkdir(exist_ok=True)
    manifest = []
    for d, parent, info in selected:
        cid = f"{d.formula_pretty}_{d.material_id}".replace(" ", "")
        cdir = out_dir / cid
        cdir.mkdir(exist_ok=True)
        json.dump(d.structure.as_dict(), open(cdir / "polar.json", "w"))
        json.dump(parent.as_dict(), open(cdir / "nonpolar.json", "w"))
        json.dump(info, open(cdir / "pair_info.json", "w"), indent=2, default=str)
        manifest.append({"cid": cid, "mp_id": str(d.material_id),
                         "formula": d.formula_pretty, "band_gap": d.band_gap,
                         "e_above_hull": d.energy_above_hull, "nsites": d.nsites,
                         "polar_sg": info["polar_sg"], "nonpolar_sg": info["nonpolar_sg"]})
    json.dump(manifest, open(out_dir / "candidates.json", "w"), indent=2)
    print(f"\n选定 {len(manifest)} 个候选 -> {out_dir}/candidates.json")


if __name__ == "__main__":
    main()
