#!/usr/bin/env python3
"""
端到端测试候选准备 (含稳健端点匹配)
=====================================================================
扫描 MP 极性稳定绝缘体 (排除已标注铁电), 仅保留能成功构建"匹配端点 +
可插值畸变路径"的候选 (这是 DFT 验证的前提)。优先位移型 (in-cell 去畸变),
其次原胞匹配。目标: 凑齐 5 个可直接进 09 DFT 验证的候选。
"""
import json, gzip, sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "09_dft_validation"))
from mp_api.client import MPRester
from pymatgen.core import Structure
from match_endpoints import symmetrize_to_parent

API_KEY = "1tIeczIIf3CycCZ5P7V6Z2zndcZeGgFq"
POLAR_PG = {"1", "2", "m", "mm2", "4", "4mm", "3", "3m", "6", "6mm"}
TARGET = 5


def polar_sgs():
    from pymatgen.symmetry.groups import SpaceGroup
    return [n for n in range(1, 231)
            if SpaceGroup.from_int_number(n).point_group in POLAR_PG]


def fe_ids():
    wf = HERE.parent / "data_files" / "workflow_data.json.gz"
    ids = set()
    for r in json.load(gzip.open(wf)):
        for k in ("polar_id", "nonpolar_id"):
            if r.get(k):
                ids.add(r[k])
    return ids


def try_match(polar: Structure):
    """返回 (nonpolar, images, info) 或 None。"""
    # 方法 A: in-cell 去畸变 (位移型, 最干净)
    nonpolar, info = symmetrize_to_parent(polar)
    if nonpolar is not None:
        try:
            imgs = nonpolar.interpolate(polar, nimages=9, interpolate_lattices=True,
                                        autosort_tol=0.5)
            return nonpolar, imgs, info
        except Exception:
            pass
    return None


def main():
    sgs, labeled = polar_sgs(), fe_ids()
    print(f"极性空间群 {len(sgs)} | 已标注铁电 {len(labeled)}")
    with MPRester(API_KEY) as m:
        docs = m.materials.summary.search(
            spacegroup_number=sgs, energy_above_hull=(0, 0.02),
            band_gap=(0.3, 6.0), num_sites=(2, 10), theoretical=False,
            fields=["material_id", "formula_pretty", "energy_above_hull",
                    "band_gap", "structure", "nsites"])
    pool = [d for d in docs if str(d.material_id) not in labeled]
    pool.sort(key=lambda d: d.nsites)        # 小胞优先
    print(f"候选池 (排除已标注铁电): {len(pool)}; 扫描匹配...")

    out = HERE / "test_run"; out.mkdir(exist_ok=True)
    manifest, n = [], 0
    for d in pool:
        if n >= TARGET:
            break
        res = try_match(d.structure)
        if res is None:
            continue
        nonpolar, imgs, info = res
        n += 1
        cid = f"{d.formula_pretty}_{d.material_id}".replace(" ", "")
        cdir = out / cid; cdir.mkdir(exist_ok=True)
        json.dump(d.structure.as_dict(), open(cdir / "polar.json", "w"))
        json.dump(nonpolar.as_dict(), open(cdir / "nonpolar.json", "w"))
        json.dump(info, open(cdir / "pair_info.json", "w"), indent=2, default=str)
        manifest.append({"cid": cid, "mp_id": str(d.material_id),
                         "formula": d.formula_pretty, "band_gap": round(d.band_gap, 3),
                         "e_above_hull": round(d.energy_above_hull, 4), "nsites": d.nsites,
                         "polar": f"{info['polar_pg']}#{info['polar_sg']}",
                         "nonpolar": f"{info['nonpolar_pg']}#{info['nonpolar_sg']}",
                         "max_frac_disp": round(info["max_frac_displacement"], 4)})
        print(f"  [{n}] {d.formula_pretty:12} {d.material_id:12} gap={d.band_gap:.2f} "
              f"{info['polar_pg']}#{info['polar_sg']}->{info['nonpolar_pg']}#{info['nonpolar_sg']}")
    json.dump(manifest, open(out / "candidates.json", "w"), indent=2)
    print(f"\n准备好 {len(manifest)} 个可验证候选 -> {out}/candidates.json")


if __name__ == "__main__":
    main()
