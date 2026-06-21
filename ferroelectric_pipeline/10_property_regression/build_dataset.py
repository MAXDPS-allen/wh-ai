#!/usr/bin/env python3
"""
从 Smidt et al. (Sci. Data 2020) 第一性原理铁电数据库构建回归数据集
=====================================================================
输入: data_files/workflow_data.json.gz  (413 对极性/非极性结构的 DFT 工作流数据)
输出: 每个 COMPLETED 候选的结构 + 关键铁电物性回归标签

回归目标 (key ferroelectric metrics):
  - Ps           自发极化 |P(polar) - P(nonpolar)| (同一分支)        [μC/cm²]
  - dw_depth     双势阱深度 E(nonpolar) - E(polar)                  [meV/atom]  (正=极性相为基态=好铁电)
  - path_barrier 沿畸变路径的能垒 max(E_path) - E(polar)            [meV/atom]
  - gap_polar    极性相带隙                                          [eV]
  - gap_min      路径上最小带隙 (金属性判据: <0.01 → 路径含金属)     [eV]
辅助分类标签:
  - is_switchable   能量/极化路径平滑且单分支 (高质量铁电)           {0,1}

路径索引约定 (已由 static_task_labels 验证):
  index 0 = 非极性(高对称), index 9 = 极性; same_branch_polarization 从 ~0 升到 Ps。

用法:
  python build_dataset.py --src ../../data_files/workflow_data.json.gz --out dataset/
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

# 论文判据
METAL_GAP_EV = 0.01            # DFT-PBE 带隙 < 10 meV 判为金属
POL_SMOOTH_TOL = 0.1           # 极化-样条偏差 μC/cm²
ENERGY_SMOOTH_TOL = 0.01       # 能量-样条偏差 eV


def load_records(src: Path):
    with gzip.open(src) as f:
        data = json.load(f)
    return data


def extract_targets(rec: dict) -> dict:
    """从单条 workflow 记录提取回归标签与质量标志。"""
    sb = np.array(rec["same_branch_polarization"], dtype=float)        # (10, 3) μC/cm²
    e = np.array(rec["energies_per_atom"], dtype=float)                # (10,)   eV/atom
    gaps = np.array([g if g is not None else 0.0 for g in rec["bandgaps"]], dtype=float)

    # 自发极化: 极性端点相对非极性端点 (同一极化分支)
    Ps = float(np.linalg.norm(sb[-1] - sb[0]))

    # 能量 (相对极性相, meV/atom)
    e_rel = (e - e[-1]) * 1000.0
    dw_depth = float(e_rel[0])                     # 非极性 - 极性
    path_barrier = float(e_rel.max())              # 路径最高点 - 极性

    gap_polar = float(gaps[-1])
    gap_min = float(gaps.min())

    # 质量/可切换性: 论文的平滑度判据
    pol_smooth = rec.get("polarization_smoothness", [None, None, None])
    pol_smooth_max = max([s for s in pol_smooth if s is not None], default=np.inf)
    e_smooth = rec.get("energies_per_atom_smoothness", np.inf)
    is_switchable = int(
        (pol_smooth_max < POL_SMOOTH_TOL)
        and (e_smooth < ENERGY_SMOOTH_TOL)
        and (gap_min >= METAL_GAP_EV)
    )

    return {
        "Ps": Ps,
        "polarization_change_norm": float(rec.get("polarization_change_norm", Ps)),
        "dw_depth": dw_depth,
        "path_barrier": path_barrier,
        "gap_polar": gap_polar,
        "gap_min": gap_min,
        "is_switchable": is_switchable,
        "pol_smoothness_max": float(pol_smooth_max),
        "energy_smoothness": float(e_smooth),
    }


def build(src: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    struct_dir = out / "structures"
    struct_dir.mkdir(exist_ok=True)

    records = load_records(src)
    rows = []
    n_done = 0
    for rec in records:
        if rec.get("workflow_status") != "COMPLETED":
            continue
        n_done += 1
        try:
            tgt = extract_targets(rec)
        except Exception as exc:  # 跳过缺字段的记录
            print(f"  skip {rec.get('pretty_formula','?')}: {exc}")
            continue

        cid = rec.get("cid", rec.get("wfid", f"rec{n_done}"))
        # 保存极性 / 非极性结构 (pymatgen dict, 供后续等变 GNN 特征化)
        polar = rec["structures"][-1]
        nonpolar = rec["structures"][0]
        with open(struct_dir / f"{cid}_polar.json", "w") as f:
            json.dump(polar, f)
        with open(struct_dir / f"{cid}_nonpolar.json", "w") as f:
            json.dump(nonpolar, f)

        row = {
            "cid": cid,
            "formula": rec.get("pretty_formula", ""),
            "polar_id": rec.get("polar_id", ""),
            "nonpolar_id": rec.get("nonpolar_id", ""),
            "polar_spacegroup": rec.get("polar_spacegroup", None),
            "nonpolar_spacegroup": rec.get("nonpolar_spacegroup", None),
            "structure_file": f"structures/{cid}_polar.json",
            **tgt,
        }
        rows.append(row)

    # 写 CSV (不依赖 pandas, 保持轻量)
    import csv
    keys = list(rows[0].keys())
    with open(out / "regression_dataset.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # 汇总统计
    Ps_arr = np.array([r["Ps"] for r in rows])
    gap_arr = np.array([r["gap_polar"] for r in rows])
    sw = np.array([r["is_switchable"] for r in rows])
    print(f"\n{'='*60}")
    print(f"COMPLETED workflows: {n_done} | usable rows: {len(rows)}")
    print(f"Ps (μC/cm²):  mean={Ps_arr.mean():.2f}  median={np.median(Ps_arr):.2f}  max={Ps_arr.max():.2f}")
    print(f"gap_polar(eV): mean={gap_arr.mean():.2f}  min={gap_arr.min():.2f}  max={gap_arr.max():.2f}")
    print(f"high-quality switchable: {int(sw.sum())}/{len(rows)}")
    print(f"output -> {out/'regression_dataset.csv'}  (+ {struct_dir})")
    print('='*60)


def main():
    ap = argparse.ArgumentParser(description="Build FE-property regression dataset from DFT database")
    ap.add_argument("--src", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data_files" / "workflow_data.json.gz")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "dataset")
    args = ap.parse_args()
    build(args.src, args.out)


if __name__ == "__main__":
    main()
