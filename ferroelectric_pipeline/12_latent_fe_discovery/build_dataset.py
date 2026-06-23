#!/usr/bin/env python3
"""
潜在铁电发现 (Latent Ferroelectric Discovery) — 数据集构建
=====================================================================
方法学创新 (A+B): 从高对称非极性母相**单独**预测
  (A) 是否存在极性失稳 + 失稳本征矢 (极性软模, 原子位移场)
  (B) 由此产生的(假想)极性相的自发极化 Ps (经 Born 电荷路径, 无极化量子歧义)
而**无需**计算极性结构、声子或 Berry 相 —— 从而在海量稳定非极性晶体中筛选
"隐藏的"铁电体。

本脚本从 Smidt et al. (Sci.Data 2020) 数据库构建训练样本:
  parent  : 高对称非极性母相结构 (structures[0])
  mode    : 极性畸变原子位移场 u_i = r_i(polar) - r_i(parent), 母相笛卡尔系 (l=1 等变靶)
  Ps      : 量子约化后的 Berry 相极化矢量 (μC/cm²)  —— 训练/验证 B 的端到端靶
  dw_depth: 双势阱深度 (meV/atom), 失稳强度的能量度量
  label=1 : 正样本 (该母相确有极性失稳)

负样本 (label=0, 稳定非极性、无极性失稳) 由 add_negatives.py 从 MP 拉取。

输出: dataset/ (parent 结构 json + modes.npz + labels.csv)
"""
from __future__ import annotations
import argparse, gzip, json, csv
from pathlib import Path
import numpy as np

E_TO_UCCM2 = 1602.176634


def polarization_quanta(lattice):
    V = lattice.volume
    return np.array([np.linalg.norm(lattice.matrix[i]) / V * E_TO_UCCM2 for i in range(3)])


def quantum_reduce(dP_vec, quanta):
    return dP_vec - np.round(dP_vec / quanta) * quanta


def build(src: Path, out: Path):
    from pymatgen.core import Structure
    out.mkdir(parents=True, exist_ok=True)
    (out / "parents").mkdir(exist_ok=True)
    records = json.load(gzip.open(src))
    rows, modes = [], []
    n = 0
    for r in records:
        if r.get("workflow_status") != "COMPLETED":
            continue
        try:
            parent = Structure.from_dict(r["structures"][0])
            polar = Structure.from_dict(r["structures"][-1])
            if len(parent) != len(polar):
                continue
            # 极性畸变原子位移场 (母相笛卡尔系, 最小镜像)
            du = polar.frac_coords - parent.frac_coords
            du -= np.round(du)
            u_cart = parent.lattice.get_cartesian_coords(du)            # (N,3)

            # Ps 矢量 (量子约化)
            sb = np.array(r["same_branch_polarization"])
            dP = sb[-1] - sb[0]
            quanta = polarization_quanta(polar.lattice)
            Ps_vec = quantum_reduce(dP, quanta)

            # 能量度量
            e = np.array(r["energies_per_atom"])
            dw = float((e[0] - e[-1]) * 1000.0)
            gap_min = float(min(g for g in r["bandgaps"] if g is not None))
        except Exception as exc:
            continue

        cid = r.get("cid", f"rec{n}")
        json.dump(parent.as_dict(), open(out / "parents" / f"{cid}.json", "w"))
        modes.append(u_cart.astype(np.float32))
        rows.append({
            "cid": cid, "formula": r.get("pretty_formula", ""),
            "natoms": len(parent),
            "label": 1,
            "Ps_x": Ps_vec[0], "Ps_y": Ps_vec[1], "Ps_z": Ps_vec[2],
            "Ps_norm": float(np.linalg.norm(Ps_vec)),
            "dw_depth_meV": dw, "gap_min": gap_min,
            "mode_norm": float(np.linalg.norm(u_cart)),
            "max_disp_A": float(np.abs(u_cart).max()),
            "parent_file": f"parents/{cid}.json",
        })
        n += 1

    # 变长 mode 数组用 object 保存
    np.savez(out / "modes.npz", **{rows[i]["cid"]: modes[i] for i in range(len(rows))})
    keys = list(rows[0].keys())
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

    Ps = np.array([r["Ps_norm"] for r in rows])
    print(f"positives (latent FE parents): {len(rows)}")
    print(f"  Ps (μC/cm²): mean={Ps.mean():.2f} median={np.median(Ps):.2f} max={Ps.max():.2f}")
    print(f"  mode |u| (Å): mean={np.mean([r['mode_norm'] for r in rows]):.2f}")
    print(f"output -> {out}/labels.csv, modes.npz, parents/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path,
                    default=Path(__file__).resolve().parents[2] / "data_files" / "workflow_data.json.gz")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "dataset")
    args = ap.parse_args()
    build(args.src, args.out)


if __name__ == "__main__":
    main()
